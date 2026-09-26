# AutoResearcher — Design Spec

**Date:** 2026-09-17
**Status:** Draft for review
**Scope:** `AutoResearcher/` (this repo). Reads from `../WorldModel` (AlayaWorld) and `../WBench`; never modifies them.

---

## 1. Goal

AutoResearcher is a recursively self-improving multi-agent system that improves the
AlayaWorld world model's **WBench (full split)** performance **through training-data
construction**. The loop grows a tree of nodes; every node is a new version of the agent
code (produced by its parent's self-edit) plus a training recipe (data + data-coupled
training config), trained from the **released base checkpoint** and scored on a cheap
stratified **40-case WBench proxy**.

The tree and parent selection draw on HGM (clade-metaproductivity: value a node by its
subtree) and HyperAgents (score- and child-count-weighted parent sampling; a
self-referential agent whose meta level can rewrite itself). Both codebases were read:
HGM `hgm.py`, `tree.py`, `hgm_utils.py`, `self_improve_step.py`; HyperAgents
`generate_loop.py`, `select_next_parent.py`, `meta_agent.py`, `task_agent.py`,
`run_meta_agent.py`, `utils/gl_utils.py`, `utils/git_utils.py`, `utils/docker_utils.py`,
`agent/llm_withtools.py`.

### 1.1 Hard rules

1. **Data-construction focus.** Improvements must come from training data. Hyperparameter
   changes are allowed only together with a data change; architecture changes are not
   allowed. Model code, sampler and benchmark are never modified by agents.
2. **One training interface.** Training uses only the WorldModel standard fine-tuning
   interface (`WorldModel/docs/TRAINING.md`): the three standard data formats, declared
   under `data.datasets`, checked by `scripts/tools/check_dataset.py`, prompts cached by
   `scripts/tools/precache_train_text_embeds.py`, launched by
   `scripts/finetune/lowcompute_4x4090.sh`. No other data format is accepted.
3. **Kernel/agent split.** The kernel holds mechanism only (archive/tree, selection, VCS,
   sandbox, contract verification, training/eval execution, minimal privileged tools,
   telemetry). It contains no agent logic. The agent layer is fully editable by agents,
   including prompts, graphs and tools.
4. **Fixed entry points.** Every agent version exposes `edit_self` and `improve_recipe`,
   verified programmatically after every self-edit.
5. **Isolation.** Agents never touch WBench, the model inference path or the kernel.
6. **No chaining.** Every node trains from the same released checkpoint.
7. **Proxy only.** The loop scores nodes on the 40-case proxy only. A full WBench run is a
   manual command, never triggered by the loop.
8. **Telemetry is the most critical component.** Every input/output that could be useful
   is recorded, at choke points agents cannot reach.
9. **Frameworks.** The single-agent harness is an explicit LangGraph ReAct graph in the
   agent's own code: it reproduces `langchain.agents.create_agent` (no middleware), plus two
   additions: tool errors are reported to the model, and Claude-Code-style auto-compaction
   (§9.1.2). Multi-agent orchestration is plain Python. Both are agent code, so
   self-improvement can change them. Agent LLMs are reached through an OpenAI-compatible
   Chat Completions endpoint (e.g. OpenAI or DeepSeek) at `OPENAI_BASE_URL`, via
   `langchain-openai`'s `ChatOpenAI` pointed at the gateway; a provider's `reasoning_content`
   is kept and sent back when it returns one, and the gateway enforces `OPENAI_EFFORT`. *(Amended 2026-09-21, Plan 2 revision: the OpenAI Agents SDK is dropped. Amended 2026-09-24, Plan 2 Task 19: Chat Completions instead of the Responses API, any OpenAI-compatible provider.)*

### 1.2 Non-goals

- Changing AlayaWorld architecture, the DMD sampler, WBench code, or WBench cases.
- Data formats other than the three standard formats.
- Concurrent node expansion (nodes run strictly one at a time).
- Using any generator whose license forbids training AlayaWorld on its outputs
  (e.g. MiniMax-H3, HunyuanVideo-1.5) or whose native output is below 24 fps
  (e.g. Wan 2.2 A14B at 16 fps).

---

## 2. Environment and prerequisites

| Item | Value |
|---|---|
| Project root | `/mnt/biometrics/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch` |
| Disk | `/mnt/biometrics`, ~2.4 TB free at design time |
| GPUs | This machine: 6x RTX 4090 (24 GB). **GPU policy (identical for every GPU task: precache, training, rendering, eval, rollouts, camera annotation):** if `CUDA_VISIBLE_DEVICES` is set when the loop starts, the kernel uses exactly that list; if unset it falls back to `gpus.default` from `kernel.yaml`. Any indices are allowed. Fewer than `gpus.min_count` GPUs is refused at run start. Nothing in the kernel assumes a GPU count, index set or card model: the list length flows into the rank count (`train.sh`), `--gpus` for WBench, and the two-GPU slice used for prompt precache; larger or different hardware needs only the config values changed. |
| Host RAM | 251 GB |
| Conda envs (existing) | `alayaworld` (training, rendering, rollouts, data checks), `wbench-main` (WBench metrics), `wbench-vp` (visual plausibility) |
| Conda env (new) | `autoresearcher` (Python 3.12): kernel, gateway, tool server, dashboard |
| Docker | 27.3.1, user in `docker` group, `nvidia` runtime available (agent containers are CPU-only) |
| Secrets | `AutoResearcher/.env` (mode 600, git-ignored), symlinked as `WorldModel/.env` and `WBench/.env` |
| WorldModel revision | Recorded per run (git SHA + `git status --porcelain` hash). The kernel warns at run start if WorldModel has uncommitted changes. |

### 2.1 `.env` keys

| Key | Use |
|---|---|
| `OPENAI_API_KEY` | Upstream key, held only by the gateway (never passed into containers). |
| `OPENAI_BASE_URL` | Upstream base URL of an OpenAI-compatible API (default `https://api.openai.com/v1`). Used exactly as given: the endpoint path (`/chat/completions`, `/responses`) is appended, nothing else (OpenAI's base includes `/v1`; other providers may have no version segment). |
| `OPENAI_MODEL` | Default model for agents that do not choose one. |
| `OPENAI_EFFORT` | Optional reasoning effort (e.g. `low`). When set, the gateway overrides every forwarded request with it (`reasoning_effort` for Chat Completions, `reasoning.effort` for Responses); empty leaves requests untouched. |
| `VLM_API_KEY`, `VLM_API_URL`, `VLM_MODEL_NAME` | WBench VLM metrics. The 6 VLM metrics are computed **iff `VLM_API_KEY` is non-empty**. |

Existing shell variables (e.g. `HF_TOKEN`) take precedence; loaders never override them.

*(Amended 2026-09-25, Follow-up B: the gateway rejects, with 400 and never forwards, any chat
`messages` content part or Responses `input` item carrying video — type `video_url`, `video`,
`input_video`, or a `file`/`input_file` part whose mime type or data URL says video; images are
unaffected. The error text tells the agent to caption the clip (`caption_clip`) instead of
sending video frames/urls directly. See §13.1. Amended again 2026-09-25, Follow-up C: the text
now names the `caption_videos` kernel tool (§10); `caption_clip` is gone.)*

### 2.3 Upstream patches (applied by a human/Claude, never by agents)

Two small fixes to the sibling repos; both are bug fixes that stand on their own. Agents
never modify `WorldModel/` or `WBench/`.

| id | File | Change | Why |
|---|---|---|---|
| U1 | `WBench/tools/run_visual_plausibility.py` | Add `--work_dir` (default `work_dirs`, joined with `PROJECT_ROOT`, an absolute value winning) exactly as `main.py` does, and resolve `--model_path` against `PROJECT_ROOT`. | The script hardcodes relative `work_dirs/{model}/videos`, so it only works when run from the repo root and cannot score videos held anywhere else. |
| U2 | `WorldModel/scripts/finetune/lowcompute_4x4090.sh` | Replace the "refuse GPU 5" check with a count check: at least 4 entries in `CUDA_VISIBLE_DEVICES`, default `0,1,2,3` when unset. | GPU 5 is usable; what the recipe actually needs is 4+ GPUs. |

Both are verified by §16.3 items 9 and 10 before the first real run.

### 2.2 Weights

| Path | Content | Use |
|---|---|---|
| `WorldModel/weights/alaya-world-ar` | Released v1.1 stage2b (AR teacher) | Base for every node's training; AR rollouts |
| `WorldModel/weights/alaya-world-dmd` | Released v1.1 stage3 LoRA (4-step student) | Proxy rendering; fast rollouts |
| `WorldModel/weights/ltx-2.3` | LTX-2.3 base + Gemma text encoder | Training/rendering dependencies |
| `AutoResearcher/weights/ltx-2.5` | Lightricks LTX-2.5, dev + distilled (LTX-2.x Community License), 24 fps | Video-generation data source |
| `AutoResearcher/weights/wan2.2-ti2v-5b` | Wan 2.2 TI2V-5B (Apache-2.0), 720p 24 fps | Video-generation data source |

License notes: training AlayaWorld (a Derivative of LTX-2) on LTX-2.5 outputs is permitted
by the LTX-2.x license (the "train other models" restriction applies to commercial use and
exempts Derivatives of LTX-2; entities with >= $10M annual revenue need a commercial
license). Wan 2.2 is Apache-2.0 with no output restrictions.

### 2.4 Caches

Large caches stay inside the project (user rule, 2026-09-25): `ar_kernel.subproc.cache_env()`
sets `HF_HOME`, `XDG_CACHE_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `PIP_CACHE_DIR` and `UV_CACHE_DIR` on every
`run_in_env` child (an `extra_env` override still wins) and on the `ar` CLI process itself
at startup, all rooted at `cache_dir()` -- `AutoResearcher/.cache/` (gitignored) by default,
or `AR_CACHE_DIR` to relocate the whole tree; a new machine needs the captioner model
(`Qwen/Qwen3.8-27B-FP8`, ~29 GB) copied or downloaded into
`AutoResearcher/.cache/huggingface/hub/` before the captioner can start.

---

## 3. Glossary

| Term | Meaning |
|---|---|
| **Node** | One tree entry: `(agent code commit, data commit, recipe, proxy score, subtree value)`, plus status and artifacts. |
| **Cycle** | One pass of the loop producing one node: select → self-edit → improve recipe → train → eval → record. |
| **Attempt** | One try inside a retry loop (`edit_self#k`, `improve_recipe#k`). |
| **Lineage** | A node's chain of ancestors up to the root. |
| **Subtree** | A node plus all of its descendants. |
| **Clip (sample)** | One training clip: `video` (mp4), `caption` (json), optional `pose` (npz), plus metadata, provenance and format eligibility. |
| **Format** | One of `video_caption_camera`, `video_timed_prompts_camera`, `video_caption_static` (§6). |
| **Dataset** | A named group of clips in a data commit with a format, an optional `prompt_mode`, and a sampling weight. Becomes one `data.datasets` entry. |
| **Data commit** | An immutable, content-hashed manifest of datasets, with a parent commit. |
| **View** | The on-disk dataset roots materialized from a data commit for training. |
| **Recipe** | The node's training hyperparameters (`recipe.yaml`); combined with the data commit into the resolved training config. |

---

## 4. Architecture

```
                          ┌──────────────────────── KERNEL (not agent-editable) ─────────────────────────┐
  ar run / ar stop  ───►  │ control ─► loop ─► select ─► sandbox(edit_self) ─► contract ─┐ (retry ≤ N)    │
                          │                     ▲                                          │               │
                          │                     │        sandbox(improve_recipe) ─► gate ─┤ (retry ≤ N)    │
                          │                     │                                          ▼               │
                          │   archive (SQLite, agents.git, blob store)   train ─► merge ─► render ─► eval  │
                          │   telemetry (JSONL + payloads + SQLite index) ◄── every component              │
                          │   gateway (OpenAI proxy)      tools (MCP server)      dashboard                │
                          └───────▲──────────────────────────▲─────────────────────────────────────────────┘
                                  │ LLM calls                │ privileged tools
                          ┌───────┴──────────────────────────┴──────┐
                          │  AGENT CONTAINER (Docker, CPU-only)       │
                          │  ar_contract runner (read-only)           │
                          │  /agent: orchestration + ReAct harness    │
                          │  prompts, tools, knowledge, memory (edit) │
                          └──────────────────────────────────────────┘
```

### 4.1 Repository layout

```
AutoResearcher/
  kernel/                 # Python package ar_kernel (mechanism only)
    archive/  select/  sandbox/  gateway/  tools/  contract/  train/  eval/
    telemetry/  control/  dashboard/
  contract/               # Python package ar_contract (mounted read-only into containers)
  seed_agent/             # initial agent repo content (becomes the root node's code)
  docker/                 # agent base image
  configs/
    kernel.yaml           # all kernel defaults (§17)
    base_recipe.yaml      # copy of WorldModel/configs/examples/finetune_video_caption_camera.yaml
    proxy_cases.txt       # the 40 proxy case ids (§11.1)
  tests/
  weights/                # git-ignored
  runs/<run_id>/          # git-ignored: archive.db, agents.git, store/, nodes/, cache/, telemetry/
  docs/superpowers/specs/
  .env                    # git-ignored
```

### 4.2 Kernel components

| Component | Responsibility |
|---|---|
| `archive` | SQLite (`nodes`, `attempts`, `clips`, `blobs`, `data_commits`, `selection_events`), `agents.git`, content-addressed blob store. Single writer: the loop process. |
| `select` | Parent selection (§12). |
| `sandbox` | Docker runner, one container per call (§9.5). |
| `gateway` | OpenAI-compatible HTTP proxy to `OPENAI_BASE_URL`; full recording; per-container auth tokens; model allowlist (§13). |
| `tools` | MCP (streamable HTTP) server exposing privileged tools only (§10). |
| `contract` | Contract verification of child code (§9.4). |
| `train` | Recipe gate (§8), view materialization (§5.6), prompt precache and training launch through the standard interface (§7.3). |
| `eval` | Merge, proxy render, WBench phases, scoring, agent-facing aggregates (§11). |
| `telemetry` | Event bus and stores (§13). |
| `control` | CLI `ar run`, `ar run --resume`, `ar status`, `ar stop --graceful|--force`, `ar full-eval <node>`. |
| `dashboard` | Read-only FastAPI app on localhost (§13.4). |

---

## 5. Archive and data store

### 5.1 Node record (`nodes` table)

`node_id`, `parent_id`, `depth`, `created_at`, `status` (`running`, `scored`,
`invalid_code`, `invalid_recipe`, `train_failed`, `crashed`), `agent_commit`,
`data_commit`, `recipe_hash`, `recipe_path`, `resolved_config_path`, `score`,
`metric_set` (ordered metric names used), `metrics_json` (per-metric means),
`subtree_value` (§12), `checkpoint_path`, `lora_rank`, `lora_alpha`,
`phase_timings_json`, `attempt_counts_json`, `rationale_path`.

### 5.2 Agent code

`runs/<run>/agents.git` is a real git repository. The root node's commit is the content of
`seed_agent/`. A child's code is committed on branch `node/<id>`, branched from the
parent's commit. Every attempt (including failed ones) is committed under
`refs/attempts/<node>/<phase>-<k>` (e.g. `refs/attempts/n7/edit_self-2`). *(Amended
2026-09-24, Plan 2 as built: the phase is part of the ref, since both phases have attempts.)*
*(Amended 2026-09-25: attempt refs exist for `edit_self` attempts only. `improve_recipe`
mounts `/agent` read-only, so its attempts produce no code; the agent commit it ran is
recorded in telemetry as `code_commit` on its `phase.start` event.)*

### 5.3 Blob store

- Location: `runs/<run>/store/blobs/<kind>/<sha256>.<ext>` with `kind ∈ {video, caption,
  pose}` and `ext ∈ {mp4, json, npz}`.
- Written to a temp file, fsynced, renamed; then `chmod 0444`.
- Immutable and deduplicated by content hash. Containers mount the store read-only.

### 5.4 Clips

A **clip** row stores:

- `clip_id = sha256(canonical JSON {"video": h_v, "caption": h_c, "pose": h_p or null,
  "camera_motion": "moving"|"static"})`.
- File hashes by role. `pose` is required for `camera_motion: moving` and absent for
  `camera_motion: static`.
- `camera_motion`, declared by the agent at ingest. `static` means a truly fixed camera
  (tripod, surveillance); it is recorded as a declaration.
- Probed metadata: frame count, fps, width, height, duration, `has_segments`,
  `has_intrinsics`.
- **Format eligibility** (computed at ingest, §5.5): `video_caption_camera`,
  `video_timed_prompts_camera:segment`, `video_timed_prompts_camera:per_chunk`,
  `video_caption_static`, each with the checker's warnings.
- Provenance (required): `{"kind": "hf_dataset", "repo", "revision", "files"}`,
  `{"kind": "rollout", "generator": "alayaworld-dmd4|alayaworld-ar30|ltx-2.5-dev|ltx-2.5-distilled|wan2.2-ti2v-5b", "job_id", "inputs_hash", "seed"}`,
  or `{"kind": "derived", "from": [clip_ids], "transform": "<agent description>"}`.
- License (from the HF dataset card/tag or the generator's license), `derived_from`,
  `ingested_by_node`.

Modifying data never mutates a clip: a re-captioned, cropped, trimmed or re-posed clip is a
new clip with `derived_from` set.

### 5.5 Ingest pipeline (kernel tool `data.ingest`)

Agents write candidate files under `/workspace/staging/` and call `data.ingest` with a list
of candidates (`video`, `caption`, optional `pose` staging paths; `camera_motion`;
provenance). For each candidate the kernel:

1. Hashes files; dedupes against the store.
2. **Aspect-ratio check** (a TRAINING.md requirement the checker does not enforce):
   `width/height` within ±2% of 16/9, else rejected.
3. **Standard checker, per eligible format.** Builds a single-clip temporary root in the
   standard layout and runs `alaya.data.standard_check.check_dataset` (in the `alayaworld`
   env) with `layout_from_config(base recipe)`:
   - `camera_motion: moving` → formats `video_caption_camera`; plus
     `video_timed_prompts_camera` with `prompt_mode` `segment` and `per_chunk` if the
     caption has `segments`.
   - `camera_motion: static` → format `video_caption_static`.
   A format is eligible iff the report has zero errors. Errors and warnings are stored.
   A candidate eligible for no format is rejected with the checker's messages.
4. **Leakage check** (two signals must agree, to avoid false rejections on flat frames):
   for the candidate's first frame and frames at 25/50/75%, compare against the first
   frames of all 289 WBench cases using (a) 64-bit pHash Hamming distance <= 4 **and**
   (b) normalized cross-correlation of 32x32 grayscale >= 0.95. A frame whose Shannon
   entropy is below `leakage.min_entropy` (flat sky, white wall, dark corridor) cannot
   trigger a rejection on its own; it is logged as a near-match for review. Rejections and
   near-matches are always logged with both scores. The kernel reads WBench; agents never
   do. **Known limit:** WBench ships only each case's first frame, so later turns of
   multi-turn cases cannot be compared; this check reduces leakage, it does not prove its
   absence.
5. Moves blobs into the store, inserts the clip row, deletes the staging files.

Returns, per candidate, `accepted` + `clip_id` + eligible formats + warnings, or
`rejected` + reasons.

### 5.6 Data commits and views

- A **data commit** stores a manifest blob (canonical JSON, keys sorted):

  ```json
  {"datasets": {
     "<name>": {"format": "video_timed_prompts_camera", "prompt_mode": "per_chunk",
                "weight": 1.0, "clips": ["<clip_id>", "..."]}
  }}
  ```

  plus `parent_commit`, `node_id`, `attempt`, `message`, `created_at`.
  `commit_id = sha256(parent_commit || manifest_hash || message)`.
- **Validation at commit:** dataset names match `^[a-z][a-z0-9_]{0,39}$` and are not
  built-in source names; `prompt_mode` is present iff the format is
  `video_timed_prompts_camera`; `weight >= 0`; every clip is visible (below) and eligible
  for its dataset's format (and `prompt_mode`); at least one dataset has `weight > 0` and
  clips. A clip may appear in several datasets.
- **Weights** are the standard interface's sampling probabilities between datasets
  (`weight / sum(weights)`). Agents weight a subset by putting it in its own dataset.
- Agents may create **any number** of commits during `improve_recipe`; the node trains on
  the commit named in its result.
- **Visibility: an archive-wide pool.** Every node may select from every clip ingested by
  any node of the run, plus the clips it ingests itself, each with full provenance (which
  node ingested it, generator or HF source, license, transform chain, eligible formats) and
  the scores of nodes that trained on it. Data work is therefore never lost when a node
  scores badly or fails. Agents may freely select subsets, filter (including their own new
  clips) and regroup; parent commits are never changed. Selection and subtree values stay
  lineage-based (§12); only the data pool is shared.
- **View materialization** happens **before the recipe gate runs** (and inside
  `recipe.check`, into a temporary view that is discarded), because `check_dataset.py`,
  the precache dry run and `--describe` all read the dataset roots. The training phase
  reuses the view materialized by the gate. It is a hardlink farm (same ext4 filesystem,
  zero copy), one standard root per dataset:

  ```
  runs/<run>/nodes/<n>/view/<dataset>/videos/<clip_id>.mp4
  runs/<run>/nodes/<n>/view/<dataset>/captions/<clip_id>.json
  runs/<run>/nodes/<n>/view/<dataset>/poses/<clip_id>.npz      # camera formats only
  ```

### 5.7 Caches

- **Text-embedding cache**, shared across nodes of a run: `runs/<run>/cache/text_embed/`
  (keyed by `sha1(prompt)` in WorldModel). Grow-only.
- **Dataset sample-list cache:** per node, `ALAYA_DATASET_CACHE_DIR=runs/<run>/nodes/<n>/dataset_cache`
  (the loader's pickle is keyed by source name, not content, and dataset names repeat
  across nodes).

---

## 6. Data formats (the only accepted formats)

Source of truth: `WorldModel/docs/TRAINING.md` §1 and `alaya/data/standard.py`,
`alaya/data/standard_check.py`.

### 6.1 Layout and formats

```
<root>/videos/<id>.mp4
<root>/captions/<id>.json
<root>/poses/<id>.npz        # camera formats only
```

| Format | Video | Caption | Timed prompts | Extrinsics | Intrinsics |
|---|---|---|---|---|---|
| `video_caption_camera` | required | required | – | required | optional |
| `video_timed_prompts_camera` | required | required | required | required | optional |
| `video_caption_static` | required | required | – | fixed (identity) | – |

### 6.2 Requirements

- **Training window:** 57 frames at 24 fps (25 history + 32 target); rollout rounds are
  32 frames.
- **Video:** `.mp4`; fps >= 24 (1% tolerance; higher rates are subsampled); duration
  >= 2.375 s; aspect ratio close to 16:9 (frames are resized to `sample.width` x
  `sample.height` without cropping); any resolution.
- **Caption:** `captions/<id>.json` with a non-empty `"caption"` string.
- **Timed prompts:** `"segments": [{"time_range_s": [start, end), "prompt": "..."}]`,
  non-overlapping, `0 <= start < end <= duration`, non-empty prompts. `per_chunk` mode:
  every boundary inside the clip lies on a round boundary `25/24 + k·32/24 s`
  (within 0.5 frame) and at least one round has a prompt. `segment` mode: segments shorter
  than 2.375 s are never trained (warning).
- **Extrinsics:** `poses/<id>.npz` array `cam_c2w`, shape `[N,4,4]` with N = the mp4's
  frame count, camera-to-world, OpenCV convention, finite, bottom row `[0,0,0,1]`,
  orthonormal rotation with det +1 (tolerance 1e-2). Translation units as given, consistent
  within a clip.
- **Intrinsics:** optional `intrinsics` `[3,3]` or `[N,3,3]` in pixels of the mp4, fx, fy > 0,
  principal point at the image centre (5% tolerance, warning). Missing → warning, training
  assumes fx = width, fy = height.
- **Static format:** identity poses are used; any `poses/` directory is ignored.
- **Per run:** each enabled dataset needs at least as many clips as training GPUs.

### 6.3 Prompt modes

- `per_chunk`: windows start on round boundaries; each window trains with the prompt active
  at the midpoint of its target. Teaches mid-rollout prompt switches.
- `segment`: a window is placed inside one segment and trains with that segment's prompt,
  or with `caption` with probability `data.overall_caption_prob` (default 0.1).

---

## 7. Node lifecycle

One node at a time. GPU phases never overlap.

### 7.1 Bootstrap (run start)

1. Create `runs/<run_id>/`, snapshot `configs/`, record versions (kernel git SHA, WorldModel
   and WBench git SHAs and dirty flags, `pip freeze` of each env), resolve the GPU list (§2)
   and the **metric set** (§11.3).
2. Root node: code = `seed_agent/`, data commit = empty, recipe = none. Its score is the
   released checkpoint rendered and scored on the proxy without training.

### 7.2 Cycle

```
 1  select parent p                                         (§12)
 2  prepare child c: git branch from p; build /context bundle
 3  edit_self      : run p's code (container) on a copy of p's code        ┐ loop
 4  contract check : verify c's code (fresh container)                     ┘ ≤ N_edit (default 3)
 5  improve_recipe : run c's code (container); tools; data commits; recipe  ┐ loop
 6  recipe gate    : materialize view, then checks + dry runs (§8)           ┘ ≤ N_recipe (default 3)
 7  precache prompts; train on the GPU list, reusing the gate's view (§7.3)
 8  merge → render proxy (40 cases) → WBench phases → score (§11)
 9  record; update subtree values; delete transient files; store aggregates
10  stop checks (node count, graceful/force flags)
```

- **Retry semantics (3↔4, 5↔6):** a retry continues from the failed attempt's code or
  workspace; the failure report (verifier or gate output) is provided at
  `/context/retry.json`. The agent may fix or reset. Clips ingested during a failed recipe
  attempt remain in the archive-wide clip pool. *(Plan 2 as built: both phases take the
  failed attempt's workspace, so an `edit_self` retry keeps its edit plan; the workspace is
  copied, its stale `result.json` removed, and the attempt's staging directory is moved,
  not copied, because downloads can be tens of GiB.)*
- **Exhaustion:** `invalid_code` after `N_edit` failed attempts; `invalid_recipe` after
  `N_recipe`.
- **Recipe-caused training failures** (CUDA OOM, NaN/inf loss, training wall-time cap
  exceeded) return to step 5 and consume one of the same `N_recipe` attempts; on
  exhaustion the node is `train_failed`.

### 7.3 Training execution (standard interface only)

All commands run in `WorldModel/` in the `alayaworld` env.

0. **View** already materialized by the gate (§5.6).
1. **Resolved config** `runs/<run>/nodes/<n>/train_config.yaml` = base recipe, overlaid with
   the node's recipe (tunable keys, §8), plus kernel-injected keys:
   `data.sources: {}`; `data.datasets` generated from the data commit (name, `format`,
   `prompt_mode`, `weight`, `root` = absolute view path); `run.name = node_<n>`;
   `run.output_dir` and `run.log_dir` under `runs/<run>/nodes/<n>/train/`;
   `runtime.text_embed_cache_dir = runs/<run>/cache/text_embed`;
   `optimizer.checkpoint_steps = optimizer.max_steps`; `optimizer.max_checkpoints = 1`.
   The validation mode's `dataset.source` is set to the first dataset name
   (`validation.enabled` stays `false`).
2. **Check:** `python scripts/tools/check_dataset.py --config <resolved> --max-messages 0`
   (exit 0 required; already run by the gate).
3. **Precache** (requires a GPU list of at least 2 GPUs; the kernel refuses to start
   otherwise): `CUDA_VISIBLE_DEVICES=<first two GPUs of the list> ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" python scripts/tools/precache_train_text_embeds.py --config <resolved> --device-map auto`.
4. **Train:** `CONFIG_PATH=<resolved> CUDA_VISIBLE_DEVICES=<GPU list> LOG_FILTER=all ALAYA_LOG_MEMORY=1 ALAYA_DATASET_CACHE_DIR=<node dataset_cache> bash scripts/finetune/lowcompute_4x4090.sh`.
5. **Outputs:** the newest `<run.output_dir>/checkpoint-<step>/` (training stops at
   `optimizer.max_steps` or `optimizer.epochs`, whichever comes first) with
   `lora.safetensors` and `history_encoder.pt`, kept as the node checkpoint; a run that
   exits 0 without a checkpoint directory is a recipe-caused failure. `trainer_state.pt` is deleted after
   training. The training log is `<run.log_dir>/<config name>/train_node0_<timestamp>.log`.

---

## 8. Recipe gate

The agent's `recipe.yaml` contains only tunable keys. Data (datasets, formats, prompt modes,
weights, clips) lives in the data commit. The gate checks, in order:

1. **Tunable keys only.** Allowed keys:
   - `data.overall_caption_prob`
   - `optimizer.lr`, `optimizer.weight_decay`, `optimizer.max_grad_norm`,
     `optimizer.warmup_steps`, `optimizer.max_steps`, `optimizer.epochs`,
     `optimizer.grad_accum_steps`
   - `sample.height`, `sample.width` — pair in `train.resolution_allowlist`
   - `lora.rank`, `lora.alpha` — pair in `train.lora_allowlist`

   Any other key is rejected. Everything else comes from the base recipe unchanged,
   including `layout`, `spatial_memory`, `training.mode`, `next_forcing` (TRAINING.md: do
   not change for a fine-tune), `memory` (incl. `drop_prob`), `anti_drift`, `dmd`,
   `control`, `sample.fps`, `sample.temporal_stride`, `optimizer.batch_size` (must stay 1),
   `data.camera_norm_mode`, `lora.targets`, and the kernel-owned `paths`, `runtime`,
   `validation`, `run`.
2. **Data change:** the data commit's manifest hash differs from the parent's (root
   children: any valid non-empty commit).
3. **Schema:** the resolved config (§7.3.1) loads with `alaya.config.loader.load_config`
   (this also validates `data.datasets` through `parse_dataset_specs`).
4. **Clip count:** every dataset with `weight > 0` has at least as many clips as GPUs in
   the GPU list.
4b. **Step budget.** With `epoch_windows = max over enabled datasets of
   ceil(clips_d / share_d)` (the standard interface's epoch size),
   `steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps)`:
   - `steps_per_epoch >= 1` is required. Otherwise no optimizer step can ever complete, no
     matter how large `optimizer.epochs` is (12 clips on 4 GPUs with
     `grad_accum_steps: 4` gives 3 micro-batches per rank per epoch and 0 steps). The
     rejection message states the minimum: `epoch_windows >= n_gpus *
     optimizer.grad_accum_steps`, i.e. raise the clip count or lower
     `grad_accum_steps`.
   - `optimizer.epochs * steps_per_epoch >= optimizer.max_steps` is required, so training
     cannot stop early on the epoch limit.
   Both rejections report all four numbers.
5. **Checker:** `check_dataset.py --config <resolved> --max-messages 0` exits 0.
6. **Prompt enumeration:** `precache_train_text_embeds.py --config <resolved> --dry-run`
   exits 0.
7. **Describe:** `DESCRIBE=1 CONFIG_PATH=<resolved> CUDA_VISIBLE_DEVICES=<GPU list> bash scripts/finetune/lowcompute_4x4090.sh`
   exits 0.

Both diffs (data manifest vs. parent; recipe vs. parent and base) are stored on the node,
so near-pure hyperparameter changes are visible in analysis.

---

## 9. Agent layer

### 9.1 Seed agent repo (`seed_agent/`, fully editable by agents)

*(Amended 2026-09-21, Plan 2 revision: the OpenAI Agents SDK is dropped. Amended 2026-09-24,
Plan 2 as built: agent code prefers simplicity over everything, since every cycle reads and
rewrites it, so the seed is four modules instead of packages.)*

```
agent/
  entry.py            # def edit_self(ctx) -> EditResult ; def improve_recipe(ctx) -> RecipeResult,
                      #   plus the settings: model, context window, compaction threshold, round limits
  requirements.txt
  harness.py          # single-agent inner loop (§9.1.2): explicit LangGraph ReAct graph with
                      #   auto-compaction
  orchestration.py    # multi-agent workflow in plain Python: roles (a role = system prompt
                      #   + knowledge + tools, run on the harness), improve_recipe, edit_self
  prompts/            # one Markdown system prompt per role, plus compact.md
  tools.py            # agent-local tools (file read/list/write/edit, bash, ffmpeg/ffprobe;
                      #   timed-prompt helper that snaps segment boundaries to round
                      #   boundaries), the MCP -> LangChain adapter for kernel tools, and
                      #   submit_* result tools. Captioning is the kernel's caption.videos
                      #   GPU job (§10; Follow-up C removed the frame-sending caption_clip)
  knowledge/          # reference material that roles read: data_building.md (formats,
                      #   conversion recipes)
  memory/             # agent-owned notes (in the code repo, inherited by children): README.md
```

**Roles and results.** Each LLM step is a role run on the harness. A role returns its result
by calling a `submit_<x>` tool whose arguments are validated against a schema; invalid
arguments come back to the model as a tool error it can fix. A role that stops without
submitting gets one reminder, then fails the attempt.

**Seed orchestration:**
- **meta** (`edit_self`): an edit planner reads the code (read-only tools) and the history,
  and submits a plan for exactly one component (§9.1.1) → a coder implements it (file/bash
  tools on `/agent`) → self-test in the container (static + import) → fix if needed → finalize.
- **task** (`improve_recipe`): diagnose from aggregates and plan data hypotheses → build data
  (search HF / request rollouts → convert to a standard format: crop to 16:9, trim, caption,
  timed prompts on round boundaries, poses via `annotate.camera` → ingest → commit, repeatable)
  → write the recipe → `recipe.check` → loop back on failures.

There are no framework checkpoints: the kernel retries a failed attempt from its workspace
(§7.2), and the gateway records every model call.

#### 9.1.1 Edit components

An agent version has five components. Every `edit_self` plan names exactly one, with one
focused change, so the next node's score can be attributed to it:

| Component | Where (seed layout) | What changes |
|---|---|---|
| `prompts` | `agent/prompts/` | The system prompt of any role |
| `tools` | `agent/tools.py` | Agent-local tools and the kernel-tool adapter (never the kernel tools themselves) |
| `harness` | `agent/harness.py` | The single-agent inner loop: graph, tool execution, context management |
| `orchestration` | `agent/orchestration.py` (+ settings in `agent/entry.py`) | Which roles run, in what order, with which tools and context |
| `knowledge` | `agent/knowledge/` | Reference material roles read |

The choice is returned as `EditResult.component`, stored with the node and shown in the
lineage the next planner sees. It is **not enforced**: the kernel never rejects a diff that
touches other components. Notes in `memory/` are bookkeeping and may be written by any edit.

#### 9.1.2 Seed harness

`build_react_agent(model, tools, system_prompt, *, context_window, compact_at, compact_prompt)` behaves
exactly like `create_agent(model, tools, system_prompt=...)` from `langchain` 1.4.2 with no
middleware and no `response_format`: START → model → END if the last AI message has no tool
calls, otherwise one parallel `Send("tools", [call])` per call → model; tools are bound on
every model call; unknown tools and invalid arguments get `ToolNode`'s messages; recursion
limit 9999. A test compares both message for message and prompt for prompt. It differs in
exactly two ways:

1. **Tool errors.** A tool that raises is reported to the model as an error `ToolMessage`
   (`ToolNode`'s `handle_tool_errors=True` text) instead of ending the run.
2. **Auto-compaction.** Before every model call, inside the model node, the harness estimates
   the context on the merged state (after all parallel tool results are in): the last
   reply's reported `usage.total_tokens` plus about 4 characters per token for the messages
   after it, with each image block counted as a fixed 1,500 tokens. Below
   `compact_at × context_window` (`agents.compact_at`, default 0.85, and
   `agents.context_window_tokens`, passed to the container) messages append linearly. At or
   above it, a dedicated summarizer call writes a structured summary, the history is
   replaced by one user message (a continuation preamble plus the summary), and the model
   call runs on it. The graph itself stays exactly `create_agent`'s: there is no separate
   compaction node. *(Amended 2026-09-24, Plan 2 as built: a routing function on the edge
   from the tool node runs once per parallel `Send` branch, before their results merge, so
   it could compact beside a normal model call or skip a due compaction.)* *(Amended
   2026-09-25, Follow-up B: the summarizer call branches off the ORIGINAL conversation using
   the SAME `bound` model as the normal call (same system prompt, tools, `tool_choice=None`)
   with one instruction message (`prompts/compact.md`) appended — byte-identical to the
   agent's normal request up to that append, maximizing prompt-cache hits. A live probe
   found `tool_choice="none"` on the summarizer alone changes the provider-side prompt (a
   DeepSeek cache hit fell from 3200 to 2560 of ~3.3k tokens). If the model calls a tool
   instead of writing the summary, the call is retried once with `tool_choice="none"` to
   force text.)*

### 9.2 Fixed contract package (`ar_contract`, kernel-owned, read-only)

- Pydantic models: `EditContext`, `EditResult`, `RecipeContext`, `RecipeResult`.
- Client helpers: `chat_model()` (a `ChatOpenAI` bound to the gateway socket with the
  per-container token) and `mcp_session()` (an MCP client session on the tool server socket,
  with a 4 h read timeout: `recipe.check` can wait on the GPU lock and then run long gate
  steps, and `hf.download` can fetch 20 GiB; a shorter client timeout would make the agent
  retry work still in progress. The phase hard cap bounds real hangs).
- Runner: `python -m ar_contract.run <edit_self|improve_recipe>`. It imports `agent.entry`,
  calls the function, validates the result against the schema, and writes
  `/workspace/result.json`. The kernel does not trust that file (agent code can write it):
  it re-validates it against the same schema, and anything unreadable or invalid is a failed
  attempt, never a kernel exception. An agent-authored `requirements.txt` that fails to
  build is also a failed attempt.

### 9.3 Contexts and results

| | Receives | Returns |
|---|---|---|
| `EditContext` | `agent_dir=/agent` (rw copy of own code); lineage history (per ancestor: code diff, recipe diff, data manifest stats, score, per-metric and per-stratum aggregates, rationale); archive-wide aggregates (tree shape, scores, subtree values, recipes; no clips); `nodes_remaining`; `attempt`, `max_attempts`, `retry` report; `dry_run` flag | `EditResult{summary, component}` (`component`: the one edit component the plan chose, optional, recorded and never enforced, §9.1.1; the kernel computes the diff) |
| `RecipeContext` | `workspace=/workspace`; archive-wide clip pool summary (per clip: provenance, license, metadata, eligible formats, ingesting node, scores of nodes that trained on it); parent's data commit and recipe; base recipe; the tunable-key schema and allowlists (§8); the standard-format rules (§6); GPU count; the same aggregates; `nodes_remaining`; attempt/retry info; `dry_run` flag | `RecipeResult{data_commit, recipe (dict of tunable keys), rationale}` |

### 9.4 Contract verification (step 4, fresh container from the child commit)

1. **Build:** `agent/requirements.txt` installs; image builds (cached by requirements hash).
2. **Static:** `agent/entry.py` defines top-level `edit_self` and `improve_recipe`, each
   with exactly one positional parameter `ctx`.
3. **Import:** the runner imports `agent.entry` within 60 s; callables match.
4. **Smoke run:** both entry points with `ctx.dry_run=True`, gateway in mock mode (scripted
   responses), mock tool server (canned results); each must return a schema-valid result
   without raising, under the contract smoke-run timeout policy (§14.5).
5. **Report:** structured pass/fail per step, delivered to the retry.

### 9.5 Sandbox

| Mount | Mode | Content |
|---|---|---|
| `/agent` | rw for `edit_self`, ro for `improve_recipe` | Copy of the code being run/edited *(amended 2026-09-24, Plan 2 as built: only `edit_self` changes code)* |
| `/workspace` | rw | Staging, scratch, results |
| `/context` | ro | Context bundle (+ `retry.json`) |
| `/store` | ro | Blob store (clips resolved via tools) |
| `/ar_contract` | ro | Contract package |

- Not mounted: `WorldModel/`, `WBench/`, `AutoResearcher/kernel`, `.env`, weights.
- Network: **none** (`--network none`). The gateway and the tool server are reached over Unix
  domain sockets in a per-run socket directory mounted read-only at `/run/ar` (connecting needs no
  write access to the directory, so an agent cannot delete the sockets; final review, 2026-09-24). *(Amended 2026-09-21, Plan 2:
  a Docker `--internal` network was tested and still exposes host services on the bridge IP; SSH
  was reachable from inside the sandbox.)*
- Limits: CPU and memory caps from `kernel.yaml`; no GPU. The image ships `ffmpeg`/
  `ffprobe`, `numpy`, `opencv-python-headless` and `Pillow`, which is what cropping to 16:9,
  trimming and pose file writing need on CPU. Captioning is the kernel's `caption.videos` GPU
  job (§10); video never goes to the agent model. *(Amended 2026-09-25, Follow-up C.)* Clips produced by `rollout.*` already carry the
  generation prompt as their caption.
- Container names carry the run prefix `ar-<run_id>-`.

---

## 10. Kernel tools (MCP)

*(Amended 2026-09-21, Plan 2: tools are registered with underscores, e.g. `data_ingest`,
`hf_search`, `job_wait`, `recipe_check`, because OpenAI function names must match
`^[a-zA-Z0-9_-]+$` and LangChain passes tool names through as function names. The dotted
names below are the spec's logical names.)*

Only operations that need privileges live in the kernel; all other logic (format
conversion, cropping, trimming, captioning, prompt timing) is agent code.

| Tool | Behavior |
|---|---|
| `hf.search(query, kind)` | Hugging Face dataset search; returns ids, license, tags, download counts and last-modified time. *(Amended 2026-09-24, Plan 2 as built: no sizes, which need a per-repo metadata call; `hf.download` checks them against the byte cap before transferring. `kind` other than `dataset` is refused: models are not training data.)* |
| `hf.download(repo, revision, patterns, max_bytes)` | Downloads into `/workspace/staging/hf/...`; records repo, revision, files, bytes, license. |
| `rollout.alayaworld(first_frame, camera, prompt_schedule, frames, variant, seed)` | Renders with the released checkpoint (`variant`: `dmd4` = 4-step student, `ar30` = 30-step teacher) through the `custom_i2v` validation path. `camera` is `{cam_c2w [N,4,4], intrinsics}` or a navigation action list. Output is a staging candidate in standard layout: 24 fps mp4, `cam_c2w` for every frame + intrinsics, caption JSON; with a per-round `prompt_schedule`, `segments` aligned to round boundaries (eligible for `per_chunk`). |
| `rollout.ltx25(prompt, image?, video?, frames, resolution, seed, variant)` | LTX-2.5 generation (T2V/I2V/V2V; `dev` or `distilled`), 24 fps, 16:9. Output: mp4 + caption JSON (no poses). |
| `rollout.wan22(prompt, image?, frames, seed)` | Wan 2.2 TI2V-5B, 720p 24 fps. Output: mp4 + caption JSON (no poses). |
| `caption.videos(paths, prompt)` | *(Added 2026-09-25, Follow-up C.)* Captions clips with a local video model (`captioner` config, §17: Qwen3.8-27B-FP8 served by vLLM from its own conda env), never the paid agent model. A GPU job on the node's GPU set under the GPU lock: it starts `vllm serve` (own session, process-group kill, `127.0.0.1` on a free port, `CUDA_VISIBLE_DEVICES` = the node GPUs, `--allowed-local-media-path` = a job-private directory the clips are hard-linked into), waits for readiness, sends each video file (a `video_url` `file://` part, thinking disabled) with the agent's `prompt`, then always stops the server and waits for GPU memory to return to its pre-job level. `paths` are workspace/staging paths (relative ones resolve against `/workspace`), validated at submit and again when the job starts, where each file is opened with `O_NOFOLLOW`, its real path re-checked against the caller's mounts and the staged link verified to be that same inode (a directory swapped for a link after the check is refused). Result: `clips: {path: {caption} or {error}}` (a per-clip failure is not a failed job), `load_s`, `gpu_memory_mib {before, peak, after}`, `gpu_memory_released`. A server that exits or is not ready within `startup_timeout_s` fails the job with its log tail. Telemetry: `caption.server_ready`, one `caption.clip` per clip (latency, caption or error), `caption.server_stopped` (`reason` done/cancelled/failed; the launcher logs every stop as `subproc.cancelled`), `caption.gpu_not_released`. A cancelled job waits at most 5 s for the memory (so `JobQueue.shutdown` keeps its deadline); the pre-phase GPU check is the real guard. |
| `annotate.camera(video)` | Estimates per-frame `cam_c2w [N,4,4]` (N = mp4 frame count, OpenCV convention) and pixel intrinsics; output passes the pose rules of §6.2. Backend per §16.3 item 6. |
| `video.probe(path)` | Frame count, fps, width, height, duration. |
| `data.ingest(candidates)` | §5.5. |
| `data.query(filter)` | Any clip in the run's pool, with provenance, metadata, eligible formats, ingesting node and the scores of nodes that used it. |
| `data.commit(parent, datasets, message)` | §5.6; returns `commit_id` and per-dataset stats. |
| `job.status(job_id)` | State of a GPU job (`queued`, `running`, `done`, `failed`, `cancelled`), progress counters, and the result payload when finished. |
| `job.wait(job_id, timeout_s)` | Blocks up to `min(timeout_s, tools.job_wait_max_s)` and returns the same payload as `job.status`, with `running` on expiry. |
| `job.cancel(job_id)` | Stops a queued or running GPU job. |
| `recipe.check(recipe, data_commit)` | Runs gate checks 1–7 without consuming an attempt (materializing a temporary view). |

**GPU tools are asynchronous jobs.** `rollout.*`, `annotate.camera` and `caption.videos` return a `job_id`
immediately; the agent polls `job.status(job_id)` or calls `job.wait(job_id, timeout_s)`
(capped at 300 s per call, returning `running` on expiry). No MCP request ever blocks on a
multi-minute GPU job, so default HTTP/MCP timeouts cannot kill one. `job.cancel(job_id)`
stops a job. The kernel runs jobs one at a time on its GPU list while the container waits.
Tool errors (e.g. OOM) return to the agent and are not node failures. Disabled generator
variants (§16.3 item 5) are omitted from tool schemas.

---

## 11. Evaluation and scoring

### 11.1 Proxy subset

40 of 289 cases, stratified by interaction mix (seed 20260913):
`2,25,47,63,66,70,78,84,89,90,91,96,109,114,133,136,139,142,145,164,167,172,177,178,180,188,195,200,204,206,215,237,242,246,247,248,260,272,284,289`.
Stored in `configs/proxy_cases.txt`; fixed for the run.

### 11.2 Eval pipeline per node

0. **Eval root.** All WBench artifacts live in a work_dirs-shaped tree inside the node
   directory: `runs/<run>/nodes/<n>/eval/work_dirs/<model>/videos/...`, with
   `<model> = ar_<run>_n<node>`. Both `main.py` and `tools/run_visual_plausibility.py` are
   given `--work_dir <abs>/runs/<run>/nodes/<n>/eval/work_dirs` and `--model <model>`
   (absolute `--work_dir` overrides WBench's own), so every phase reads the same videos and
   writes beside them. Nothing is written inside `WBench/`. This relies on upstream patch
   U1 (§2.3).
1. **Merge:** `WorldModel/scripts/tools/merge_lora_for_rollout.py --ckpt_dir <node checkpoint> --base_transformer weights/alaya-world-ar/transformer.pt --output runs/<run>/merge_slot --lora_rank <node rank> --lora_alpha <node alpha>`.
   Requires ≥ 30 GB free disk. The merge runs as its own subprocess (it holds ~52 GB of
   host RAM); after it exits the kernel calls `sync` and waits until
   `/proc/meminfo MemAvailable >= eval.free_ram_before_render_gb` (default 120 GB) before
   starting the render, alerting if that takes longer than 10 minutes. This avoids the
   host OOM killer taking down torchrun with exit code -9.
2. **Render:** `scripts/tools/run_wbench.py` semantics with a kernel copy of
   `configs/wbench_full.yaml`: `paths.resume_checkpoint` = merge slot,
   `paths.history_encoder` = **the node checkpoint's `history_encoder.pt`**,
   `paths.dmd_resume = weights/alaya-world-dmd`, `validation.per_sample_seed: true`, case
   ids = proxy subset, output `runs/<run>/nodes/<n>/eval/videos/`. The root node uses the
   released `transformer.pt` and `history_encoder.pt` unmerged.
3. **WBench phases** (`wbench-main`, explicit `--gpus`): `precompute`, `gpu`; `vlm` iff VLM
   metrics are in the metric set; `report`. `tools/run_visual_plausibility.py`
   (`wbench-vp`) iff visual_plausibility is in the metric set.
4. **Score** (§11.3).
5. **Aggregates for agents:** per-metric and per-dimension means, plus means by stratum
   (interaction type, scene category, perspective), computed from per-case JSONs. No case
   ids, prompts, images or videos.
6. **Cleanup:** delete the merged `transformer.pt` and, under the node's
   `work_dirs/<model>/`, the regenerable intermediates `da3_cache/`, `megasam/`,
   `masks/` and `_navi_videos_tmp/` (symlinks written by `main.py`). Keep `videos/`,
   `evaluation/`, `report.json` and the node checkpoint.

### 11.3 Score definition

- **Metric set** (fixed at run start): the 22 metric names of WBench `DIMENSION_MAP`,
  restricted to those computed in this run:
  - the 6 VLM metrics iff `VLM_API_KEY` is non-empty;
  - `visual_plausibility` iff the `wbench-vp` env and its weights are present.
  `navigation_trajectory` is counted as one metric (the report's composite of accuracy and
  consistency, as listed in `DIMENSION_MAP`); the report's extra component keys
  `navigation_accuracy` and `navigation_consistency` are not counted separately.
- **Score** = unweighted mean over the metric set of `report["full"][metric]["mean"]`.
- **Preflight at run start:** the kernel proves every metric in the set is producible
  before the first node — `VLM_API_KEY` present for the VLM metrics; the `wbench-vp` env
  and the `qwen3vl-a3b-visual-plausibility` weights present for `visual_plausibility`;
  each GPU metric's weights present. Metrics that fail preflight are excluded from the
  metric set at run start and recorded, so the loop cannot pause on node 1 for a metric
  that was never available.
- If a metric in the set is missing for a node after one phase retry, the loop pauses and
  alerts (it never scores a node on a different metric set).

### 11.4 Best node and full evaluation

- `ar status` reports the best node by highest own proxy score.
- `ar full-eval <node>` (manual only) runs the full 289-case WBench for a node.

---

## 12. Parent selection

Eligible parents: the root and every node with status `scored`.

1. **Floor:** `floor = min(score)` over all scored nodes (root included).
2. **Subtree score:** over the **scored** descendants `d` at depth distance `k ≥ 1`,
   weight `w_d = γ^k`: `subtree(n) = Σ w_d score(d) / Σ w_d`. Descendants that ended
   `invalid_code`, `invalid_recipe`, `train_failed` or `crashed` are **left out** of this
   mean: a child whose code failed to compile says nothing about the lineage's data. They
   are still counted by the child-count penalty in step 5, so repeated failures do reduce
   a node's selection weight. (Counting failures at the archive floor was rejected: a
   single failed child would drop a 0.82 node to near the worst score in the archive.)
3. **Value:** `value(n) = λ·score(n) + (1−λ)·subtree(n)` if `n` has **at least one scored
   descendant**; otherwise `value(n) = score(n)`. A node whose descendants all failed has
   an empty subtree mean (`Σ w = 0`), so it keeps its own score and is penalized only by
   the child-count term in step 5.
4. **Percentile rank:** sort eligible nodes by value; `x(n) = rank/(N−1)` with ties assigned
   the average rank (tolerance 1e-12); `N = 1` ⇒ select the root.
5. **Weight:** `mid = mean of the top `min(3, N)` x values` (explicitly `min(3, N)`, so
   `N = 2` is well defined); `w(n) = sigmoid(slope·(x(n) − mid)) · exp(−(children(n)/8)³)`
   with `slope = 6`, where `children(n)` counts direct children of any status.
6. **Probability:** `P(n) = (1−ε)·w(n)/Σw + ε/N`, with `ε = 0.2`.
7. **Draw:** seeded RNG; the seed and, for every candidate, `score`, `subtree`, `value`, `x`,
   `children`, `w`, `P` are recorded in `selection_events`.

Defaults: `γ = 0.5`, `λ = 0.5`, `slope = 6`, midpoint over the top `min(3, N)`, child
penalty scale 8, `ε = 0.2`. Simulated on a 6-node tree these give the top two candidates
~70% of the mass (slope 10 / `ε` 0.1 gave ~85%), leaving the rest well above the
exploration floor. The node's `subtree_value` field stores `value(n)`.

Verified behavior (toy trees, to become `tests/select/` golden tests): a node whose own score
is higher but whose child scored poorly ranks below a sibling without children; a single
root is chosen with P=1; tied values receive equal probability; all probabilities sum to 1.

---

## 13. Telemetry and monitoring

### 13.1 Principles

1. Capture at kernel choke points (gateway, MCP server, runner, sandbox, trainer,
   evaluator); agent-side logging is never relied upon.
2. **Fail closed:** the gateway persists the request before forwarding and the response
   before returning; if persistence fails, the call fails.
3. No truncation or sampling. Secrets (`OPENAI_API_KEY`, `HF_TOKEN`, `VLM_API_KEY`,
   container tokens) are redacted.
4. Every event carries `ts_wall`, `ts_mono`, `run_id`, `node_id`, `phase`, `attempt`,
   `span_id`, `parent_span_id`, `component`, `type`.

*(Amended 2026-09-25, Follow-up B: a request the gateway refuses before forwarding —
malformed JSON, an unknown/revoked token, a disallowed model, streaming, or video content in
`messages`/`input` (§2.1) — returns its error directly and is never persisted to telemetry,
consistent with the other pre-forward rejections; only requests that reach `store.begin` are
recorded.)*

### 13.2 Storage

- Source of truth: append-only JSONL `runs/<run>/telemetry/events/<node>.jsonl`
  (+ `run.jsonl` for run-level events), flushed per line.
- Large payloads: `runs/<run>/telemetry/payloads/<sha256>.json.zst`, referenced by hash.
- Query index: `runs/<run>/telemetry/index.db` (SQLite, WAL), rebuildable from JSONL.
- Nothing is uploaded (containers have no network; LangSmith tracing is never enabled).

### 13.3 Captured data

| Area | Recorded |
|---|---|
| LLM calls | Full request (all messages incl. system/developer/user, tool schemas, model, params) and full response (output items, tool calls, usage incl. cached tokens), latency, HTTP status, retries, cost; per-container token → node/phase/attempt; `conversation_id`, `turn_index`, `parent_call_id` (from `previous_response_id`/`conversation` when present, else by matching the request's message prefix to a prior call's request+response). Both `/v1/responses` and `/v1/chat/completions` are supported. |
| Tool calls | Kernel tools: args, result, duration, errors, side effects (bytes, files, GPU job ids, GPU-seconds). Agent-local tools: reconstructed from function-call/output pairs in LLM traffic. Auto-compactions appear as the summarizer calls that precede a new conversation. |
| Agent execution | `/context` bundle (hashed), container image hash, command, limits, stdout/stderr, exit code, `docker stats` samples, filesystem diff of `/agent` and `/workspace` (including `/workspace/staging`), result JSON. |
| Code | Every attempt commit; diff stats; contract step logs and verdicts. |
| Data | HF searches/downloads; rollout jobs (inputs, seeds, GPU time, outputs); ingest decisions with per-format checker reports, aspect ratio and leakage distances; data commits (manifest, parent, message, per-dataset stats); view materialization stats. |
| Recipe | Agent recipe; resolved config; diffs vs. parent and base; each gate check result with full tool output. |
| Training | `check_dataset.py`, precache and launcher logs; the full training log; parsed per-step metrics from `[Train]` lines (step, epoch, source, video, fs/fe, K, sigma, loss, grad, lr, time) and `[Mem]` lines; `nvidia-smi` samples every 5 s for all visible GPUs (util, memory, power, temperature) and an alert when a kernel process uses a GPU outside the list; host RAM and disk. |
| Eval | Merge log; per-case render timing; WBench phase logs; per-case per-metric JSONs; `report.json`; metric set and score; exact agent-facing aggregates. |
| Selection | §12.7. |
| System | Config snapshot, versions, control commands, crashes with tracebacks, recoveries, discards (§14.3), deletions (path, bytes, reason), node counters. |

### 13.4 Monitoring

- `ar status`: current node/phase/attempt, progress and ETA, recent events.
- Dashboard (read-only FastAPI, bound to 127.0.0.1): tree view (nodes colored by score,
  selection probabilities); node page (phase/attempt timeline, code diff, recipe diff, data
  commit summary per dataset, training curves, metric table); conversation viewer (threaded
  LLM calls with full prompts, responses, tool calls); data browser (clips by provenance and
  format with kernel-generated thumbnails); live panels (event tail via SSE, training loss,
  GPUs, tokens and cost by node/phase/model).
- Alerts (banner + `run.jsonl`): no events in the current phase for 30 min, GPU outside the
  list used, disk below 50 GB, gateway error rate > 20% over 5 min, retries exhausted, loop
  paused.

---

## 14. Failure handling, resume, stop

### 14.1 Node outcomes

| Status | Meaning | Counts toward `max_nodes` | Parent-eligible |
|---|---|---|---|
| `scored` | Completed | yes | yes |
| `invalid_code` | `edit_self`/contract retries exhausted | yes | no |
| `invalid_recipe` | `improve_recipe`/gate retries exhausted | yes | no |
| `train_failed` | Recipe-caused training failures exhausted retries | yes | no |
| `crashed` | Kernel-recoverable crash of the node (§14.2) | yes | no |

### 14.2 Failure classes

| Where | Failure | Handling |
|---|---|---|
| `edit_self` | container crash, timeout, missing/invalid result | failed attempt → retry loop 3↔4 |
| contract | any check fails | retry loop 3↔4 |
| `improve_recipe` | crash, timeout, invalid result | failed attempt → retry loop 5↔6 |
| kernel tools | download/rollout/annotation errors | returned to the agent as tool errors |
| gate | any check fails | retry loop 5↔6 |
| gateway | upstream 429/5xx | gateway retries with exponential backoff; upstream unavailable > 15 min ⇒ loop pauses and alerts (not charged to the node) |
| precache/train | recipe-caused (CUDA OOM, NaN/inf loss, wall-time cap) | back to 5↔6, consuming one attempt; exhausted ⇒ `train_failed` |
| precache/train, merge, render, eval | infrastructure (host OOM kill, disk full, NCCL/driver error, text-embed cache miss after a successful precache, WBench phase error) | retry the phase once; if it fails again ⇒ loop pauses and alerts |
| any phase | unexpected kernel exception while the kernel process survives | node marked `crashed`, artifacts kept, loop continues with a new cycle |

### 14.3 Resume

- **Graceful stop, or `crashed` node:** all finished nodes stay; `ar run --resume` starts a
  new cycle at *select*.
- **Unrecoverable crash (kernel process died) or forced stop:** on `ar run --resume` the
  unfinished node is removed entirely: its containers and process groups (by run prefix),
  attempt refs and branch, data commits, clips and blobs referenced only by it, view,
  dataset cache, training outputs, merge slot contents, renders, eval dirs, and its
  telemetry files. One discard record remains in `run.jsonl` (node id, parent, phase
  reached, reason, time). The run then starts a new cycle at *select*.
- Atomicity: blob writes are temp+rename; every DB state change is one transaction;
  orphaned staging directories are removed on resume.

### 14.4 Stop controls

- `ar stop --graceful` (or first Ctrl-C): finish the current node through step 9, then exit.
- `ar stop --force` (or second Ctrl-C): `docker kill` the node's containers, SIGTERM the
  training/WBench/rollout process groups, SIGKILL after 30 s, flush telemetry, exit. The
  unfinished node is discarded on resume (§14.3).

### 14.5 Timeouts with liveness checks

| Phase | Soft timeout | Liveness signals | Hard cap |
|---|---|---|---|
| `edit_self` | 2 h | gateway calls, MCP calls, container CPU, `/agent` or `/workspace` changes | 4x soft |
| `improve_recipe` | 24 h | as above + active kernel GPU jobs started by the call | 4x soft |
| contract smoke run | 15 min | gateway (mock) calls, container CPU | 4x soft |
| precache/train | 48 h | `[Train] step=` advancing, GPU util and VRAM held by the phase's processes, log growth | none |
| render/eval/rollout/annotation jobs | 12 h | finished cases or output files increasing, GPU util/VRAM of own processes, log growth | none |

At the soft timeout the kernel observes a 10-min probe window. Any liveness signal ⇒ extend
by 25% of the soft timeout and re-check at the next deadline. No signal during the whole
window ⇒ end the phase as a timeout failure (classified per §14.2). Hard caps, when set, end
the phase regardless. Stall alerts (§13.4) never terminate anything.

### 14.6 Guards before GPU phases

- GPU list resolved per §2; the kernel refuses a list it cannot see.
- No non-kernel processes on the listed GPUs; otherwise wait and alert.
- Merge requires ≥ 30 GB free disk; otherwise pause and alert.

---

## 15. Disk policy

- The kernel deletes only its own transient files without asking: merged `transformer.pt`
  after each eval, `da3_cache/`, `megasam/`, `masks/`, `trainer_state.pt`, staging after
  ingest, and the artifacts of a discarded node (§14.3).
- Nothing outside `AutoResearcher/runs/` is deleted by the kernel.
- No per-node data quotas.

---

## 16. Testing and verification

### 16.1 Kernel unit tests (pytest; no GPU, no LLM)

- Data store: hashing, dedupe, read-only blobs, clip ids, commit manifests and ids, commit
  validation (names, `prompt_mode` rules, format eligibility, clips from other branches
  accepted), hardlink views in the standard layout (poses omitted for static datasets).
- Ingest decisions from stubbed checker reports: aspect-ratio rejection, per-format
  eligibility, rejection when no format is eligible.
- Leakage check: a near-duplicate of a WBench first frame is rejected (both signals fire);
  an unrelated flat frame (low entropy, pHash within 4 of a WBench frame but low NCC) is
  accepted and logged as a near-match.
- Metric-set preflight: missing `VLM_API_KEY`, missing VP weights or missing `wbench-vp`
  env each drop their metrics at run start instead of pausing the loop later.
- Eval path layout: `main.py` and the VP script both get the node's absolute `--work_dir`
  and `--model`; cleanup removes `da3_cache/`, `megasam/`, `masks/` and
  `_navi_videos_tmp/` and keeps videos, evaluation and report.
- GPU list: unset environment resolves to `0,1,2,3`; a set list is used verbatim (any
  indices, GPU 5 included); fewer than 4 GPUs is refused at run start.
- Memory guard: render waits while `MemAvailable` is below the threshold and alerts after
  the timeout.
- GPU job API: `job.wait` returns `running` at the cap instead of blocking; `job.cancel`
  stops a running job.
- Selection: toy tree and edge cases (single root, two nodes, ties, failed descendants, all
  equal) as golden tests; seed determinism.
- Recipe gate: non-tunable key, resolution and LoRA allowlists, unchanged data commit, clip
  count below GPU count, **step budget** (`epochs x steps_per_epoch < max_steps` rejected,
  with the computed numbers), valid recipe; resolved-config generation (injected keys);
  the view exists before any gate tool runs.
- Contract: fixture agents (missing entry, wrong signature, import error, hanging smoke run,
  invalid result).
- Scoring: metric-set resolution with/without `VLM_API_KEY`; mean from the existing 40-case
  `report.json` fixture; missing metric ⇒ pause.
- Failure classifier: log fixtures for CUDA OOM, NaN loss, exit -9 host OOM, NCCL error,
  disk full, text-embed cache miss.
- Gateway: conversation linking (response id and prefix matching), fail-closed writes,
  redaction.
- Liveness: simulated active/idle signals ⇒ extend/end.

### 16.2 Loop integration test (no GPU)

Mock gateway, mock tools, fake train/eval returning synthetic scores; multi-node run in
minutes covering both retry loops and exhaustion, `crashed` node continuation, graceful
stop, forced stop and `kill -9` of the kernel mid-phase (discard + fresh cycle), telemetry
completeness (every phase has start/end events; every LLM call has a conversation id; every
discard leaves one record).

### 16.3 Real-component verification (GPU; manual, marked slow)

1. **Standard formats:** ingest every clip of `WorldModel/data/examples/{video_caption_camera,video_timed_prompts_camera,video_caption_static}`
   (all accepted with the expected eligible formats), plus one deliberately broken clip per
   checker error class and one 4:3 clip (all rejected with the expected messages).
2. **Training smoke:** a data commit mixing the three example datasets with
   `optimizer.max_steps: 2`, `optimizer.grad_accum_steps: 1` passes the gate, precaches and
   trains through §7.3, writing `lora.safetensors` and `history_encoder.pt`.
3. **Eval smoke:** merge, render 2 cases with the node's `history_encoder.pt`, WBench,
   score; then reproduce the root score on the 40-case proxy; GPU metrics match the existing
   `WBench/work_dirs/alayaworld` report within 1e-3 per metric (seeded rendering).
4. **Sandbox isolation:** from inside a container, internet access, reads of
   WorldModel/WBench/kernel/.env, and writes to `/store` all fail and are logged.
5. **Generator fit:** each rollout variant (`rollout.alayaworld` dmd4/ar30, `rollout.ltx25`
   dev/distilled, `rollout.wan22` ti2v-5b) produces a clip on the GPU list that passes
   ingest (with poses from `annotate.camera` for generated moving-camera clips). A variant
   that cannot fit is set `enabled: false` under `generators` in `kernel.yaml`.
6. **Camera annotation backend:** ViGeo (`WorldModel/third_party/ViGeo`) if it yields
   per-frame camera-to-world poses and intrinsics that pass §6.2 on the
   `video_caption_camera` example clips (compared against their provided poses);
   otherwise Depth-Anything-3 (installed under `AutoResearcher/third_party/`). If neither
   passes, `annotate.camera` is disabled, and generated clips can only be ingested as
   `camera_motion: static`.
7. **Allowlist preflight:** each resolution pair in `train.resolution_allowlist` combined
   with the largest rank in `train.lora_allowlist` completes 10 steps without OOM; pairs
   that fail are removed from the allowlist before the first run.
8. **Merge-then-render memory:** a merge immediately followed by a proxy render on the
   full GPU list completes without a host OOM kill, with `MemAvailable` sampled
   throughout; confirms the `sync` + threshold guard is sufficient on this 251 GB host.
9. **Upstream patch U1:** with the VP weights installed, the patched VP script, given the
   node's absolute `--work_dir` and `--model` from any working directory, scores the
   node's rendered proxy videos and writes `evaluation/visual_plausibility/` inside the
   node directory. Unpatched behavior (relative `work_dirs`) is what this replaces.
10. **Upstream patch U2:** `lowcompute_4x4090.sh` starts with `CUDA_VISIBLE_DEVICES=0,1,2,3,5`
   (GPU 5 included) and refuses a 3-GPU list.

### 16.4 Acceptance run

Three real nodes with small recipes before the first long run.

---

## 17. Configuration defaults (`configs/kernel.yaml`)

| Key | Default |
|---|---|
| `max_nodes` | required (no default) |
| `retries.edit_self` / `retries.improve_recipe` | 3 / 3 |
| `selection.gamma` / `lambda` / `slope` / `top_k_mid` / `child_penalty_scale` / `epsilon` | 0.5 / 0.5 / 6 / 3 (used as `min(3,N)`) / 8 / 0.2 |
| `timeouts` | §14.5 |
| `liveness.probe_window_min` / `extension_frac` | 10 / 0.25 |
| `gpus.default` / `gpus.min_count` | `0,1,2,3` / 4 — used only when `CUDA_VISIBLE_DEVICES` is unset; both are config, no GPU count is hardcoded anywhere |
| `sandbox.cpus` / `sandbox.memory_gb` | 16 / 64 |
| `gateway.model_allowlist` | `[${OPENAI_MODEL}]` plus explicitly listed models |
| `gateway.upstream_outage_pause_min` | 15 |
| `ingest.aspect_tolerance` | 0.02 |
| `leakage.phash_max_distance` / `leakage.min_ncc` / `leakage.min_entropy` | 4 / 0.95 / 4.0 bits |
| `eval.free_ram_before_render_gb` / `eval.ram_wait_alert_min` | 120 / 10 |
| `tools.job_wait_max_s` | 300 |
| `captioner` | `env: zhantaoy-vllm`, `model: Qwen/Qwen3.8-27B-FP8` (by id, HF cache, offline), `tensor_parallel: null` (= largest power of two <= node GPU count), `max_model_len: 32768`, `gpu_memory_utilization: 0.85`, `max_tokens: 512`, `media_io_kwargs: {video: {num_frames: 64, fps: 2}}`, `startup_timeout_s: 1200`, `clip_timeout_s: 300`, `memory_release_timeout_s: 120`, `extra_args: []` *(added 2026-09-25, Follow-up C)* |
| `train.resolution_allowlist` | `[[416,736],[352,608]]` |
| `train.lora_allowlist` | `[[16,16],[32,32],[64,64]]` |
| `disk.merge_min_free_gb` / `disk.alert_below_gb` | 30 / 50 |
| `telemetry.gpu_sample_sec` | 5 |
| `agents.context_window_tokens` / `agents.compact_at` | 128000 (set to the agent model's window) / 0.85 |
| `alerts.stall_min` / `gateway_error_rate` | 30 / 0.2 |
| `generators` | `alayaworld: {dmd4, ar30}`, `ltx25: {dev, distilled}`, `wan22: {ti2v-5b}`, each `enabled` per §16.3 item 5 |

All values are snapshotted into `runs/<run>/config/` at run start and logged.

---

## 18. Package versions (at design time)

`langgraph` 1.2.11, `langchain-core` 1.6.3, `langchain-openai` 1.6.2, `mcp` 2.2.0, Python 3.12
for the kernel and agent image (`langchain` 1.4.2 in the kernel env for tests only); the `alayaworld` env (Python 3.10, torch 2.7.1,
`transformers<5`) for data checks, training, rendering and rollouts.
