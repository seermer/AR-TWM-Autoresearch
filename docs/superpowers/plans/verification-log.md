# Kernel foundations — real-hardware verification log

Task 14 of `2026-09-17-autoresearcher-kernel-foundations`. Everything below was run
on the actual box against the actual repos; nothing here is inferred from unit tests.

**Why this document exists.** The unit suite was green at 79/79 while six real
defects sat in the kernel and the two upstream repos. Every one of them was
invisible to tests that used synthetic fixtures and temporary directories. One
destroyed real data before it was caught; another silently deleted five metrics
from a benchmark report and still looked like a clean run. This log records what
was proven, how, and what the misses imply for the loop.

**Outcome: all eight steps passed.** The render path reproduces the 2026-09-13
reference, training writes a checkpoint from the released base with the step
counter reset, the configured resolution allowlist fits in 24 GB, and the
ingest guard is proven against the incident that motivated it.

---

## 1. Environment

| Component | Value |
|---|---|
| GPUs | 5 x 24 GB; training on `0,1,2,3`, eval/render on `0,1,2,3,4` |
| Conda envs | `alayaworld` (train/render/data), `wbench-main`, `wbench-vp`, `autoresearcher` (kernel) |
| Base checkpoint | `weights/alaya-world-ar` (released stage2b) |
| Disk | `/mnt/biometrics`, 2.1 TB free at time of run |
| torch | 2.7.1+cu128, NCCL 2.26.2 |

Documented prerequisite, confirmed twice: WorldModel scripts driven from outside
the repo need `PYTHONPATH=<worldmodel>`, because a script's own directory — not
the caller's cwd — goes on `sys.path[0]`.

---

## 2. Commits produced by this task

| Repo | Commit | What |
|---|---|---|
| AutoResearcher | `264df51` | `BlobStore.put` survives cross-filesystem candidates |
| AutoResearcher | `bbfb38b` | ingest refuses candidates outside the run directory |
| AutoResearcher | `b9c816f` | `classify_failure` no longer fails every successful run |
| AutoResearcher | `97653bd` | shell-out failures include stderr |
| WorldModel | `96c68fd` | launcher takes a GPU count, not a GPU-5 rejection (U2) |
| WorldModel | `2d99676` | `init_process_group(device_id=)` binds rank to GPU |
| WorldModel | `dc577e9` | `run_wbench` accepts a generated config outside the repo |
| WBench | `606f0b8` | visual plausibility takes `--work_dir` (U1) |
| WBench | `cf71da5` | MegaSAM: repair stale weight symlinks, stop failing silently |
| WBench | `737549b` | seed the reconstruction point subsample |
| AutoResearcher | `f4f0aaf` | `ar doctor`, portability doc, measured tolerances |

All pushed to the project forks (`seermer/{AR,AlayaWorld,WBench}-TWM-Autoresearch`).

Suites at time of writing: AutoResearcher 83 passed, WorldModel 77 passed.

---

## 3. INCIDENT: real training data destroyed by the verification test

**This is the most important entry in this document.** It is recorded in full
because the failure was caused by the verification work itself, and because the
recovery was partial.

### What happened

The Task 14 brief had the manual ingest test pass the real example datasets —
`WorldModel/data/examples/{video_caption_camera,video_timed_prompts_camera,video_caption_static}`
— directly into `Ingestor.ingest()`. `ingest()` **moves** its input into the
content-addressed blob store; consuming the source is its designed behaviour, and
`tests/test_blobs.py` asserts exactly that.

The test had been "passing" only because `BlobStore.put` used `os.replace`, which
raises `OSError` across filesystems. That error was masking the data loss. Fixing
it (`264df51`, an approved and correct fix) unmasked the real behaviour, and the
first successful run emptied all three example datasets: 18 clips, 48 files.

`data/examples` is git-ignored, so git could not restore it, and no script
rebuilds it. The upstream sources *are* documented in `docs/TRAINING.md:187-189`
— SpatialVID-HQ (`video_caption_camera`), C3VD colonoscopy
(`video_timed_prompts_camera`), and CDnet 2012 highway/office/peopleInShade
(`video_caption_static`) — and the CDnet source frames still exist under
`data/examples/_downloads/` (9,914 files, 105 MB). So the static set could in
principle have been rebuilt from local material; the other two would have
required re-downloading from their upstream datasets, and in every case the
exact clip selection, trims and captions would have had to be reproduced by
hand. Recovery from the blob store was both faster and exact.

### Recovery

The only surviving copy was the blob store written by the pytest run, inside a
temporary directory pytest prunes on a schedule. It was preserved to
`AutoResearcher/.recovery/` (104 MB, 40 blobs, 18 clip rows) before that could
happen. With the user's explicit authorization — the subagent could not authorize
writes outside the project, and correctly did not try — 48 files were restored,
each verified by sha256 against its blob record.

Verified after restore, using WorldModel's own checker rather than our own:

```
check_dataset.py --config configs/examples/finetune_video_caption_camera.yaml
  -> 6 clips, 1.4 min, 6 usable clips, 0 errors, 0 warnings
check_dataset.py --config configs/examples/finetune_video_timed_prompts_camera.yaml
  -> 6 clips, 0.8 min, 30 round-aligned windows, 0 errors, 0 warnings
check_dataset.py --config configs/examples/finetune_video_caption_static.yaml
  -> 6 clips, 1.0 min, 6 usable clips, 0 errors, 0 warnings
```

These match the figures in `docs/TRAINING.md` exactly.

### What was NOT recovered

**The original filenames are gone.** They were not stored in the clip records or
in telemetry, and the caption JSONs carry only content descriptions, no
provenance — checked directly, not assumed. The files now carry synthetic ids
`clip_0001..0006` per dataset. The content is byte-identical and remains
scene-identifiable from the captions (e.g. the office scene is plainly the CDnet
`office` sequence), so the datasets are fully usable; what is lost is the naming,
and anything that referenced those clips by filename must be re-derived.

### What prevents recurrence

Two independent guards, both required, on the principle that the test being
correct should not be the only thing standing between a bug and real data:

1. **The test no longer hands `ingest()` a real path.** `tests/manual/test_real_pipeline.py::_ingest_examples`
   copies each clip into `<run_dir>/staging/<dataset>/<stem>/` and ingests the copies.
2. **The kernel refuses the class of call outright.** `Ingestor._outside_run_dir()`
   rejects any candidate whose video/caption/pose resolves outside the run
   directory, evaluated before `probe_video` and before any blob write.
   Covered by `tests/test_ingest.py::test_candidate_outside_run_dir_is_rejected`,
   which asserts both that the candidate is rejected **and that the source files
   still exist afterwards** — i.e. that the guard fires before any move.

`BlobStore.put`'s move semantics are deliberately unchanged. The trade accepted:
a legitimate caller passing an absolute path outside the run tree now gets a
rejection instead of silent consumption.

### Proof the guard works

Step 3 was re-run after the fix (PASSED, 119 s). All 48 files were still present
afterwards, and all three datasets still pass `check_dataset.py` with 0 errors /
0 warnings.

---

## 4. Defects found only on real hardware

All five were invisible to a green unit suite.

### 4.1 `BlobStore.put` could not cross filesystems — `264df51`

`os.replace` raises `OSError` across devices. Tests used a single tmpdir, so the
ingest path was never exercised across the real filesystem boundary. Replaced
with `shutil.move`. Fixing this is what unmasked §3.

### 4.2 Every successful training run was classified as a failure — `b9c816f`

`INFRA_SIGNATURES` contained the bare substring `"NCCL"`. A **healthy** 4-GPU run
emits `NCCL version 2.26.2+cuda12.2` and four W-level `ProcessGroupNCCL.cpp`
device-id warnings. So the happy path was misclassified **100% of the time**.

Observed on a run that in fact succeeded completely: 2 steps, loss 0.4316 -> 0.2520,
18.28 GB peak allocated per rank, `checkpoint-2` written with a 624 MB
`lora.safetensors`. The test failed only on `assert 'infra' == 'none'`.

**Loop impact had this shipped:** every node marked infra-failed and discarded;
the archive never gains a scored node. It would have presented as "the search
finds no improvements" rather than as a string-matching bug — a silent, plausible,
and entirely wrong result.

Second defect in the same tuple: `"cache miss"` matches nothing a failing run
emits; it occurs only in WorldModel source comments. The real message is
`loader.py::_text_encoder_disabled`'s `"... a prompt missed the on-disk embedding
cache"`, so an incomplete precache would have read as a clean run.

Both replaced with tokens only a failing run produces. Signature-before-returncode
ordering was kept deliberately so log content can override a lying exit code; this
is safe because `lowcompute_4x4090.sh` is `set -euo pipefail`, ends in `exec bash
train.sh`, and its only `exit 0` is the dry-run branch.

**Mitigation for the root cause (synthetic fixtures):** the verbatim log of the
real run is committed as `tests/fixtures/real_successful_train.log`, and the
regression test asserts it **still contains** the benign NCCL banner and warnings,
so the fixture cannot later be "cleaned up" back into a blind spot.

### 4.3 Ranks were not bound to their GPUs — `2d99676` (WorldModel)

Both distributed entry points called `torch.cuda.set_device()` and then left
`init_process_group` without `device_id`. torch therefore could not infer the
rank->GPU mapping and warned, on every rank, that this "can potentially cause a
hang if this rank to GPU mapping is incorrect" — a latent hang under the
multi-hour distributed timeout, not log noise. Affects training
(`alaya/utils/distributed.py`) and context-parallel rollouts used by eval
(`alaya/inference/rollout_utils.py`).

This was fixed **upstream rather than filtered in the kernel**: the kernel had
been treating a real defect's symptom as noise.

**Method note — the first verification was vacuous.** A 2-rank script that called
`init_distributed()` with `UserWarning` promoted to an error passed against the
*unpatched* code too. The device-id warning fires when the NCCL communicator is
first built, i.e. on the first collective, not at `init_process_group`. After
adding an `all_reduce`, the negative control behaved correctly:

| | unpatched | patched |
|---|---|---|
| both ranks | `FAIL: UserWarning: No device id is provided ...` | `OK ... allreduce=3.0` |

Always run the negative control. A verification that passes before the fix proves
nothing.

### 4.4 `run_wbench` could not be driven from another work tree — `dc577e9` (WorldModel)

`main()` built `CONFIG_PATH` with `out.relative_to(REPO)`, assuming the bucket
config it generates lives inside the WorldModel checkout. AutoResearcher writes
run configs under its own `runs/` tree, so Step 5 died in 1.46 s with
`ValueError: ... is not in the subpath of ...` — after printing the bucket plan,
before generating anything. Same class as U1: a WorldModel script assuming it is
driven from inside its own repo. `train.sh` already accepts an absolute
`CONFIG_PATH`, so relativisation now happens only when the config really is
inside the repo.

### 4.5 Failure messages discarded the cause — `97653bd`

Six phase helpers (render, wbench x2, merge, precache, gate) built their
`RuntimeError` from `proc.stdout` alone. Python tracebacks go to **stderr**, so
§4.4 surfaced as nothing but a progress banner, and an earlier gate failure read
as a bare `describe failed (rc=1): CUDA_VISIBLE_DEVICES=0,1,2,3`.

**Loop impact:** a failed node would have recorded a cause-free message into the
archive, leaving the meta-agent to reason about failures whose reasons were
discarded at the point of capture. Telemetry already stored both streams, so only
the surfaced message was lossy — but the surfaced message is what an agent sees.

### 4.6 A moved checkout silently deleted five metrics — `cf71da5` (WBench)

The highest-consequence defect found, and a **portability** failure.

`WBench/weights/hub/torchhub` and `.../hub/checkpoints` were symlinks still
pointing at a previous checkout location (`/home/...`) after this tree moved to
`/mnt/biometrics`. `Path.exists()` follows symlinks and returns `False` for a
dangling one, so `run_megasam.py::setup_env` believed the links were missing,
tried to create them, and raised `FileExistsError`. Because `setup_env` runs per
case, this failed **every** case: 0/22 navigation cases produced poses.

It then failed quietly at three further levels:

| level | fault |
|---|---|
| worker | counted failures, exited 0 regardless |
| `run_megasam.main()` | joined workers without inspecting `exitcode` |
| WBench `main.py` | ran each precompute tool with a bare `subprocess.run()`, ignoring rc |

So the phase printed `MegaSAM done in 0s`, the run continued, and the report came
out missing `spatial_consistency`, `gated_spatial_consistency`,
`navigation_trajectory`, `navigation_accuracy` and `navigation_consistency` — a
wrong result indistinguishable from a clean one.

All four are fixed: `_ensure_symlink()` repairs a stale or dangling link in place
(and leaves a real file or directory alone), and failures now propagate at each
level. Verified: 0/22 → 22/22 poses produced.

**This is why `ar doctor` exists** (see `docs/PORTABILITY.md`). Broken symlinks
anywhere in the three repos are now a checked, reported failure, because this is
how moving the tree breaks it.

---

## 5. Step results

### Step 3 — ingest of the three standard formats: PASSED (119 s)

All three formats ingested through the staging path; example data intact
afterwards (§3).

### Step 4 — two-step training writes a checkpoint: PASSED (531 s)

| | |
|---|---|
| steps | 2 (loss 0.4316 -> 0.2520, grad 0.0090 / 0.0845) |
| peak memory | 17.05 GB step 1, 18.28 GB step 2 alloc; 22.93 GB reserved (24 GB cards) |
| checkpoint | `checkpoint-2`, `lora.safetensors` 624 MB, `history_encoder.pt` present |
| load profile | 4 ranks serially, ~52 GB fp32 CPU build each (`ALAYA_SERIAL_MODEL_LOAD=1`) |

**Checkpoint isolation verified — evidence for a hard rule of the design.** Each
rank logged, verbatim:

```
[Resume] keep history encoder initialized from paths.history_encoder;
         resume checkpoint history_encoder.pt is ignored
[Resume] loaded weights (base=weights/alaya-world-ar); step reset to 0
```

Every node therefore starts from the released base with the optimizer step
counter reset, and the history encoder comes from configured paths rather than
being inherited from a prior node. This is what makes node scores comparable
across the tree: without it a child would inherit its parent's training state and
clade-metaproductivity would measure accumulated training rather than data quality.

**Operational note:** GPU utilization is **not** a liveness signal during model
load. A near-idle snapshot during serial rank loading was misread as "the run
died"; the run was healthy. Phase-timeout logic must key off log progress markers
(`[Setup] rank N/4`, `[Train] step=`), not device activity.

### Step 5 — proxy score reproduction: PASSED, after two upstream fixes

The first attempt died in 1.46 s (§4.4). The second ran the full 1:39:45, rendered
40/40 videos, and then failed at scoring with

```
KeyError: metrics missing from the report:
  ['spatial_consistency', 'gated_spatial_consistency', 'navigation_trajectory']
```

**That was the kernel behaving correctly** — `score_from_report` refuses to score
an incomplete report rather than quietly averaging 12 metrics instead of 17. The
defect was in WBench (§4.6). After fixing it, the metrics were recomputed on the
existing 40 videos; no re-render was needed.

**Result: all 17 reference metrics present, every case count identical.**

| class | metrics | max \|delta\| | cause of spread |
|---|---|---|---|
| deterministic | 10 | 0.0007 | GPU float noise |
| subsampled | 2 | 0.0015 | unseeded `torch.randperm` (now fixed) |
| pose-derived | 5 | 0.0084 | MegaSAM solver, irreducible |
| **aggregate score** | 17 | **0.00044** | — |

Reference aggregate 0.783312, reproduced 0.783747. The 10 deterministic metrics
agree to **+0.00001** in aggregate, six of them bit-identical — **the render path
is verified**.

#### Why the original blanket 1e-3 tolerance was unachievable

Two stochastic inputs, both demonstrated rather than assumed.

**1. MegaSAM is non-deterministic, and cannot be seeded.** Two independent runs
over the *same* videos were compared directly: `cam_c2w` differed by up to
6.8e-3 and the estimated focal length by 0.92 px. The cause is not an RNG —
DROID-SLAM's CUDA kernels accumulate with `atomicAdd` on floats
(`base/src/droid_kernels.cu:1488`, `altcorr_kernel.cu:265`). Float addition is
not associative, so the result depends on thread completion order. No seed
changes this; only deterministic kernels would.

**2. Unseeded point subsampling — fixed** (WBench `737549b`).
`reconstruction_consistency.py:154` drew points with a bare `torch.randperm`.
Now seeded per (video, frame) from a SHA-1 of the case name: stable across
machines, independent of worker/GPU assignment. Negative control, three runs
each on one case:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| unseeded | psnr 22.72 | 22.75 | 22.68 |
| seeded | 22.69 | 22.69 | 22.69 |

This does not bias the metric — the subsample is still uniform, so the estimator
is unchanged; seeding fixes *which* draw is taken, not the distribution. It also
makes any two models a paired comparison on identical points.

#### Tolerances

Set from the measured spread with headroom, **not** loosened until the test
passed: deterministic 2e-3, subsampled 5e-3, pose-derived 2e-2, aggregate 2e-3.
The deterministic band is deliberately tight — it is what would catch a real
regression in the render path. The test also asserts that no metric the
reference recorded has gone missing, which is the failure mode that started this.

**Open item:** the reference's `geometric`/`photometric` values were produced
with the unseeded code, so comparing seeded results to them carries a one-time
offset inside the old noise. Regenerating the reference under the seeded code
(~105 min) would remove it. Current tolerances cover it either way.

### Step 6 — allowlist preflight and peak memory: PASSED

Each resolution in `train.resolution_allowlist` run as a fresh 10-step training
through the manual path (ingest -> commit -> gate -> precache -> train), against
the largest `lora_allowlist` pair, on 4x24 GB.

| resolution | lora rank/alpha | failure | peak alloc | peak reserved | checkpoint |
|---|---|---|---|---|---|
| 416 x 736 | 64 / 64 | none | 18.48 GB | 22.93 GB | `checkpoint-10` |
| 352 x 608 | 64 / 64 | none | 18.25 GB | 22.96 GB | `checkpoint-10` |

Neither OOM'd, so **nothing was pruned from `configs/kernel.yaml`** — the
allowlist stands as configured.

**What was actually measured.** Only the largest LoRA pair (64/64) was run
against each resolution. The remaining pairs — (16, 16) and (32, 32) — are
covered by a fits-by-dominance argument (a smaller rank trains strictly fewer
parameters at the same resolution), **not** by measurement. Two of the six
combinations were executed.

**Reserved is not a useful headroom signal here.** Peak *reserved* is effectively
identical for both resolutions (22.93 vs 22.96 GB) while peak *allocated* differs
by 0.23 GB: PyTorch's caching allocator grows toward capacity regardless of what
is live. Scheduling decisions in the loop should key off allocated, and treat
reserved as "approximately all of the card". Headroom on 24 GB cards is roughly
5.5 GB on allocation, which is what a larger resolution or batch would consume.

Both runs also reconfirm §5/Step 4's checkpoint isolation: each started from the
released base with `step reset to 0`.

---

## 6. Findings that feed Plan 3

1. **GPU phases must be serialized with node execution.** `test_gate.py` failed
   once with `describe failed (rc=1)` purely because the unit suite was run
   concurrently with the GPU training test; it passes with idle GPUs (9/9). The
   gate's describe step is a GPU job. Phase classification must distinguish
   "GPU busy" from "recipe invalid", or contention is recorded as
   `invalid_recipe` and the agent is penalized for a scheduling artifact.
2. **Liveness ≠ GPU utilization** (§5, Step 4).
3. **A real-hardware smoke run belongs in the loop's own health checking,** not
   only in a pre-merge gate. All five defects here were invisible to a green unit
   suite; a long autonomous run that only ever exercises unit-tested paths will
   accumulate the same class of blind spot.
4. **Fixes belong upstream when the defect is upstream.** Two of the six
   (§4.3, §4.4) were initially handled, or nearly handled, by working around
   WorldModel from inside the kernel. Both were better fixed at the source, and
   §4.3's kernel-side filter was masking a genuine hang risk.
5. **The proxy score has a noise floor of ~4.4e-4 on the aggregate** (§5, Step 5),
   now entirely from MegaSAM's irreducible pose noise. **A node whose score
   improves by less than that is indistinguishable from metric noise.** Parent
   selection must either require a minimum meaningful delta or evaluate promising
   nodes more than once; otherwise the tree will chase noise and reward nothing.
   The figure is from a single re-run — a proper standard deviation needs several
   repeats of the same checkpoint, which is worth doing before the loop runs
   unattended.
6. **Phase termination must kill the process group and confirm GPU memory is
   released.** `pkill -f run_megasam` did not match its multiprocessing workers,
   whose command line is `from multiprocessing.spawn import spawn_main`. Two
   survived 35 minutes reparented to init holding ~19 GB, and caused the OOM that
   lost 3 of 22 MegaSAM cases. In the loop, a killed phase that leaks orphans
   makes the **next** node fail with an OOM that looks like a bad recipe — the
   agent would be penalized for a scheduling artifact it did not cause.
7. **A killed GPU phase leaks scratch space as well as processes.**
   `run_megasam` stages each case in `WBench/_megasam_tmp` via
   `tempfile.TemporaryDirectory`, which cleans up on normal exit and on
   exception but **not** on SIGKILL. Force-killing runs during this task left
   6.7 GB across 7 directories. The loop kills phases on timeout and on a stop
   request, so it must sweep that directory afterwards; otherwise a long
   unattended run accumulates ~1 GB per killed case on a disk that is already
   the binding constraint. Pairs with finding 6: kill the process group, then
   reclaim both the GPU memory and the scratch space before the next node.
8. **Portability is a runtime property, not a source property** (§4.6,
   `docs/PORTABILITY.md`). Tracked source was already clean; what broke was
   untracked state — symlinks left pointing at the previous checkout. `ar doctor`
   now checks this, and its tests build the broken tree rather than only asserting
   the healthy one.
9. **Run the negative control.** Two verifications in this task passed *before*
   the fix was applied and so proved nothing: the `device_id` check (the warning
   fires on the first collective, not at `init_process_group`) and the seeding
   check (compared the 4-dp rounded score instead of `details.photometric_psnr`).
   Both looked like confirmation. A verification that passes against the unfixed
   code is not evidence.

---

## Plan 2 — agent runtime

Task 18 of `2026-09-21-autoresearcher-agent-runtime`, run on 2026-09-24 against branch
`feat/agent-runtime` at `3c20f7a` (code) on this machine. Tasks 1–17 were implemented and
reviewed before this. **Merge to `main` is withheld** until the whole-branch review is done.

### Verified facts relied on

Plan 2 facts 1–18 (pre-plan spikes, 2026-09-21), in short: MCP 2.x is `MCPServer`, its client
uses `httpx2`, UDS needs `allowed_hosts=["localhost"]`, `session_idle_timeout=None`, long
client timeouts (fact 5, now 4 h); the mock Responses API envelope works with
`ChatOpenAI(use_responses_api=True)`; `--network none` + UDS is the only isolation that also
hides host services (an `--internal` network still reached host SSH); ffmpeg writes and
ffprobe reports display rotation; the pinned package set (`langgraph` 1.2.11,
`langchain-core` 1.6.3, `langchain-openai` 1.6.2, `mcp` 2.2.0, `httpx2` 2.13.0, `langchain`
1.4.2 for tests only); caller identity via `ctx.request_context.request.headers`;
underscore tool names; the 107-byte `AF_UNIX` path cap; one event loop per MCP session;
`create_agent` (langchain 1.4.2 @ 4af7ab8) as the harness reference and its tool-error
behaviour; snake_case MCP attributes; no offline tokenizer (usage + 4 chars/token, images
1,500 tokens); `ChatOpenAI` needs both sync and async HTTP clients on the socket.

### Results

| Check | Result |
|---|---|
| Default suite (`pytest -p no:cacheprovider`) | **PASS: 342 passed, 17 deselected**, 1 warning (starlette `BlockingPortal` deprecation, pre-existing), 7 m 14 s wall |
| Docker suite alone (`pytest -m docker`), GPUs idle (`nvidia-smi`: no compute processes) | **PASS: 13 passed, 346 deselected**, 71.7 s (1 m 13 s wall). Slowest: real tool server end to end 37.9 s, hanging smoke run times out 15.6 s, runner timeout kill 8.3 s, seed agent contract 3.5 s, image 1.7 s, isolation 0.31 s. No `ar-*` containers left behind |
| Harness comparison `tests/test_seed_harness.py` | **PASS: 26 passed** (17 `create_agent` comparison cases + tool-error + 8 compaction tests) |
| Harness mutation controls (on a copy of `seed_agent/agent/harness.py`, restored byte-identical, sha256 checked) | invalid-tool text **4 failed**; sequential calls in reverse **4**; system prompt after history (`[*system, *messages]` → `[*messages, *system]`) **10**; no content normalisation **2**; invalid-argument text **2**. At stage 1 (18-test file) they were 4, 4, 9, 2, 2; the extra failure is `test_compaction_replaces_history_and_continues` |
| Isolation negative control | `--network none` → `--network bridge` in `sandbox/runner.py`: `test_isolation_holds_from_inside` **FAILED** (`assert 'open' == 'blocked'`: the internet was reachable). Restored `--network none`: **PASSED** (0.44 s); `git diff kernel/` empty |
| Live LLM run (Step 4) | **Deferred: no `OPENAI_API_KEY`** (`OPENAI_API_KEY` and `OPENAI_MODEL` are unset in `.env`). Not passed; Plan 4 contract item 10 carries it |

### Spec and plan sync (Step 5)

The earlier amendments (§9.5 no network, §10 underscore names, §1.1 item 9 and §9.1–9.3
frameworks and edit components) still describe what was built. Amended now to match the
code: §5.2 attempt refs `refs/attempts/<node>/<phase>-<k>`; §7.2 retries carry the workspace
for both phases and move staging; §9.1 seed layout (`entry.py` with settings, `harness.py`,
`orchestration.py`, `tools.py`, `prompts/`, `knowledge/data_building.md`,
`memory/README.md`) and §9.1.1 paths; §9.1.2 compaction inside the model node on the merged
state; §9.2 4 h MCP client timeout, kernel re-validation of `result.json`, image build
failure is a failed attempt; §9.5 `/agent` read-only for `improve_recipe`; §10 `hf.search`
returns no sizes and searches datasets only. The plan's File structure, Tasks 3–17 notes,
test counts, the `<plan-2-branch>` placeholder and the Plan 4 contract (config keys as
constructor parameters, `JobQueue.shutdown` `RuntimeError`, live-run deferral) were updated.

### Defects found during implementation, and their fixes

Pre-flight scan (30 issues, F1–F30) rulings, applied in the named tasks:
- F1: Task 3 re-runs `pip install -e '.[dev]'` after creating `contract/ar_contract` (editable install maps packages found at install time).
- F2: Task 2's rejection message keeps "aspect ratio" so the Plan 1 ingest test stays valid.
- F3/F4: plan test counts were prose; code blocks win, prose synced here.
- F5: Task 15 re-validates `result.json` with the phase's pydantic model; invalid is `ok=False`.
- F6: Task 15 deletes a stale `workspace/result.json` before the container starts.
- F7: `run_edit_self` gained `previous_workspace`, so a retry keeps `edit_plan.json`.
- F8: on retry the previous staging dir is moved, not copied (downloads up to 20 GiB).
- F9: the runner scrubs `AR_TOKEN` from the recorded argv; Task 14 records sandbox events on the harness recorder.
- F10: MCP client timeout 900 s → 14,400 s (a shorter timeout makes agents retry running work).
- F11: `TokenRegistry.on_revoke` + `CallStore.forget` bound linking memory to live containers.
- F12: gateway/tool config keys are constructor parameters; added to the Plan 4 contract.
- F13: the gateway leak test decompresses zstd payloads before searching.
- F14: the data-tools "leaves no view" test really builds a view first.
- F15: the runner test uses config roots, not absolute paths.
- F16: the service test asserts both sockets are 0600.
- F17: per-dataset stats helpers are module-level in `data_tools`; Task 13 imports them.
- F18: `EditComponent` Literal defined once in `ar_contract.models`; the seed imports it.
- F19: the GPU job cancel path reuses the Plan 1 launcher (`run_in_env(cancel=)`).
- F20: hard-coded phase tool list kept; pinned by a test (deferred minor).
- F21/F29/F30: doc sync (attempt refs, branch name, `/agent` ro, `hf_search` sizes), done here.
- F22: seed self-test also rejects keyword-only parameters.
- F23: seed-agent tests isolate `sys.path` and drop `agent*` modules.
- F24: `httpx2==2.13.0` declared.
- F25: the seed MCP adapter prefers `structured_content` over text blocks.
- F26: the job test waits for the pid file, not a fixed timer.
- F27/F28: unused import dropped; `run_tool`/summarizer marked async in the interface text.

Review rulings and fix rounds:
- Task 5 ruling + fix: the plan's `r.text[:2000]` truncated non-JSON upstream error bodies (violates §13.1.3); now the full body is kept (`358feb4`).
- Task 6 ruling: `ToolError` subclasses the MCP SDK's `ToolError`; MCP 2.2 otherwise masks the message as "Error executing tool".
- Task 6 ruling: `timeout_keep_alive=900` on `tools.sock` kept (sessions are keyed by header; in-flight responses are not idle).
- Task 6 fix: `RunServices.stop()` silently dropped live threads; it now raises on a stuck server thread (`d352a96`).
- Task 8 fix: repo-reported file names that are absolute or contain `..` could escape the `hf_download` destination; refused (`db82ac4`).
- Task 9 fix: NaN `timeout_s` bypassed the `job_wait` cap; `shutdown` join shorter than the kill path and silent (now raises `RuntimeError`); `run_cancellable` dropped telemetry and the -15 contract (`311c673`).
- Task 11 ruling + fix: a failed `docker run -d` left a Created container; now removed on every path (`cfae713`).
- Task 14 ruling + fix: agent-controlled inputs (non-UTF-8 `entry.py`, `RecursionError` in `ast.parse`, malformed or non-object `result.json`) escaped `verify_contract` as kernel exceptions; now failed steps, with the traceback tail in the smoke detail (`e987837`).
- Task 15 ruling: `ImageBuildError` from an agent-authored `requirements.txt` becomes a failed attempt, not a Plan 4 exception.
- Task 15 fix: malformed / non-UTF-8 / non-object / symlinked / directory `result.json` raised and skipped the `edit_self` commit; `ImageBuildError`; re-run staging not cleared (move nested); token issued before the `try` and a revoke listener could skip `cancel_for_token`; `ok` must be literally `true` and `error` coerced to a string (`701aa1a`).
- Seed simplicity ruling (user requirement): the seed agent collapses to `entry.py`, `harness.py`, `orchestration.py`, `tools.py`, `prompts/`, `knowledge/data_building.md`, `memory/README.md`.
- Task 16 ruling + fix: compaction routing on conditional edges was evaluated per `Send` branch under parallel tool calls (reproduced: a summarizer beside a model call on stale history, and a skipped due compaction); the check moved into `call_model` on the merged state and the graph became exactly `create_agent`'s, with two parallel-call tests (`3c20f7a`).

Deferred minors (logged in the ledger, not fixed in Plan 2) include: `JobQueue` marks a job
running before it holds the GPU lock; a backend raising after cancel ends `failed`;
`run_command` timeout kills only bash; one pre-existing unexplained warning in the suite.

### Process note

During this task's doc sync, one text-substitution script was run with the system
`python3` instead of the conda env (a rule breach; it only edited the plan Markdown and was
re-checked). All later scripts and every test ran in the `autoresearcher` env.
