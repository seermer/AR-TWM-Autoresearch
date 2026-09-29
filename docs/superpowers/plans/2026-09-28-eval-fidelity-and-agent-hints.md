# Eval fidelity, tool fixes and agent hints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every node on all 22 WBench metrics over a 50-case proxy (local Qwen judge when no API key), fix the tool and prompt friction found in the `acceptance_20260928` audit, give the recipe writer real hyperparameter hints, and verify parent selection.

**Architecture:** Eval changes live in the kernel (`kernel/ar_kernel/eval/*`, `run.py`, `configs/`) plus one small WBench patch; the judge reuses the captioner's vLLM launch code. Tool fixes are kernel-side (take effect on resume/new run). Prompt, knowledge and orchestration changes are in `seed_agent/` (only reach a fresh run, because nodes inherit their parent's agent code). Context additions (process digest, recipe guide) are computed by the kernel in `context_bundle.py` and travel in the contract models.

**Tech Stack:** Python 3.12, pytest, sqlite, docker, vLLM (env `vllm`), WBench (`wbench-main`, `wbench-vp` envs), huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-09-17-autoresearcher-design.md` (§11 eval, §10 tools, §8 sandbox). Audit evidence: this conversation's report on run `acceptance_20260928` (telemetry under `runs/acceptance_20260928/telemetry`).

## Global Constraints

- Python is always `AutoResearcher/.envs/autoresearcher/bin/python` (never base/system Python). Tests: `cd AutoResearcher && AutoResearcher/.envs/autoresearcher/bin/python -m pytest tests/<file> -q`.
- **GPU count is never assumed.** Act as if the machine has exactly the GPUs the run was given (`0,1,2,3` here); the project may move to a 2- or 8-GPU machine. GPU device lists are passed explicitly everywhere, and every GPU job, tool, train or eval phase uses **all** the GPUs it is passed. No hard-coded counts or indices (`min_count` is configuration: set `gpus.min_count` to match the machine). vLLM servers (captioner, judge) derive tensor-parallel size from the passed list.
- Commit messages carry **no** Claude/co-author attribution (user rule, overrides any default). Finished work is fast-forwarded to `main` and pushed in every repo touched (AutoResearcher, WBench).
- Disk: ask before freeing space; never delete outside the project folder. The VP weights download is ~62 GB (2.0 TB free at time of writing).
- Agent-layer code (`seed_agent/`) stays simple: fewer/shorter files, straightforward implementations.
- The acceptance run is restarted as a **new run** (new run id, new git remote): the metric set, proxy set and judge change, so `acceptance_20260928` scores are not comparable and its records stay as they are.
- A missing eval prerequisite (VP weights, judge model, vLLM env) is a `PreflightError` at run creation, never a silent metric exclusion.

## Review Focus

1. Local judge server dies or hangs mid-eval → the node ends `eval_failed` with the reason; metrics are never dropped silently.
2. `--resume` of a run whose stored judge/metric set differs from the current environment → refuse with a clear message.
3. Judge answers that cannot be parsed shrink a metric's `n` → `ScoreError` (expected_n check), not a quietly different mean.
4. Aux GPUs absent, busy, or precompute crashing while render runs → fall back to the serial post-render precompute, same results.
5. `hf_search` with an empty string, punctuation, or one very long query; `hf_list_files` on a repo the token cannot open.
6. Job id prefix that matches two jobs → error listing both; mistyped id → error listing the caller's own jobs.
7. Process digest for a node with no events, a killed phase, or a conversation that never completed → empty/partial digest, never an exception in context building.
8. Selection with one scored node, with interrupted children, and with all scores equal.

## Decisions taken with the user (2026-09-28)

| # | Decision |
|---|---|
| 1 | All 22 metrics, 50-case proxy; API judge if `VLM_API_KEY` set, else local `Qwen/Qwen3.8-27B-FP8`. |
| 2 | Process digest in the edit planner's lineage; edit plans cite evidence. |
| 3 | Tool-quirk fixes; more tools in the image. Network: internet for both agent containers on docker's default bridge (Task 8). No license enforcement for URL sources. |
| 4 | Planner gets read-only HF tools; gated access is granted. |
| 5 | One sentence on quality checking in `data_builder.md`. |
| 6 | Recipe writer gets base values, a parameter guide and cost model; data-identical nodes stay rejected by the gate. |
| 7 | Verify selection by test and simulation; no rewrite unless a bug shows. |
| 8 | No push retry. Captioner server stays warm and yields to other GPU jobs, no idle timer (Task 6). The eval precompute overlap is **dropped**: it depended on idle spare GPUs, and every phase must use all the GPUs it is given. |
| D1 | Skipped: no data-driven noise floor; `selection.noise_floor` stays as configured. |
| D2 | The 50-case proxy contains the original 40. |
| D3 | Internet for both agent containers (improve_recipe and edit_self); contract verification containers stay offline. |

One input is still open: **D4**, the run id and empty git remote for the fresh run.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/make_proxy_cases.py` (new) | Deterministically extend the 40-case proxy to 50 (superset). |
| `configs/proxy_cases.txt` | The 50 ids. |
| `kernel/ar_kernel/eval/score.py` | Strict 22-metric set, judge-aware; `expected_n` helpers. |
| `kernel/ar_kernel/eval/judge.py` (new) | `Judge` description, `resolve_judge`, `JudgeServer` (local vLLM as an OpenAI endpoint). |
| `kernel/ar_kernel/tools/vllm_server.py` (new) | `VllmServer` extracted from `CaptionBackend` (start, ready-wait, stop, memory release), shared by captioner and judge. |
| `kernel/ar_kernel/eval/wbench.py` | Phase order incl. judge server. |
| `kernel/ar_kernel/run.py` | Preflight (strict), run.json (judge), resume checks. |
| `WBench/src/metrics/vlm/vlm_evaluator.py` | Accept a non-Doubao model name and extra request body when `VLM_API_URL` is set. |
| `kernel/ar_kernel/tools/{hf_tools,data_tools,jobs}.py` | Tool-quirk fixes (Task 7). |
| `kernel/ar_kernel/context_bundle.py`, `contract/ar_contract/models.py` | `process` digest per lineage node; `recipe_guide`, `parent_train` in the recipe context. |
| `kernel/ar_kernel/process_digest.py` (new) | Digest of a node's telemetry. |
| `kernel/ar_kernel/train/recipe_guide.py` (new) | Static parameter guide + train-log summary. |
| `seed_agent/agent/{tools.py,orchestration.py}`, `prompts/*.md`, `knowledge/data_building.md` | Agent-side fixes and hints. |
| `docker/agent.Dockerfile` | More CLI tools. |
| `tests/…` | One test file per unit above. |

---

### Task 1: 50-case proxy set

**Files:**
- Create: `scripts/make_proxy_cases.py`, `tests/test_make_proxy_cases.py`
- Modify: `configs/proxy_cases.txt`, `docs/superpowers/specs/2026-09-17-autoresearcher-design.md` (§11.1, 40→50), `reference/wbench_alayaworld_proxy/README.md` (note it describes the old 40)

**Interfaces:**
- Produces: `extend_proxy(cases_dir: Path, base_ids: list[str], target: int, seed: int) -> list[str]` — returns `base_ids` plus `target - len(base_ids)` new ids, sorted numerically, chosen greedily so the interaction-type mix of the result is as close as possible to the mix over all 289 cases; deterministic for a seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_make_proxy_cases.py
import json
from scripts.make_proxy_cases import extend_proxy

def _case(i, types):
    return {"interactions": [{"type": t} for t in types]}

def test_extend_keeps_base_and_is_deterministic(tmp_path):
    kinds = [["navigation"], ["event_edit"], ["subject_action"], ["perspective_switch"]]
    for i in range(1, 41):
        (tmp_path / f"case_{i}.json").write_text(json.dumps(_case(i, kinds[i % 4])))
    base = [str(i) for i in range(1, 9)]
    a = extend_proxy(tmp_path, base, 12, seed=1)
    assert a == extend_proxy(tmp_path, base, 12, seed=1)
    assert set(base) <= set(a) and len(a) == 12 and len(set(a)) == 12

def test_extend_balances_interaction_mix(tmp_path):
    for i in range(1, 41):
        (tmp_path / f"case_{i}.json").write_text(json.dumps(_case(i, ["navigation"] if i <= 20 else ["event_edit"])))
    base = [str(i) for i in range(1, 9)]           # all navigation: the extension must add event_edit
    out = extend_proxy(tmp_path, base, 12, seed=1)
    added = [i for i in out if i not in base]
    assert all(int(i) > 20 for i in added)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_make_proxy_cases.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `scripts/make_proxy_cases.py`: load all `case_<id>.json`, compute each case's interaction-type set, target mix = share of each type over all cases; loop `target - len(base)` times: for every remaining case (iterate in `random.Random(seed)`-shuffled order to break ties) compute the L1 distance between the selected-set mix (with that case added) and the target mix, pick the minimum. CLI: `python scripts/make_proxy_cases.py --target 50 --seed 20260928 --write` reads `configs/proxy_cases.txt` as base and rewrites it (comma-separated, numerically sorted).
- [ ] **Step 4: Run** the test → PASS. Then run the script with `--write`, check `wc -c`/id count is 50 and the first 40 ids are the old ones (`git diff configs/proxy_cases.txt`).
- [ ] **Step 5: Docs + commit**: update spec §11.1 ("50 of 289 … the original 40 plus 10 added by `scripts/make_proxy_cases.py`, seed 20260928"), then `git add scripts tests configs docs reference && git commit -m "feat(eval): 50-case proxy set (the 40 plus 10 stratified)"`.

---

### Task 2: Strict full metric set, expected case counts from the root

**Files:**
- Modify: `kernel/ar_kernel/eval/score.py` (`resolve_metric_set`), `kernel/ar_kernel/run.py` (`preflight_metrics`, `_expected_case_counts`), `configs/kernel.yaml`
- Test: `tests/test_score.py`, `tests/test_run_bootstrap.py`

**Interfaces:**
- Produces: `resolve_metric_set(cfg, env) -> list[str]` returns all 22 `DIMENSION_METRICS` always; `preflight_metrics(cfg, env) -> tuple[list[str], list[str]]` raises `PreflightError` listing every unmet prerequisite (VP weights path, judge availability from Task 3) instead of returning exclusions; `UNIVERSAL_METRICS: tuple[str,...]` = the ten metrics computed for every case (`aesthetic_quality, imaging_quality, temporal_flickering, dynamic_degree, motion_smoothness, hpsv3_quality, background_consistency, segment_continuity, geometric_consistency, photometric_consistency`); `expected_n` for a fresh run = `{m: len(case_ids) for m in UNIVERSAL_METRICS}`, and after the root scores, `run.json["expected_n"]` is extended with every metric's root `n` (nodes after the root must match).
- Consumes: Task 3's `resolve_judge`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_score.py
def test_metric_set_is_always_all_22(cfg):
    from ar_kernel.eval.score import resolve_metric_set, DIMENSION_METRICS
    assert resolve_metric_set(cfg, {"VLM_API_KEY": ""}) == DIMENSION_METRICS
    assert resolve_metric_set(cfg, {"VLM_API_KEY": "abc"}) == DIMENSION_METRICS

# tests/test_run_bootstrap.py
def test_preflight_fails_when_vp_weights_missing(cfg_without_vp_weights):
    import pytest
    from ar_kernel.run import preflight_metrics, PreflightError
    with pytest.raises(PreflightError, match="visual_plausibility"):
        preflight_metrics(cfg_without_vp_weights, {"VLM_API_KEY": "abc"})

def test_expected_n_universal_metrics_use_case_count(cfg):
    from ar_kernel.run import initial_expected_n
    n = initial_expected_n(["1", "2", "3"])
    assert n["aesthetic_quality"] == 3 and "perspective_consistency" not in n
```

- [ ] **Step 2: Run** both files → FAIL.
- [ ] **Step 3: Implement.** `resolve_metric_set` returns `list(DIMENSION_METRICS)`. `preflight_metrics` checks `cfg.wbench / eval.vp_weights` exists and (Task 3) that a judge can be resolved; collects all problems into one `PreflightError`. Replace `_expected_case_counts` (reference-report based) with `initial_expected_n(case_ids)`; in the node scoring path (`score_node`), when the node is the root and its report validates, write the report's per-metric `n` into `run.json["expected_n"]` (atomic write via tmp file + `os.replace`). Delete the `REFERENCE_REPORT` dependency (keep the reference folder for history).
- [ ] **Step 4: Run** `pytest tests/test_score.py tests/test_run_bootstrap.py tests/test_loop.py -q` → PASS (fix fixtures that relied on exclusions).
- [ ] **Step 5: Commit** `git commit -m "feat(eval): always score all 22 metrics; prerequisites are preflight errors; expected case counts come from the root"`.

---

### Task 3: VP weights and the local Qwen judge

**Files:**
- Create: `kernel/ar_kernel/tools/vllm_server.py`, `kernel/ar_kernel/eval/judge.py`, `tests/test_vllm_server.py`, `tests/test_judge.py`
- Modify: `kernel/ar_kernel/tools/captioner.py` (use `VllmServer`), `WBench/src/metrics/vlm/vlm_evaluator.py`, `configs/kernel.yaml` (`eval.judge`), `kernel/ar_kernel/run.py`
- Test: existing `tests/test_captioner.py` must keep passing unchanged.

**Interfaces:**
- Produces:
  - `class VllmServer: __init__(self, cfg, gpus: list[int], port: int, extra_env: dict|None = None, log_path: Path)`; `start(self, timeout_s) -> None` (spawns `vllm serve` via the configured `captioner.env`, polls `GET /v1/models`; `--tensor-parallel-size` = `captioner.tensor_parallel` or the largest power of two ≤ `len(gpus)`, and `--data-parallel-size = len(gpus) // tp` so a non-power-of-two count still uses every GPU); `stop(self) -> None` (SIGTERM, wait, then wait for GPU memory release via `wait_gpu_release`); `base_url` property. Context manager (`with VllmServer(...) as s`). Command construction is the existing `CaptionBackend.server_command`, moved here as `serve_command(cfg, port, media_dir=None)`.
  - `@dataclass(frozen=True) class Judge: kind: str  # "api" | "local"; model: str; url: str | None`; `resolve_judge(cfg, env) -> Judge` — `api` iff `env["VLM_API_KEY"].strip()`, model from `VLM_MODEL_NAME` (default `doubao-seed-2-0-lite-260215`), url from `VLM_API_URL` (default Ark); otherwise `local` with model `captioner.model`, url `None`.
  - `judge_env(judge: Judge, base_url: str | None) -> dict[str, str]` — env for WBench's vlm phase: for `local`: `VLM_API_URL=<base_url>/v1/chat/completions`, `VLM_API_KEY=local`, `VLM_MODEL_NAME=<model>`, `VLM_EXTRA_BODY=<json from eval.judge.extra_body>`; for `api`: `{}`.
  - WBench: `VLMClient` skips the `ALLOWED_MODELS` check when `VLM_API_URL` is set to a non-default URL, and merges `json.loads(os.environ["VLM_EXTRA_BODY"])` into both request payloads (it replaces the Doubao-only `"thinking"` field when present).
- Weights: `hf download meituan-longcat/WBench-weights --include "qwen3vl-a3b-visual-plausibility/*" --local-dir WBench/weights`.

- [ ] **Step 1: Download the VP weights** (user runs, or with approval): `/home/zhantaoy/miniforge3/envs/autoresearcher/bin/hf download meituan-longcat/WBench-weights --include "qwen3vl-a3b-visual-plausibility/*" --local-dir /mnt/biometrics/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WBench/weights`. Expected: ~62 GB under `WBench/weights/qwen3vl-a3b-visual-plausibility/`.
- [ ] **Step 2: VP smoke on existing videos**: run the kernel's own VP step on the old root's 40 videos: `CUDA_VISIBLE_DEVICES=<the run's GPUs> <wbench-vp python> tools/run_visual_plausibility.py --model ar_acceptance_20260928_nroot --work_dir <run>/nodes/root/eval/work_dirs --model_path weights/qwen3vl-a3b-visual-plausibility` (copy the work dir first so the old run's tree stays untouched). Record wall time and score range in the verification log. Expected: one JSON per case with a numeric `score`.
- [ ] **Step 3: Failing tests for `VllmServer` and `Judge`**

```python
# tests/test_judge.py
from ar_kernel.eval.judge import Judge, resolve_judge, judge_env

def test_api_judge_when_key_present(cfg):
    j = resolve_judge(cfg, {"VLM_API_KEY": "k", "VLM_MODEL_NAME": "Doubao-Seed-2.0-lite"})
    assert (j.kind, j.model) == ("api", "Doubao-Seed-2.0-lite")
    assert judge_env(j, None) == {}

def test_local_judge_when_key_blank(cfg):
    j = resolve_judge(cfg, {"VLM_API_KEY": "  "})
    assert j.kind == "local" and j.model == cfg.get("captioner.model")
    env = judge_env(j, "http://127.0.0.1:8123")
    assert env["VLM_API_URL"] == "http://127.0.0.1:8123/v1/chat/completions"
    assert env["VLM_API_KEY"] == "local" and env["VLM_MODEL_NAME"] == j.model
```

`tests/test_vllm_server.py`: with `serve_command` monkeypatched to `[sys.executable, "-m", "http.server", str(port)]`-style fake that answers `/v1/models`, assert `start` returns once ready, `stop` terminates the process, and `serve_command` derives parallelism from the GPU list (2 GPUs → tp 2, dp 1; 4 → tp 4, dp 1; 8 → tp 8, dp 1; 6 → tp 4, dp 1 and a logged warning that 2 GPUs stay idle, because vLLM cannot shard this model across a non-power-of-two count; the warning is the only way to say so, since the rule is dp = `len(gpus) // tp`), and `start` raises `TimeoutError` (with the log tail in the message) when the process exits early.
- [ ] **Step 4: Run** → FAIL. **Step 5: Implement** the module split (move `server_command`, the serve thread and release wait out of `CaptionBackend.run` into `VllmServer`; `CaptionBackend` becomes a client of it) and `judge.py`. **Step 6: Run** `pytest tests/test_vllm_server.py tests/test_judge.py tests/test_captioner.py -q` → PASS.
- [ ] **Step 7: WBench patch** (in the WBench repo): in `VLMClient.__init__`, enforce `ALLOWED_MODELS` only when `api_url` equals `DEFAULT_API_URL`; add `_extra_body()` reading `VLM_EXTRA_BODY` and `payload.update(...)` in `_call_openai` and `_call_ark`. Add a unit test under `WBench` if it has a tests dir, else a 10-line script check. Commit in WBench: `feat(vlm): allow a self-hosted OpenAI-compatible judge (VLM_API_URL, VLM_EXTRA_BODY)`.
- [ ] **Step 8: Judge spike (gate for Task 4)** — start `VllmServer` on the run's GPUs (`0,1,2,3` here), run WBench `--phase vlm` on the old root's 40 videos with `judge_env`, choose `eval.judge.extra_body` (candidates: `{"chat_template_kwargs": {"enable_thinking": false}}`, or the template's `reasoning_effort: low`), and record in the verification log: answer parse rate (must be ≥ 99% of requests), per-metric `n`, wall time, and the run-to-run spread over two runs. There is no Doubao reference to calibrate against; the log states that local-judge scores are a different scale from API-judge scores. If wall time > 30 min for 50 cases, raise it with the user before Task 4.
- [ ] **Step 9: Commit** `git commit -m "feat(eval): local Qwen judge for the VLM metrics; VllmServer shared with the captioner"`.

---

### Task 4: Eval phases with the judge and VP; run records the judge

**Files:**
- Modify: `kernel/ar_kernel/eval/wbench.py`, `kernel/ar_kernel/run.py`, `kernel/ar_kernel/cli.py` (status line), `panel` run header (show judge kind/model)
- Test: `tests/test_wbench_phases.py` (new), `tests/test_run_bootstrap.py`, `tests/test_cli_run.py`

**Interfaces:**
- Consumes: `Judge`, `judge_env`, `VllmServer` (Task 3).
- Produces: `run_wbench_phases(cfg, work_dir, model, gpus, metric_set, recorder, node_id, judge: Judge) -> dict`; phase order: `precompute`, `gpu`, `gpu`, then (if `judge.kind == "local"`) start `VllmServer` on `gpus` inside a `wbench.judge_server` span, run `vlm` with `extra_env=judge_env(...)`, stop the server, then VP, then `report`. `run.json` gains `"judge": {"kind","model","url"}`; `attach_run` refuses (`PreflightError`) if `resolve_judge(cfg, env)` differs from the stored judge in kind or model.

- [ ] **Step 1: Failing tests** — with `run_in_env` and `VllmServer` replaced by recorders, assert (a) phase call order for a local judge is `precompute, gpu, gpu, server_start, vlm, server_stop, vp, report`, (b) the server is stopped when the vlm phase raises, (c) for an API judge no server is started, (d) `attach_run` with a different judge raises with both judges named.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with `try/finally` around the server; record `judge` in `run.start` and `run.json`. **Step 4: Run** the three test files → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(eval): judge server phase and run-level judge record"`.

---

### Task 5: Selection verification

**Files:**
- Modify: `tests/test_selection.py` (and `kernel/ar_kernel/selection.py` only if a test exposes a bug)
- Create: `docs/superpowers/plans/selection-simulation.md` (results)

**Interfaces:**
- Consumes: `candidates(nodes, cfg)` and `select_parent(conn, cfg, child_id, seed, recorder)` from `selection.py`.
- Produces: tests only. `selection.noise_floor` stays as configured (decision D1: no data-driven floor).

Findings that motivate this task (measured 2026-09-28 with the shipped `candidates`): on the observed chain the newest node held P = 0.25 and the best node 0.37 — the "always the newest" picks were luck (joint probability ≈ 0.09). The size penalty gives the newest leaf weight 1 by construction, so with pure-noise scores in a 6-node chain P(newest) = 0.21 against a uniform 0.167; a node that is truly +0.02 better receives P = 0.35 (0.30 with `noise_floor` 0.008). The logic does what the design says.

- [ ] **Step 1: Write tests that pin the behaviour** (append to `tests/test_selection.py`, which already defines `CFG`, `node`, `candidates`, `select_parent`, `open_db`, `NodeStore`, `Recorder`)

```python
import random
from collections import Counter


def chain(scores):
    return [node(f"n{i}", None if i == 0 else f"n{i-1}", score=s) for i, s in enumerate(scores)]


def test_truly_better_node_beats_uniform_in_a_chain():
    ps = [r["P"] for r in candidates(chain([0.787, 0.807, 0.787, 0.787, 0.787, 0.787]), CFG)]
    assert ps[1] == max(ps) and ps[1] > 2 * (1 / 6)


def test_pure_noise_does_not_strongly_prefer_the_newest():
    rng, newest = random.Random(0), 0.0
    for _ in range(500):
        newest += candidates(chain([0.79 + rng.gauss(0, 0.008) for _ in range(6)]), CFG)[-1]["P"]
    assert newest / 500 < 0.24                  # measured 0.206; uniform would be 0.167


def test_draw_frequencies_follow_the_recorded_probabilities(tmp_path):
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    for i, score in enumerate((0.7871, 0.7945, 0.7917, 0.7907)):
        nid = "root" if i == 0 else f"n{i}"
        nodes.create(nid, None if i == 0 else ("root" if i == 1 else f"n{i-1}"), i)
        nodes.record_score(nid, score, ["m"], {"m": score})
    expected = {r["node_id"]: r["P"] for r in candidates(NodeStore(conn).all(), CFG)}
    picks = Counter(select_parent(conn, CFG, f"c{seed}", seed, rec) for seed in range(4000))
    for nid, p in expected.items():
        assert abs(picks[nid] / 4000 - p) < 0.03
```
- [ ] **Step 2: Run** `pytest tests/test_selection.py -q` → all PASS (behaviour already correct); any failure is a real bug, fix `selection.py` and note it in the log.
- [ ] **Step 3: Save the simulation table** above (script inline in the test file docstring) into `docs/superpowers/plans/selection-simulation.md`.
- [ ] **Step 4: Commit** `git commit -m "test(selection): pin probabilities, recency bias and draw frequencies; record simulation"`.

---

### Task 6: Caption server reuse across caption calls (implemented, measured, reverted)

**Files:**
- Modify: `kernel/ar_kernel/tools/captioner.py`, `kernel/ar_kernel/tools/gpu_jobs.py`, `kernel/ar_kernel/tools/jobs.py`
- Test: `tests/test_captioner.py`, `tests/test_gpu_jobs.py`

**Interfaces:**
- Produces: the caption job leaves its `VllmServer` running after its clips are done (user decision: stay warm, but never block other GPU work). A shared holder object (`CaptionServerHolder`, in `captioner.py`) is asked by the job queue to `release()` **before any other GPU job starts on the same GPUs** (rollout, annotate, images, eval merge/render) and at phase end. A later `caption_videos` call reuses the running server (no load). There is no idle timer (user decision).

Why on-demand release rather than a keep-alive for the whole phase: the server holds ~0.90 of every GPU, and GPU jobs run one at a time on the same devices, so an unconditional keep-alive would block generation. Expected saving is small: `caption.server_ready` loads took 72–167 s; n2 paid it 4 times (~5 min of a ~2.5 h node).

- [ ] **Step 1: Failing tests**: (a) two caption jobs in a row start the fake server once; (b) a rollout job submitted while the server idles causes `release()` before the rollout starts and `gpu_memory_released` is checked; (c) with no other job the server keeps running (no timer); (d) run/phase end stops it; (e) an eval merge/render that needs the GPUs also triggers `release()`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with a lock in the queue's "start job" path. **Step 4: Run** `pytest tests/test_captioner.py tests/test_gpu_jobs.py tests/test_jobs.py -q` → PASS.
- [ ] **Step 5: Docker-free real check**: caption 3 clips twice, 30 s apart, on the run's GPUs — one `caption.server_ready` event, second call latency < 60 s; then a `rollout_wan22` call starts after the server stopped. Record in the verification log.
- [ ] **Step 6: Commit** `git commit -m "feat(tools): captioner server stays up between caption jobs and yields to other GPU jobs"`.

---

### Task 7: Tool-interface fixes (kernel side)

**Files:**
- Modify: `kernel/ar_kernel/tools/hf_tools.py`, `data_tools.py`, `jobs.py`, `annotate.py`, `kernel/ar_kernel/data/ingest.py` (error text), `docker/agent.Dockerfile`
- Test: `tests/test_hf_tools.py`, `test_data_tools.py`, `test_jobs.py`, `test_ingest.py`, `test_sandbox_image.py`

**Interfaces:**
- Produces:
  - `HfTools.search(caller, query, kind="dataset", limit=20)`: split `query` into words; ask the Hub for the longest word, keep hits whose id/tags contain every word (case-insensitive); each row gains `"gated": bool|str` from `full=True`. Empty/blank query → `ToolError("query is empty")`.
  - `HfTools.list_files(...)` result gains `"gated"` and `"accessible"` (from `api.auth_check`, false when it raises `GatedRepoError`/`RepositoryNotFoundError`).
  - `JobTable.get(caller, job_id)`: exact id, else unique prefix (≥ 8 chars), else `ToolError("no job '<id>' for this caller; yours: <ids>")`; ambiguous prefix lists the matches.
  - Tool descriptions (schema text the agent reads): `data_ingest` — "moves each staged file into the archive; copy first if you still need it"; `data_commit` — "prompt_mode only for video_timed_prompts_camera; omit it otherwise"; `recipe_check` — "recipe is a flat {tunable key: value} map, e.g. {\"optimizer.lr\": 1e-4}; no wrapper key"; `annotate_camera` — "at most 64 items per job; split larger batches".
  - `ingest.py`: `FileNotFoundError` messages show container paths (`/workspace/staging/...`), never host paths.
  - Image: `apt-get install -y --no-install-recommends ffmpeg unzip curl wget git p7zip-full` (curl/wget stay unusable while the container has no network, until Task 8).

- [ ] **Step 1: Failing tests**

```python
# tests/test_hf_tools.py
def test_search_matches_all_words(fake_api):
    fake_api.datasets = [{"id": "a/TartanAir-videos", "tags": []}, {"id": "b/TartanAir", "tags": []}]
    rows = tools.search(caller, "TartanAir videos")
    assert [r["id"] for r in rows] == ["a/TartanAir-videos"]

def test_search_blank_query_is_an_error(tools, caller):
    import pytest
    with pytest.raises(ToolError, match="empty"):
        tools.search(caller, "  ")

def test_list_files_reports_gated_and_access(fake_api, tools, caller):
    fake_api.gated, fake_api.no_access = True, True
    out = tools.list_files(caller, "x/y", "main")
    assert out["gated"] and out["accessible"] is False

# tests/test_jobs.py
def test_job_prefix_and_helpful_miss(table, caller):
    j = table.submit(caller, "annotate_camera", {})
    assert table.get(caller, j.id[:8]).id == j.id
    import pytest
    with pytest.raises(ToolError, match=j.id):
        table.get(caller, "0" * 32)
```
plus `test_ingest.py::test_missing_source_error_uses_container_path` (ingest a candidate whose file was already consumed; message contains `/workspace/staging` and not `runs/`) and `test_sandbox_image.py` asserting the Dockerfile lists the new packages.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** each item. **Step 4: Run** the five files → PASS. **Step 5: Real check** against the Hub: `hf_list_files` on `Kunho/RealEstate10K-videos` reports `accessible: true` once the token has access (the user granted it); if `false`, stop and tell the user the token in `.env` is not the account with access.
- [ ] **Step 6: Commit** `git commit -m "fix(tools): word-AND hf_search, gated/accessible flags, job id prefixes, clearer tool docs and error paths"`.

---

### Task 8: Sandbox network access for the agent containers (done)

Decision D3 (user): internet for both agent containers (`edit_self` and `improve_recipe`), which docker's default
bridge already gives without sudo. An earlier draft of this task added a LAN/host firewall (`DOCKER-USER`
iptables rules, an `ar-egress` network, `scripts/setup_egress_network.sh`, sudo once). The user never asked
for it, and it is not part of the plan.

- `sandbox.network: bridge` in `configs/kernel.yaml`; `run_container(..., network="bridge")` and `_docker_args` default
  to the bridge (network on is the default);
  `agent_phase.py` passes the configured value; contract verification passes `network="none"` explicitly and
  stays offline. The rest of the sandbox (`--read-only`, `--cap-drop ALL`, non-root user, `no-new-privileges`)
  is unchanged. `sandbox.network: none` restores the old behaviour.
- Tools that need root, `apt-get` among them, do not work in this container; they are in the image instead (Task 7).
- Data fetched with curl/wget has no pinned revision: its provenance is `{"kind": "url", "url": ...}`; no license
  is required or checked (user decision).
- Tests: `tests/test_sandbox_network.py`, `tests/test_agent_phase.py::test_agent_phases_get_the_configured_network`.
  Real check: from the sandbox flags on the bridge, `curl https://pypi.org` returns 200 and `pip install --user` works.

---

### Task 9: Process digest for the edit planner; evidence rule; atomic edits

**Files:**
- Create: `kernel/ar_kernel/process_digest.py`, `tests/test_process_digest.py`
- Modify: `kernel/ar_kernel/context_bundle.py` (lineage entry key `"process"`), `contract/ar_contract/models.py` if lineage is typed, `seed_agent/agent/tools.py` (`write_file`, `edit_file`), `seed_agent/agent/prompts/edit_planner.md`, `seed_agent/agent/prompts/compact.md`
- Test: `tests/test_context_bundle.py`, `tests/test_seed_agent.py`

**Interfaces:**
- Produces: `process_digest(run_dir: Path, node_id: str) -> dict` with keys `phases` (`{phase: seconds}` from `phase.start/end`, plus `train_s`, `render_s`, `eval_s`), `llm` (`{phase: {"conversations": n, "turns": n, "compactions": conversations-1}}`), `tool_errors` (`[{"tool", "count", "example"}]`, at most 8, examples normalised: hex ids and host paths removed, 160 chars), `local_errors` (`{"run_command_nonzero": n, "tool_error_messages": n}` counted from tool results appended to each gateway `llm.request`, deduplicated by message index per conversation), `gates` (`{"failed": n, "passed": n}`), `job_wait_s`. Serialised size < 2 KB. Never raises: missing files give `{}`.
- Agent side: `edit_file`/`write_file` take a per-path `threading.Lock` (module-level dict) and write via temp file + `os.replace`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_process_digest.py
import json
from ar_kernel.process_digest import process_digest

def write_events(run_dir, node, events):
    d = run_dir / "telemetry" / "events"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{node}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

def test_counts_tool_errors_by_message_shape(tmp_path):
    write_events(tmp_path, "n1", [
        {"type": "tool.error", "tool": "hf_download", "ts_wall": 1.0, "payload": None},
    ])
    d = process_digest(tmp_path, "n1")
    assert d["tool_errors"][0]["tool"] == "hf_download" and d["tool_errors"][0]["count"] == 1

def test_missing_events_is_empty_not_an_error(tmp_path):
    assert process_digest(tmp_path, "n9") == {}
```
plus a payload-backed case (write a payload with the recorder helper from `tests/conftest.py`) checking the error example is normalised (32-hex ids replaced by `<id>`, `/mnt/...` paths by `<path>`), a two-conversation phase gives `compactions == 1`, and `test_seed_agent.py::test_edit_file_parallel_calls_do_not_lose_edits` running two `edit_file` calls on one file from two threads 200 times and asserting both replacements land every time.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**; add `"process": process_digest(run_dir, node["node_id"])` to each lineage entry. **Step 4: Prompt edits** — `edit_planner.md` gets: "Each lineage node has `process`: phase times, tool errors, turns and compactions. Prefer fixing friction that repeats there over guessing from scores. Score differences below about 0.01 are noise: state a causal claim about metrics in a prompt only if the same effect appears in at least two nodes, and name the node ids and numbers you rely on in the plan." `compact.md` gets: "Keep a short list of tool quirks you have learned (for example that data_ingest moves files)." **Step 5: Run** the three test files → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(context): per-node process digest for the edit planner; atomic edit_file; evidence rule in prompts"`.

---

### Task 10: Data-builder and planner hints; planner tools; knowledge

**Files:**
- Modify: `seed_agent/agent/prompts/data_builder.md`, `prompts/planner.md`, `knowledge/data_building.md`, `seed_agent/agent/orchestration.py`
- Test: `tests/test_seed_agent.py`

**Interfaces:**
- Consumes: kernel tools `hf_search`, `hf_list_files`, `data_query` (already exposed by `kernel_tools(session)`).
- Produces: the planner role receives `[plan_tool, *read_only_kernel_tools]` where `read_only_kernel_tools = [t for t in ktools if t.name in {"hf_search", "hf_list_files", "data_query"}]`.

- [ ] **Step 1: Failing test** — `test_planner_gets_only_read_only_kernel_tools`: run `improve_recipe` with the fake harness used in `test_seed_agent.py` and assert the planner's tool names are `{submit_plan, hf_search, hf_list_files, data_query}`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** in `orchestration.py` (planner call at the `run_role(system_prompt("planner"), [plan_tool], …)` site; `ktools` is already loaded above it).
- [ ] **Step 4: Prompt text** —
  - `data_builder.md` (one sentence, per the user): "Before you ingest anything, look at frames and captions of what you built and drop clips that are blurry, static when they should move, or whose caption disagrees with the video or its pose."
  - `data_builder.md` also: "`data_ingest` moves the staged files into the archive; keep a copy if you still need them." and "hf_list_files shows `accessible`; do not try to download a repo where it is false."
  - `planner.md`: remove the "negative control / analysis only" wording (the loop cannot run analysis-only controls); add "You may call hf_search, hf_list_files and data_query to check that a source you propose exists and is accessible."
  - `knowledge/data_building.md`: a "Known interface facts" list — ingest consumes files; absolute `/workspace/...` paths and a caption JSON *path*; `prompt_mode` only for timed datasets; `annotate_camera` ≤ 64 items; no `unzip` unless Task 7's image is used, otherwise `python -m zipfile`; sources that worked (TartanAirVideos zips, `Kunho/RealEstate10K-videos` mirror clips now accessible, Wan generation via `rollout_wan22`).
- [ ] **Step 5: Run** `pytest tests/test_seed_agent.py tests/test_contract_verify.py -q` → PASS (the contract verifier runs the agent's import and smoke).
- [ ] **Step 6: Commit** `git commit -m "feat(seed-agent): planner can check sources; quality-check sentence; known interface facts"`.

---

### Task 11: Recipe guide for the recipe writer

**Files:**
- Create: `kernel/ar_kernel/train/recipe_guide.py`, `tests/test_recipe_guide.py`
- Modify: `kernel/ar_kernel/context_bundle.py` (`build_recipe_context`), `contract/ar_contract/models.py` (`RecipeContext.recipe_guide: dict`, `RecipeContext.parent_train: dict`), `seed_agent/agent/orchestration.py`, `seed_agent/agent/prompts/recipe_writer.md`
- Test: `tests/test_context_bundle.py`, `tests/test_seed_agent.py`

**Interfaces:**
- Produces:
  - `RECIPE_GUIDE: dict[str, dict]` — for each key in `TUNABLE_KEYS`: `{"base": <value from base_recipe>, "meaning": str, "try": str, "cost": str}`; built by `recipe_guide(base_recipe: dict) -> dict` so `base` comes from the run's snapshot, not a constant.
  - `train_summary(train_log: Path) -> dict` — from `[Train] step=… loss=… grad=…` lines: `{"steps", "loss_first_quarter", "loss_last_quarter", "mean_grad_norm", "sec_per_step"}` (empty dict when the log is missing).
  - `RecipeContext.parent_train` = `train_summary` of the parent's `improve_recipe` attempt; the orchestration passes `base_recipe`, `recipe_guide`, `parent_train` to the recipe writer.
- Measured facts to encode in the guide text: the base recipe is lr 5e-5, weight_decay 1e-3, max_steps 300, grad_accum 4, warmup 50, LoRA 64/64, max_grad_norm 5; the acceptance run's agents instead used lr 1e-4, wd 1e-2, max_steps 100, grad_accum 1, warmup 10, rank 32, clip 1.0 in every node without seeing the base values, i.e. 1/12 of the base sample budget at twice the learning rate. Cost model: one optimizer step at `grad_accum_steps = 1` took ≈ 13–15 s on 4 GPUs (100 steps ≈ 22–25 min); time scales linearly with `max_steps × grad_accum_steps`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_recipe_guide.py
from ar_kernel.train.recipe_guide import recipe_guide, train_summary
from ar_kernel.train.recipe import TUNABLE_KEYS

def test_guide_covers_every_tunable_key_with_base_values():
    base = {"optimizer": {"lr": 5e-5, "max_steps": 300, "grad_accum_steps": 4, "warmup_steps": 50,
                          "weight_decay": 0.001, "max_grad_norm": 5.0, "epochs": 5000},
            "lora": {"rank": 64, "alpha": 64}, "sample": {"height": 416, "width": 736},
            "data": {"overall_caption_prob": 0.0}}
    g = recipe_guide(base)
    assert set(g) == set(TUNABLE_KEYS)
    assert g["optimizer.lr"]["base"] == 5e-5 and g["lora.rank"]["base"] == 64
    assert all(g[k]["meaning"] and g[k]["try"] for k in g)

def test_train_summary_reads_loss_quarters(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("".join(f"[Train] step={i} epoch=0 loss={1.0 if i <= 25 else 0.5} grad=0.1 time=14.0\n"
                           for i in range(1, 101)))
    s = train_summary(log)
    assert s["steps"] == 100 and s["loss_first_quarter"] > s["loss_last_quarter"]

def test_train_summary_missing_log_is_empty(tmp_path):
    assert train_summary(tmp_path / "nope.log") == {}
```
- [ ] **Step 2: Run** → FAIL (check the real log format first: `grep "\[Train\] step=" runs/acceptance_20260928/nodes/n1/attempts/improve_recipe-1/train/train.log | head -2`, and adapt the regex and the `time=` field name to what is there).
- [ ] **Step 3: Implement** the module; add the two `RecipeContext` fields (defaults `{}`, so old bundles still load); fill them in `build_recipe_context`; pass them in `seed_agent/agent/orchestration.py` where the recipe writer's task JSON is built (currently `parent_recipe` and `previous_failures` only).
- [ ] **Step 4: Prompt** — append to `recipe_writer.md`: "The context has `base_recipe` (the defaults), `recipe_guide` (what each tunable does, its base value, what is worth trying and what it costs) and `parent_train` (how the parent's training went). Do not just copy the parent's recipe: choose the settings that fit the data you built and say in the rationale which ones you changed from the base and why, or why you kept them. Good candidates: `optimizer.max_steps` and `grad_accum_steps` scaled to the number of windows and the time budget, `optimizer.lr` for how far the new data is from the model's own outputs, `lora.rank`. Recipe changes alone never count: the data commit must differ from the parent's."
- [ ] **Step 5: Run** `pytest tests/test_recipe_guide.py tests/test_context_bundle.py tests/test_seed_agent.py tests/test_contract_package.py -q` → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat(recipe): base values, parameter guide and parent training summary for the recipe writer"`.

---

### Task 12: Documentation, memory, full suite, fresh acceptance run

**Files:**
- Modify: `docs/superpowers/specs/2026-09-17-autoresearcher-design.md` (§11.2 phases + judge, §8 network, §10 tool notes), `docs/superpowers/plans/verification-log.md`, `README.md`, memory file `proxy-metric-set-15.md` (rewrite as "all 22 metrics, 50 cases, judge kind")

- [ ] **Step 1:** Run the full unit suite `pytest tests -q -x --ignore=tests/manual` and the docker tests `pytest tests -m docker -q`; expected: all pass (record counts).
- [ ] **Step 2:** Update spec and README; add verification-log entries for Tasks 3, 6, 7 and 8 with the measured numbers each task calls for.
- [ ] **Step 3:** Update the memory note (the 15-metric note becomes wrong once Task 2 lands) and `MEMORY.md`.
- [ ] **Step 4:** Fast-forward `main` and push in AutoResearcher and WBench (no attribution lines).
- [ ] **Step 5 (user-supplied inputs — **D4**):** new run id and an empty git remote for it. Start with `ar run --run-id <new> --max-nodes 6 --git-remote <remote>`; the root score must complete with all 22 metrics, `n` per metric recorded in `run.json["expected_n"]`, and `judge` recorded. There is no earlier reference for the 50-case/22-metric root; the first root score becomes the reference (record it in the verification log with its per-metric `n`).

---

## Open input

- **D4:** run id and an empty git remote for the fresh acceptance run (Task 12, step 5). Everything else is decided (see the table above).

## Execution notes (2026-09-29)

Tasks 1-11 are on `main`; Task 8 is a plain default bridge (no firewall, no sudo). Task 12 steps
1-4 are done; step 5 (fresh run) is the user's. Differences from the text above: `VllmServer` takes a prebuilt
command (`serve_command(c, gpus, port, served_name, media_dir, mm_limits)`), not `cfg`; `preflight_metrics`
returns the metric list (it raises instead of returning exclusions); `judge_env(cfg, judge, base_url)` takes the
config for `eval.judge.extra_body`; the judge server needs `eval.judge.max_images` (the interaction and causal
metrics send frames as images); the warm caption server hangs off a `GpuLock` (`tools/jobs.py`) instead of a
separate holder; the process digest reports `gpu_job_s` rather than `job_wait_s`; the planner prompt in the seed
never contained the "negative control" wording (only a node's own edit had), so nothing was removed.
Open finding: the token in the environment (`HF_TOKEN`, account `seermer`) reports `accessible: false` for
`Kunho/RealEstate10K-videos` (gated `auto`), so access is not in place for that account.

Task 6 (warm caption server) was implemented and then reverted at the user's request: from the acceptance
run's telemetry it would have saved 147 s of 759 s of model loading across four nodes (one reload in n2; every
other load followed a different GPU job, which stops a warm server anyway), under 1% of the run, for about 145
net lines including a `GpuLock`. `VllmServer` (Task 3) stays: the local judge needs it.
