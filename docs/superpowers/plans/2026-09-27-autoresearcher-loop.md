# AutoResearcher Loop Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ar run` grows the node tree unattended. Each cycle selects a parent, self-edits (with contract retries), improves the recipe (with gate and training retries), trains, scores on the proxy and records the result. `ar stop` stops the run (gracefully or forced), `ar run --resume` continues it, and `ar status` shows the state. The run spends LLM money only up to an optional cap.

**Architecture:**
- One new module, `loop.py`, sequences the Plan 1–3 pieces (agent phases, contract, gate, training, scoring).
- The rest are small, separately tested helpers, each one module:
  - `selection.py`: parent selection.
  - `budget.py`: LLM spend ledger and cap, enforced at the gateway.
  - `liveness.py`: soft timeouts with a probe window.
  - `guards.py`: alerts, the GPU visibility check at run start, and `nvidia-smi` helpers.
  - `run_kit.py`: services built from config.
  - `control.py`: pid file, stop requests, signals, and marking an unfinished node `interrupted` on resume.
  - `monitor.py`: GPU samples and alerts.
  - `status.py`: `ar status`.
- Every failure is recorded and handled in place; nothing pauses the loop (user decision 2026-09-27).
  - A failure the agent can act on goes back to it as the next attempt's `retry.json`.
  - A failure after the agent phases ends the node with a status and an error, which later agents see in the lineage.
- `status.py` returns plain JSON-serializable data. A dashboard could be added later on top of it without refactoring, but none is planned.

**Tech Stack:** Python 3.12 (`autoresearcher` env), SQLite, the Docker 27.3.1 CLI, `nvidia-smi`, and the existing kernel packages. No new dependencies (`psutil` is not installed; process ancestry is read from `/proc`).

**Spec:** `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`, sections implemented:
- §7.1–7.2: bootstrap and the cycle.
- §12: selection.
- §13.3 (selection row) and §13.4 (alerts and `ar status`; no dashboard).
- §14: failure handling, resume, stop and liveness. §14.6 is reduced to the start-time GPU visibility check and the merge disk check.
- §15: transient-file deletion.
- §16.1 (selection, liveness and GPU list tests) and §16.2 (loop integration test).
- §16.4: the acceptance run, after the user's go-ahead.

This plan amends the spec where the user decided otherwise on 2026-09-27 (Task 13 writes the amendments):
- **No pause anywhere.** Infrastructure failures are handled like any other failure.
- **Optional run spend cap, in dollars only.** There is none by default.
- **Parent selection is a continuous softmax** over a subtree-aware value, with a subtree-size penalty and a uniform floor (Task 3). It replaces the spec's percentile-rank method.
- **No GPU-idle wait** before GPU phases. The plain GPU lock stays.
- **An unfinished node is never cleaned up or resumed.** On resume it is marked `interrupted`, keeps all its files and is left alone, and a fresh cycle starts. This replaces the spec's §14.3 discard.
- **New statuses `eval_failed` and `interrupted`.**
- **No dashboard.**

Also read:
- `docs/superpowers/plans/verification-log.md`. Findings 1–9 bind every task that launches or kills a process.
- The "Interface contract for Plan 4" at the end of `docs/superpowers/plans/2026-09-21-autoresearcher-agent-runtime.md`.
- The Plan 3 plan's Task 9 interfaces (`build_gpu_backends`).

## Global Constraints

- **Agent boundaries.** Agents never touch WBench, the model inference path, the kernel or `.env`. The loop changes nothing that is mounted into containers.
- **The API key stays in the gateway process.** It never reaches containers, telemetry or `ar status` output. `OPENAI_API_KEY`, `VLM_API_KEY`, `HF_TOKEN` and container tokens are redacted.
- **Real money.** No test and no verification step calls the paid LLM, with two exceptions:
  - the acceptance run in Task 13;
  - a capped live check, which also needs the user's go-ahead.
  
  Never send video to the paid model.
- **No pause** (user, 2026-09-27). The loop never blocks waiting for an operator. The only wait is the upstream HTTP retries inside the gateway.
  - Training, precache and gate failures of any class go back to the agent as the next `improve_recipe` attempt's retry report.
  - Merge, render, WBench and scoring failures end the node as `eval_failed`.
  - Unexpected kernel exceptions end the node as `crashed`.
  - Each of these writes an `alert` event, and the loop continues.
  - Two conditions **stop** the run instead, because they are not the agent's doing (Task 9, `StopRun`):
    - the dollar budget is spent;
    - an LLM provider outage: an agent attempt failed while at least `gateway.outage_error_rate` (0.8) of the last `gateway.outage_window_min` (10) minutes' LLM calls, and at least `gateway.outage_min_calls` (3) of them, failed with 429/5xx or connection errors (user decision 2026-09-27, option A).
    
    The node in progress is marked `interrupted` on resume and does not count. The operator resumes with `ar run --resume`.
- **GPU policy.** Device lists are always explicit. The default is `0,1,2,3`; any indices are allowed, with at least 4. Another user sometimes runs vLLM on these GPUs.
  - No GPU-idle wait (user, 2026-09-27: prefer the simpler implementation). The kernel's own GPU phases are serialized by the plain run `gpu_lock`.
  - GPU tests read the list from `AR_TEST_GPUS` (default `0,1,2,3`). Check `nvidia-smi` first.
- **Python envs.** Never use system or `base` Python.
  - Kernel and tests: `conda run --no-capture-output -n autoresearcher ...`.
  - WorldModel: `alayaworld`. WBench: `wbench-main` and `wbench-vp`.
  - Do not use or modify `gen-alaya`.
- **Disk is the binding constraint.** The kernel deletes only its own transient files under `AutoResearcher/runs/` (spec §15), and only for nodes that finished. An `interrupted` node's files are never deleted. Ask the user before freeing anything else, and never delete outside the project folder.
- **Portability.**
  - No absolute paths in tracked code or config.
  - Node paths stored in the archive are **run-relative**, so a moved run still resolves (Task 2).
  - Everything else resolves from `KernelConfig` / `REPO_ROOT` (`docs/PORTABILITY.md`).
- **One config per run.** The loop reads the run's frozen `runs/<id>/config/kernel.yaml` (Task 2, `KernelConfig.for_run`), never the live `configs/kernel.yaml`.
- **Tests.**
  - The default unit suite uses no GPU, no network and no Docker.
  - Docker tests are marked `docker`, GPU tests `gpu`, and manual runs `manual`.
  - Run the suite with `conda run --no-capture-output -n autoresearcher python -m pytest`.
- **Commits.** End every commit with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Use explicit pathspecs; never `git add -A`.
- **Branching.** Work on branch `feat/loop` in AutoResearcher. No WorldModel or WBench changes are planned; a genuine bug fix there needs the user told first. When done, fast-forward `main` and push all three repos (memory `main-always-current`).

## Verified facts this plan relies on (checked 2026-09-27)

1. **Plan 2 entry points.**
   - `run_edit_self(env, *, conn, node, parent_id, base_commit, attempt, max_attempts, retry, nodes_remaining, previous_workspace=None, mock_script=None, dry_run=False) -> PhaseOutcome`.
   - `run_improve_recipe(env, *, conn, node, parent_id, agent_commit, attempt, max_attempts, retry, nodes_remaining, previous_workspace=None, ...) -> PhaseOutcome`.
   - `PhaseOutcome` fields: `ok, result, error, exit_code, timed_out, commit, attempt_dir, duration_s`.
   - `attempt_dir` is `runs/<run>/nodes/<n>/attempts/<phase>-<k>/`.
   - `edit_self` returns `commit=None` when the checkout or commit fails; a retry must then keep the previous base commit.
2. **Contract.** `verify_contract(*, cfg, run_dir, run_id, repo, commit, harness, recorder, node, attempt, ...) -> ContractReport` (`.ok`, `.to_retry()`). The `ContractHarness(cfg, run_dir, recorder)` is started once per run (`.start()` / `.stop()`).
3. **Gate.**
   - `Gate(cfg, commits, recorder).check(recipe, commit_id, parent_commit, node_id, node_dir, run_dir, gpus) -> GateResult(ok, failures, resolved_path, view_roots)`.
   - It materializes the view at `node_dir/view` and writes `node_dir/train_config.yaml`.
   - The resolved config's `run.output_dir` / `run.log_dir` are under `node_dir/train/`.
   - Passing an **attempt directory** as `node_dir` gives per-attempt training directories (Plan 2 contract item 2) with no Gate change.
4. **Training.**
   - `TrainRunner(cfg, recorder).precache(resolved, gpus, node_id)` raises `RuntimeError` on failure.
   - `TrainRunner.train(resolved, gpus, node_id, node_dir) -> TrainOutcome(checkpoint, failure, log_path, metrics, detail)`. It writes `node_dir/train/train.log`, uses `node_dir/dataset_cache`, and currently has a fixed 48 h timeout (`TRAIN_TIMEOUT_SECONDS`).
5. **Scoring.**
   - `run.score_node(cfg, ctx, node_id, checkpoint, rank, alpha) -> (score, {"metrics", "aggregates", "report"})`.
   - Its cleanup of `merge_slot` and the regenerable eval directories runs only on success. The deferred Plan 1 item asks for try/finally.
   - `render_proxy` and `run_wbench_phases` use a fixed 12 h timeout.
6. **The GPU lock is a `threading.Lock`**, used as a context manager (`with self.gpu_lock:`) in `JobQueue._loop` and `DataTools.recipe_check`. The loop shares the same lock for gate, precache, training and scoring.
7. **`run_container`** blocks in `docker wait` with `timeout=timeout_s` (the hard cap), then `docker kill` and `docker rm -f`. It samples `docker stats` every 30 s into `RunResult.stats`; each sample has a `CPUPerc` string such as `"12.34%"`.
8. **`run_in_env`** runs each job in its own session. With `cancel` it polls every `poll_s` seconds. On `TimeoutExpired` from `_wait` it kills the process group, records `subproc.error` and raises `SubprocTimeout`. A `BaseException` in the wait also kills the group, so a signal handler raising in the main thread cannot orphan a training job.
9. **Telemetry.**
   - `Recorder.event(type, *, node, phase, attempt, payload, **fields)` appends to `telemetry/events/<node>.jsonl`.
   - `payload` is stored as a content-addressed zstd file, so a 5-second GPU sample must put its numbers in inline **fields**, not in `payload`.
   - `llm.response` events carry `usage` inline, in either Chat Completions form (`prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, or DeepSeek's `prompt_cache_hit_tokens`) or Responses form (`input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`).
10. **`nvidia-smi` works on this host.**
    - `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits` lists all 6 GPUs. Idle memory is 15–113 MiB.
    - `--query-compute-apps=pid,gpu_uuid --format=csv,noheader` lists compute processes, including other users' processes (they are in the same PID namespace).
11. **`psutil` is not installed** in `autoresearcher`. Ancestry comes from `/proc/<pid>/stat` field 4 (ppid).
12. **Container names** are `ar-<run_id>-<node>-<phase>-<attempt>-<hex>`, with characters outside `[A-Za-z0-9_.-]` replaced by `-` (`sandbox/runner.py:container_name`).
13. **Plan 2 live run** (verification log "Live improve_recipe run"): killing a container does not cancel an in-flight upstream call, so a spend cap can overshoot by one call.
14. **Proxy noise.** The aggregate score moved by 4.4e-4 between two evaluations of the same videos (MegaSAM's non-deterministic CUDA kernels). This is the default `noise_floor` of the selection temperature (Task 3). The user chose no evaluation re-runs.
15. **Nothing else reads the `selection:` config keys** (`grep -rn "selection\." kernel/` finds none), so Task 2 can replace them.

## Review Focus

1. **A force stop that arrives while the loop is blocked must still take effect within seconds.** Examples: waiting on a container, waiting on the GPU lock, a `run_in_env` training job. Task 12 tests SIGTERM during a phase that sleeps; the loop must exit in under 10 s.
2. **Resuming after `kill -9` must not trust a stale or recycled pid.**
   - The pid file stores the pid together with its `/proc/<pid>/stat` start time. A command-line check would not work: the `ar` console script's command line is `python .../bin/ar run ...`, with no `ar_kernel` in it.
   - Task 10 tests a pid file naming a live, unrelated process.
3. **An `interrupted` node must keep every file and archive row, and must stay out of the loop's arithmetic.**
   - It is never a parent.
   - It does not count toward `max_nodes`.
   - It does not count toward any node's subtree size or subtree mean.
   - Its leftover containers are removed on resume, but its files stay.
   
   Tasks 3, 9 and 10 test each point.
4. **Settings must come from the run's snapshot, not the live `configs/kernel.yaml`.** A budget raised in the snapshot must take effect on resume, and an edit to the live file must not affect a running run. Task 2 tests `KernelConfig.for_run`; Task 8 tests that the kit uses it.
5. **Selection must change smoothly with scores and never collapse onto one node unless its lead is large.**
   - A tiny score change must give a tiny probability change.
   - Values within the noise floor must be near-uniform.
   - One badly scored child must not sink its parent.
   
   Task 3 tests each.

---

### Task 1: Plan 3 leftovers

Small fixes deferred from Plan 3's final review (ledger: `.superpowers/sdd/2026-09-25-autoresearcher-gpu-data-sources/progress.md`).

**Files:**
- Modify: `kernel/ar_kernel/tools/rollouts.py:249`
- Modify: `kernel/ar_kernel/bridges/zimage_generate.py:29-30` and module docstring lines 12–13
- Modify: `kernel/ar_kernel/doctor.py` (`_envs`, `run_checks`)
- Test: `tests/test_rollouts.py`, `tests/test_doctor.py`

**Interfaces:**
- Produces `doctor._prefix_envs(cfg) -> list[Finding]`: one finding per enabled tool whose `env` is a prefix path (contains `/`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_rollouts.py`, extend the two existing bad-input assertions (lines 426 and 439):

```python
    assert by[1]["error"].startswith("input: ")
    assert "Error" in by[1]["error"].split(":")[1]     # the exception type is named
```

In `tests/test_doctor.py`:

```python
from ar_kernel.doctor import _prefix_envs


def test_missing_prefix_env_of_an_enabled_tool_is_a_failure(tmp_path):
    cfg = KernelConfig(raw={"images": {"enabled": True, "env": ".envs/gen-zimage"},
                            "generators": {"wan22": {"env": ".envs/gen-wan22",
                                                     "variants": {"ti2v-5b": {"enabled": False}}}}},
                       repo_root=tmp_path)
    found = {f.check: f.level for f in _prefix_envs(cfg)}
    assert found == {"env.images": "fail"}                # disabled wan22 is not checked
    (tmp_path / ".envs" / "gen-zimage" / "conda-meta").mkdir(parents=True)
    assert {f.check: f.level for f in _prefix_envs(cfg)} == {"env.images": "ok"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_rollouts.py tests/test_doctor.py -q`
Expected: FAIL. The error string has no type, and `_prefix_envs` is not defined.

- [ ] **Step 3: Implement**

`rollouts.py:249`:

```python
                (out / f"{i}.json").write_text(json.dumps(
                    {"ok": False, "error": f"input: {type(exc).__name__}: {exc}"}))
```

`zimage_generate.py`: give `--offload` real choices, and fix the docstring. The Task 5 spike showed `none` OOMs at 1920×1088.

```python
    for name in ("--items", "--out", "--weights"):
        ap.add_argument(name, required=True)
    ap.add_argument("--offload", required=True, choices=("none", "model"))
```

Docstring lines 12–13 become:

```
`--offload model` (the kernel default) calls `enable_model_cpu_offload()`: every allowed size fits
(peak 12.8 GB). `--offload none` keeps the pipeline on the GPU and OOMs at 1920x1088 (Task 5 spike).
```

In `doctor.py`, add the helper and call it from `run_checks` after `_envs()`:

```python
def _prefix_envs(cfg: KernelConfig) -> list[Finding]:
    """Enabled tools whose `env` is a conda-prefix path inside the repo (Plan 3 disk ruling)."""
    blocks = {"annotate": cfg.get("annotate") or {}, "images": cfg.get("images") or {},
              **{f"generators.{k}": v for k, v in (cfg.get("generators") or {}).items()}}
    out = []
    for name, block in blocks.items():
        env = str(block.get("env") or "")
        variants = block.get("variants")
        enabled = (any((v or {}).get("enabled") for v in variants.values()) if variants
                   else bool(block.get("enabled")))
        if "/" not in env or not enabled:
            continue
        path = Path(env) if Path(env).is_absolute() else cfg.repo_root / env
        out.append(Finding("ok", f"env.{name.split('.')[-1]}", str(path))
                   if (path / "conda-meta").is_dir()
                   else Finding("fail", f"env.{name.split('.')[-1]}",
                                f"prefix env missing at {path}; see docs/PORTABILITY.md"))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_rollouts.py tests/test_doctor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/tools/rollouts.py kernel/ar_kernel/bridges/zimage_generate.py kernel/ar_kernel/doctor.py tests/test_rollouts.py tests/test_doctor.py
git commit -m "fix(tools,doctor): Plan 3 leftovers (input error type, --offload choices, prefix-env check)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Archive schema, run config, node paths

**Files:**
- Modify: `kernel/ar_kernel/archive/db.py`, `kernel/ar_kernel/archive/nodes.py`, `kernel/ar_kernel/config.py`, `kernel/ar_kernel/context_bundle.py`, `configs/kernel.yaml`
- Test: `tests/test_archive_nodes.py`, `tests/test_config.py`, `tests/test_context_bundle.py`

**Interfaces:**
- Produces:
  - `NodeStore` statuses: add `"eval_failed"` and `"interrupted"`.
  - `NodeStore.set_fields` also accepts `edit_component`, `recipe_path`, `attempt_counts`, `error`.
  - `nodes.run_rel(run_dir, path) -> str` and `nodes.run_abs(run_dir, value) -> Path | None`.
  - Table `selection_events(id, child_id, chosen, seed, candidates, created_at)`.
  - `KernelConfig.for_run(run_dir) -> KernelConfig`: the run's frozen `kernel.yaml`, with the repo root being this checkout.
  - Lineage and archive-summary entries gain `"component"` and `"error"`.
- New `kernel.yaml` keys used by later tasks (all added here, in one place):

```yaml
budget:                          # run-wide LLM spend cap in dollars, enforced by the gateway (user decision 2026-09-27)
  max_usd: null                  # null = no cap (the default)
  usd_per_mtok: {input: null, cached_input: null, output: null}   # needed for max_usd and for cost in `ar status`
liveness:                        # spec 14.5
  probe_window_min: 10
  extension_frac: 0.25
  probe_every_s: 30              # during a probe window, read the progress signals at most this often
alerts:                          # spec 13.4
  stall_min: 30
  gateway_error_rate: 0.2
  gateway_error_window_min: 5
```

Replace the whole `selection:` block (the old rank-based keys are read nowhere; fact 15) with:

```yaml
selection:                       # continuous softmax parent selection (Task 3; user decision 2026-09-27)
  decay: 0.5                     # weight per generation below a node: child 1x decay, grandchild decay^2 ...
  prior_weight: 1.0              # the node's own score as one virtual child: pulls the subtree mean toward it
  subtree_share: 0.3             # value = (1 - share) * own score + share * subtree mean
  size_scale: 4                  # size penalty 1 / (1 + size / scale); size = descendants, decayed per generation
  temperature: 2.0               # softmax temperature, in units of the spread of values across candidates
  noise_floor: 4.4e-4            # the spread never counts as smaller than proxy noise (verification-log finding 5)
  epsilon: 0.2                   # this share of probability is spread uniformly over all candidates
```

In `gateway:`, replace the unused `upstream_outage_pause_min: 15` with:

```yaml
  outage_error_rate: 0.8         # stop the run (option A, no pause) when an agent attempt fails while at least
  outage_window_min: 10          #   this share of the LLM calls in this window failed upstream (429/5xx/connection)
  outage_min_calls: 3            #   and at least this many calls were made in the window
```

In `timeouts:` add:

```yaml
  train_s: 172800                # 48 h soft; liveness = train.log growth (spec 14.5)
  eval_s: 43200                  # 12 h soft per render / WBench phase; liveness = output files
```

- [ ] **Step 1: Write the failing tests**

`tests/test_archive_nodes.py`:

```python
import sqlite3

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore, run_abs, run_rel


def test_new_fields_and_eval_failed(tmp_path):
    nodes = NodeStore(open_db(tmp_path))
    nodes.create("n1", None, 0)
    nodes.set_fields("n1", edit_component="prompts", recipe_path="nodes/n1/recipe.yaml",
                     attempt_counts='{"edit_self": 2}', error="render failed")
    nodes.set_status("n1", "eval_failed")
    got = nodes.get("n1")
    assert (got["edit_component"], got["error"], got["status"]) == ("prompts", "render failed", "eval_failed")


def test_old_database_gains_the_new_columns(tmp_path):
    old = sqlite3.connect(tmp_path / "archive.db")
    old.execute("CREATE TABLE nodes (node_id TEXT PRIMARY KEY, parent_id TEXT, depth INTEGER NOT NULL, "
                "created_at REAL NOT NULL, status TEXT NOT NULL, agent_commit TEXT, data_commit TEXT, "
                "recipe_hash TEXT, resolved_config_path TEXT, checkpoint_path TEXT, lora_rank INTEGER, "
                "lora_alpha INTEGER, score REAL, metric_set TEXT, metrics TEXT, subtree_value REAL, "
                "phase_timings TEXT, rationale_path TEXT)")
    old.commit(), old.close()
    conn = open_db(tmp_path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}
    assert {"edit_component", "recipe_path", "attempt_counts", "error"} <= cols
    assert conn.execute("SELECT count(*) FROM selection_events").fetchone()[0] == 0


def test_interrupted_is_a_status(tmp_path):
    nodes = NodeStore(open_db(tmp_path))
    nodes.create("n1", None, 0)
    nodes.set_status("n1", "interrupted")
    assert nodes.get("n1")["status"] == "interrupted"


def test_run_relative_paths(tmp_path):
    p = tmp_path / "nodes" / "n1" / "train" / "checkpoint-2"
    assert run_rel(tmp_path, p) == "nodes/n1/train/checkpoint-2"
    assert run_abs(tmp_path / "moved", "nodes/n1/x") == tmp_path / "moved" / "nodes/n1/x"
    assert run_abs(tmp_path, None) is None
```

`tests/test_config.py`:

```python
def test_for_run_reads_the_frozen_snapshot_with_this_checkout_as_root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text("budget: {max_usd: 5}\npaths: {runs_dir: runs}\n")
    cfg = KernelConfig.for_run(tmp_path)
    assert cfg.get("budget.max_usd") == 5
    assert cfg.repo_root == REPO_ROOT                       # not runs/<id>: sibling paths still resolve
```

`tests/test_context_bundle.py`: add a test that creates `root` (scored) and `n1` (`invalid_code`, `error="contract import failed"`, `edit_component="tools"`). It asserts that `lineage(conn, run_dir, repo, "n1")[-1]` has `component == "tools"` and `error == "contract import failed"`, and that `archive_summary(conn)["nodes"]` entries carry `"error"`. Build `repo` as the other tests in that file do.

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_archive_nodes.py tests/test_config.py tests/test_context_bundle.py -q`
Expected: FAIL. The new columns, the new statuses, `run_rel` and `for_run` are missing.

- [ ] **Step 3: Implement**

`db.py`:
- Add `edit_component TEXT, recipe_path TEXT, attempt_counts TEXT, error TEXT` to the `nodes` CREATE statement (after `rationale_path TEXT`).
- Append to `SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS selection_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL, chosen TEXT NOT NULL, seed INTEGER NOT NULL,
  candidates TEXT NOT NULL, created_at REAL NOT NULL
);
```

and after `conn.executescript(SCHEMA)`:

```python
    _migrate(conn)
    return conn


# Columns added after Plan 1; CREATE TABLE IF NOT EXISTS never adds columns to an existing table.
NODE_COLUMNS_ADDED = {"edit_component": "TEXT", "recipe_path": "TEXT", "attempt_counts": "TEXT",
                      "error": "TEXT"}


def _migrate(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}
    for column, kind in NODE_COLUMNS_ADDED.items():
        if column not in have:
            conn.execute(f"ALTER TABLE nodes ADD COLUMN {column} {kind}")
```

`nodes.py`:

```python
# interrupted: unfinished when the loop stopped (forced stop, kernel death, spent budget); kept
# untouched, never resumed, never a parent, not counted toward max_nodes (user decision 2026-09-27).
STATUSES = {"running", "scored", "invalid_code", "invalid_recipe", "train_failed", "eval_failed",
            "crashed", "interrupted"}


def run_rel(run_dir: Path, path: Path | str) -> str:
    """A path under the run, stored relative to it so a moved run still resolves."""
    return str(Path(path).resolve().relative_to(Path(run_dir).resolve()))


def run_abs(run_dir: Path, value: str | None) -> Path | None:
    return None if value is None else Path(run_dir) / value
```

(`from pathlib import Path` at the top.) In `set_fields`, add `"edit_component", "recipe_path", "attempt_counts", "error"` to `allowed`.

`config.py`, in `KernelConfig`:

```python
    @classmethod
    def for_run(cls, run_dir: Path) -> "KernelConfig":
        """The run's frozen kernel.yaml (spec 17: snapshotted at run start), with THIS checkout as
        the repo root. load(path) would take runs/<id> as the root and break every sibling path."""
        raw = yaml.safe_load((Path(run_dir) / "config" / "kernel.yaml").read_text(encoding="utf-8"))
        return cls(raw=raw, repo_root=REPO_ROOT)
```

`context_bundle.py`:
- In `lineage`, add `"component": node["edit_component"], "error": node["error"],` after `"status"`.
- In `archive_summary`, add `"error": n["error"], "component": n["edit_component"]` to each node entry.

`configs/kernel.yaml`: add the keys listed under **Interfaces**.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`
Expected: the whole default suite passes.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/archive/db.py kernel/ar_kernel/archive/nodes.py kernel/ar_kernel/config.py kernel/ar_kernel/context_bundle.py configs/kernel.yaml tests/test_archive_nodes.py tests/test_config.py tests/test_context_bundle.py
git commit -m "feat(archive): node fields for the loop, selection_events, run snapshot config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Parent selection — continuous softmax (spec §12, replaced by user decision 2026-09-27)

User requirements (2026-09-27):
- Balance exploration and exploitation.
- Down-weight a node whose subtree is already large.
- Value both the node's own score (more important) and its subtree.
- One outlier in a subtree must not be catastrophic.
- The distribution must not be sharp: no one or two nodes dominate unless their lead is large.
- Use no ranking. Ranks are discontinuous: they ignore how large a lead is, and they amplify a lead of 0.0001.

The method takes HGM's clade value (the subtree counts) and HyperAgents' score × usage-penalty weight with a uniform floor, but works on the scores directly. For each candidate `n` (a `scored` node), with descendants `d` at generation `k ≥ 1`:

1. **Subtree mean, shrunk toward the node's own score.**
   `m(n) = (prior_weight·s(n) + Σ_scored decay^k·s(d)) / (prior_weight + Σ_scored decay^k)`.
   - Only `scored` descendants enter the mean.
   - With the defaults, one child moves `m` at most a third of the way toward its score.
2. **Value.** `v(n) = (1 − subtree_share)·s(n) + subtree_share·m(n)`. The own score's effective weight is at least 0.7.
3. **Subtree-size penalty.**
   `size(n) = Σ decay^(k−1)` over descendants of every status except `interrupted`; a child counts 1, a grandchild 0.5, and so on.
   `q(n) = 1 / (1 + size(n)/size_scale)`.
   - Failed children still count here, so repeated failures reduce selection.
   - `interrupted` nodes are not the agent's doing and count nowhere.
4. **Softmax.** `σ = max(pstdev of v over candidates, noise_floor)`, `τ = temperature·σ` and `w(n) = exp((v(n) − max v)/τ)·q(n)`.
   - Because τ follows the spread of values, the sharpness does not depend on the score scale.
   - Values within the noise floor are near-uniform.
5. **Uniform floor.** `P(n) = (1 − ε)·w(n)/Σw + ε/N`. A single candidate gets `P = 1`.

Worked example (the defaults, four candidates with no descendants):

| Values | Probabilities |
|---|---|
| 0.780 / 0.790 / 0.800 / 0.830 | 0.152 / 0.184 / 0.225 / 0.440 |

**Files:**
- Create: `kernel/ar_kernel/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `NodeStore.all()` rows (`node_id, parent_id, status, score`), the `selection_events` table, and the `selection.*` config (Task 2).
- Produces:
  - `candidates(nodes: list[dict], cfg) -> list[dict]`: one row per scored node, `{node_id, score, subtree_mean, value, size, penalty, w, P}`.
  - `select_parent(conn, cfg, child_id: str, seed: int, recorder) -> str`: draws the parent, inserts a `selection_events` row and records a `select` event in `run.jsonl`.
  - `update_values(conn, cfg) -> None`: writes `v(n)` to `nodes.subtree_value` for every scored node.
  - `selection_seed(run_id: str, child_id: str) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
import math

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.selection import candidates, select_parent, selection_seed, update_values
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig(raw={"selection": {"decay": 0.5, "prior_weight": 1.0, "subtree_share": 0.3,
                                      "size_scale": 4, "temperature": 2.0, "noise_floor": 4.4e-4,
                                      "epsilon": 0.2}}, repo_root=None)


def node(nid, parent, status="scored", score=None):
    return {"node_id": nid, "parent_id": parent, "status": status, "score": score}


def by_id(rows):
    return {r["node_id"]: r for r in rows}


def roots(*scores):                       # independent candidates with no descendants
    return [node(f"r{i}", None, score=s) for i, s in enumerate(scores)]


def test_single_candidate_is_chosen_with_certainty():
    assert [(r["node_id"], r["P"]) for r in candidates([node("root", None, score=0.78)], CFG)] == [("root", 1.0)]


def test_probabilities_sum_to_one():
    tree = [node("root", None, score=0.78), node("a", "root", score=0.80), node("b", "root", score=0.70),
            node("c", "a", score=0.82)]
    assert math.isclose(sum(r["P"] for r in candidates(tree, CFG)), 1.0)


def test_worked_example_is_not_too_sharp():
    ps = [r["P"] for r in candidates(roots(0.78, 0.79, 0.80, 0.83), CFG)]
    assert ps == pytest.approx([0.152, 0.184, 0.225, 0.440], abs=2e-3)


def test_probabilities_are_continuous_in_scores():
    before = candidates(roots(0.78, 0.79, 0.80, 0.83), CFG)[2]["P"]
    after = candidates(roots(0.78, 0.79, 0.80001, 0.83), CFG)[2]["P"]
    assert 0 < after - before < 1e-3


def test_values_within_the_noise_floor_are_near_uniform():
    ps = [r["P"] for r in candidates(roots(0.7500, 0.7502, 0.7503), CFG)]
    assert max(ps) / min(ps) < 1.5


def test_one_outlier_child_is_not_catastrophic():
    rows = by_id(candidates([node("a", None, score=0.80), node("a1", "a", score=0.40)], CFG))
    assert rows["a"]["value"] == pytest.approx(0.7 * 0.80 + 0.3 * (0.80 + 0.5 * 0.40) / 1.5)   # 0.76


def test_own_score_matters_more_than_the_subtree():
    tree = [node("a", None, score=0.80), node("a1", "a", score=0.70),
            node("b", None, score=0.76), node("b1", "b", score=0.86)]
    rows = by_id(candidates(tree, CFG))
    assert rows["a"]["value"] > rows["b"]["value"]


def test_large_subtree_is_down_weighted():
    tree = [node("a", None, score=0.8), node("b", None, score=0.8),
            *[node(f"a{i}", "a", score=0.8) for i in range(4)]]
    rows = by_id(candidates(tree, CFG))
    assert rows["a"]["value"] == pytest.approx(rows["b"]["value"])
    assert rows["a"]["w"] / rows["b"]["w"] == pytest.approx(0.5)              # size 4, scale 4


def test_failed_children_count_in_size_but_not_the_mean_and_interrupted_counts_nowhere():
    tree = [node("a", None, score=0.8), node("a1", "a", status="invalid_code"),
            node("a2", "a", status="crashed"), node("a3", "a", status="interrupted")]
    (row,) = candidates(tree, CFG)
    assert row["value"] == pytest.approx(0.8) and row["size"] == 2 and row["penalty"] == pytest.approx(2 / 3)


def test_select_parent_is_seeded_and_recorded(tmp_path):
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    for nid, parent, score in (("root", None, 0.7), ("n1", "root", 0.8), ("n2", "root", 0.6)):
        nodes.create(nid, parent, 0 if parent is None else 1)
        nodes.record_score(nid, score, ["m"], {"m": score})
    seed = selection_seed("run1", "n3")
    first = select_parent(conn, CFG, "n3", seed, rec)
    assert all(select_parent(conn, CFG, "n3", seed, rec) == first for _ in range(5))
    row = conn.execute("SELECT * FROM selection_events ORDER BY id LIMIT 1").fetchone()
    assert (row["child_id"], row["chosen"], row["seed"]) == ("n3", first, seed)
    assert [e["type"] for e in rec.read_events()].count("select") == 6


def test_update_values_writes_the_value(tmp_path):
    conn = open_db(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.7, ["m"], {"m": 0.7})
    nodes.create("n1", "root", 1), nodes.record_score("n1", 0.9, ["m"], {"m": 0.9})
    update_values(conn, CFG)
    assert nodes.get("root")["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.45) / 1.5)
    assert nodes.get("n1")["subtree_value"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_selection.py -q`
Expected: FAIL with `ModuleNotFoundError: ar_kernel.selection`.

- [ ] **Step 3: Implement `kernel/ar_kernel/selection.py`**

```python
"""Parent selection (user decision 2026-09-27, replacing spec 12's percentile ranks): a softmax
over each scored node's value -- mostly its own score, partly its subtree mean shrunk toward that
score -- with a temperature set by the spread of values (floored at proxy noise), a penalty for
large subtrees, and a uniform floor. Continuous in the scores; never sharp unless a lead is large."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time

from .archive.nodes import NodeStore


def candidates(nodes: list[dict], cfg) -> list[dict]:
    s = cfg.get("selection")
    decay, prior, share = float(s["decay"]), float(s["prior_weight"]), float(s["subtree_share"])
    kids: dict[str | None, list[dict]] = {}
    for n in nodes:
        kids.setdefault(n["parent_id"], []).append(n)
    rows = []
    for n in nodes:
        if n["status"] != "scored" or n["score"] is None:
            continue
        num, den, size = prior * n["score"], prior, 0.0
        frontier, k = [n["node_id"]], 0
        while frontier:
            k += 1
            nxt = []
            for pid in frontier:
                for child in kids.get(pid, []):
                    nxt.append(child["node_id"])
                    if child["status"] == "interrupted":
                        continue
                    size += decay ** (k - 1)
                    if child["status"] == "scored" and child["score"] is not None:
                        num += decay ** k * child["score"]
                        den += decay ** k
            frontier = nxt
        mean = num / den
        rows.append({"node_id": n["node_id"], "score": n["score"], "subtree_mean": mean,
                     "value": (1 - share) * n["score"] + share * mean, "size": size,
                     "penalty": 1 / (1 + size / float(s["size_scale"]))})
    if not rows:
        raise ValueError("no scored node to select a parent from")
    if len(rows) == 1:
        rows[0].update(w=1.0, P=1.0)
        return rows
    values = [r["value"] for r in rows]
    # the 1e-12 guard keeps tau > 0 even if noise_floor is configured as 0 and all values tie
    tau = float(s["temperature"]) * max(statistics.pstdev(values), float(s["noise_floor"]), 1e-12)
    top = max(values)
    for r in rows:
        r["w"] = math.exp((r["value"] - top) / tau) * r["penalty"]
    total, eps = sum(r["w"] for r in rows), float(s["epsilon"])
    for r in rows:
        r["P"] = (1 - eps) * r["w"] / total + eps / len(rows)
    return rows


def selection_seed(run_id: str, child_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_id}:{child_id}".encode()).digest()[:4], "big")


def select_parent(conn, cfg, child_id: str, seed: int, recorder) -> str:
    rows = candidates(NodeStore(conn).all(), cfg)
    chosen = random.Random(seed).choices([r["node_id"] for r in rows], weights=[r["P"] for r in rows])[0]
    conn.execute("INSERT INTO selection_events (child_id, chosen, seed, candidates, created_at) "
                 "VALUES (?,?,?,?,?)", (child_id, chosen, seed, json.dumps(rows), time.time()))
    recorder.event("select", payload={"child": child_id, "chosen": chosen, "seed": seed,
                                      "candidates": rows}, chosen=chosen, child=child_id)
    return chosen


def update_values(conn, cfg) -> None:
    nodes = NodeStore(conn)
    for row in candidates(nodes.all(), cfg):
        nodes.set_fields(row["node_id"], subtree_value=row["value"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_selection.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/selection.py tests/test_selection.py
git commit -m "feat(selection): continuous softmax parent selection with subtree value and size penalty

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: LLM spend ledger and optional run cap, enforced by the gateway

User decision 2026-09-27: one optional run cap in dollars, with **no cap by default**. Tokens are counted for `ar status` only.
- The gateway refuses non-mock calls once the cap is reached (HTTP 402).
- The loop checks the same ledger and stops the run.
- Costs are computed from the provider's reported usage and the configured prices. No provider is assumed.

**Files:**
- Create: `kernel/ar_kernel/budget.py`
- Modify: `kernel/ar_kernel/gateway/app.py` (`create_gateway_app`, `handle`), `kernel/ar_kernel/gateway/store.py` (`CallStore.end`)
- Test: `tests/test_budget.py`, `tests/test_gateway_app.py`

**Interfaces:**
- Produces:
  - `usage_tokens(usage) -> tuple[int, int, int]`: `(uncached input, cached input, output)`.
  - `Budget(max_usd=None, prices=None)` with:
    - `.record(status: int, usage) -> float | None`: the call's cost, or `None` without prices.
    - `.exhausted() -> str | None`: a human-readable reason once the cap is reached.
    - `.error_rate(window_s) -> tuple[float, int]`: error fraction and call count in the window.
    - `.load(run_dir)`: re-counts non-mock `llm.response` events already on disk (resume, status).
    - `.snapshot() -> dict`: `{usd, tokens, calls, max_usd}`.
  - `Budget.from_config(cfg)` raises `BudgetError` when `max_usd` is set but a price is missing.
  - `create_gateway_app(..., budget: Budget | None = None)`.
  - `CallStore.end(..., cost_usd=None, mock=False)` adds `cost_usd` and `mock` fields to `llm.response`.

- [ ] **Step 1: Write the failing tests**

`tests/test_budget.py`:

```python
import json

import pytest

from ar_kernel.budget import Budget, BudgetError, usage_tokens
from ar_kernel.config import KernelConfig

PRICES = {"input": 2.0, "cached_input": 0.5, "output": 8.0}


def test_usage_forms():
    assert usage_tokens({"prompt_tokens": 100, "completion_tokens": 10,
                         "prompt_tokens_details": {"cached_tokens": 60}}) == (40, 60, 10)
    assert usage_tokens({"prompt_tokens": 100, "completion_tokens": 10,
                         "prompt_cache_hit_tokens": 90}) == (10, 90, 10)
    assert usage_tokens({"input_tokens": 50, "output_tokens": 5,
                         "input_tokens_details": {"cached_tokens": 50}}) == (0, 50, 5)
    assert usage_tokens(None) == (0, 0, 0)


def test_cost_and_dollar_cap():
    b = Budget(max_usd=0.001, prices=PRICES)
    cost = b.record(200, {"prompt_tokens": 400, "completion_tokens": 50,
                          "prompt_tokens_details": {"cached_tokens": 200}})
    assert cost == pytest.approx((200 * 2 + 200 * 0.5 + 50 * 8) / 1e6)
    assert b.exhausted() is None
    b.record(200, {"prompt_tokens": 1000, "completion_tokens": 0})
    assert "budget" in b.exhausted()


def test_without_prices_cost_is_unknown_and_nothing_is_capped():
    b = Budget()
    assert b.record(200, {"prompt_tokens": 60, "completion_tokens": 50}) is None
    assert b.exhausted() is None and b.snapshot()["tokens"] == 110 and b.snapshot()["usd"] is None


def test_no_cap_by_default():
    b = Budget.from_config(KernelConfig.load())
    b.record(200, {"prompt_tokens": 10 ** 9, "completion_tokens": 10 ** 9})
    assert b.exhausted() is None


def test_dollar_cap_needs_prices():
    cfg = KernelConfig(raw={"budget": {"max_usd": 5, "usd_per_mtok": {"input": 1}}}, repo_root=None)
    with pytest.raises(BudgetError, match="usd_per_mtok"):
        Budget.from_config(cfg)


def test_load_recounts_real_calls_only(tmp_path):
    events = tmp_path / "telemetry" / "events"
    events.mkdir(parents=True)
    lines = [{"type": "llm.response", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "mock": False},
             {"type": "llm.response", "usage": {"prompt_tokens": 999, "completion_tokens": 9}, "mock": True},
             {"type": "tool.call"}]
    (events / "n1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    b = Budget(prices=PRICES)
    b.load(tmp_path)
    assert b.snapshot()["tokens"] == 15 and b.snapshot()["calls"] == 1


def test_load_twice_does_not_double_count(tmp_path):
    events = tmp_path / "telemetry" / "events"
    events.mkdir(parents=True)
    (events / "n1.jsonl").write_text(json.dumps({"type": "llm.response", "mock": False,
                                                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}) + "\n")
    b = Budget()
    b.load(tmp_path), b.load(tmp_path)
    assert b.snapshot()["tokens"] == 15 and b.snapshot()["calls"] == 1


def test_error_rate_window():
    b = Budget()
    for status in (200, 500, 502, 200):
        b.record(status, None)
    rate, n = b.error_rate(300)
    assert (rate, n) == (0.5, 4)
```

In `tests/test_gateway_app.py`, add a test that builds the app the way the existing upstream tests do (a stub `httpx.MockTransport` upstream returning a 200 with `usage`), passing `budget=Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})` (the stub's usage costs more than that):
- The first real call succeeds, and its `llm.response` event has a positive `cost_usd` and `mock is False`.
- The second call returns **402** with "budget" in the message. The upstream transport saw exactly one request, and no second `llm.request` event was recorded.
- A mock-token call (`mock_script="smoke"`) still succeeds after exhaustion, with `mock is True`.

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_budget.py tests/test_gateway_app.py -q`
Expected: FAIL (`ar_kernel.budget` missing; `create_gateway_app` has no `budget`).

- [ ] **Step 3: Implement `kernel/ar_kernel/budget.py`**

```python
"""LLM spend ledger and the optional run cap (user decision 2026-09-27: dollars only, none by
default). The gateway records every real call here and refuses new ones once the cap is
reached; the loop reads the same ledger to stop the run. Mock calls never count."""
from __future__ import annotations

import collections
import json
import threading
import time
from pathlib import Path

PRICE_KEYS = ("input", "cached_input", "output")


class BudgetError(ValueError):
    """The budget configuration cannot be enforced."""


def usage_tokens(usage) -> tuple[int, int, int]:
    """(uncached input, cached input, output) from a Chat Completions or Responses usage block."""
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int((details.get("cached_tokens") if isinstance(details, dict) else 0)
                 or usage.get("prompt_cache_hit_tokens") or 0)
    cached = min(cached, prompt)
    output = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return prompt - cached, cached, output


class Budget:
    def __init__(self, *, max_usd: float | None = None, prices: dict | None = None) -> None:
        self.prices = {k: (prices or {}).get(k) for k in PRICE_KEYS}
        if max_usd is not None and any(v is None for v in self.prices.values()):
            raise BudgetError("budget.max_usd needs budget.usd_per_mtok.{input,cached_input,output}")
        self.max_usd = max_usd
        self.usd, self.tokens, self.calls = 0.0, 0, 0
        self._statuses: collections.deque = collections.deque(maxlen=10000)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(max_usd=cfg.get("budget.max_usd"),
                   prices=cfg.get("budget.usd_per_mtok") or {})

    def cost(self, usage) -> float | None:
        if any(v is None for v in self.prices.values()):
            return None
        fresh, cached, output = usage_tokens(usage)
        return (fresh * self.prices["input"] + cached * self.prices["cached_input"]
                + output * self.prices["output"]) / 1e6

    def record(self, status: int, usage) -> float | None:
        cost = self.cost(usage)
        with self._lock:
            self.tokens += sum(usage_tokens(usage))
            self.calls += 1
            self.usd += cost or 0.0
            self._statuses.append((time.monotonic(), int(status)))
        return cost

    def exhausted(self) -> str | None:
        with self._lock:
            if self.max_usd is not None and self.usd >= self.max_usd:
                return f"run LLM budget spent: ${self.usd:.4f} of ${self.max_usd}"
        return None

    def error_rate(self, window_s: float) -> tuple[float, int]:
        cutoff = time.monotonic() - window_s
        with self._lock:
            recent = [s for t, s in self._statuses if t >= cutoff]
        if not recent:
            return 0.0, 0
        # Only upstream unavailability counts (rate limits, server and connection errors, 599 in
        # Upstream.post): a 400 caused by the agent's own request is not an outage.
        return sum(s == 429 or s >= 500 for s in recent) / len(recent), len(recent)

    def load(self, run_dir: Path) -> None:
        """Re-count the real calls already recorded for this run (resume and `ar status`). Resets
        the totals first, so calling it twice never double-counts. Statuses are not re-added: the
        error-rate window is about the live process."""
        with self._lock:
            self.usd, self.tokens, self.calls = 0.0, 0, 0
        for path in sorted((Path(run_dir) / "telemetry" / "events").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if '"llm.response"' not in line:
                    continue
                event = json.loads(line)
                if event.get("type") != "llm.response" or event.get("mock"):
                    continue
                cost = self.cost(event.get("usage"))
                with self._lock:
                    self.tokens += sum(usage_tokens(event.get("usage")))
                    self.calls += 1
                    self.usd += cost or 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {"usd": self.usd if all(v is not None for v in self.prices.values()) else None,
                    "tokens": self.tokens, "calls": self.calls,
                    "max_usd": self.max_usd}
```

Gateway changes (`app.py`):
- Add the `budget: Budget | None = None` parameter to `create_gateway_app` (import `from ..budget import Budget`).
- In `handle`, right after `mock = bool(caller.mock_script) or upstream is None`:

```python
        if not mock and budget is not None:
            why = budget.exhausted()
            if why:       # rejected like the checks above: not recorded, never forwarded
                return JSONResponse({"error": {"message": why}}, status_code=402)
```

- After the upstream/mock call, before `store.end`:

```python
        cost = budget.record(status, payload.get("usage") if isinstance(payload, dict) else None) \
            if (budget is not None and not mock) else None
```

- Pass `cost_usd=cost, mock=mock` to `store.end`.

`store.py`: `end(self, meta, caller, *, status, body, latency_s, attempts, cost_usd=None, mock=False)`, adding `cost_usd=cost_usd, mock=mock` to the `llm.response` event's inline fields.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_budget.py tests/test_gateway_app.py tests/test_gateway_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/budget.py kernel/ar_kernel/gateway/app.py kernel/ar_kernel/gateway/store.py tests/test_budget.py tests/test_gateway_app.py
git commit -m "feat(budget): LLM spend ledger and optional run cap enforced by the gateway

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Soft timeouts with liveness (spec §14.5)

**Files:**
- Create: `kernel/ar_kernel/liveness.py`
- Modify:
  - `kernel/ar_kernel/sandbox/runner.py` (`run_container`)
  - `kernel/ar_kernel/subproc.py` (`run_in_env`, `_wait`)
  - `kernel/ar_kernel/agent_phase.py` (`_run`)
  - `kernel/ar_kernel/tools/jobs.py` (`JobQueue.active_for_token`)
  - `kernel/ar_kernel/contract/verify.py` (smoke runs)
- Test: `tests/test_liveness.py`, `tests/test_subproc.py`, `tests/test_agent_phase.py`, `tests/test_jobs.py`, `tests/test_sandbox_runner.py` (docker)

**Interfaces:**
- Produces:
  - `Liveness(soft_s, *, probe_window_s, extension_frac, signals, hard_s=None, clock=time.monotonic)`:
    - `.expired() -> str | None`, called periodically. A non-`None` result is the reason, also kept in `.reason`.
    - `.add_signal(fn)` adds another progress signal.
    - `.extensions` counts extensions.
  - `Liveness.from_config(cfg, soft_s, signals, hard_s=None)`.
  - `tree_mark(*paths) -> tuple[int, int, int]`: `(files, bytes, newest mtime_ns)` under the paths.
  - `run_container(..., liveness=None, poll_s=5.0)`. `timeout_s` stays the hard cap.
  - `run_in_env(..., liveness=None)`. A stall raises `SubprocTimeout` with the reason in the `subproc.error` event.
  - `JobQueue.active_for_token(token) -> int`: jobs queued or running for that token.

- [ ] **Step 1: Write the failing tests**

`tests/test_liveness.py`:

```python
from ar_kernel.liveness import Liveness, tree_mark


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(clock, marks, **kw):
    it = iter(marks)
    return Liveness(100, probe_window_s=10, extension_frac=0.25, signals=[lambda: next(it)],
                    clock=clock, **kw)


def test_before_the_soft_deadline_nothing_is_probed():
    c = Clock()
    lv = make(c, [])                                   # a signal call would raise StopIteration
    c.t = 99
    assert lv.expired() is None


def test_progress_in_the_probe_window_extends_by_a_quarter_of_soft():
    c = Clock()
    lv = make(c, [1, 2, 2, 2])
    c.t = 100
    assert lv.expired() is None                        # probe starts, baseline 1
    c.t = 105
    assert lv.expired() is None and lv.extensions == 1 # changed -> deadline 105 + 25
    c.t = 129
    assert lv.expired() is None                        # before the new deadline: not probed
    c.t = 130
    assert lv.expired() is None                        # new probe, baseline 2
    c.t = 140
    assert "no sign of progress" in lv.expired()


def test_signals_are_throttled_during_the_probe_window():
    c, calls = Clock(), []
    lv = Liveness(100, probe_window_s=60, extension_frac=0.25, probe_every_s=30, clock=c,
                  signals=[lambda: calls.append(c.t) or 0])
    for t in range(100, 161):
        c.t = t
        lv.expired()
    assert calls == [100, 130, 160] and lv.reason                 # the window end is still checked


def test_hard_cap_ends_regardless():
    c = Clock()
    lv = make(c, [1, 2, 3, 4, 5], hard_s=150)
    c.t = 150
    assert "hard cap" in lv.expired() and lv.reason


def test_tree_mark_sees_new_and_grown_files(tmp_path):
    a = tree_mark(tmp_path)
    (tmp_path / "x").write_text("1")
    b = tree_mark(tmp_path)
    (tmp_path / "x").write_text("12")
    assert a != b != tree_mark(tmp_path)
    assert tree_mark(tmp_path / "missing") == (0, 0, 0)
```

`tests/test_subproc.py`:

```python
from ar_kernel.liveness import Liveness


def test_a_stalled_job_is_killed_by_liveness(tmp_path):
    rec = Recorder(tmp_path)
    lv = Liveness(1, probe_window_s=1, extension_frac=0.25, signals=[lambda: 0])
    started = time.monotonic()
    with pytest.raises(SubprocTimeout):
        run_in_env(ENV, ["python", "-c", "import time; time.sleep(120)"], cwd=tmp_path,
                   recorder=rec, liveness=lv, poll_s=0.2)
    assert time.monotonic() - started < 30
    err = [e for e in rec.read_events() if e["type"] == "subproc.error"][-1]
    assert "no sign of progress" in rec.load_payload(err["payload"])["message"]


def test_a_job_that_keeps_writing_is_extended(tmp_path):
    log = tmp_path / "log.txt"
    lv = Liveness(1, probe_window_s=1.5, extension_frac=0.5, signals=[lambda: tree_mark(log)])
    code = "import time\nfor i in range(8):\n    print(i, flush=True); time.sleep(0.5)\n"
    proc = run_in_env(ENV, ["python", "-c", code], cwd=tmp_path, liveness=lv, poll_s=0.2, log_path=log)
    assert proc.returncode == 0 and lv.extensions >= 1
```

(add `from ar_kernel.liveness import tree_mark`.)

`tests/test_jobs.py`: submit a job to a backend that blocks on an event. Assert `active_for_token(token) == 1` while it runs and `0` after it finishes. Use the existing blocking-backend fixture pattern in that file.

`tests/test_agent_phase.py`: add a test that `FakeRunner` receives a `liveness` kwarg whose `soft_s` equals `timeouts.edit_self_s`:

```python
def test_runner_gets_a_liveness_with_the_phase_soft_timeout(env):
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}})
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                  max_attempts=3, retry=None, nodes_remaining=5)
    lv = runner.calls[0]["liveness"]
    assert lv.soft_s == float(CFG.get("timeouts.edit_self_s")) and runner.calls[0]["timeout_s"] == 4 * lv.soft_s
```

`tests/test_sandbox_runner.py` (docker): the container `print('go', flush=True); time.sleep(600)` with `timeout_s=600` and a `Liveness(3, probe_window_s=3, extension_frac=0.25, signals=[lambda: 0])` (the CPU signal is added inside `run_container`; `sleep` uses no CPU). Assert `timed_out`, a duration under 60 s, and that the container is gone.

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_liveness.py tests/test_subproc.py tests/test_jobs.py tests/test_agent_phase.py -q`
Expected: FAIL (module missing, unknown kwargs).

- [ ] **Step 3: Implement**

`kernel/ar_kernel/liveness.py`:

```python
"""Soft timeouts with a probe window (spec 14.5). At the soft deadline the kernel watches the
phase's progress signals for one probe window: any change extends the deadline by
extension_frac x soft; none ends the phase. A hard cap, when set, ends it regardless.
GPU utilization is never a signal (verification-log finding 2)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


class Liveness:
    def __init__(self, soft_s: float, *, probe_window_s: float, extension_frac: float,
                 signals: list[Callable[[], object]], hard_s: float | None = None,
                 probe_every_s: float = 0.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.soft_s, self.probe_window_s, self.extension_frac = float(soft_s), probe_window_s, extension_frac
        # Signals can walk large trees (tree_mark over a WBench work dir), and callers poll every
        # second or so: during a probe window, evaluate them at most every probe_every_s.
        self.probe_every_s, self._last_eval = probe_every_s, float("-inf")
        self.hard_s, self.clock, self.signals = hard_s, clock, list(signals)
        self.started = clock()
        self.deadline = self.started + self.soft_s
        self.extensions = 0
        self.reason: str | None = None
        self._probe_start: float | None = None
        self._baseline = None

    @classmethod
    def from_config(cls, cfg, soft_s: float, signals: list, hard_s: float | None = None) -> "Liveness":
        return cls(soft_s, probe_window_s=60 * float(cfg.get("liveness.probe_window_min")),
                   extension_frac=float(cfg.get("liveness.extension_frac")), signals=signals, hard_s=hard_s,
                   probe_every_s=float(cfg.get("liveness.probe_every_s")))

    def add_signal(self, fn: Callable[[], object]) -> None:
        self.signals.append(fn)

    def expired(self) -> str | None:
        now = self.clock()
        if self.hard_s is not None and now - self.started >= self.hard_s:
            self.reason = f"hard cap of {self.hard_s:.0f}s reached"
            return self.reason
        if now < self.deadline:
            return None
        if (self._probe_start is not None and now - self._last_eval < self.probe_every_s
                and now - self._probe_start < self.probe_window_s):
            return None
        self._last_eval = now
        mark = tuple(fn() for fn in self.signals)
        if self._probe_start is None:
            self._probe_start, self._baseline = now, mark
            return None
        if mark != self._baseline:
            self.deadline = now + self.extension_frac * self.soft_s
            self.extensions += 1
            self._probe_start = None
            return None
        if now - self._probe_start >= self.probe_window_s:
            self.reason = (f"no sign of progress for {self.probe_window_s:.0f}s after the "
                           f"{self.soft_s:.0f}s soft timeout ({self.extensions} extensions)")
            return self.reason
        return None


def tree_mark(*paths) -> tuple[int, int, int]:
    """(files, bytes, newest mtime_ns) of the regular files at or under `paths`."""
    count = size = newest = 0
    for p in map(Path, paths):
        files = [p] if p.is_file() else (p.rglob("*") if p.is_dir() else [])
        for f in files:
            try:
                if f.is_symlink() or not f.is_file():
                    continue
                st = f.stat()
            except OSError:
                continue
            count, size, newest = count + 1, size + st.st_size, max(newest, st.st_mtime_ns)
    return count, size, newest
```

`runner.py`, `run_container(..., liveness=None, poll_s: float = 5.0, stats_every_s=30.0)`. Replace the blocking `docker wait` block with:

```python
        if liveness is not None:
            # Container CPU is a liveness signal (spec 14.5): count samples above 1%.
            liveness.add_signal(lambda: sum(1 for s in stats if _cpu(s) > 1.0))
        waiter = subprocess.Popen(["docker", "wait", name], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        hard = time.monotonic() + timeout_s
        reason = None
        while True:
            try:
                out, _ = waiter.communicate(timeout=poll_s)
                exit_code = int(out.strip() or -1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= hard:
                    reason = f"hard cap of {timeout_s:.0f}s reached"
                elif liveness is not None:
                    reason = liveness.expired()
                if reason:
                    timed_out = True
                    subprocess.run(["docker", "kill", name], capture_output=True)
                    waiter.kill()
                    waiter.communicate()
                    break
```

Add the helper and include `reason=reason` in the `sandbox.end` event fields:

```python
def _cpu(sample: dict) -> float:
    try:
        return float(str(sample.get("CPUPerc", "0")).rstrip("%") or 0)
    except ValueError:
        return 0.0
```

`subproc.py`:
- `run_in_env(..., liveness: "Liveness | None" = None)`.
- Call `_wait(proc, timeout, cancel, poll_s, liveness)`.
- In `_wait`, the fast path applies only when `cancel is None and liveness is None`. In the polling loop, after the `cancel` check, add:

```python
            if liveness is not None and liveness.expired():
                raise subprocess.TimeoutExpired(proc.args, timeout or 0)
```

- In the `except subprocess.TimeoutExpired:` branch of `run_in_env`, make the message `liveness.reason if (liveness is not None and liveness.reason) else f"timed out after {timeout}s"`, followed by `"; process group killed"`.

`jobs.py`:

```python
    def active_for_token(self, token: str) -> int:
        with self._cond:
            return sum(1 for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL)
```

`agent_phase._run`: after `soft = ...`, build the liveness and pass `liveness=liveness` to `env.runner(...)`:

```python
        liveness = Liveness.from_config(env.cfg, soft, signals=[
            lambda: tree_mark(dirs["workspace"], dirs["staging"]),          # workspace changes
            lambda: tree_mark(env.recorder.events_path(node)),               # gateway + tool calls
            lambda: env.queue.active_for_token(caller.token) and time.monotonic()])  # own GPU jobs
```

`contract/verify.py`: in `run()`, for the two smoke runs only (not the import probe), pass the following to `runner(...)`:

```python
liveness=Liveness.from_config(cfg, float(cfg.get("timeouts.contract_smoke_s")), signals=[lambda: tree_mark(ws), lambda: tree_mark(recorder.events_path(node))])
```

Pass `None` for the import probe.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`, then `conda run --no-capture-output -n autoresearcher python -m pytest -m docker tests/test_sandbox_runner.py -q`.
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/liveness.py kernel/ar_kernel/sandbox/runner.py kernel/ar_kernel/subproc.py kernel/ar_kernel/agent_phase.py kernel/ar_kernel/tools/jobs.py kernel/ar_kernel/contract/verify.py tests/test_liveness.py tests/test_subproc.py tests/test_jobs.py tests/test_agent_phase.py tests/test_sandbox_runner.py
git commit -m "feat(liveness): soft timeouts with a probe window for agent phases and subprocesses

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Alerts, GPU visibility check and container sweep

There is no GPU-idle wait (user, 2026-09-27: prefer the simpler implementation). The run's GPU lock stays a plain `threading.Lock`.

**Files:**
- Create: `kernel/ar_kernel/guards.py`
- Modify: `kernel/ar_kernel/sandbox/runner.py` (add `container_prefix`, `kill_run_containers`)
- Test: `tests/test_guards.py`, `tests/test_sandbox_runner.py`

**Interfaces:**
- Produces:
  - `alert(recorder, kind, message, *, level="error", **payload)`: an `alert` event in `run.jsonl` with inline fields `kind`, `level` and `message`.
  - `smi(args) -> str | None`: `nvidia-smi` stdout, or `None` when it is missing or fails. The monitor uses it too (Task 11).
  - `visible_gpus() -> set[int] | None`.
  - `check_visible(gpus, visible=visible_gpus)`: raises `GpuPolicyError` for a listed GPU that `nvidia-smi` does not show. It does nothing when `nvidia-smi` is unavailable.
  - `runner.container_prefix(run_id, node=None) -> str`.
  - `runner.kill_run_containers(run_id, node=None) -> list[str]`: force-removes the matching containers and returns their names.

- [ ] **Step 1: Write the failing tests**

`tests/test_guards.py`:

```python
import pytest

from ar_kernel.config import GpuPolicyError
from ar_kernel.guards import alert, check_visible
from ar_kernel.telemetry.recorder import Recorder


def test_alert_is_a_run_event(tmp_path):
    rec = Recorder(tmp_path)
    alert(rec, "node_failed", "n1 ended crashed", node_id="n1")
    (event,) = [e for e in rec.read_events() if e["type"] == "alert"]
    assert (event["kind"], event["level"], event["message"]) == ("node_failed", "error", "n1 ended crashed")


def test_invisible_gpu_is_refused():
    with pytest.raises(GpuPolicyError, match="7"):
        check_visible([0, 7], visible=lambda: {0, 1, 2, 3, 4, 5})
    check_visible([0, 5], visible=lambda: {0, 1, 2, 3, 4, 5})
    check_visible([0, 7], visible=lambda: None)          # no nvidia-smi: cannot check, no refusal
```

`tests/test_sandbox_runner.py`:

```python
from ar_kernel.sandbox.runner import container_prefix, kill_run_containers


def test_container_prefix_matches_container_name():
    assert container_name("r:1", "n2", "edit_self", 1).startswith(container_prefix("r:1", "n2"))
    assert container_prefix("r:1") == "ar-r-1-"


def test_kill_run_containers_removes_only_the_prefix(monkeypatch):
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        out = "ar-r1-n2-edit_self-1-abc\nar-r10-n2-edit_self-1-def\nother\n" if args[1] == "ps" else ""
        return subprocess.CompletedProcess(args, 0, out, "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert kill_run_containers("r1", "n2") == ["ar-r1-n2-edit_self-1-abc"]
    assert ["docker", "rm", "-f", "ar-r1-n2-edit_self-1-abc"] in calls
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_guards.py tests/test_sandbox_runner.py -q`
Expected: FAIL (module and functions missing).

- [ ] **Step 3: Implement**

`kernel/ar_kernel/guards.py`:

```python
"""Run alerts (spec 13.4) and the start-time GPU visibility check (14.6). There is deliberately no
wait for idle GPUs (user decision 2026-09-27)."""
from __future__ import annotations

import subprocess

from .config import GpuPolicyError


def alert(recorder, kind: str, message: str, *, level: str = "error", **payload) -> None:
    recorder.event("alert", payload={"kind": kind, "level": level, "message": message, **payload},
                   kind=kind, level=level, message=message)


def smi(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def visible_gpus() -> set[int] | None:
    out = smi(["--query-gpu=index", "--format=csv,noheader"])
    return None if out is None else {int(x) for x in out.split() if x.strip().isdigit()}


def check_visible(gpus: list[int], visible=visible_gpus) -> None:
    seen = visible()
    if seen is None:
        return
    missing = [g for g in gpus if g not in seen]
    if missing:
        raise GpuPolicyError(f"GPU(s) {missing} are not visible to nvidia-smi (visible: {sorted(seen)})")
```

`runner.py`:

```python
def container_prefix(run_id: str, node: str | None = None) -> str:
    raw = f"ar-{run_id}-" + (f"{node}-" if node else "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def kill_run_containers(run_id: str, node: str | None = None) -> list[str]:
    """Force-remove this run's (or node's) containers: on force stop, and on resume for what a
    killed kernel left running (spec 14.4). Files are never touched."""
    prefix = container_prefix(run_id, node)
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
                            capture_output=True, text=True).stdout
    names = [n for n in listed.split() if n.startswith(prefix)]
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return names
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/guards.py kernel/ar_kernel/sandbox/runner.py tests/test_guards.py tests/test_sandbox_runner.py
git commit -m "feat(guards): alerts, GPU visibility check, run container sweep

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Training and eval under liveness; `score_node` cleanup

**Files:**
- Modify:
  - `kernel/ar_kernel/train/runner.py`
  - `kernel/ar_kernel/train/recipe.py` (`lora_of`)
  - `kernel/ar_kernel/eval/render.py`
  - `kernel/ar_kernel/eval/wbench.py`
  - `kernel/ar_kernel/run.py` (`score_node`)
- Test: `tests/test_train_runner.py`, `tests/test_recipe.py`, `tests/test_run_bootstrap.py`

**Interfaces:**
- Produces:
  - `TrainRunner.train(resolved, gpus, node_id, node_dir)`:
    - No fixed timeout; liveness is `timeouts.train_s` soft with the signal `tree_mark(train.log)` and no hard cap.
    - A stall is `TrainOutcome(checkpoint=None, failure="infra", detail="training stalled: <reason>")`.
    - After success, `trainer_state.pt` is deleted from the checkpoint (spec §7.3.5).
  - `classify_timeout` and `TRAIN_TIMEOUT_SECONDS` are removed.
  - `lora_of(resolved_path) -> tuple[int, int]` returns `(lora.rank, lora.alpha)` of a resolved config.
  - `render_proxy(..., cfg)` and `run_wbench_phases(...)` run with liveness: `timeouts.eval_s` soft, signal `tree_mark` of their output tree, no hard cap.
  - `score_node` cleans up `merge_slot` and the regenerable eval directories on every exit path (try/finally).

- [ ] **Step 1: Write the failing tests**

`tests/test_train_runner.py`: delete `classify_timeout` from the import and delete its tests. Add:

```python
from ar_kernel.config import KernelConfig
from ar_kernel.subproc import SubprocTimeout
from ar_kernel.train import runner as runner_mod
from ar_kernel.train.runner import TrainRunner
from ar_kernel.telemetry.recorder import Recorder


def _resolved(tmp_path):
    out = tmp_path / "node" / "train" / "outputs"
    cfg = tmp_path / "train_config.yaml"
    cfg.write_text(f"run: {{output_dir: {out}}}\n")
    return cfg, out


def test_a_stalled_training_run_is_an_infra_failure_reported_to_the_agent(tmp_path, monkeypatch):
    resolved, _ = _resolved(tmp_path)
    seen = {}

    def fake_run(env, args, **kw):
        seen.update(kw)
        kw["liveness"].reason = "no sign of progress for 600s after the 172800s soft timeout"
        raise SubprocTimeout(args, 0, output="[Setup] rank 0/4\n")
    monkeypatch.setattr(runner_mod, "run_in_env", fake_run)
    out = TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert seen["timeout"] is None and seen["liveness"].soft_s == 172800
    assert (out.checkpoint, out.failure) == (None, "infra") and "stalled" in out.detail


def test_trainer_state_is_deleted_after_success(tmp_path, monkeypatch):
    resolved, outputs = _resolved(tmp_path)

    def fake_run(env, args, **kw):
        ck = outputs / "checkpoint-2"
        ck.mkdir(parents=True)
        for name in ("lora.safetensors", "history_encoder.pt", "trainer_state.pt"):
            (ck / name).write_text("x")
        kw["log_path"].write_text("[Train] step=2 epoch=0 loss=0.25 grad=0.05 lr=5.00e-05 time=7.8s\n")
        return subprocess.CompletedProcess(args, 0, kw["log_path"].read_text(), "")
    monkeypatch.setattr(runner_mod, "run_in_env", fake_run)
    out = TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert out.failure == "none" and not (out.checkpoint / "trainer_state.pt").exists()
    assert (out.checkpoint / "lora.safetensors").exists()
```

(add `import subprocess`.)

`tests/test_recipe.py`:

```python
def test_lora_of_reads_the_resolved_config(tmp_path):
    p = tmp_path / "train_config.yaml"
    p.write_text("lora: {rank: 32, alpha: 16}\n")
    assert lora_of(p) == (32, 16)
```

`tests/test_run_bootstrap.py`: add a test in the style of `test_score_node_degrades_...`. Monkeypatch `merge_lora` to create `ctx.run_dir / "merge_slot"` and return it, and `run_wbench_phases` to raise `RuntimeError("wbench gpu failed")`. Assert that `score_node(...)` raises, that `merge_slot` no longer exists, and that an `eval.cleanup` event was recorded.

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_train_runner.py tests/test_recipe.py tests/test_run_bootstrap.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`train/runner.py`:
- Delete `TRAIN_TIMEOUT_SECONDS` and `classify_timeout`.
- Import `from ..liveness import Liveness, tree_mark`.
- In `train`:

```python
        liveness = Liveness.from_config(self.cfg, float(self.cfg.get("timeouts.train_s")),
                                        signals=[lambda: tree_mark(log_path)])
        with self.recorder.span(...):
            try:
                proc = run_in_env(..., timeout=None, liveness=liveness, recorder=self.recorder,
                                  node=node_id, phase="train", log_path=log_path)
            except SubprocTimeout as exc:
                log = exc.output or ""
                return TrainOutcome(checkpoint=None, failure="infra", log_path=log_path,
                                    metrics=parse_train_lines(log),
                                    detail=f"training stalled: {liveness.reason}")
        ...
        if checkpoint is not None and failure == "none":
            (checkpoint / "trainer_state.pt").unlink(missing_ok=True)     # spec 7.3.5 / 15
```

`train/recipe.py`:

```python
def lora_of(resolved: Path) -> tuple[int, int]:
    lora = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))["lora"]
    return int(lora["rank"]), int(lora["alpha"])
```

`eval/render.py`, in `render_proxy`: replace `timeout=int(12 * 3600)` with `timeout=None`, and add

```python
liveness=Liveness.from_config(cfg, float(cfg.get("timeouts.eval_s")), signals=[lambda: tree_mark(videos_dir, render_config.parent / "logs")])
```

`eval/wbench.py`: same for every `main.py` phase and the VP script, with the signal `lambda: tree_mark(Path(work_dir) / model)`. Build a fresh `Liveness` per phase; `report2` keeps its fixed 3600 s timeout.

`run.py`, `score_node`: wrap everything after `merged = ...` so cleanup always runs.

```python
    merged = None
    try:
        if checkpoint is None:
            history = cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
        else:
            merged = merge_lora(cfg, checkpoint, rank, alpha, ctx.run_dir, ctx.recorder, node_id)
            history = Path(checkpoint) / "history_encoder.pt"
        ...                                                    # render, wbench, score, aggregates as before
        return score, {"metrics": per_metric, "aggregates": agg, "report": report}
    finally:
        removed = cleanup_eval(work_dir, model)
        slot = ctx.run_dir / "merge_slot"
        if slot.exists():
            shutil.rmtree(slot)
            removed.append("merge_slot")
        ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})
```

`merge_lora` creates the slot even when it fails midway, so the `finally` checks the slot path rather than `merged`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/train/runner.py kernel/ar_kernel/train/recipe.py kernel/ar_kernel/eval/render.py kernel/ar_kernel/eval/wbench.py kernel/ar_kernel/run.py tests/test_train_runner.py tests/test_recipe.py tests/test_run_bootstrap.py
git commit -m "feat(train,eval): liveness instead of fixed timeouts; score_node always cleans up

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The run kit (services built from config)

**Files:**
- Create: `kernel/ar_kernel/run_kit.py`
- Test: `tests/test_run_kit.py`

**Interfaces:**
- Consumes:
  - `build_gpu_backends`, `register_gpu_tools`, `register_caption_tool` (Plan 3).
  - `DataTools`, `HfTools`, `JobQueue`, `register_*`, `create_gateway_app`, `Upstream.from_env`, `CallStore`, `MockBook`, `RunServices`, `socket_dir_for` and `ContractHarness` (Plan 2).
  - `Budget` (Task 4) and `alert` (Task 6).
- Produces:
  - `RunKit` dataclass: `registry, queue, gpu_lock, budget, services, harness, socket_dir, default_model, gateway_app, tools_app`.
  - `build_run_kit(cfg, run_dir, gpus, recorder, environ) -> RunKit`. It raises `ValueError` when `OPENAI_API_KEY` or `OPENAI_MODEL` is empty, and `BudgetError` from `Budget.from_config`.
  - `RunKit.start()` starts the services and the contract harness.
  - `RunKit.stop()`:
    - It always stops the services and the harness, even when `JobQueue.shutdown()` raises.
    - It records an `alert` (`kind="shutdown"`) for that `RuntimeError` and returns normally (Plan 2 contract item 9).

- [ ] **Step 1: Write the failing tests**

```python
import threading

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.run_kit import build_run_kit
from ar_kernel.telemetry.recorder import Recorder

ENV = {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "model-x", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}


def test_kit_uses_the_run_config_and_registers_enabled_tools(tmp_path, monkeypatch):
    cfg = KernelConfig(raw={**KernelConfig.load().raw, "budget": {"max_usd": 7, "usd_per_mtok": {"input": 1, "cached_input": 1, "output": 1}}}, repo_root=KernelConfig.load().repo_root)
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    kit = build_run_kit(cfg, tmp_path, [0, 1, 2, 3], Recorder(tmp_path), ENV)
    try:
        assert kit.default_model == "model-x" and kit.budget.max_usd == 7
        assert kit.queue.gpu_lock is kit.gpu_lock and kit.queue.wait_cap_s == float(cfg.get("tools.job_wait_max_s"))
    finally:
        kit.queue.shutdown()


def test_missing_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_run_kit(KernelConfig.load(), tmp_path, [0, 1, 2, 3], Recorder(tmp_path), {"OPENAI_MODEL": "m"})


def test_stop_survives_a_stuck_job_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    rec = Recorder(tmp_path)
    kit = build_run_kit(KernelConfig.load(), tmp_path, [0, 1, 2, 3], rec, ENV)
    stopped = []
    monkeypatch.setattr(kit.queue, "shutdown", lambda: (_ for _ in ()).throw(RuntimeError("stuck")))
    def services_stop():
        stopped.append("services")
        raise RuntimeError("service thread did not join")
    monkeypatch.setattr(kit.services, "stop", services_stop)
    monkeypatch.setattr(kit.harness, "stop", lambda: stopped.append("harness"))
    with pytest.raises(RuntimeError, match="join"):
        kit.stop()
    assert stopped == ["services", "harness"]                     # the harness still stopped
    assert [e["kind"] for e in rec.read_events() if e["type"] == "alert"] == ["shutdown"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_run_kit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `kernel/ar_kernel/run_kit.py`**

```python
"""Everything a run serves to its agents, built from the run's config (Plan 2 contract items 8-9)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .budget import Budget
from .contract.verify import ContractHarness
from .gateway.app import Upstream, create_gateway_app
from .gateway.mock import MockBook
from .gateway.store import CallStore
from .guards import alert
from .services import RunServices, socket_dir_for
from .tools.captioner import register_caption_tool
from .tools.context import TokenRegistry
from .tools.data_tools import DataTools, register_data_tools
from .tools.gpu_jobs import build_gpu_backends, register_gpu_tools
from .tools.hf_tools import HfTools, register_hf_tools
from .tools.jobs import JobQueue, register_job_tools
from .tools.server import ToolKit, build_tool_app, new_mcp


@dataclass
class RunKit:
    registry: TokenRegistry
    queue: JobQueue
    gpu_lock: threading.Lock
    budget: Budget
    services: RunServices
    harness: ContractHarness
    socket_dir: Path
    default_model: str
    gateway_app: object
    tools_app: object
    recorder: object

    def start(self) -> None:
        self.services.start(self.gateway_app, self.tools_app)
        self.harness.start()

    def stop(self) -> None:
        try:
            self.queue.shutdown()
        except RuntimeError as exc:           # a job may still hold a GPU; recorded, never fatal
            alert(self.recorder, "shutdown", str(exc))
        finally:
            try:
                self.services.stop()        # raises if a service thread will not join
            finally:
                self.harness.stop()


def build_run_kit(cfg, run_dir: Path, gpus: list[int], recorder, environ) -> RunKit:
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if not environ.get(key, "").strip():
            raise ValueError(f"{key} is empty; set it in AutoResearcher/.env before `ar run`")
    budget = Budget.from_config(cfg)
    budget.load(run_dir)
    registry = TokenRegistry(recorder)
    gpu_lock = threading.Lock()
    queue = JobQueue(recorder, gpu_lock, wait_cap_s=float(cfg.get("tools.job_wait_max_s")))
    for backend in build_gpu_backends(cfg, run_dir, gpus, registry, recorder):
        queue.register(backend)
    kit, mcp = ToolKit(registry, recorder), new_mcp()
    register_data_tools(mcp, kit, DataTools(cfg, run_dir, recorder, gpus, gpu_lock))
    register_hf_tools(mcp, kit, HfTools(cfg, Path(run_dir) / "hf_tmp"))
    register_job_tools(mcp, kit, queue)
    register_gpu_tools(mcp, kit, queue)
    register_caption_tool(mcp, kit, queue)
    model = environ["OPENAI_MODEL"]
    gateway = create_gateway_app(
        registry=registry, store=CallStore(recorder),
        allowed_models=set(cfg.get("gateway.model_allowlist") or []) | {model},
        upstream=Upstream.from_env(environ, timeout_s=float(cfg.get("gateway.upstream_timeout_s")),
                                   retries=int(cfg.get("gateway.upstream_retries"))),
        mocks=MockBook.default(), budget=budget)
    socket_dir = socket_dir_for(run_dir)
    return RunKit(registry=registry, queue=queue, gpu_lock=gpu_lock, budget=budget,
                  services=RunServices(socket_dir), harness=ContractHarness(cfg, run_dir, recorder),
                  socket_dir=socket_dir, default_model=model, gateway_app=gateway,
                  tools_app=build_tool_app(mcp), recorder=recorder)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_run_kit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/run_kit.py tests/test_run_kit.py
git commit -m "feat(run_kit): build gateway, tool server, GPU jobs and contract harness from the run config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The loop (spec §7.1–7.2)

**Files:**
- Create: `kernel/ar_kernel/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes:
  - Tasks 2–8.
  - `run_edit_self`, `run_improve_recipe`, `PhaseEnv`, `verify_contract`, `Gate`, `TrainRunner`, `TrainOutcome`, `score_node`, `lora_of`, `AgentsRepo`, `CommitStore`, `BlobStore`, `ClipStore`.
- Produces:
  - `Phases` dataclass of callables. The defaults are the real steps; tests swap them:
    - `edit_self(env, **kw) -> PhaseOutcome`
    - `contract(**kw) -> ContractReport`
    - `improve_recipe(env, **kw) -> PhaseOutcome`
    - `gate(loop, recipe, data_commit, parent_commit, node, attempt_dir) -> GateResult`
    - `train(loop, resolved, node, attempt_dir) -> TrainOutcome`
    - `score(loop, node, checkpoint, resolved) -> (score, detail)`
  - `StopRun(Exception)`: raised with a reason when the budget is exhausted mid-node, or when an agent attempt fails during an LLM provider outage (`_check_outage`, thresholds under `gateway.outage_*`). The node stays `running`, and resume marks it `interrupted`.
  - `Loop(cfg, ctx, kit, repo, *, max_nodes, phases=None)`:
    - `.run() -> str` returns the exit reason: `"graceful stop"`, `"max_nodes reached"` or the budget reason.
    - `.graceful` is a `threading.Event`.
    - `.state_path` is `runs/<id>/control/state.json`, written as `{node, phase, attempt, since}` at every step change.
  - Node artifacts (read by `context_bundle`):
    - `nodes/<n>/edit.json` (the passing `EditResult`)
    - `nodes/<n>/recipe.yaml`
    - `nodes/<n>/rationale.md`
    - `nodes/<n>/eval/aggregates.json`
  - Events:
    - `node.created` in `run.jsonl`. Node ids come from the archive (`n<max k + 1>`): rows are never deleted, so ids are never reused, and a kill between the insert and the event cannot produce a duplicate id.
    - `node.end` in `run.jsonl` with the status and error.
    - `alert` for every failed node, exhausted retries and the budget stop.

- [ ] **Step 1: Write the failing tests** (`tests/test_loop.py`; the §16.2 scenarios are in Task 12)

```python
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ar_kernel.agent_phase import PhaseOutcome
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractReport
from ar_kernel.loop import Loop, Phases
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import GateResult
from ar_kernel.train.runner import TrainOutcome
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
SEED = Path(__file__).resolve().parents[1] / "seed_agent"


def outcome(ok, attempt_dir, result=None, error=None, commit=None):
    return PhaseOutcome(ok, result, error, 0, False, commit, attempt_dir, 1.0)


class Script:
    """Scripted phases: each list is consumed per call; default is success."""
    def __init__(self, tmp, edit=(), contract=(), recipe=(), gate=(), train=(), score=()):
        self.tmp, self.q = tmp, {k: list(v) for k, v in dict(edit=edit, contract=contract, recipe=recipe,
                                                               gate=gate, train=train, score=score).items()}
        self.retries = []

    def _next(self, key, default):
        return self.q[key].pop(0) if self.q[key] else default

    def edit_self(self, env, *, node, base_commit, attempt, retry, **kw):
        self.retries.append(("edit_self", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"edit_self-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        ok = self._next("edit", True)
        return outcome(ok, d, {"summary": "s", "component": "prompts"} if ok else None,
                       None if ok else "agent failed", commit=base_commit)

    def contract(self, *, node, attempt, **kw):
        return ContractReport(ok=self._next("contract", True))

    def improve_recipe(self, env, *, node, attempt, retry, **kw):
        self.retries.append(("improve_recipe", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"improve_recipe-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        ok = self._next("recipe", True)
        return outcome(ok, d, {"data_commit": "c" * 64, "recipe": {"optimizer.max_steps": 2},
                               "rationale": "why"} if ok else None, None if ok else "no commit")

    def gate(self, loop, recipe, data_commit, parent_commit, node, attempt_dir):
        if not self._next("gate", True):
            return GateResult(ok=False, failures=["dataset d has 1 clips, fewer than 4 GPUs"])
        resolved = attempt_dir / "train_config.yaml"
        resolved.write_text("lora: {rank: 16, alpha: 16}\n")
        return GateResult(ok=True, resolved_path=resolved)

    def train(self, loop, resolved, node, attempt_dir):
        log = attempt_dir / "train" / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA out of memory\n")
        if not self._next("train", True):
            return TrainOutcome(None, "recipe", log, detail="CUDA OOM")
        ck = attempt_dir / "train" / "outputs" / "checkpoint-2"
        ck.mkdir(parents=True)
        return TrainOutcome(ck, "none", log)

    def score(self, loop, node, checkpoint, resolved):
        s = self._next("score", 0.8)
        if isinstance(s, Exception):
            raise s
        return s, {"metrics": {"m": s}, "aggregates": {"metrics": {"m": s}}}

    def phases(self):
        return Phases(edit_self=self.edit_self, contract=self.contract, improve_recipe=self.improve_recipe,
                      gate=self.gate, train=self.train, score=self.score)


@pytest.fixture
def make_loop(tmp_path):
    run = tmp_path / "run"

    def make(script, max_nodes=1, budget=None):
        rec = Recorder(run)
        ctx = RunContext(run_dir=run, conn=open_db(run), recorder=rec, gpus=[0, 1, 2, 3],
                         metric_set=["m"], case_ids=["1"], versions={})
        kit = SimpleNamespace(registry=None, queue=None, gpu_lock=threading.Lock(),
                              budget=budget or Budget(), harness=None, socket_dir=tmp_path / "sock",
                              default_model="mock-model")
        return Loop(CFG, ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=max_nodes,
                    phases=script.phases())
    return run, make


def test_root_then_one_scored_child(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])                      # root 0.7, n1 0.9
    loop = make(script, max_nodes=1)
    assert loop.run() == "max_nodes reached"
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert nodes["root"]["status"] == "scored" and nodes["n1"]["status"] == "scored"
    assert nodes["n1"]["edit_component"] == "prompts" and nodes["n1"]["checkpoint_path"].startswith("nodes/n1/")
    assert json.loads((run / "nodes" / "n1" / "edit.json").read_text())["summary"] == "s"
    assert (run / "nodes" / "n1" / "recipe.yaml").exists() and (run / "nodes" / "n1" / "rationale.md").exists()
    assert json.loads((run / "nodes" / "n1" / "eval" / "aggregates.json").read_text())["metrics"]["m"] == 0.9
    assert nodes["root"]["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.5 * 0.9) / 1.5)
    assert not (run / "staging" / "n1").exists()


def test_contract_failure_retries_with_the_report(make_loop):
    run, make = make_loop
    script = Script(run, contract=[False, True])
    make(script).run()
    edits = [r for r in script.retries if r[0] == "edit_self"]
    assert [a for _, _, a, _ in edits] == [1, 2] and edits[1][3]["kind"] == "contract"


def test_edit_exhaustion_is_invalid_code(make_loop):
    run, make = make_loop
    loop = make(Script(run, edit=[False, False, False]))
    loop.run()
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "invalid_code" and "agent failed" in n1["error"]


def test_training_failure_goes_back_to_the_agent_then_train_failed(make_loop):
    run, make = make_loop
    script = Script(run, train=[False, False, False])
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert [r[3]["kind"] if r[3] else None for r in recipes] == [None, "train", "train"]
    assert "CUDA out of memory" in recipes[1][3]["log_tail"]
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "train_failed"


def test_gate_failure_then_success(make_loop):
    run, make = make_loop
    script = Script(run, gate=[False, True])
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert recipes[1][3] == {"kind": "gate", "failures": ["dataset d has 1 clips, fewer than 4 GPUs"]}
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "scored"


def test_eval_failure_is_eval_failed_and_the_loop_continues(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("wbench gpu failed")]), max_nodes=2)
    loop.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert nodes["n1"]["status"] == "eval_failed" and "wbench gpu failed" in nodes["n1"]["error"]
    assert nodes["n2"]["status"] == "scored"
    kinds = [e["kind"] for e in loop.ctx.recorder.read_events() if e["type"] == "alert"]
    assert "node_failed" in kinds


def test_unexpected_exception_is_crashed(make_loop):
    run, make = make_loop
    script = Script(run)
    script.gate = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
    loop = make(script, max_nodes=1)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "crashed"


def test_provider_outage_stops_the_run_instead_of_failing_the_node(make_loop):
    run, make = make_loop
    budget = Budget()
    script = Script(run, edit=[False, False, False])
    real_edit = script.edit_self

    def outage(env, **kw):
        for _ in range(3):
            budget.record(503, None)                    # what the gateway records during an outage
        return real_edit(env, **kw)
    loop = make(script, max_nodes=3, budget=budget)
    loop.phases.edit_self = outage
    assert "outage" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"      # -> interrupted on resume
    assert NodeStore(loop.ctx.conn).attempts("n1", "edit_self") == []     # the attempt was not charged


def test_agent_side_400s_are_not_an_outage(make_loop):
    run, make = make_loop
    budget = Budget()
    script = Script(run, edit=[False, True])
    real_edit = script.edit_self

    def bad_requests(env, **kw):
        for _ in range(3):
            budget.record(400, None)
        return real_edit(env, **kw)
    loop = make(script, max_nodes=1, budget=budget)
    loop.phases.edit_self = bad_requests
    assert loop.run() == "max_nodes reached"


def test_budget_spent_during_the_last_attempt_is_a_stop_not_invalid_code(make_loop):
    run, make = make_loop
    budget = Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})
    script = Script(run, edit=[False, False, False])
    real_edit = script.edit_self

    def spend_on_third(env, **kw):
        if kw["attempt"] == 3:
            budget.record(200, {"prompt_tokens": 5, "completion_tokens": 0})
        return real_edit(env, **kw)
    loop = make(script, max_nodes=3, budget=budget)
    loop.phases.edit_self = spend_on_third
    assert "budget" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"


def test_budget_exhaustion_stops_the_run_and_leaves_the_node_running(make_loop):
    run, make = make_loop
    budget = Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})
    script = Script(run)
    real_edit = script.edit_self

    def spend(env, **kw):
        budget.record(200, {"prompt_tokens": 5, "completion_tokens": 0})
        return real_edit(env, **kw)
    script.edit_self = spend
    loop = make(script, max_nodes=3, budget=budget)
    assert "budget" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"


def test_graceful_flag_finishes_the_current_node(make_loop):
    run, make = make_loop
    script = Script(run)
    loop = make(script, max_nodes=5)
    real_score = script.score

    def score_and_stop(*a, **k):
        loop.graceful.set()
        return real_score(*a, **k)
    script.score = score_and_stop
    loop.phases.score = score_and_stop
    assert loop.run() == "graceful stop"
    # root's score call sets the flag, so no child is started
    assert [n["node_id"] for n in NodeStore(loop.ctx.conn).all()] == ["root"]


def test_interrupted_nodes_do_not_count_and_ids_are_never_reused(make_loop):
    run, make = make_loop
    loop = make(Script(run), max_nodes=1)
    loop.run()
    NodeStore(loop.ctx.conn).set_status("n1", "interrupted")   # as resume marks an unfinished node
    loop2 = make(Script(run), max_nodes=1)
    loop2.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop2.ctx.conn).all()}
    assert {k: v["status"] for k, v in nodes.items()} == {"root": "scored", "n1": "interrupted", "n2": "scored"}
    assert nodes["n2"]["parent_id"] == "root"                  # an interrupted node is never a parent
    assert (run / "nodes" / "n1" / "edit.json").exists()       # and keeps its files
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_loop.py -q`
Expected: FAIL (`ar_kernel.loop` missing).

- [ ] **Step 3: Implement `kernel/ar_kernel/loop.py`**

```python
"""The loop (spec 7.1-7.2): one node at a time. Nothing pauses (user decision 2026-09-27): a
failure the agent can act on goes back to it as the next attempt's retry report; any other
failure ends the node with a status and an error that later agents see in the lineage."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from .agent_phase import PhaseEnv, run_edit_self, run_improve_recipe
from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.nodes import NodeStore, run_rel
from .contract.verify import verify_contract
from .guards import alert
from .run import score_node
from .selection import select_parent, selection_seed, update_values
from .train.gate import Gate
from .train.recipe import lora_of
from .train.runner import TrainOutcome, TrainRunner


class StopRun(Exception):
    """End the run now (budget spent); resume marks the running node `interrupted`."""


def _gate(loop, recipe, data_commit, parent_commit, node, attempt_dir):
    conn = loop.ctx.conn
    store = CommitStore(conn, BlobStore(loop.ctx.run_dir, conn), ClipStore(conn))
    with loop.kit.gpu_lock:
        return Gate(loop.cfg, store, loop.ctx.recorder).check(
            recipe, data_commit, parent_commit, node, attempt_dir, loop.ctx.run_dir, loop.ctx.gpus)


def _train(loop, resolved, node, attempt_dir):
    runner = TrainRunner(loop.cfg, loop.ctx.recorder)
    with loop.kit.gpu_lock:
        try:
            runner.precache(resolved, loop.ctx.gpus, node)
        except RuntimeError as exc:
            return TrainOutcome(None, "infra", Path(attempt_dir) / "train" / "train.log",
                                detail=f"prompt precache failed: {exc}")
        return runner.train(resolved, loop.ctx.gpus, node, attempt_dir)


def _score(loop, node, checkpoint, resolved):
    rank, alpha = lora_of(resolved) if resolved is not None else (0, 0)
    with loop.kit.gpu_lock:
        return score_node(loop.cfg, loop.ctx, node, checkpoint, rank, alpha)


@dataclass
class Phases:
    edit_self: Callable = run_edit_self
    contract: Callable = verify_contract
    improve_recipe: Callable = run_improve_recipe
    gate: Callable = _gate
    train: Callable = _train
    score: Callable = _score


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


class Loop:
    def __init__(self, cfg, ctx, kit, repo, *, max_nodes: int, phases: Phases | None = None) -> None:
        self.cfg, self.ctx, self.kit, self.repo, self.max_nodes = cfg, ctx, kit, repo, int(max_nodes)
        self.phases = phases or Phases()
        self.nodes = NodeStore(ctx.conn)
        self.graceful = threading.Event()
        self.state_path = Path(ctx.run_dir) / "control" / "state.json"
        self.env = PhaseEnv(cfg=cfg, run_dir=ctx.run_dir, run_id=Path(ctx.run_dir).name,
                            recorder=ctx.recorder, registry=kit.registry, queue=kit.queue, repo=repo,
                            socket_dir=kit.socket_dir, gpus=ctx.gpus, default_model=kit.default_model)

    # -- bookkeeping -------------------------------------------------------------------------
    def _state(self, node: str | None, phase: str, attempt: int = 0) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"node": node, "phase": phase, "attempt": attempt, "since": time.time()}))
        tmp.replace(self.state_path)

    def _children(self) -> int:
        """Nodes counted toward max_nodes: every child except `interrupted` ones (user decision)."""
        return sum(1 for n in self.nodes.all() if n["parent_id"] is not None and n["status"] != "interrupted")

    def _next_id(self) -> str:
        """From the archive, not telemetry: rows are never deleted (interrupted nodes stay), and a
        kill between nodes.create and the node.created event must not hand out the id again."""
        nums = [int(n["node_id"][1:]) for n in self.nodes.all()
                if n["node_id"].startswith("n") and n["node_id"][1:].isdigit()]
        return f"n{max(nums, default=0) + 1}"

    def _check_budget(self) -> None:
        why = self.kit.budget.exhausted()
        if why:
            alert(self.ctx.recorder, "budget", why)
            raise StopRun(why)

    def _check_outage(self) -> None:
        """After a failed agent attempt: if the LLM provider is down, stop the run instead of
        charging the attempt (user decision 2026-09-27, option A: a stop, never a pause)."""
        rate, calls = self.kit.budget.error_rate(60 * float(self.cfg.get("gateway.outage_window_min")))
        if calls >= int(self.cfg.get("gateway.outage_min_calls")) and \
                rate >= float(self.cfg.get("gateway.outage_error_rate")):
            why = f"LLM provider outage: {rate:.0%} of the last {calls} calls failed upstream; resume later"
            alert(self.ctx.recorder, "llm_outage", why)
            raise StopRun(why)

    def _node_dir(self, node: str) -> Path:
        return Path(self.ctx.run_dir) / "nodes" / node

    # -- the run -----------------------------------------------------------------------------
    def run(self) -> str:
        self.ensure_root()
        while True:
            if self.graceful.is_set():
                return "graceful stop"
            if self._children() >= self.max_nodes:
                return "max_nodes reached"
            try:
                self._check_budget()
                self.cycle()
            except StopRun as exc:
                return str(exc)

    def ensure_root(self) -> None:
        try:
            root = self.nodes.get("root")
        except KeyError:
            commit = self.repo.resolve(self.repo.branch_ref("root")) or \
                self.repo.init(self.cfg.repo_root / "seed_agent")
            self.nodes.create("root", None, 0)
            self.nodes.set_fields("root", agent_commit=commit)
            root = self.nodes.get("root")
        if root["status"] == "scored":
            return
        self._state("root", "eval")
        score, detail = self.phases.score(self, "root", None, None)   # a failure here ends the run
        self._record_score("root", score, detail)

    def cycle(self) -> None:
        child = self._next_id()
        self._state(child, "select")
        parent_id = select_parent(self.ctx.conn, self.cfg, child,
                                  selection_seed(Path(self.ctx.run_dir).name, child), self.ctx.recorder)
        parent = self.nodes.get(parent_id)
        self.nodes.create(child, parent_id, parent["depth"] + 1)
        self.ctx.recorder.event("node.created", payload={"node": child, "parent": parent_id}, child=child)
        started, status, error = time.time(), "scored", None
        try:
            code, error = self._edit(child, parent)
            if code is None:
                status = "invalid_code"
                return
            trained, status, error = self._recipe(child, parent, code)
            if trained is None:
                return
            checkpoint, resolved = trained
            self._state(child, "eval")
            try:
                score, detail = self.phases.score(self, child, checkpoint, resolved)
            except Exception as exc:                              # noqa: BLE001 -- merge/render/WBench
                status, error = "eval_failed", f"{type(exc).__name__}: {exc}"
                return
            self._record_score(child, score, detail)
        except StopRun:
            status = None                                         # left running: interrupted on resume
            raise
        except Exception as exc:                                  # noqa: BLE001 -- spec 14.2 last row
            status, error = "crashed", f"{type(exc).__name__}: {exc}"
        except BaseException:                                     # ForceStop / KeyboardInterrupt
            status = None                                         # left running: interrupted on resume
            raise
        finally:
            if status is not None:
                self._finish(child, status, error, started)

    # -- steps 3-4 ---------------------------------------------------------------------------
    def _edit(self, child: str, parent: dict) -> tuple[str | None, str | None]:
        n = int(self.cfg.get("retries.edit_self"))
        base, retry, prev_ws = parent["agent_commit"], None, None
        for k in range(1, n + 1):
            self._check_budget()
            self._state(child, "edit_self", k)
            out = self.phases.edit_self(self.env, conn=self.ctx.conn, node=child, parent_id=parent["node_id"],
                                        base_commit=base, attempt=k, max_attempts=n, retry=retry,
                                        nodes_remaining=self.max_nodes - self._children(),
                                        previous_workspace=prev_ws)
            base = out.commit or base
            prev_ws = Path(out.attempt_dir) / "workspace"
            if not out.ok:
                self._check_budget()        # a 402 from the gateway is a spent budget, not an agent failure
                self._check_outage()        # a provider outage is not an agent failure either
                retry = {"kind": "edit_self", "error": out.error}
                self.nodes.add_attempt(child, "edit_self", k, "failed", retry)
                continue
            self._state(child, "contract", k)
            report = self.phases.contract(cfg=self.cfg, run_dir=self.ctx.run_dir,
                                          run_id=Path(self.ctx.run_dir).name, repo=self.repo,
                                          commit=out.commit, harness=self.kit.harness,
                                          recorder=self.ctx.recorder, node=child, attempt=k)
            if not report.ok:
                retry = report.to_retry()
                self.nodes.add_attempt(child, "edit_self", k, "contract_failed", retry)
                continue
            self.nodes.add_attempt(child, "edit_self", k, "passed", {"commit": out.commit})
            self.repo.set_ref(self.repo.branch_ref(child), out.commit)
            self._node_dir(child).mkdir(parents=True, exist_ok=True)
            (self._node_dir(child) / "edit.json").write_text(json.dumps(out.result, indent=1))
            self.nodes.set_fields(child, agent_commit=out.commit, edit_component=out.result.get("component"))
            return out.commit, None
        return None, f"edit_self retries exhausted; last failure: {json.dumps(retry)[:2000]}"

    # -- steps 5-7 ---------------------------------------------------------------------------
    def _recipe(self, child: str, parent: dict, code: str):
        n = int(self.cfg.get("retries.improve_recipe"))
        retry, prev_ws, status = None, None, "invalid_recipe"
        for k in range(1, n + 1):
            self._check_budget()
            self._state(child, "improve_recipe", k)
            out = self.phases.improve_recipe(self.env, conn=self.ctx.conn, node=child,
                                             parent_id=parent["node_id"], agent_commit=code, attempt=k,
                                             max_attempts=n, retry=retry,
                                             nodes_remaining=self.max_nodes - self._children(),
                                             previous_workspace=prev_ws)
            adir = Path(out.attempt_dir)
            prev_ws = adir / "workspace"
            if not out.ok:
                self._check_budget()        # a 402 from the gateway is a spent budget, not an agent failure
                self._check_outage()        # a provider outage is not an agent failure either
                retry, status = {"kind": "improve_recipe", "error": out.error}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "failed", retry)
                continue
            res = out.result
            self._state(child, "gate", k)
            gate = self.phases.gate(self, res["recipe"], res["data_commit"], parent["data_commit"], child, adir)
            if not gate.ok:
                retry, status = {"kind": "gate", "failures": gate.failures}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "gate_failed", retry)
                continue
            self._record_recipe(child, res, gate.resolved_path)
            self._state(child, "train", k)
            trained = self.phases.train(self, gate.resolved_path, child, adir)
            if trained.checkpoint is None:
                retry = {"kind": "train", "failure": trained.failure, "detail": trained.detail,
                         "log_tail": _tail(trained.log_path)}
                status = "train_failed"
                self.nodes.add_attempt(child, "improve_recipe", k, "train_failed", retry)
                alert(self.ctx.recorder, "train_failed", f"{child} attempt {k}: {trained.failure} "
                      f"{trained.detail}".strip(), level="warning", node_id=child)
                continue
            self.nodes.add_attempt(child, "improve_recipe", k, "passed", {"checkpoint": str(trained.checkpoint)})
            self.nodes.set_fields(child, checkpoint_path=run_rel(self.ctx.run_dir, trained.checkpoint))
            return (trained.checkpoint, gate.resolved_path), "scored", None
        return None, status, f"improve_recipe retries exhausted; last failure: {json.dumps(retry)[:2000]}"

    def _record_recipe(self, child: str, res: dict, resolved: Path) -> None:
        d = self._node_dir(child)
        d.mkdir(parents=True, exist_ok=True)
        (d / "recipe.yaml").write_text(yaml.safe_dump(res["recipe"], sort_keys=True))
        (d / "rationale.md").write_text(res["rationale"])
        rank, alpha = lora_of(resolved)
        self.nodes.set_fields(
            child, data_commit=res["data_commit"],
            recipe_hash=hashlib.sha256(json.dumps(res["recipe"], sort_keys=True).encode()).hexdigest(),
            recipe_path=run_rel(self.ctx.run_dir, d / "recipe.yaml"),
            rationale_path=run_rel(self.ctx.run_dir, d / "rationale.md"),
            resolved_config_path=run_rel(self.ctx.run_dir, resolved), lora_rank=rank, lora_alpha=alpha)

    # -- steps 8-9 ---------------------------------------------------------------------------
    def _record_score(self, node: str, score: float, detail: dict) -> None:
        self.nodes.record_score(node, score, self.ctx.metric_set, detail["metrics"])
        agg = self._node_dir(node) / "eval" / "aggregates.json"
        agg.parent.mkdir(parents=True, exist_ok=True)
        agg.write_text(json.dumps(detail["aggregates"], indent=1))
        update_values(self.ctx.conn, self.cfg)

    def _finish(self, node: str, status: str, error: str | None, started: float) -> None:
        if status != "scored":
            self.nodes.set_status(node, status)
            alert(self.ctx.recorder, "node_failed", f"{node} ended {status}: {error}", node_id=node)
        counts = {p: len(self.nodes.attempts(node, p)) for p in ("edit_self", "improve_recipe")}
        self.nodes.set_fields(node, error=error, attempt_counts=json.dumps(counts),
                              phase_timings=json.dumps({"total_s": time.time() - started}))
        shutil.rmtree(Path(self.ctx.run_dir) / "staging" / node, ignore_errors=True)   # spec 15
        self.ctx.recorder.event("node.end", payload={"node": node, "status": status, "error": error},
                                child=node, status=status)
        self._state(None, "idle")
```

Notes for the implementer:
- `_score` passes `resolved=None` for the root, so rank and alpha are unused there (`checkpoint=None` scores the released model unmerged).
- `_record_recipe` runs on every gate pass, so the node's `recipe.yaml`, `rationale.md`, `data_commit` and `resolved_config_path` always describe **the last attempt that passed the gate**, overwritten by any later passing attempt.
  - This is deliberate. For a `train_failed` node it is exactly the recipe and data that failed to train, which the next agent should see in the lineage.
  - For a node that ends `invalid_recipe` after an earlier gate pass, the node's `error` names the last failure, and every attempt is in the `attempts` table and under `attempts/improve_recipe-<k>/`.
  - `data_commit` does not affect selection, and a failed node is never a parent, so it is never used as `parent_data_commit`.
- `self.env` holds `kit.registry` and `kit.queue`. The unit tests pass `None` for both because their phases are fakes.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_loop.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/loop.py tests/test_loop.py
git commit -m "feat(loop): the node cycle with retries, failure statuses and node artifacts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Stop and resume; `ar run` and `ar stop`

**Files:**
- Create: `kernel/ar_kernel/control.py`
- Modify: `kernel/ar_kernel/cli.py`, `kernel/ar_kernel/subproc.py` (process-group registry)
- Test: `tests/test_control.py`, `tests/test_cli_run.py`, `tests/test_subproc.py`

**Interfaces:**
- Produces:
  - `control.Control(run_dir)`:
    - `.claim()` writes `control/loop.pid` (the pid and its `/proc` start time). It raises `RuntimeError` if a live loop already holds it; a stale or recycled pid is ignored.
    - `.release()`.
    - `.alive_pid() -> int | None`.
    - `.request_stop()` touches `control/stop`, and `.stop_requested() -> bool` checks for it.
    - `.clear_stop()`.
    - `.args() -> dict` and `.save_args(**kw)` read and write `control/run_args.json` (`max_nodes`).
  - `control.ForceStop(BaseException)`.
  - `control.drive(loop, kit, control, recorder) -> str`:
    - It installs signal handlers: first SIGINT sets `loop.graceful`, a second SIGINT or any SIGTERM raises `ForceStop`.
    - A thread mirrors the stop file into `loop.graceful`.
    - It runs `loop.run()`.
    - Always: it kills the run's containers on force stop, calls `kit.stop()`, releases the pid and records `run.stopped` with the reason.
  - Process-group registry, closing the gap left by the kernel's own-session launches:
    - Every GPU process the kernel starts (training, eval, rollouts, captioner vLLM, gate) goes through `run_in_env` in its own session, so it survives the kernel's death. The same happens when `JobQueue.shutdown()` times out and the process exits while its kill sequence is still running.
    - When `AR_PGID_DIR` is set, `run_in_env` writes `<dir>/<pgid>` containing the process start time right after `Popen`, and removes the file when the wait ends.
    - `subproc.proc_start_time(pid) -> str | None` reads that start time.
    - `control.kill_recorded_groups(control) -> list[int]` SIGTERMs, then after 30 s SIGKILLs, every recorded group whose leader still has the recorded start time, then removes the files.
    - `drive` sets `AR_PGID_DIR=control/pgids` and calls `kill_recorded_groups` after `kit.stop()`. `ar run --resume` calls it before starting.
  - `control.mark_interrupted(ctx, reason) -> list[str]`: on resume, every non-root `running` node becomes `interrupted`. Its files and rows are kept, and only its leftover containers are removed. It records `node.interrupted` and returns the node ids.
  - CLI:
    - `ar run --max-nodes N [--run-id ID]` creates a new run. It refuses an existing id without `--resume`.
    - `ar run --resume --run-id ID [--max-nodes N]`.
    - `ar stop --run-id ID [--force]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_control.py`:

```python
import os
import subprocess
import sys

import pytest

from ar_kernel.control import Control


def test_claim_ignores_a_recycled_pid(tmp_path):
    c = Control(tmp_path)
    (tmp_path / "control").mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # alive, but its start time is not the recorded one: a recycled pid
        (tmp_path / "control" / "loop.pid").write_text(f"{sleeper.pid} 1")
        c.claim()
        assert c.alive_pid() == os.getpid()
    finally:
        sleeper.kill()


def test_a_live_loop_is_detected_and_blocks_a_second_claim(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        Control(tmp_path).dir.mkdir(parents=True)
        (tmp_path / "control" / "loop.pid").write_text(f"{other.pid} {Control._start_time(other.pid)}")
        assert Control(tmp_path).alive_pid() == other.pid
        with pytest.raises(RuntimeError, match="already running"):
            Control(tmp_path).claim()
    finally:
        other.kill()


def test_stop_request_round_trip(tmp_path):
    c = Control(tmp_path)
    assert not c.stop_requested()
    c.request_stop()
    assert c.stop_requested()
    c.clear_stop()
    assert not c.stop_requested()
```

The force-stop latency test goes in `tests/test_loop_integration.py` (Task 12), because it needs the fake loop in a subprocess.

Add to `tests/test_subproc.py`:

```python
def test_process_groups_are_recorded_while_running(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_PGID_DIR", str(tmp_path / "pgids"))
    seen = []

    def check():
        seen.extend(p.name for p in (tmp_path / "pgids").iterdir())
    t = threading.Timer(3.0, check)
    t.start()
    run_in_env(ENV, ["python", "-c", "import time; time.sleep(6)"], cwd=tmp_path)
    t.join()
    assert len(seen) == 1 and not any((tmp_path / "pgids").iterdir())
```

(add `import threading`.) Add to `tests/test_control.py`:

```python
from ar_kernel.control import kill_recorded_groups
from ar_kernel.subproc import proc_start_time


def test_recorded_orphan_groups_are_killed_and_recycled_pids_are_not(tmp_path):
    c = Control(tmp_path)
    (c.dir / "pgids").mkdir(parents=True)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (c.dir / "pgids" / str(orphan.pid)).write_text(proc_start_time(orphan.pid))
        (c.dir / "pgids" / str(bystander.pid)).write_text("1")          # recycled pid: start time differs
        assert kill_recorded_groups(c) == [orphan.pid]
        assert orphan.wait(timeout=40) is not None and bystander.poll() is None
        assert not any((c.dir / "pgids").iterdir())
    finally:
        orphan.kill(), bystander.kill()
```

A killed child of the test itself stays a zombie, so its `/proc` entry remains until `orphan.wait()`. `kill_recorded_groups` therefore treats "still present after SIGKILL + 10 s" as done rather than looping forever. A real orphan is reaped by init.

Add to `tests/test_control.py`:

```python
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.control import mark_interrupted
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder


def test_resume_marks_unfinished_nodes_interrupted_and_keeps_their_files(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr("ar_kernel.control.kill_run_containers",
                        lambda run_id, node=None: killed.append(node) or [f"ar-{run_id}-{node}-x"])
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.7, ["m"], {"m": 0.7})
    nodes.create("n1", "root", 1)                                   # left running by a forced stop
    work = tmp_path / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    work.mkdir(parents=True), (work / "notes.txt").write_text("keep me")
    (tmp_path / "staging" / "n1").mkdir(parents=True)
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "state.json").write_text('{"node": "n1", "phase": "train", "attempt": 1}')
    ctx = RunContext(run_dir=tmp_path, conn=conn, recorder=rec, gpus=[0, 1, 2, 3], metric_set=["m"],
                     case_ids=["1"], versions={})

    assert mark_interrupted(ctx, "resume after forced stop") == ["n1"]

    n1 = nodes.get("n1")
    assert n1["status"] == "interrupted" and "phase train" in n1["error"]
    assert (work / "notes.txt").read_text() == "keep me" and (tmp_path / "staging" / "n1").exists()
    assert killed == ["n1"] and nodes.get("root")["status"] == "scored"
    (event,) = [e for e in rec.read_events() if e["type"] == "node.interrupted"]
    assert rec.load_payload(event["payload"])["phase_reached"] == "train"
```

`tests/test_cli_run.py`:

```python
from ar_kernel import cli


def test_run_refuses_an_existing_run_without_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["run", "--run-id", "r1", "--max-nodes", "3"]) == 2
    assert "--resume" in capsys.readouterr().err


def test_stop_writes_the_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["stop", "--run-id", "r1"]) == 0
    assert (tmp_path / "r1" / "control" / "stop").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_control.py tests/test_cli_run.py tests/test_subproc.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`kernel/ar_kernel/control.py`:

```python
"""Run control (spec 14.4): the loop's pid file, graceful stop requests, and signal handling."""
from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

from .sandbox.runner import kill_run_containers


class ForceStop(BaseException):
    """Raised in the main thread by SIGTERM or a second SIGINT. A BaseException, so no
    `except Exception` in a phase swallows it; run_in_env and run_container clean up on it."""


class Control:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir) / "control"

    def _pid_file(self) -> Path:
        return self.dir / "loop.pid"

    @staticmethod
    def _start_time(pid: int) -> str | None:
        return proc_start_time(pid)

    def alive_pid(self) -> int | None:
        try:
            pid_s, started = self._pid_file().read_text().split()
            pid = int(pid_s)
        except (OSError, ValueError):
            return None
        return pid if self._start_time(pid) == started else None

    def claim(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        other = self.alive_pid()
        if other is not None and other != os.getpid():
            raise RuntimeError(f"a loop is already running for this run (pid {other})")
        self._pid_file().write_text(f"{os.getpid()} {self._start_time(os.getpid())}")

    def release(self) -> None:
        if self.alive_pid() == os.getpid():
            self._pid_file().unlink(missing_ok=True)

    def request_stop(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "stop").touch()

    def stop_requested(self) -> bool:
        return (self.dir / "stop").exists()

    def clear_stop(self) -> None:
        (self.dir / "stop").unlink(missing_ok=True)

    def args(self) -> dict:
        path = self.dir / "run_args.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_args(self, **kw) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run_args.json").write_text(json.dumps({**self.args(), **kw}))


def drive(loop, kit, control: Control, recorder) -> str:
    """Run `loop` with stop handling. Returns the exit reason."""
    control.claim()
    os.environ["AR_PGID_DIR"] = str(control.dir / "pgids")     # run_in_env records every group here
    control.clear_stop()
    presses = {"n": 0}

    def on_int(signum, frame):
        presses["n"] += 1
        if presses["n"] == 1:
            loop.graceful.set()
            recorder.event("control", payload={"command": "graceful stop (SIGINT)"})
        else:
            raise ForceStop("second Ctrl-C")

    def on_term(signum, frame):
        raise ForceStop("SIGTERM")

    old = {s: signal.signal(s, h) for s, h in ((signal.SIGINT, on_int), (signal.SIGTERM, on_term))}
    done = threading.Event()

    def watch_stop_file():
        while not done.wait(2.0):
            if control.stop_requested():
                loop.graceful.set()

    threading.Thread(target=watch_stop_file, daemon=True, name="ar-stop-file").start()
    reason = "unknown"
    try:
        kit.start()
        reason = loop.run()
    except ForceStop as exc:
        reason = f"force stop ({exc})"
        kill_run_containers(Path(loop.ctx.run_dir).name)
    finally:
        done.set()
        try:
            kit.stop()
        finally:
            kill_recorded_groups(control)       # e.g. a job whose kill outlived JobQueue.shutdown's join
            for s, h in old.items():
                signal.signal(s, h)
            control.release()
            recorder.event("run.stopped", payload={"reason": reason})
    return reason
```

`subproc.py`: the start-time reader lives here (`Control` uses it):

```python
def proc_start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat: with the pid it names one process, so a recycled pid
    (after kill -9 or a reboot) is never mistaken for the process that was recorded."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None
```

In `run_in_env`, right after the successful `Popen`:

```python
    registry = os.environ.get("AR_PGID_DIR")
    marker = None
    if registry:
        marker = Path(registry) / str(proc.pid)          # start_new_session: pgid == pid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(proc_start_time(proc.pid) or "")
```

In the existing `finally:` of the wait, add `if marker is not None: marker.unlink(missing_ok=True)`.

Also in `control.py` (imports `time` and `from .subproc import proc_start_time`):

```python
def kill_recorded_groups(control: Control) -> list[int]:
    """Kill every process group the kernel started and never saw end (the subproc registry).
    Only groups whose leader still has the recorded start time: a recycled pid is left alone."""
    folder = control.dir / "pgids"
    killed = []
    for marker in sorted(folder.glob("*")) if folder.is_dir() else []:
        pgid, started = int(marker.name), marker.read_text().strip()
        if started and proc_start_time(pgid) == started:
            for sig, wait in ((signal.SIGTERM, 30.0), (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.2)
            killed.append(pgid)
        marker.unlink(missing_ok=True)
    return killed
```

Also in `control.py` (import `json`, `NodeStore` from `.archive.nodes` and `alert` from `.guards`):

```python
def mark_interrupted(ctx, reason: str) -> list[str]:
    """On resume: every non-root node still `running` was cut off (forced stop, kernel death or a
    spent budget). It is never resumed and never cleaned up (user decision 2026-09-27): its status
    becomes `interrupted`, its files and archive rows stay, and only containers a killed kernel left
    running are removed. A fresh cycle then starts."""
    nodes = NodeStore(ctx.conn)
    state_file = Path(ctx.run_dir) / "control" / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    marked = []
    for node in nodes.all():
        if node["status"] != "running" or node["parent_id"] is None:
            continue
        nid = node["node_id"]
        phase = state.get("phase") if state.get("node") == nid else None
        containers = kill_run_containers(Path(ctx.run_dir).name, nid)
        nodes.set_status(nid, "interrupted")
        nodes.set_fields(nid, error=f"{reason} (phase {phase})")
        ctx.recorder.event("node.interrupted", child=nid, payload={
            "node": nid, "reason": reason, "phase_reached": phase, "containers": containers})
        alert(ctx.recorder, "node_interrupted", f"{nid} was interrupted in phase {phase}; kept as is",
              level="warning", node_id=nid)
        marked.append(nid)
    return marked
```

A root left `running` (its first scoring failed or was killed) is not marked; `ensure_root` scores it again.

`cli.py`, `run` and `stop` subcommands:

```python
    runp = sub.add_parser("run", help="start (or --resume) the loop")
    runp.add_argument("--run-id", default=None)
    runp.add_argument("--max-nodes", type=int, default=None)
    runp.add_argument("--resume", action="store_true")
    stop = sub.add_parser("stop", help="stop a running loop")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--force", action="store_true")
```

Handlers (before the existing `attach_run` block; `init-run` stays as is):

```python
    if args.command == "run":
        return _run(cfg, args)
    if args.command == "stop":
        run_dir = cfg.runs_dir / args.run_id
        if not (run_dir / "config" / "run.json").exists():
            print(f"error: no run {args.run_id!r}", file=sys.stderr)
            return 2
        control = Control(run_dir)
        if args.force:
            pid = control.alive_pid()
            if pid is None:
                print("no loop is running for this run", file=sys.stderr)
                return 1
            os.kill(pid, signal.SIGTERM)
        else:
            control.request_stop()
        return 0
```

```python
def _run(cfg, args) -> int:
    existing = args.run_id and (cfg.runs_dir / args.run_id / "config" / "run.json").exists()
    if existing and not args.resume:
        print(f"error: run {args.run_id!r} exists; use --resume", file=sys.stderr)
        return 2
    if args.resume and not existing:
        print(f"error: no run {args.run_id!r} to resume", file=sys.stderr)
        return 2
    ctx = attach_run(cfg, args.run_id, os.environ) if args.resume else bootstrap_run(cfg, args.run_id, os.environ)
    control = Control(ctx.run_dir)
    if control.alive_pid() not in (None, os.getpid()):       # before touching any run file
        print(f"error: a loop is already running (pid {control.alive_pid()})", file=sys.stderr)
        return 2
    if args.max_nodes is not None:
        control.save_args(max_nodes=args.max_nodes)
    max_nodes = control.args().get("max_nodes")
    if max_nodes is None:
        print("error: --max-nodes is required for a new run", file=sys.stderr)
        return 2
    run_cfg = KernelConfig.for_run(ctx.run_dir)
    check_visible(ctx.gpus)
    repo = AgentsRepo(ctx.run_dir / "agents.git")
    killed = kill_recorded_groups(control)              # GPU jobs a killed kernel left running
    slot = ctx.run_dir / "merge_slot"                   # ~52 GB run-level transient (spec 15), not node data
    removed_slot = slot.exists()
    if removed_slot:
        shutil.rmtree(slot)
    if killed or removed_slot:
        ctx.recorder.event("run.resume_cleanup", payload={"killed_groups": killed, "merge_slot": removed_slot})
    mark_interrupted(ctx, "unfinished when the loop stopped (forced stop, kernel death or spent budget)")
    kit = build_run_kit(run_cfg, ctx.run_dir, ctx.gpus, ctx.recorder, os.environ)
    loop = Loop(run_cfg, ctx, kit, repo, max_nodes=max_nodes)
    reason = drive(loop, kit, control, ctx.recorder)
    print(f"loop stopped: {reason}")
    return 0
```

(Import `shutil`, `signal`, `Control`, `drive`, `kill_recorded_groups`, `mark_interrupted`, `check_visible`, `build_run_kit`, `Loop`, `AgentsRepo`.)

Resume cleanup is limited to what is not node data:
- process groups the killed kernel left running;
- the run-level `merge_slot`, which the next merge would replace anyway, but which would otherwise hold ~52 GB through the next node's training.

The interrupted node's own files stay (user decision).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/control.py kernel/ar_kernel/cli.py kernel/ar_kernel/subproc.py tests/test_control.py tests/test_cli_run.py tests/test_subproc.py
git commit -m "feat(control): ar run / --resume / ar stop, force stop, interrupted nodes on resume

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Monitor and `ar status`

**Files:**
- Create: `kernel/ar_kernel/monitor.py`, `kernel/ar_kernel/status.py`
- Modify: `kernel/ar_kernel/control.py` (`drive` starts and stops the monitor), `kernel/ar_kernel/cli.py` (`status`)
- Test: `tests/test_monitor.py`, `tests/test_status.py`

**Interfaces:**
- Produces:
  - `Monitor(cfg, run_dir, recorder, gpus, budget, *, usage=gpu_usage_detailed, clock=time.time)`, with:
    - `.tick()`: one GPU sample. It writes a `gpu.sample` event to `telemetry/events/gpu.jsonl`, with inline fields: per GPU index, util, memory, power and temperature.
    - `.check()`: all alert checks.
    - `.start()` / `.stop()`: a daemon thread that ticks every `telemetry.gpu_sample_sec` and checks every 60 s.
  - The alerts `.check()` raises, at most once per condition:
    - `stall`: no events for the current node for `alerts.stall_min` while a phase is running.
    - `disk_low`: free space under `disk.alert_below_gb`.
    - `gateway_errors`: error rate above `alerts.gateway_error_rate` over `alerts.gateway_error_window_min`, with at least 5 calls.
    - `gpu_outside_list`: a process descended from the loop runs on an unlisted GPU.
  - `gpu_usage_detailed(gpus=None) -> dict[int, dict] | None` (in `monitor.py`).
  - `status.run_status(run_dir) -> dict`: plain JSON, the one data source for `ar status` (and for any future UI):
    - `run_id`, `loop_pid`, `state`, `max_nodes`, `nodes` (with `P`, the probability the next draw would give each scored node, computed live by `selection.candidates`), `best`, `spend` (a `Budget.snapshot()` recomputed from telemetry with the run's prices), `alerts` (the last 20).
  - `status.format_status(d) -> str`.
  - `ar status --run-id ID [--json]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_monitor.py`:

```python
import os

from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.monitor import Monitor
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()


def kinds(rec):
    return [e["kind"] for e in rec.read_events() if e["type"] == "alert"]


def test_gpu_sample_goes_to_its_own_file(tmp_path):
    rec = Recorder(tmp_path)
    m = Monitor(CFG, tmp_path, rec, [0], Budget(),
                usage=lambda gpus: {0: {"util": 97, "memory_mib": 20000, "power_w": 300, "temp_c": 70, "pids": []}})
    m.tick()
    (event,) = rec.read_events("gpu")
    assert event["gpus"]["0"]["util"] == 97 and event["payload"] is None


def test_stall_alert_once(tmp_path):
    rec = Recorder(tmp_path)
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "state.json").write_text('{"node": "n1", "phase": "train", "attempt": 1, "since": 0}')
    now = [10_000.0]
    m = Monitor(CFG, tmp_path, rec, [0], Budget(), usage=lambda gpus: {}, clock=lambda: now[0])
    m.check()
    m.check()
    assert kinds(rec).count("stall") == 1


def test_gateway_error_alert(tmp_path):
    rec, b = Recorder(tmp_path), Budget()
    for _ in range(6):
        b.record(503, None)
    Monitor(CFG, tmp_path, rec, [0], b, usage=lambda gpus: {}).check()
    assert "gateway_errors" in kinds(rec)


def test_gpu_outside_the_list_alert(tmp_path):
    rec = Recorder(tmp_path)
    m = Monitor(CFG, tmp_path, rec, [0], Budget(),
                usage=lambda gpus: {3: {"util": 5, "memory_mib": 900, "power_w": 50, "temp_c": 40,
                                        "pids": [os.getpid()]}})
    m.check()
    assert "gpu_outside_list" in kinds(rec)
```

`tests/test_status.py`:

```python
import json

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.guards import alert
from ar_kernel.status import format_status, run_status
from ar_kernel.telemetry.recorder import Recorder


def test_status_is_plain_json(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text(
        "budget: {max_usd: 10, usd_per_mtok: {input: 1, cached_input: 0.1, output: 4}}\n"
        "selection: {decay: 0.5, prior_weight: 1.0, subtree_share: 0.3, size_scale: 4, temperature: 2.0,"
        " noise_floor: 4.4e-4, epsilon: 0.2}\n")
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.78, ["m"], {"m": 0.78})
    nodes.create("n1", "root", 1), nodes.set_status("n1", "invalid_code")
    nodes.set_fields("n1", error="contract import failed")
    nodes.create("n2", "root", 1), nodes.set_status("n2", "interrupted")
    rec.event("llm.response", node="n1", usage={"prompt_tokens": 1000, "completion_tokens": 100}, mock=False)
    alert(rec, "node_failed", "n1 ended invalid_code")
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "run_args.json").write_text('{"max_nodes": 5}')
    d = run_status(tmp_path)
    json.dumps(d)                                               # serializable
    assert d["best"]["node_id"] == "root" and d["max_nodes"] == 5 and d["loop_pid"] is None
    assert {n["node_id"]: n["P"] for n in d["nodes"]} == {"root": 1.0, "n1": None, "n2": None}
    assert d["spend"]["usd"] == (1000 * 1 + 100 * 4) / 1e6
    assert [a["kind"] for a in d["alerts"]] == ["node_failed"]
    text = format_status(d)
    assert "n1" in text and "invalid_code" in text and "contract import failed" in text
    assert "nodes: 1 of 5" in text                              # interrupted n2 is not counted
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_monitor.py tests/test_status.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`kernel/ar_kernel/monitor.py`:

```python
"""Run monitoring (spec 13.3 GPU samples, 13.4 alerts). Alerts never stop anything."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .eval.merge import free_disk_gb
from .guards import alert, smi


def gpu_usage_detailed(gpus: list[int] | None = None) -> dict[int, dict] | None:
    table = smi(["--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
                  "--format=csv,noheader,nounits"])
    apps = smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"])
    if table is None or apps is None:
        return None
    out, by_uuid = {}, {}
    for line in table.splitlines():
        if not line.strip():
            continue
        index, uuid, util, mem, power, temp = (s.strip() for s in line.split(","))
        by_uuid[uuid] = int(index)
        num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None   # "[N/A]" -> None
        out[int(index)] = {"util": num(util), "memory_mib": num(mem), "power_w": num(power),
                           "temp_c": num(temp), "pids": []}
    for line in apps.splitlines():
        if line.strip():
            pid, uuid = (s.strip() for s in line.split(","))
            if uuid in by_uuid:
                out[by_uuid[uuid]]["pids"].append(int(pid))
    return out if gpus is None else {g: v for g, v in out.items() if g in gpus}


def _descends_from(pid: int, ancestor: int) -> bool:
    for _ in range(64):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


class Monitor:
    def __init__(self, cfg, run_dir: Path, recorder, gpus: list[int], budget, *,
                 usage=gpu_usage_detailed, clock=time.time) -> None:
        self.cfg, self.run_dir, self.recorder, self.gpus, self.budget = cfg, Path(run_dir), recorder, list(gpus), budget
        self.usage, self.clock = usage, clock
        self._raised: set = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _once(self, key, kind: str, message: str, **payload) -> None:
        if key not in self._raised:
            self._raised.add(key)
            alert(self.recorder, kind, message, **payload)

    def tick(self) -> None:
        usage = self.usage(None)
        if usage is not None:
            self.recorder.event("gpu.sample", node="gpu",
                                gpus={str(g): {k: v for k, v in u.items() if k != "pids"} for g, u in usage.items()})

    def check(self) -> None:
        now = self.clock()
        state_file = self.run_dir / "control" / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            node = state.get("node")
            if node and state.get("phase") not in (None, "idle"):
                events = self.recorder.events_path(node)
                last = max(state.get("since", 0), events.stat().st_mtime if events.exists() else 0)
                if now - last > 60 * float(self.cfg.get("alerts.stall_min")):
                    self._once(("stall", node, state.get("phase"), state.get("attempt")), "stall",
                               f"no events from {node} {state.get('phase')} for "
                               f"{(now - last) / 60:.0f} min", level="warning")
        free = free_disk_gb(self.run_dir)
        if free < float(self.cfg.get("disk.alert_below_gb")):
            self._once(("disk", int(now // 3600)), "disk_low", f"{free:.0f} GB free under {self.run_dir}")
        rate, calls = self.budget.error_rate(60 * float(self.cfg.get("alerts.gateway_error_window_min")))
        if calls >= 5 and rate > float(self.cfg.get("alerts.gateway_error_rate")):
            self._once(("gateway", int(now // 300)), "gateway_errors",
                       f"{rate:.0%} of the last {calls} LLM calls failed")
        usage = self.usage(None) or {}
        me = os.getpid()
        for gpu, u in usage.items():
            if gpu in self.gpus:
                continue
            ours = [p for p in u["pids"] if _descends_from(p, me)]
            if ours:
                self._once(("outside", gpu, tuple(ours)), "gpu_outside_list",
                           f"kernel process(es) {ours} run on GPU {gpu}, outside the list {self.gpus}")

    def start(self) -> None:
        sample_s = float(self.cfg.get("telemetry.gpu_sample_sec"))

        def loop():
            last_check = 0.0
            while not self._stop.wait(sample_s):
                try:
                    self.tick()
                    if time.monotonic() - last_check >= 60:
                        last_check = time.monotonic()
                        self.check()
                except Exception as exc:                       # noqa: BLE001 -- never kill the run
                    self.recorder.event("monitor.error", payload={"error": repr(exc)})

        self._thread = threading.Thread(target=loop, daemon=True, name="ar-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
```

In `control.drive`, take an optional `monitor=None`. Call `monitor.start()` after `kit.start()`, and `monitor.stop()` first in the `finally` block. `cli._run` builds it as `Monitor(run_cfg, ctx.run_dir, ctx.recorder, ctx.gpus, kit.budget)`.

`kernel/ar_kernel/status.py`:

```python
"""`ar status` (spec 13.4): one plain-JSON snapshot of a run. Any future UI reads this."""
from __future__ import annotations

import json
from pathlib import Path

from .archive.db import open_db
from .archive.nodes import NodeStore
from .budget import Budget
from .config import KernelConfig
from .control import Control
from .selection import candidates


def _events(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        return []
    return [e for e in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if e.get("type") == kind]


def run_status(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    conn = open_db(run_dir)
    try:
        nodes = NodeStore(conn).all()
    finally:
        conn.close()
    cfg = KernelConfig.for_run(run_dir)
    try:      # the probabilities the NEXT draw would use, computed live from the current tree
        probs = {c["node_id"]: c["P"] for c in candidates(nodes, cfg)}
    except ValueError:                                         # no scored node yet
        probs = {}
    control = Control(run_dir)
    state_file = control.dir / "state.json"
    budget = Budget.from_config(cfg)
    budget.load(run_dir)
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    alerts = _events(run_dir / "telemetry" / "events" / "run.jsonl", "alert")[-20:]
    return {
        "run_id": run_dir.name, "loop_pid": control.alive_pid(),
        "state": json.loads(state_file.read_text()) if state_file.exists() else None,
        "max_nodes": control.args().get("max_nodes"),
        "nodes": [{"node_id": n["node_id"], "parent_id": n["parent_id"], "depth": n["depth"],
                   "status": n["status"], "score": n["score"], "value": n["subtree_value"],
                   "P": probs.get(n["node_id"]), "component": n["edit_component"], "error": n["error"]}
                  for n in nodes],
        "best": {"node_id": best["node_id"], "score": best["score"]} if best else None,
        "spend": budget.snapshot(),
        "alerts": [{"ts": a["ts_wall"], "kind": a.get("kind"), "level": a.get("level"),
                    "message": a.get("message")} for a in alerts],
    }


def format_status(d: dict) -> str:
    lines = [f"run {d['run_id']}: loop {'running (pid %s)' % d['loop_pid'] if d['loop_pid'] else 'not running'}"]
    if d["state"]:
        s = d["state"]
        lines.append(f"now: {s.get('node')} {s.get('phase')} attempt {s.get('attempt')}")
    spend = d["spend"]
    usd = "n/a (no prices)" if spend["usd"] is None else f"${spend['usd']:.4f}"
    cap = "none" if spend["max_usd"] is None else f"${spend['max_usd']}"
    lines.append(f"spend: {usd}, {spend['tokens']} tokens, {spend['calls']} calls; cap: {cap}")
    done = sum(1 for n in d["nodes"] if n["parent_id"] is not None and n["status"] != "interrupted")
    lines.append(f"nodes: {done} of {d['max_nodes']}; best: {d['best']}")
    for n in d["nodes"]:
        score = "-" if n["score"] is None else f"{n['score']:.4f}"
        p = "" if n["P"] is None else f" P={n['P']:.2f}"
        extra = f" [{n['component']}]" if n["component"] else ""
        err = f"  ! {n['error'][:120]}" if n["error"] else ""
        lines.append(f"  {'  ' * n['depth']}{n['node_id']:<6} {n['status']:<14} {score}{p}{extra}{err}")
    if d["alerts"]:
        lines.append("alerts (latest last):")
        lines += [f"  [{a['level']}] {a['kind']}: {a['message']}" for a in d["alerts"]]
    return "\n".join(lines)
```

`cli.py` `status`: add `--json`. Replace the old node loop with:

```python
    if args.command == "status":
        d = run_status(ctx.run_dir)
        print(json.dumps(d, indent=1) if args.json else format_status(d))
        return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/monitor.py kernel/ar_kernel/status.py kernel/ar_kernel/control.py kernel/ar_kernel/cli.py tests/test_monitor.py tests/test_status.py
git commit -m "feat(status,monitor): ar status snapshot, GPU samples and run alerts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Loop integration test (spec §16.2)

Multi-node runs in about a minute, with no GPU, LLM or Docker. They use the Task 9 scripted phases and a real `drive` in a subprocess, which makes the signal and `kill -9` tests possible.

**Files:**
- Create: `tests/fixtures/fake_loop.py`, `tests/test_loop_integration.py`
- Modify: `tests/test_loop.py`. Move `Script` into `tests/fixtures/fake_loop.py` and import it with `from fixtures.fake_loop import Script`, so both test files share one definition. This import works because `tests/` has no `__init__.py`, so pytest puts `tests/` on `sys.path`.

**Interfaces:**
- `tests/fixtures/fake_loop.py`:
  - `Script` (moved from Task 9) gains a `slow` argument:
    - `None`: no delay.
    - A phase name (`"improve_recipe"`, `"train"`): that phase sleeps in `time.sleep(0.5)` steps for up to 600 s.
    - `"pace"`: every phase sleeps 0.5 s.
  - `build(run_dir, max_nodes, script) -> (loop, kit)`: `kit` has `start`/`stop` no-ops and a `Budget`.
  - `main()`: reads `sys.argv[1]` (run dir), `sys.argv[2]` (max nodes) and `sys.argv[3]` (`none`, `pace` or a phase name), then runs `drive(loop, kit, Control(run_dir), rec)`. The file is run directly as `python tests/fixtures/fake_loop.py ...`, with `if __name__ == "__main__": main()`.

- [ ] **Step 1: Write the failing tests** (`tests/test_loop_integration.py`)

```python
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.control import mark_interrupted
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder

FIX = Path(__file__).parent / "fixtures"


def spawn(run, max_nodes, slow):
    # sys.executable is the autoresearcher env's python running this test suite
    return subprocess.Popen([sys.executable, str(FIX / "fake_loop.py"), str(run), str(max_nodes), slow])


def wait_phase(run, phase, timeout=60):
    state = run / "control" / "state.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.exists() and json.loads(state.read_text()).get("phase") == phase:
            return
        time.sleep(0.2)
    raise AssertionError(f"never reached {phase}")


def resume_mark(run):
    ctx = RunContext(run_dir=run, conn=open_db(run), recorder=Recorder(run), gpus=[0, 1, 2, 3],
                     metric_set=["m"], case_ids=["1"], versions={})
    mark_interrupted(ctx, "resume")
    return ctx


def test_multi_node_run_telemetry_is_complete(tmp_path):
    run = tmp_path / "run"
    proc = spawn(run, 4, "none")
    assert proc.wait(timeout=120) == 0
    events = Recorder(run).read_events()
    created = [e["child"] for e in events if e["type"] == "node.created"]
    ended = [e["child"] for e in events if e["type"] == "node.end"]
    assert created == ended == ["n1", "n2", "n3", "n4"]
    assert [e for e in events if e["type"] == "run.stopped"]
    assert len({e["child"] for e in events if e["type"] == "select"}) == 4


def test_sigterm_mid_phase_stops_within_seconds_and_resume_marks_interrupted(tmp_path):
    run = tmp_path / "run"
    proc = spawn(run, 3, "improve_recipe")
    wait_phase(run, "improve_recipe")
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0 and time.monotonic() - t0 < 10
    stopped = [e for e in Recorder(run).read_events() if e["type"] == "run.stopped"]
    assert "force stop" in Recorder(run).load_payload(stopped[-1]["payload"])["reason"]
    ctx = resume_mark(run)
    assert {n["node_id"]: n["status"] for n in NodeStore(ctx.conn).all()} == {"root": "scored", "n1": "interrupted"}
    assert (run / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace").exists()   # files kept


def test_kill_9_then_resume_starts_a_fresh_cycle_with_a_new_id(tmp_path):
    run = tmp_path / "run"
    proc = spawn(run, 3, "train")
    wait_phase(run, "train")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    resume_mark(run)
    proc = spawn(run, 1, "none")                           # max_nodes 1: exactly one more node
    assert proc.wait(timeout=120) == 0
    ids = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert ids == {"root": "scored", "n1": "interrupted", "n2": "scored"}   # n1 kept; ids not reused


def test_graceful_stop_file_finishes_the_current_node(tmp_path):
    run = tmp_path / "run"
    proc = spawn(run, 5, "pace")                       # ~3 s per node, so the request lands in n1
    wait_phase(run, "edit_self")
    (run / "control" / "stop").touch()
    assert proc.wait(timeout=120) == 0
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "n1": "scored"}
```

The "none" script in `fake_loop.main` must exercise both retry loops and every outcome within 4 nodes:
- `n1`: contract fails once, then passes; the gate fails once, then scores.
- `n2`: edit exhausted (`invalid_code`).
- `n3`: training fails 3 times (`train_failed`).
- `n4`: scoring raises (`eval_failed`).

Add these assertions to `test_multi_node_run_telemetry_is_complete`:
- `{n: status}` equals `{"root": "scored", "n1": "scored", "n2": "invalid_code", "n3": "train_failed", "n4": "eval_failed"}`.
- There is one `node_failed` alert per failed node.

A `slow` phase sleeps in a loop of `time.sleep(0.5)` for up to 600 s, so a signal is handled within half a second.

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_loop_integration.py -q`
Expected: FAIL (`tests/fixtures/fake_loop.py` missing).

- [ ] **Step 3: Implement the fixture**

`tests/fixtures/fake_loop.py` holds `Script` (moved from `tests/test_loop.py`, with the `slow` argument described under **Interfaces**), then:

```python
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ar_kernel.archive.db import open_db
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.control import Control, drive
from ar_kernel.loop import Loop
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.vcs.agents_repo import AgentsRepo

SCENARIO = dict(contract=[False, True], gate=[False, True],           # n1: both retry loops, then scored
                edit=[True, True, False, False, False],               # n2: invalid_code (after n1's 2 edits)
                train=[True, False, False, False],                    # n3: train_failed
                score=[0.78, 0.80, RuntimeError("wbench gpu failed")])  # root, n1, n4 -> eval_failed


def build(run: Path, max_nodes: int, script):
    rec = Recorder(run)
    ctx = RunContext(run_dir=run, conn=open_db(run), recorder=rec, gpus=[0, 1, 2, 3],
                     metric_set=["m"], case_ids=["1"], versions={})
    kit = SimpleNamespace(registry=None, queue=None, gpu_lock=threading.Lock(), budget=Budget(),
                          harness=None, socket_dir=run / "sock", default_model="mock-model",
                          start=lambda: None, stop=lambda: None)
    return Loop(KernelConfig.load(), ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=max_nodes,
                phases=script.phases()), kit


def main() -> None:
    run, max_nodes, slow = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    script = Script(run, slow=None if slow == "none" else slow, **(SCENARIO if slow == "none" else {}))
    loop, kit = build(run, max_nodes, script)
    drive(loop, kit, Control(run), loop.ctx.recorder)


if __name__ == "__main__":
    main()
```

(The `edit` list is consumed across nodes. `n1`'s contract retry makes two edit calls (both `True`), and `n2`'s three calls get the three `False` values. Adjust the list if the per-node consumption differs, and assert the final statuses rather than the list.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_loop.py tests/test_loop_integration.py -q`
Expected: PASS (in under 3 minutes).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/fake_loop.py tests/test_loop.py tests/test_loop_integration.py
git commit -m "test(loop): integration runs covering retries, failures, signals, kill -9 and resume

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Real verification, docs, acceptance run, merge

**Files:**
- Modify:
  - `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`
  - `docs/superpowers/plans/2026-09-21-autoresearcher-agent-runtime.md` (plan sequence row 4: done)
  - `docs/superpowers/plans/verification-log.md`
  - `docs/PORTABILITY.md` (run-relative node paths; `ar run` needs `.env` with `OPENAI_API_KEY`/`OPENAI_MODEL`)
  - `seed_agent/agent/knowledge/data_building.md`: one line saying training failures come back as `retry.kind == "train"` with a log tail.

- [ ] **Step 1: Spec amendments** (each dated 2026-09-27, marked "Plan 4 as built / user decision")
  - **§7.2:** "Recipe-caused training failures" becomes **all** training and precache failures, returned to the agent with a log tail. Merge, render and WBench failures end the node `eval_failed`. There is no pause. Add a note that per-attempt training directories sit under `attempts/improve_recipe-<k>/`.
  - **§12 (replaced):** the continuous softmax selection of Task 3: its formula, defaults, worked example and requirements. The percentile-rank method and its golden tests are superseded. There are no evaluation re-runs (user decision); proxy noise enters as `noise_floor`.
  - **§13.4:** there is no dashboard, and none is planned. `ar status` (`status.run_status`, plain JSON) is the monitoring interface, and a UI can be built on it later. The alert list is as implemented.
  - **§14.1:** add the `eval_failed` and `interrupted` rows.
  - **§14.2:**
    - The precache/train infrastructure row: "reported to the agent as a failed attempt".
    - The merge/render/eval row: "node `eval_failed`, alert, loop continues".
    - The gateway row: "upstream unavailable (≥ `gateway.outage_error_rate` of the last `outage_window_min` minutes' calls failing with 429/5xx/connection errors) when an agent attempt fails ⇒ the run **stops** (a stop, not a pause); the node is marked `interrupted` on resume and not charged; `llm_outage` alert".
    - Remove every "pause".
  - **§14.3 (replaced):** an unfinished node (forced stop, kernel death, spent budget) is never resumed and never cleaned up. On resume it is marked `interrupted` and kept as is; only its leftover containers are removed. It is not a parent, does not count toward `max_nodes` and counts nowhere in selection. A fresh cycle starts. Node ids are never reused.
  - **§14.6:** reduced to the start-time GPU visibility check and the merge disk check. There is no wait for idle GPUs (user decision: simpler); a foreign process on a listed GPU shows up as a failure of that phase.
  - **§17:** the `budget` (dollars only), `liveness`, `alerts` and `timeouts.train_s`/`eval_s` rows, and the new `selection` block (Task 3).

- [ ] **Step 2: Full default suite, then the Docker suite**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -q`, then `conda run --no-capture-output -n autoresearcher python -m pytest -m docker -q`.
Expected: all pass. Record the counts in the verification log.

- [ ] **Step 3: Real-component check without the paid model** (GPU; ~4 h; `AR_TEST_GPUS` free)
  - Create a run with `ar init-run --run-id loopcheck_<date>`.
  - Run the real `Loop.ensure_root()` through `drive` with `max_nodes=0`:

```bash
conda run --no-capture-output -n autoresearcher python -m ar_kernel.cli run --run-id loopcheck_<date> --max-nodes 0
```

  - Set `OPENAI_MODEL` to any name. `OPENAI_API_KEY` must be non-empty for `build_run_kit`, but no call is made, because `max_nodes=0` means no agent phase runs.
  - Record:
    - the root proxy score, which must match the reference 0.7833 within the aggregate tolerance 2e-3;
    - that the monitor samples (`gpu.jsonl`) and `ar status` work while it runs;
    - that `ar stop --force` during the render kills the render process group, and that the GPUs return to idle (`nvidia-smi`);
    - that `ar run --resume` then scores the root again from scratch.

- [ ] **Step 4: Acceptance run (spec §16.4) — only with the user's explicit go-ahead**
  - Ask the user for:
    - the model and endpoint in `.env`;
    - `budget.max_usd`, with the `usd_per_mtok` prices for the model;
    - permission to start (real money; about a day of GPU time for 3 nodes).
  - Then run `ar run --max-nodes 3 --run-id accept_<date>`.
  - Watch it with `ar status --run-id accept_<date>`.
  - Record in the verification log: each node's status and score, attempts, spend, wall time per phase, and any alert.
  - Any defect found here is fixed with a failing test first.

- [ ] **Step 5: Final review and merge**
  - Run the final whole-branch review, then fast-forward AutoResearcher `main` to `feat/loop` and push.
  - WorldModel and WBench have no changes in this plan: confirm with `git status` that they are clean and match their remotes.
  - Commit the docs:

```bash
git add docs/superpowers/specs/2026-09-17-autoresearcher-design.md docs/superpowers/plans/2026-09-21-autoresearcher-agent-runtime.md docs/superpowers/plans/verification-log.md docs/PORTABILITY.md seed_agent/agent/knowledge/data_building.md
git commit -m "docs: Plan 4 as built (no pause, dollar cap, softmax selection, interrupted/eval_failed, status-only monitoring)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage.**

  | Spec | Task |
  |---|---|
  | §7.1 bootstrap | 9 `ensure_root`, 10 `ar run` |
  | §7.2 steps 1–10 | 9 (steps 1–9); 9 and 10 (step 10: stop checks) |
  | §7.3 per-attempt dirs, `trainer_state.pt` | 9 (attempt dir as the Gate's `node_dir`), 7 |
  | §12 | 3 |
  | §13.3 selection row, GPU samples | 3, 11 |
  | §13.4 status and alerts | 11, 6 |
  | §14.1 | 2, 9 |
  | §14.2 (amended) | 9 |
  | §14.3 | 10 |
  | §14.4 | 10 |
  | §14.5 | 5 (agent phases, contract smoke), 7 (train, render, eval) |
  | §14.6 (reduced) | 6 (visibility), 7 (merge disk check already in `merge_lora`) |
  | §15 | 7 (`merge_slot`, `trainer_state.pt`), 9 (staging of finished nodes) |
  | §16.1 selection, liveness, GPU list | 3, 5, 6 |
  | §16.2 | 12 |
  | §16.4 | 13 |

  **Plan 2 "Plan 4 must" items:**
  - 1 liveness → Task 5.
  - 2 per-attempt dirs → Task 9.
  - 3 termination → Tasks 6 and 10.
  - 4 noise → Task 3's `noise_floor` on the softmax temperature. The user chose no re-evaluation.
  - 5 outage → no pause (user); the `gateway_errors` alert is in Task 11.
  - 6 component → Tasks 2, 9 and 11.
  - 7 deferred Plan 1 items → Tasks 7 and 2 (`resolve_gpus` visibility in Task 6; run-relative paths in Task 2).
  - 8 config to constructors → Task 8.
  - 9 shutdown errors → Task 8.
  - 10 live LLM run → Task 13 Step 4.
- **Placeholders.** None, with three deliberate exceptions:
  - Task 2's context-bundle test and Task 4's gateway test are described against the fixtures in their existing test files, so they follow those files' helpers.
  - Task 12 says to assert the final statuses if list consumption differs.
- **Type consistency.**
  - `Phases` callables are defined in Task 9. The Task 9 and Task 12 fakes use the same signatures, and `run_edit_self`, `run_improve_recipe` and `verify_contract` are called with the keyword names of verified facts 1–2.
  - `Budget.record` / `exhausted` / `error_rate` / `load` / `snapshot` are defined in Task 4 and used in Tasks 8, 9, 11 and 12.
  - `alert(recorder, kind, message, *, level, **payload)` is defined in Task 6 and used in Tasks 8, 9 and 11.
  - `Liveness.from_config(cfg, soft_s, signals, hard_s=None)` is defined in Task 5 and used in Tasks 5 and 7.
- **Deliberately out of scope:**
  - A dashboard (user decision).
  - Telemetry `index.db` (spec §13.2 "rebuildable"; nothing reads it yet).
  - `pip freeze` of each env at bootstrap (§7.1). `versions.json` already records git SHAs; add the freeze only if a user asks.
