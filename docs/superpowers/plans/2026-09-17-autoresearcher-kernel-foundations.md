# AutoResearcher Kernel Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AutoResearcher kernel up to the point where it can ingest training clips, commit a data view, gate a recipe, fine-tune the released AlayaWorld checkpoint through the standard training interface, and score the result on the 40-case WBench proxy.

**Architecture:** A plain-Python kernel package (`ar_kernel`) in its own conda env drives two sibling repos as subprocesses: WorldModel (training, rendering, rollouts) in the `alayaworld` env and WBench (metrics) in `wbench-main`/`wbench-vp`. State lives in one SQLite database plus a content-addressed blob store; training data is materialized as hardlink farms in the three standard formats. Every kernel operation emits an append-only telemetry event before and after it acts.

**Tech Stack:** Python 3.12, SQLite (stdlib), PyYAML, NumPy, Pillow, ImageHash, pytest, conda, ffmpeg/ffprobe, Docker (later plans).

**Spec:** `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`

**Follow-on plans (not this one):** Plan 2 — agent layer (gateway, MCP tool server, sandbox, contract, seed agent). Plan 3 — the loop (selection, retries, failure handling, stop/resume, dashboard).

## Global Constraints

- **Python:** kernel runs on Python 3.12 in conda env `autoresearcher`. WorldModel work runs in `alayaworld` (Python 3.10, torch 2.7.1), WBench metrics in `wbench-main`, visual plausibility in `wbench-vp`.
- **Repo boundaries:** the kernel never modifies `WorldModel/` or `WBench/` except the two upstream patches U1 and U2 in Task 3. Agents (later plans) never modify them at all.
- **Training interface:** training happens only through `WorldModel/docs/TRAINING.md` — the three standard formats, `data.datasets`, `scripts/tools/check_dataset.py`, `scripts/tools/precache_train_text_embeds.py`, `scripts/finetune/lowcompute_4x4090.sh`.
- **Data formats:** `video_caption_camera`, `video_timed_prompts_camera`, `video_caption_static` only. Layout `<root>/videos/<id>.mp4`, `<root>/captions/<id>.json`, `<root>/poses/<id>.npz`. Video ≥ 24 fps, ≥ 2.375 s, aspect ratio within 2% of 16/9. `cam_c2w` is `[N,4,4]` with N = mp4 frame count.
- **GPU policy:** use `CUDA_VISIBLE_DEVICES` when set, else `gpus.default` (`0,1,2,3`); any indices allowed; fewer than `gpus.min_count` (4) is refused. No GPU count, index or card model is hardcoded in code.
- **Paths:** project root `/mnt/biometrics/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch`; kernel repo `AutoResearcher/`; run state under `AutoResearcher/runs/<run_id>/`.
- **Deletion policy:** the kernel deletes only its own transient files under `AutoResearcher/runs/`.
- **Telemetry:** fail-closed. If an event cannot be persisted, the operation raises.
- **Secrets:** `.env` values (`OPENAI_API_KEY`, `VLM_API_KEY`, `HF_TOKEN`) are redacted in every recorded payload.
- **Proxy subset:** the fixed 40 case ids in `configs/proxy_cases.txt`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `configs/kernel.yaml` | All kernel defaults (§17 of the spec) |
| `configs/proxy_cases.txt` | The 40 proxy case ids |
| `configs/base_recipe.yaml` | Copy of WorldModel's `configs/examples/finetune_video_caption_camera.yaml` |
| `kernel/ar_kernel/config.py` | Load `kernel.yaml`, resolve paths, resolve the GPU list |
| `kernel/ar_kernel/telemetry/recorder.py` | Append-only JSONL events + payload store + redaction |
| `kernel/ar_kernel/archive/db.py` | SQLite schema and connection |
| `kernel/ar_kernel/archive/nodes.py` | Node and attempt records |
| `kernel/ar_kernel/archive/blobs.py` | Content-addressed blob store |
| `kernel/ar_kernel/archive/clips.py` | Clip records and queries |
| `kernel/ar_kernel/archive/commits.py` | Data commits, visibility, view materialization |
| `kernel/ar_kernel/subproc.py` | Run commands in a named conda env, capture output, record events |
| `kernel/ar_kernel/data/probe.py` | ffprobe metadata + aspect-ratio rule |
| `kernel/ar_kernel/data/checker.py` | Bridge to `alaya.data.standard_check` for per-format eligibility |
| `kernel/ar_kernel/bridges/check_clip.py` | Script executed inside `alayaworld` by `checker.py` |
| `kernel/ar_kernel/data/leakage.py` | pHash + NCC + entropy leakage check against WBench first frames |
| `kernel/ar_kernel/data/ingest.py` | Staging → validated clip rows |
| `kernel/ar_kernel/train/recipe.py` | Resolved-config builder |
| `kernel/ar_kernel/train/gate.py` | Recipe gate checks |
| `kernel/ar_kernel/train/runner.py` | Launch training, parse logs, classify failures |
| `kernel/ar_kernel/eval/merge.py` | LoRA merge + host-RAM guard |
| `kernel/ar_kernel/eval/render.py` | Proxy rendering via `run_wbench.py` |
| `kernel/ar_kernel/eval/wbench.py` | WBench phases, VP, report |
| `kernel/ar_kernel/eval/score.py` | Metric set, score, agent-facing aggregates |
| `kernel/ar_kernel/run.py` | Run bootstrap, metric preflight, root node |
| `kernel/ar_kernel/cli.py` | `ar` command entry point |
| `tests/…` | One test module per kernel module |

---

### Task 1: Project scaffolding, config and GPU policy

**Files:**
- Create: `pyproject.toml`, `kernel/ar_kernel/__init__.py`, `kernel/ar_kernel/config.py`, `configs/kernel.yaml`, `configs/proxy_cases.txt`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `KernelConfig.load(path: Path | None = None) -> KernelConfig` with attributes `raw: dict`, `project_root: Path`, `worldmodel: Path`, `wbench: Path`, `runs_dir: Path`; `KernelConfig.get(dotted: str, default=None)`; `resolve_gpus(cfg: KernelConfig, env: Mapping[str, str]) -> list[int]` raising `GpuPolicyError`.

- [ ] **Step 1: Create the conda env and repo skeleton**

```bash
cd /mnt/biometrics/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/AutoResearcher
conda create -y -n autoresearcher python=3.12
mkdir -p kernel/ar_kernel tests configs
touch kernel/ar_kernel/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "ar-kernel"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pyyaml>=6.0", "numpy>=1.26", "pillow>=10.0", "imagehash>=4.3"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
ar = "ar_kernel.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["kernel"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Write `configs/kernel.yaml`**

```yaml
paths:
  worldmodel: ../WorldModel
  wbench: ../WBench
  runs_dir: runs
gpus:
  default: "0,1,2,3"
  min_count: 4
retries:
  edit_self: 3
  improve_recipe: 3
selection:
  gamma: 0.5
  lambda: 0.5
  slope: 6
  top_k_mid: 3
  child_penalty_scale: 8
  epsilon: 0.2
ingest:
  aspect_tolerance: 0.02
leakage:
  phash_max_distance: 4
  min_ncc: 0.95
  min_entropy: 4.0
train:
  resolution_allowlist: [[416, 736], [352, 608]]
  lora_allowlist: [[16, 16], [32, 32], [64, 64]]
eval:
  free_ram_before_render_gb: 120
  ram_wait_alert_min: 10
  vp_weights: weights/qwen3vl-a3b-visual-plausibility
disk:
  merge_min_free_gb: 30
  alert_below_gb: 50
telemetry:
  gpu_sample_sec: 5
tools:
  job_wait_max_s: 300
```

- [ ] **Step 4: Write `configs/proxy_cases.txt`**

```text
2,25,47,63,66,70,78,84,89,90,91,96,109,114,133,136,139,142,145,164,167,172,177,178,180,188,195,200,204,206,215,237,242,246,247,248,260,272,284,289
```

- [ ] **Step 5: Write the failing test**

```python
# tests/test_config.py
import pytest
from ar_kernel.config import KernelConfig, GpuPolicyError, resolve_gpus

def test_loads_defaults_and_resolves_paths():
    cfg = KernelConfig.load()
    assert cfg.get("gpus.min_count") == 4
    assert cfg.worldmodel.is_dir() and cfg.wbench.is_dir()
    assert cfg.get("train.lora_allowlist") == [[16, 16], [32, 32], [64, 64]]

def test_gpu_list_defaults_when_env_unset():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {}) == [0, 1, 2, 3]

def test_gpu_list_uses_env_verbatim_including_gpu5():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,2,4,5"}) == [0, 2, 4, 5]

def test_gpu_list_below_min_count_is_refused():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="at least 4"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2"})

def test_gpu_list_rejects_non_integer_entries():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="not an integer"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2,gpu3"})
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel'`

- [ ] **Step 7: Install the package into the env**

```bash
conda run -n autoresearcher pip install -e ".[dev]"
```

- [ ] **Step 8: Implement `kernel/ar_kernel/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

class GpuPolicyError(ValueError):
    """The GPU list is unusable for this run."""

@dataclass(frozen=True)
class KernelConfig:
    raw: dict
    repo_root: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "KernelConfig":
        path = path or REPO_ROOT / "configs" / "kernel.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(raw=raw, repo_root=REPO_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _path(self, dotted: str) -> Path:
        value = Path(self.get(dotted))
        return value if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def project_root(self) -> Path:
        return self.repo_root.parent

    @property
    def worldmodel(self) -> Path:
        return self._path("paths.worldmodel")

    @property
    def wbench(self) -> Path:
        return self._path("paths.wbench")

    @property
    def runs_dir(self) -> Path:
        return self._path("paths.runs_dir")

def resolve_gpus(cfg: KernelConfig, env: Mapping[str, str]) -> list[int]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "").strip() or str(cfg.get("gpus.default"))
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    gpus: list[int] = []
    for entry in entries:
        if not entry.isdigit():
            raise GpuPolicyError(f"CUDA_VISIBLE_DEVICES entry {entry!r} is not an integer")
        gpus.append(int(entry))
    minimum = int(cfg.get("gpus.min_count"))
    if len(gpus) < minimum:
        raise GpuPolicyError(f"need at least {minimum} GPUs, got {len(gpus)}: {raw!r}")
    return gpus
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml kernel configs tests
git commit -m "feat(kernel): project scaffolding, kernel config and GPU policy"
```

---

### Task 2: Telemetry recorder

**Files:**
- Create: `kernel/ar_kernel/telemetry/__init__.py`, `kernel/ar_kernel/telemetry/recorder.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `KernelConfig`.
- Produces: `Recorder(run_dir: Path, redact: Iterable[str] = ())` with `event(type: str, *, node: str = "run", phase: str = "-", attempt: int = 0, payload: dict | None = None, **fields) -> str` (returns the event id), `span(type, **fields)` context manager emitting `<type>.start` / `<type>.end` (and `<type>.error` on exception, re-raising), `events_path(node) -> Path`, `read_events(node) -> list[dict]`, `store_payload(obj) -> str` (sha256), `load_payload(digest) -> dict`. Raises `TelemetryError` when persistence fails.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
import json, pytest
from ar_kernel.telemetry.recorder import Recorder, TelemetryError

def test_event_is_appended_with_identity_fields(tmp_path):
    rec = Recorder(tmp_path)
    rec.event("train.start", node="n1", phase="train", attempt=1, payload={"lr": 5e-5})
    lines = [json.loads(l) for l in rec.events_path("n1").read_text().splitlines()]
    assert len(lines) == 1
    e = lines[0]
    assert e["type"] == "train.start" and e["node"] == "n1" and e["phase"] == "train"
    assert e["attempt"] == 1 and e["ts_wall"] and e["ts_mono"] and e["span_id"]
    assert rec.load_payload(e["payload"]) == {"lr": 5e-5}

def test_large_payload_is_content_addressed_and_deduped(tmp_path):
    rec = Recorder(tmp_path)
    a = rec.store_payload({"prompt": "x" * 10_000})
    b = rec.store_payload({"prompt": "x" * 10_000})
    assert a == b
    assert rec.load_payload(a)["prompt"].startswith("xxx")

def test_secrets_are_redacted(tmp_path):
    rec = Recorder(tmp_path, redact=["sk-supersecret"])
    digest = rec.store_payload({"headers": {"authorization": "Bearer sk-supersecret"}})
    assert "sk-supersecret" not in json.dumps(rec.load_payload(digest))
    assert "[REDACTED]" in json.dumps(rec.load_payload(digest))

def test_span_emits_start_and_end(tmp_path):
    rec = Recorder(tmp_path)
    with rec.span("merge", node="n1", phase="eval"):
        pass
    types = [json.loads(l)["type"] for l in rec.events_path("n1").read_text().splitlines()]
    assert types == ["merge.start", "merge.end"]

def test_span_records_error_and_reraises(tmp_path):
    rec = Recorder(tmp_path)
    with pytest.raises(RuntimeError):
        with rec.span("merge", node="n1", phase="eval"):
            raise RuntimeError("boom")
    events = [json.loads(l) for l in rec.events_path("n1").read_text().splitlines()]
    assert events[-1]["type"] == "merge.error"
    assert "boom" in json.dumps(rec.load_payload(events[-1]["payload"]))

def test_fail_closed_when_events_cannot_be_written(tmp_path):
    rec = Recorder(tmp_path)
    (tmp_path / "telemetry" / "events").chmod(0o500)
    try:
        with pytest.raises(TelemetryError):
            rec.event("train.start", node="n2")
    finally:
        (tmp_path / "telemetry" / "events").chmod(0o700)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.telemetry'`

- [ ] **Step 3: Implement the recorder**

```python
# kernel/ar_kernel/telemetry/recorder.py
from __future__ import annotations
import hashlib, json, os, time, traceback, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

class TelemetryError(RuntimeError):
    """Telemetry could not be persisted; the caller must not proceed."""

class Recorder:
    def __init__(self, run_dir: Path, redact: Iterable[str] = ()) -> None:
        self.run_dir = Path(run_dir)
        self.redact = [s for s in redact if s]
        self._events = self.run_dir / "telemetry" / "events"
        self._payloads = self.run_dir / "telemetry" / "payloads"
        for directory in (self._events, self._payloads):
            directory.mkdir(parents=True, exist_ok=True)

    def events_path(self, node: str = "run") -> Path:
        return self._events / f"{node}.jsonl"

    def _scrub(self, obj: Any) -> Any:
        if isinstance(obj, str):
            for secret in self.redact:
                obj = obj.replace(secret, "[REDACTED]")
            return obj
        if isinstance(obj, dict):
            return {k: self._scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._scrub(v) for v in obj]
        return obj

    def store_payload(self, obj: dict) -> str:
        blob = json.dumps(self._scrub(obj), sort_keys=True, default=str).encode()
        digest = hashlib.sha256(blob).hexdigest()
        target = self._payloads / f"{digest}.json"
        if not target.exists():
            try:
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(blob)
                os.replace(tmp, target)
            except OSError as exc:
                raise TelemetryError(f"cannot write payload {digest}: {exc}") from exc
        return digest

    def load_payload(self, digest: str) -> dict:
        return json.loads((self._payloads / f"{digest}.json").read_text())

    def event(self, type: str, *, node: str = "run", phase: str = "-", attempt: int = 0,
              payload: dict | None = None, span_id: str | None = None,
              parent_span_id: str | None = None, **fields: Any) -> str:
        record = {
            "ts_wall": time.time(),
            "ts_mono": time.monotonic(),
            "run_dir": str(self.run_dir),
            "node": node,
            "phase": phase,
            "attempt": attempt,
            "span_id": span_id or uuid.uuid4().hex,
            "parent_span_id": parent_span_id,
            "type": type,
            "payload": self.store_payload(payload) if payload is not None else None,
            **self._scrub(fields),
        }
        try:
            with self.events_path(node).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TelemetryError(f"cannot append event {type}: {exc}") from exc
        return record["span_id"]

    def read_events(self, node: str = "run") -> list[dict]:
        path = self.events_path(node)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @contextmanager
    def span(self, type: str, *, node: str = "run", phase: str = "-", attempt: int = 0,
             payload: dict | None = None, **fields: Any):
        span_id = self.event(f"{type}.start", node=node, phase=phase, attempt=attempt,
                             payload=payload, **fields)
        started = time.monotonic()
        try:
            yield span_id
        except BaseException as exc:
            self.event(f"{type}.error", node=node, phase=phase, attempt=attempt,
                       span_id=span_id,
                       payload={"error": repr(exc), "traceback": traceback.format_exc()},
                       duration_s=time.monotonic() - started)
            raise
        self.event(f"{type}.end", node=node, phase=phase, attempt=attempt, span_id=span_id,
                   duration_s=time.monotonic() - started)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_telemetry.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/telemetry tests/test_telemetry.py
git commit -m "feat(kernel): fail-closed telemetry recorder with payload store and redaction"
```

---

### Task 3: Upstream patches U1 and U2

**Files:**
- Modify: `../WBench/tools/run_visual_plausibility.py:19-35`
- Modify: `../WorldModel/scripts/finetune/lowcompute_4x4090.sh:1-25`
- Test: `tests/test_upstream_patches.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `run_visual_plausibility.py --work_dir <abs|rel> --model <name>` writing to `<work_dir>/<model>/evaluation/visual_plausibility`; `lowcompute_4x4090.sh` accepting any GPU indices with a `>= ALAYA_MIN_GPUS` (default 4) count check and honoring `ALAYA_LAUNCH_DRY_RUN=1` to exit 0 before launching training.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_upstream_patches.py
import os, subprocess
from pathlib import Path
from ar_kernel.config import KernelConfig

CFG = KernelConfig.load()

def test_vp_accepts_absolute_work_dir_from_any_cwd(tmp_path):
    videos = tmp_path / "wd" / "mymodel" / "videos"
    videos.mkdir(parents=True)
    out = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "wbench-main", "python",
         str(CFG.wbench / "tools" / "run_visual_plausibility.py"),
         "--work_dir", str(tmp_path / "wd"), "--model", "mymodel"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert str(videos) in out.stdout, out.stdout + out.stderr

def test_launcher_refuses_fewer_gpus_than_minimum():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2", "ALAYA_LAUNCH_DRY_RUN": "1"}
    proc = subprocess.run(["bash", str(CFG.worldmodel / "scripts/finetune/lowcompute_4x4090.sh")],
                          cwd=CFG.worldmodel, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "at least 4" in (proc.stdout + proc.stderr)

def test_launcher_accepts_any_indices_including_gpu5():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,5", "ALAYA_LAUNCH_DRY_RUN": "1"}
    proc = subprocess.run(["bash", str(CFG.worldmodel / "scripts/finetune/lowcompute_4x4090.sh")],
                          cwd=CFG.worldmodel, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,5" in proc.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_upstream_patches.py -v`
Expected: FAIL — VP errors on the unknown `--work_dir` flag, and the launcher exits 1 on the list containing GPU 5.

- [ ] **Step 3: Apply U1 to WBench's VP script**

Replace the argument block and the two path assignments (currently lines 19-27):

```python
    parser.add_argument("--model", required=True, help="Model name (e.g. hunyuan)")
    parser.add_argument("--work_dir", default="work_dirs",
                        help="work_dirs root; absolute paths are used as-is")
    parser.add_argument("--model_path", default="weights/qwen3vl-a3b-visual-plausibility")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Force re-evaluate existing results")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_dir = os.path.join(project_root, args.work_dir)  # absolute args.work_dir wins
    model_path = os.path.join(project_root, args.model_path)
    video_dir = os.path.join(work_dir, args.model, "videos")
    out_dir = os.path.join(work_dir, args.model, "evaluation", "visual_plausibility")
```

Then use `model_path` where `args.model_path` was passed to `PhysicalPlausibilityEvaluator`.

- [ ] **Step 4: Apply U2 to the WorldModel launcher**

Replace the GPU-5 rejection block with a count check plus the dry-run guard:

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
ALAYA_MIN_GPUS=${ALAYA_MIN_GPUS:-4}
IFS=',' read -ra _alaya_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#_alaya_gpus[@]}" -lt "$ALAYA_MIN_GPUS" ]; then
    echo "ERROR: this recipe needs at least $ALAYA_MIN_GPUS GPUs;" \
         "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has ${#_alaya_gpus[@]}." >&2
    exit 1
fi
unset _alaya_gpus
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

and immediately before the final `exec bash "$(dirname "$0")/train.sh"`:

```bash
if [ "${ALAYA_LAUNCH_DRY_RUN:-0}" = "1" ]; then
    echo "ALAYA_LAUNCH_DRY_RUN=1: environment validated, not launching training"
    exit 0
fi
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_upstream_patches.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit all three repos**

```bash
cd ../WBench && git add tools/run_visual_plausibility.py && \
  git commit -m "fix(tools): let run_visual_plausibility take --work_dir and resolve paths from the repo root"
cd ../WorldModel && git add scripts/finetune/lowcompute_4x4090.sh && \
  git commit -m "fix(launcher): require a GPU count instead of rejecting GPU 5; add ALAYA_LAUNCH_DRY_RUN"
cd ../AutoResearcher && git add tests/test_upstream_patches.py && \
  git commit -m "test(kernel): cover upstream patches U1 and U2"
```

---

### Task 4: Archive database and node records

**Files:**
- Create: `kernel/ar_kernel/archive/__init__.py`, `kernel/ar_kernel/archive/db.py`, `kernel/ar_kernel/archive/nodes.py`
- Test: `tests/test_archive_nodes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `open_db(run_dir: Path) -> sqlite3.Connection` (WAL, foreign keys, schema applied); `NodeStore(conn)` with `create(node_id: str, parent_id: str | None, depth: int) -> None`, `set_status(node_id, status)`, `record_score(node_id, score: float, metric_set: list[str], metrics: dict[str, float])`, `get(node_id) -> dict`, `all() -> list[dict]`, `children(node_id) -> list[dict]`, `add_attempt(node_id, phase: str, index: int, outcome: str, detail: dict)`, `attempts(node_id, phase) -> list[dict]`. Status values: `running`, `scored`, `invalid_code`, `invalid_recipe`, `train_failed`, `crashed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archive_nodes.py
import pytest
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore

def test_create_and_read_node(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    node = store.get("root")
    assert node["status"] == "running" and node["parent_id"] is None and node["depth"] == 0

def test_score_roundtrip_preserves_metric_set_order(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    store.record_score("root", 0.8123, ["aesthetic_quality", "hpsv3_quality"],
                       {"aesthetic_quality": 0.80, "hpsv3_quality": 0.82})
    node = store.get("root")
    assert node["score"] == pytest.approx(0.8123)
    assert node["metric_set"] == ["aesthetic_quality", "hpsv3_quality"]
    assert node["metrics"]["hpsv3_quality"] == pytest.approx(0.82)
    assert node["status"] == "scored"

def test_children_and_status_updates(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    store.create("n1", "root", 1)
    store.create("n2", "root", 1)
    store.set_status("n2", "invalid_code")
    assert {c["node_id"] for c in store.children("root")} == {"n1", "n2"}
    assert store.get("n2")["status"] == "invalid_code"

def test_unknown_status_is_rejected(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    with pytest.raises(ValueError, match="unknown status"):
        store.set_status("root", "finished")

def test_attempts_are_ordered_per_phase(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("n1", None, 0)
    store.add_attempt("n1", "improve_recipe", 1, "failed", {"reason": "gate"})
    store.add_attempt("n1", "improve_recipe", 2, "ok", {})
    outcomes = [a["outcome"] for a in store.attempts("n1", "improve_recipe")]
    assert outcomes == ["failed", "ok"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_archive_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.archive'`

- [ ] **Step 3: Implement `db.py`**

```python
# kernel/ar_kernel/archive/db.py
from __future__ import annotations
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES nodes(node_id),
  depth INTEGER NOT NULL,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,
  agent_commit TEXT, data_commit TEXT, recipe_hash TEXT,
  resolved_config_path TEXT, checkpoint_path TEXT,
  lora_rank INTEGER, lora_alpha INTEGER,
  score REAL, metric_set TEXT, metrics TEXT, subtree_value REAL,
  phase_timings TEXT, rationale_path TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL REFERENCES nodes(node_id),
  phase TEXT NOT NULL, idx INTEGER NOT NULL,
  outcome TEXT NOT NULL, detail TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_node_phase ON attempts(node_id, phase, idx);
CREATE TABLE IF NOT EXISTS blobs (
  digest TEXT PRIMARY KEY, kind TEXT NOT NULL, bytes INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
  clip_id TEXT PRIMARY KEY,
  video_digest TEXT NOT NULL, caption_digest TEXT NOT NULL, pose_digest TEXT,
  camera_motion TEXT NOT NULL, metadata TEXT NOT NULL, formats TEXT NOT NULL,
  warnings TEXT NOT NULL, provenance TEXT NOT NULL, license TEXT,
  derived_from TEXT NOT NULL, ingested_by TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS data_commits (
  commit_id TEXT PRIMARY KEY, parent_commit TEXT, node_id TEXT, attempt INTEGER,
  manifest TEXT NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL
);
"""

def open_db(run_dir: Path) -> sqlite3.Connection:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(run_dir / "archive.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
```

- [ ] **Step 4: Implement `nodes.py`**

```python
# kernel/ar_kernel/archive/nodes.py
from __future__ import annotations
import json, sqlite3, time

STATUSES = {"running", "scored", "invalid_code", "invalid_recipe", "train_failed", "crashed"}

class NodeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, node_id: str, parent_id: str | None, depth: int) -> None:
        self.conn.execute(
            "INSERT INTO nodes (node_id, parent_id, depth, created_at, status) VALUES (?,?,?,?,'running')",
            (node_id, parent_id, depth, time.time()),
        )

    def set_status(self, node_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; allowed: {sorted(STATUSES)}")
        self.conn.execute("UPDATE nodes SET status=? WHERE node_id=?", (status, node_id))

    def record_score(self, node_id: str, score: float, metric_set: list[str],
                     metrics: dict[str, float]) -> None:
        self.conn.execute(
            "UPDATE nodes SET score=?, metric_set=?, metrics=?, status='scored' WHERE node_id=?",
            (float(score), json.dumps(list(metric_set)), json.dumps(metrics), node_id),
        )

    def set_fields(self, node_id: str, **fields: object) -> None:
        allowed = {"agent_commit", "data_commit", "recipe_hash", "resolved_config_path",
                   "checkpoint_path", "lora_rank", "lora_alpha", "subtree_value",
                   "phase_timings", "rationale_path"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown node fields: {sorted(unknown)}")
        for key, value in fields.items():
            self.conn.execute(f"UPDATE nodes SET {key}=? WHERE node_id=?", (value, node_id))

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        node = dict(row)
        node["metric_set"] = json.loads(node["metric_set"]) if node["metric_set"] else []
        node["metrics"] = json.loads(node["metrics"]) if node["metrics"] else {}
        return node

    def get(self, node_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        return self._row(row)

    def all(self) -> list[dict]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM nodes ORDER BY created_at")]

    def children(self, node_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM nodes WHERE parent_id=? ORDER BY created_at", (node_id,))
        return [self._row(r) for r in rows]

    def add_attempt(self, node_id: str, phase: str, index: int, outcome: str, detail: dict) -> None:
        self.conn.execute(
            "INSERT INTO attempts (node_id, phase, idx, outcome, detail, created_at) VALUES (?,?,?,?,?,?)",
            (node_id, phase, index, outcome, json.dumps(detail), time.time()),
        )

    def attempts(self, node_id: str, phase: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE node_id=? AND phase=? ORDER BY idx", (node_id, phase))
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_archive_nodes.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/archive tests/test_archive_nodes.py
git commit -m "feat(archive): sqlite schema, node records and attempt log"
```

---

### Task 5: Content-addressed blob store

**Files:**
- Create: `kernel/ar_kernel/archive/blobs.py`
- Test: `tests/test_blobs.py`

**Interfaces:**
- Consumes: `open_db`.
- Produces: `BlobStore(run_dir: Path, conn)` with `put(path: Path, kind: str) -> str` (moves the file in, returns sha256), `path(digest: str, kind: str) -> Path`, `exists(digest) -> bool`, `size(digest) -> int`. Kinds: `video` (`.mp4`), `caption` (`.json`), `pose` (`.npz`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blobs.py
import hashlib, pytest
from ar_kernel.archive.blobs import BlobStore, BlobError
from ar_kernel.archive.db import open_db

def _clip(tmp_path, name, data=b"video-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return p

def test_put_returns_sha256_and_moves_file(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    src = _clip(tmp_path, "a.mp4")
    digest = store.put(src, "video")
    assert digest == hashlib.sha256(b"video-bytes").hexdigest()
    assert not src.exists()
    assert store.path(digest, "video").read_bytes() == b"video-bytes"

def test_identical_content_is_stored_once(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    d1 = store.put(_clip(tmp_path, "a.mp4"), "video")
    d2 = store.put(_clip(tmp_path, "b.mp4"), "video")
    assert d1 == d2
    assert len(list((tmp_path / "store" / "blobs" / "video").iterdir())) == 1

def test_stored_blob_is_read_only(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    digest = store.put(_clip(tmp_path, "a.mp4"), "video")
    assert store.path(digest, "video").stat().st_mode & 0o222 == 0

def test_unknown_kind_is_rejected(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    with pytest.raises(BlobError, match="unknown kind"):
        store.put(_clip(tmp_path, "a.bin"), "depth")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_blobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.archive.blobs'`

- [ ] **Step 3: Implement the blob store**

```python
# kernel/ar_kernel/archive/blobs.py
from __future__ import annotations
import hashlib, os, sqlite3, time
from pathlib import Path

KINDS = {"video": ".mp4", "caption": ".json", "pose": ".npz"}

class BlobError(ValueError):
    """The blob cannot be stored."""

class BlobStore:
    def __init__(self, run_dir: Path, conn: sqlite3.Connection) -> None:
        self.root = Path(run_dir) / "store" / "blobs"
        self.conn = conn
        for kind in KINDS:
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def path(self, digest: str, kind: str) -> Path:
        if kind not in KINDS:
            raise BlobError(f"unknown kind {kind!r}; allowed: {sorted(KINDS)}")
        return self.root / kind / f"{digest}{KINDS[kind]}"

    def exists(self, digest: str) -> bool:
        return self.conn.execute("SELECT 1 FROM blobs WHERE digest=?", (digest,)).fetchone() is not None

    def size(self, digest: str) -> int:
        row = self.conn.execute("SELECT bytes FROM blobs WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        return int(row["bytes"])

    def put(self, path: Path, kind: str) -> str:
        target_dir = self.path("0" * 64, kind).parent  # validates the kind
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        target = target_dir / f"{hexdigest}{KINDS[kind]}"
        nbytes = path.stat().st_size
        if target.exists():
            path.unlink()
        else:
            os.replace(path, target)
            target.chmod(0o444)
        self.conn.execute(
            "INSERT OR IGNORE INTO blobs (digest, kind, bytes, created_at) VALUES (?,?,?,?)",
            (hexdigest, kind, nbytes, time.time()),
        )
        return hexdigest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_blobs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/archive/blobs.py tests/test_blobs.py
git commit -m "feat(archive): content-addressed blob store with immutable blobs"
```

---

### Task 6: Clip probing and per-format eligibility

**Files:**
- Create: `kernel/ar_kernel/subproc.py`, `kernel/ar_kernel/data/__init__.py`, `kernel/ar_kernel/data/probe.py`, `kernel/ar_kernel/data/checker.py`, `kernel/ar_kernel/bridges/check_clip.py`
- Test: `tests/test_probe.py`, `tests/test_checker.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `KernelConfig`, `Recorder`.
- Produces: `run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None, timeout: int | None = None, recorder=None, node="run", phase="-") -> subprocess.CompletedProcess`; `probe_video(path: Path) -> VideoInfo(frames, fps, width, height, duration)`; `aspect_ok(info, tolerance) -> bool`; `check_clip_formats(cfg, clip_dir: Path, camera_motion: str, base_recipe: Path) -> dict[str, dict]` mapping `"video_caption_camera" | "video_timed_prompts_camera:segment" | "video_timed_prompts_camera:per_chunk" | "video_caption_static"` to `{"ok": bool, "errors": [...], "warnings": [...]}`.

- [ ] **Step 1: Write the shared clip fixtures**

```python
# tests/conftest.py
import json, subprocess
import numpy as np
import pytest

def make_mp4(path, seconds=4.0, fps=30, width=736, height=414):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path

def write_caption(path, caption="A camera moves slowly through a bright room.", segments=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"caption": caption}
    if segments is not None:
        payload["segments"] = segments
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def write_poses(path, n_frames, width=736, height=414, moving=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    if moving:
        c2w[:, 2, 3] = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    k = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]], dtype=np.float32)
    np.savez(path, cam_c2w=c2w, intrinsics=k)
    return path

@pytest.fixture
def clip_dir(tmp_path):
    """A one-clip standard root: 4 s, 30 fps, 16:9, moving camera."""
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4")
    write_caption(root / "captions" / "c1.json")
    write_poses(root / "poses" / "c1.npz", n_frames=120)
    return root
```

- [ ] **Step 2: Write the failing probe test**

```python
# tests/test_probe.py
import pytest
from ar_kernel.data.probe import probe_video, aspect_ok
from conftest import make_mp4

def test_probe_reads_fps_frames_and_size(clip_dir):
    info = probe_video(clip_dir / "videos" / "c1.mp4")
    assert info.fps == pytest.approx(30.0, abs=0.01)
    assert info.frames == 120
    assert (info.width, info.height) == (736, 414)
    assert info.duration == pytest.approx(4.0, abs=0.05)

def test_aspect_ok_accepts_16_9_and_rejects_4_3(tmp_path):
    wide = probe_video(make_mp4(tmp_path / "w.mp4", width=1280, height=720))
    narrow = probe_video(make_mp4(tmp_path / "n.mp4", width=640, height=480))
    assert aspect_ok(wide, 0.02) is True
    assert aspect_ok(narrow, 0.02) is False
```

- [ ] **Step 3: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.data'`

- [ ] **Step 4: Implement `subproc.py` and `probe.py`**

```python
# kernel/ar_kernel/subproc.py
from __future__ import annotations
import os, subprocess
from pathlib import Path

def run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None,
               timeout: int | None = None, recorder=None, node: str = "run",
               phase: str = "-") -> subprocess.CompletedProcess:
    command = ["conda", "run", "--no-capture-output", "-n", env, *args]
    process_env = {**os.environ, **(extra_env or {})}
    if recorder is not None:
        recorder.event("subproc.start", node=node, phase=phase,
                       payload={"env": env, "args": args, "cwd": str(cwd),
                                "extra_env": extra_env or {}})
    proc = subprocess.run(command, cwd=str(cwd), env=process_env, capture_output=True,
                          text=True, timeout=timeout)
    if recorder is not None:
        recorder.event("subproc.end", node=node, phase=phase, returncode=proc.returncode,
                       payload={"stdout": proc.stdout, "stderr": proc.stderr})
    return proc
```

```python
# kernel/ar_kernel/data/probe.py
from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_ASPECT = 16 / 9

@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int
    duration: float

def probe_video(path: Path) -> VideoInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    frames = int(stream["nb_read_frames"])
    duration = float(stream.get("duration") or (frames / fps if fps else 0.0))
    return VideoInfo(frames=frames, fps=fps, width=int(stream["width"]),
                     height=int(stream["height"]), duration=duration)

def aspect_ok(info: VideoInfo, tolerance: float) -> bool:
    return abs((info.width / info.height) - TARGET_ASPECT) <= TARGET_ASPECT * tolerance
```

- [ ] **Step 5: Run the probe tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_probe.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Write the failing checker test**

```python
# tests/test_checker.py
from ar_kernel.config import KernelConfig
from ar_kernel.data.checker import check_clip_formats
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"

def test_moving_camera_clip_is_eligible_for_camera_format(clip_dir):
    result = check_clip_formats(CFG, clip_dir, "moving", BASE)
    assert result["video_caption_camera"]["ok"] is True
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is False  # no segments

def test_short_clip_is_rejected_with_the_checker_message(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=1.0)
    write_caption(root / "captions" / "c1.json")
    write_poses(root / "poses" / "c1.npz", n_frames=30)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_caption_camera"]["ok"] is False
    assert any("shorter than one training window" in e
               for e in result["video_caption_camera"]["errors"])

def test_round_aligned_segments_pass_per_chunk(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=8.0, fps=24)
    write_caption(root / "captions" / "c1.json", segments=[
        {"time_range_s": [0.0, 3.7083], "prompt": "A bright room seen head on."},
        {"time_range_s": [3.7083, 8.0], "prompt": "The same room as a pencil sketch."},
    ])
    write_poses(root / "poses" / "c1.npz", n_frames=192)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is True

def test_misaligned_segments_fail_per_chunk_but_pass_segment(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=10.0, fps=24)
    write_caption(root / "captions" / "c1.json", segments=[
        {"time_range_s": [0.0, 5.0], "prompt": "A bright room seen head on."},
        {"time_range_s": [5.0, 10.0], "prompt": "The same room as a pencil sketch."},
    ])
    write_poses(root / "poses" / "c1.npz", n_frames=240)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is False
    assert result["video_timed_prompts_camera:segment"]["ok"] is True

def test_static_clip_needs_no_poses(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4")
    write_caption(root / "captions" / "c1.json")
    result = check_clip_formats(CFG, root, "static", BASE)
    assert result["video_caption_static"]["ok"] is True
    assert "video_caption_camera" not in result
```

- [ ] **Step 7: Copy the base recipe**

```bash
cp ../WorldModel/configs/examples/finetune_video_caption_camera.yaml configs/base_recipe.yaml
```

- [ ] **Step 8: Run the checker test to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_checker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.data.checker'`

- [ ] **Step 9: Implement the bridge script**

```python
# kernel/ar_kernel/bridges/check_clip.py
"""Run WorldModel's standard-dataset checker for one clip root, per format.

Executed inside the `alayaworld` env:
    python check_clip.py --config <recipe> --root <clip root> --camera-motion moving|static
Writes a JSON report to stdout between the markers so conda banners cannot corrupt it.
"""
from __future__ import annotations
import argparse, json, sys

MARK = "===AR_JSON==="

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera-motion", required=True, choices=["moving", "static"])
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.standard import DatasetSpec
    from alaya.data.standard_check import check_dataset, layout_from_config

    cfg = load_config(args.config)
    layout = layout_from_config(cfg)
    if args.camera_motion == "static":
        wanted = [("video_caption_static", None)]
    else:
        wanted = [("video_caption_camera", None),
                  ("video_timed_prompts_camera", "segment"),
                  ("video_timed_prompts_camera", "per_chunk")]
    out: dict[str, dict] = {}
    for fmt, mode in wanted:
        spec = DatasetSpec(name="probe", root=args.root, format=fmt, weight=1.0, prompt_mode=mode)
        report = check_dataset(spec, **layout)
        key = fmt if mode is None else f"{fmt}:{mode}"
        out[key] = {"ok": not report.errors, "errors": list(report.errors),
                    "warnings": list(report.warnings), "clips": report.clips,
                    "windows": report.windows}
    print(MARK + json.dumps(out) + MARK)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10: Implement `checker.py`**

```python
# kernel/ar_kernel/data/checker.py
from __future__ import annotations
import json
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "check_clip.py"
MARK = "===AR_JSON==="

class CheckerError(RuntimeError):
    """The checker bridge did not return a report."""

def check_clip_formats(cfg: KernelConfig, clip_dir: Path, camera_motion: str,
                       base_recipe: Path, recorder=None, node: str = "run") -> dict[str, dict]:
    proc = run_in_env(
        "alayaworld",
        ["python", str(BRIDGE), "--config", str(base_recipe), "--root", str(clip_dir),
         "--camera-motion", camera_motion],
        cwd=cfg.worldmodel, timeout=600, recorder=recorder, node=node, phase="ingest",
    )
    if MARK not in proc.stdout:
        raise CheckerError(f"checker bridge failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.split(MARK)[1])
```

- [ ] **Step 11: Run the checker tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_checker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 12: Commit**

```bash
git add kernel/ar_kernel/subproc.py kernel/ar_kernel/data kernel/ar_kernel/bridges \
        configs/base_recipe.yaml tests/conftest.py tests/test_probe.py tests/test_checker.py
git commit -m "feat(data): ffprobe metadata, aspect rule and per-format eligibility via the standard checker"
```

---

### Task 7: Benchmark leakage check

**Files:**
- Create: `kernel/ar_kernel/data/leakage.py`
- Test: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `KernelConfig`, `probe_video`.
- Produces: `LeakageChecker(cfg)` with `check(video: Path) -> LeakageVerdict(rejected: bool, matches: list[dict], near_matches: list[dict])`; helper `frame_signals(img) -> (phash, ncc_vector, entropy)`. Reference frames come from `<wbench>/data/images/case_<id>.jpg` (289 files).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leakage.py
import shutil, subprocess
from ar_kernel.config import KernelConfig
from ar_kernel.data.leakage import LeakageChecker
from conftest import make_mp4

CFG = KernelConfig.load()

def _video_from_image(image, out, seconds=3.0, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
                    "-t", str(seconds), "-r", str(fps), "-vf", "scale=736:414",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out

def test_wbench_first_frame_is_rejected(tmp_path):
    case_image = CFG.wbench / "data" / "images" / "case_2.jpg"
    video = _video_from_image(case_image, tmp_path / "leak.mp4")
    verdict = LeakageChecker(CFG).check(video)
    assert verdict.rejected is True
    assert verdict.matches and verdict.matches[0]["case_id"] == "2"

def test_unrelated_synthetic_clip_is_accepted(tmp_path):
    video = make_mp4(tmp_path / "ok.mp4")
    assert LeakageChecker(CFG).check(video).rejected is False

def test_flat_frame_cannot_trigger_rejection_alone(tmp_path):
    flat = tmp_path / "flat.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=gray:size=736x414:rate=24:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(flat)], check=True)
    verdict = LeakageChecker(CFG).check(flat)
    assert verdict.rejected is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.data.leakage'`

- [ ] **Step 3: Implement the leakage check**

```python
# kernel/ar_kernel/data/leakage.py
from __future__ import annotations
import subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from ..config import KernelConfig
from .probe import probe_video

SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
NCC_SIZE = 32

@dataclass
class LeakageVerdict:
    rejected: bool
    matches: list[dict] = field(default_factory=list)
    near_matches: list[dict] = field(default_factory=list)

def _grayscale_vector(image: Image.Image) -> np.ndarray:
    small = np.asarray(image.convert("L").resize((NCC_SIZE, NCC_SIZE)), dtype=np.float64).ravel()
    centered = small - small.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered

def _entropy(image: Image.Image) -> float:
    hist = np.asarray(image.convert("L").histogram(), dtype=np.float64)
    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())

def _extract(video: Path, timestamp: float) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                        "-i", str(video), "-frames:v", "1", str(out)], check=True)
        return Image.open(out).copy()

class LeakageChecker:
    def __init__(self, cfg: KernelConfig) -> None:
        self.max_distance = int(cfg.get("leakage.phash_max_distance"))
        self.min_ncc = float(cfg.get("leakage.min_ncc"))
        self.min_entropy = float(cfg.get("leakage.min_entropy"))
        self.references: list[tuple[str, imagehash.ImageHash, np.ndarray]] = []
        for path in sorted((cfg.wbench / "data" / "images").glob("case_*.jpg")):
            image = Image.open(path)
            case_id = path.stem.replace("case_", "")
            self.references.append((case_id, imagehash.phash(image), _grayscale_vector(image)))

    def check(self, video: Path) -> LeakageVerdict:
        info = probe_video(video)
        verdict = LeakageVerdict(rejected=False)
        for fraction in SAMPLE_FRACTIONS:
            frame = _extract(video, fraction * info.duration)
            frame_hash = imagehash.phash(frame)
            frame_vector = _grayscale_vector(frame)
            flat = _entropy(frame) < self.min_entropy
            for case_id, ref_hash, ref_vector in self.references:
                distance = frame_hash - ref_hash
                if distance > self.max_distance:
                    continue
                ncc = float(np.dot(frame_vector, ref_vector))
                record = {"case_id": case_id, "fraction": fraction,
                          "phash_distance": int(distance), "ncc": ncc, "flat": flat}
                if ncc >= self.min_ncc and not flat:
                    verdict.matches.append(record)
                    verdict.rejected = True
                else:
                    verdict.near_matches.append(record)
        return verdict
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_leakage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/data/leakage.py tests/test_leakage.py
git commit -m "feat(data): two-signal WBench leakage check with flat-frame guard"
```

---

### Task 8: Ingest pipeline

**Files:**
- Create: `kernel/ar_kernel/data/ingest.py`, `kernel/ar_kernel/archive/clips.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `BlobStore`, `check_clip_formats`, `LeakageChecker`, `probe_video`, `aspect_ok`, `Recorder`.
- Produces: `ClipStore(conn)` with `add(record: dict) -> None`, `get(clip_id) -> dict`, `all() -> list[dict]`, `eligible(fmt: str, prompt_mode: str | None) -> list[dict]`; `Ingestor(cfg, run_dir, conn, recorder)` with `ingest(candidates: list[Candidate], node_id: str) -> list[IngestResult]`, where `Candidate(video: Path, caption: Path, pose: Path | None, camera_motion: str, provenance: dict, license: str | None = None, derived_from: list[str] = [])` and `IngestResult(accepted: bool, clip_id: str | None, formats: list[str], warnings: list[str], reasons: list[str])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import pytest
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test fixture"}

def _ingestor(tmp_path):
    return Ingestor(CFG, tmp_path, open_db(tmp_path), Recorder(tmp_path))

def _candidate(tmp_path, name="c1", seconds=4.0, fps=30, width=736, height=414, moving=True):
    stage = tmp_path / "staging" / name
    video = make_mp4(stage / "v.mp4", seconds=seconds, fps=fps, width=width, height=height)
    caption = write_caption(stage / "c.json")
    pose = write_poses(stage / "p.npz", n_frames=int(seconds * fps)) if moving else None
    return Candidate(video=video, caption=caption, pose=pose,
                     camera_motion="moving" if moving else "static", provenance=PROV)

def test_accepted_clip_records_formats_and_moves_blobs(tmp_path):
    ing = _ingestor(tmp_path)
    [result] = ing.ingest([_candidate(tmp_path)], node_id="n1")
    assert result.accepted and "video_caption_camera" in result.formats
    clip = ing.clips.get(result.clip_id)
    assert clip["camera_motion"] == "moving" and clip["ingested_by"] == "n1"
    assert not (tmp_path / "staging" / "c1" / "v.mp4").exists()

def test_duplicate_content_yields_the_same_clip_id(tmp_path):
    ing = _ingestor(tmp_path)
    [first] = ing.ingest([_candidate(tmp_path, "a")], node_id="n1")
    [second] = ing.ingest([_candidate(tmp_path, "b")], node_id="n2")
    assert first.clip_id == second.clip_id
    assert len(ing.clips.all()) == 1

def test_four_by_three_clip_is_rejected_before_the_checker(tmp_path):
    ing = _ingestor(tmp_path)
    [result] = ing.ingest([_candidate(tmp_path, width=640, height=480)], node_id="n1")
    assert result.accepted is False
    assert any("aspect ratio" in r for r in result.reasons)

def test_clip_failing_every_format_is_rejected_with_checker_messages(tmp_path):
    ing = _ingestor(tmp_path)
    [result] = ing.ingest([_candidate(tmp_path, seconds=1.0)], node_id="n1")
    assert result.accepted is False
    assert any("shorter than one training window" in r for r in result.reasons)

def test_missing_provenance_is_rejected(tmp_path):
    ing = _ingestor(tmp_path)
    candidate = _candidate(tmp_path)
    candidate = Candidate(video=candidate.video, caption=candidate.caption, pose=candidate.pose,
                          camera_motion="moving", provenance={})
    [result] = ing.ingest([candidate], node_id="n1")
    assert result.accepted is False and any("provenance" in r for r in result.reasons)

def test_static_candidate_with_poses_is_rejected(tmp_path):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    static = Candidate(video=c.video, caption=c.caption, pose=c.pose, camera_motion="static",
                       provenance=PROV)
    [result] = ing.ingest([static], node_id="n1")
    assert result.accepted is False and any("static" in r for r in result.reasons)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.data.ingest'`

- [ ] **Step 3: Implement `clips.py`**

```python
# kernel/ar_kernel/archive/clips.py
from __future__ import annotations
import json, sqlite3, time

class ClipStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, record: dict) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO clips (clip_id, video_digest, caption_digest, pose_digest,
               camera_motion, metadata, formats, warnings, provenance, license, derived_from,
               ingested_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record["clip_id"], record["video_digest"], record["caption_digest"],
             record.get("pose_digest"), record["camera_motion"],
             json.dumps(record["metadata"]), json.dumps(record["formats"]),
             json.dumps(record["warnings"]), json.dumps(record["provenance"]),
             record.get("license"), json.dumps(record.get("derived_from", [])),
             record["ingested_by"], time.time()),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        clip = dict(row)
        for key in ("metadata", "formats", "warnings", "provenance", "derived_from"):
            clip[key] = json.loads(clip[key])
        return clip

    def get(self, clip_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        return self._row(row)

    def all(self) -> list[dict]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM clips ORDER BY created_at")]

    def eligible(self, fmt: str, prompt_mode: str | None = None) -> list[dict]:
        key = fmt if prompt_mode is None else f"{fmt}:{prompt_mode}"
        return [c for c in self.all() if key in c["formats"]]
```

- [ ] **Step 4: Implement `ingest.py`**

```python
# kernel/ar_kernel/data/ingest.py
from __future__ import annotations
import hashlib, json, shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig
from .checker import check_clip_formats
from .leakage import LeakageChecker
from .probe import aspect_ok, probe_video

@dataclass
class Candidate:
    video: Path
    caption: Path
    pose: Path | None
    camera_motion: str
    provenance: dict
    license: str | None = None
    derived_from: list[str] = field(default_factory=list)

@dataclass
class IngestResult:
    accepted: bool
    clip_id: str | None = None
    formats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

class Ingestor:
    def __init__(self, cfg: KernelConfig, run_dir: Path, conn, recorder) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.blobs = BlobStore(run_dir, conn)
        self.clips = ClipStore(conn)
        self.recorder = recorder
        self.leakage = LeakageChecker(cfg)
        self.base_recipe = cfg.repo_root / "configs" / "base_recipe.yaml"
        self.tolerance = float(cfg.get("ingest.aspect_tolerance"))

    def ingest(self, candidates: list[Candidate], node_id: str) -> list[IngestResult]:
        return [self._one(c, node_id) for c in candidates]

    def _one(self, candidate: Candidate, node_id: str) -> IngestResult:
        reasons: list[str] = []
        if not candidate.provenance:
            reasons.append("provenance is required")
        if candidate.camera_motion not in {"moving", "static"}:
            reasons.append(f"camera_motion must be moving or static, got {candidate.camera_motion!r}")
        if candidate.camera_motion == "static" and candidate.pose is not None:
            reasons.append("static clips must not carry poses (video_caption_static uses identity poses)")
        if candidate.camera_motion == "moving" and candidate.pose is None:
            reasons.append("moving clips need poses/<id>.npz")
        if reasons:
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        info = probe_video(candidate.video)
        if not aspect_ok(info, self.tolerance):
            reasons.append(f"aspect ratio {info.width}/{info.height} is not within "
                           f"{self.tolerance:.0%} of 16:9")
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        probe_root = self.run_dir / "tmp" / f"probe_{_digest(candidate.video)[:12]}"
        shutil.rmtree(probe_root, ignore_errors=True)
        (probe_root / "videos").mkdir(parents=True)
        (probe_root / "captions").mkdir(parents=True)
        shutil.copy2(candidate.video, probe_root / "videos" / "c.mp4")
        shutil.copy2(candidate.caption, probe_root / "captions" / "c.json")
        if candidate.pose is not None:
            (probe_root / "poses").mkdir(parents=True)
            shutil.copy2(candidate.pose, probe_root / "poses" / "c.npz")
        try:
            reports = check_clip_formats(self.cfg, probe_root, candidate.camera_motion,
                                         self.base_recipe, recorder=self.recorder, node=node_id)
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)

        formats = [key for key, report in reports.items() if report["ok"]]
        warnings = sorted({w for report in reports.values() for w in report["warnings"]})
        if not formats:
            reasons = sorted({e for report in reports.values() for e in report["errors"]})
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons, "reports": reports})
            return IngestResult(accepted=False, reasons=reasons)

        verdict = self.leakage.check(candidate.video)
        self.recorder.event("ingest.leakage", node=node_id, phase="ingest",
                            payload={"matches": verdict.matches, "near": verdict.near_matches})
        if verdict.rejected:
            reasons = [f"matches WBench case {m['case_id']} "
                       f"(phash {m['phash_distance']}, ncc {m['ncc']:.3f})" for m in verdict.matches]
            return IngestResult(accepted=False, reasons=reasons)

        video_digest = self.blobs.put(candidate.video, "video")
        caption_digest = self.blobs.put(candidate.caption, "caption")
        pose_digest = self.blobs.put(candidate.pose, "pose") if candidate.pose else None
        clip_id = hashlib.sha256(json.dumps(
            {"video": video_digest, "caption": caption_digest, "pose": pose_digest,
             "camera_motion": candidate.camera_motion}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, "video_digest": video_digest, "caption_digest": caption_digest,
            "pose_digest": pose_digest, "camera_motion": candidate.camera_motion,
            "metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                         "height": info.height, "duration": info.duration},
            "formats": formats, "warnings": warnings, "provenance": candidate.provenance,
            "license": candidate.license, "derived_from": candidate.derived_from,
            "ingested_by": node_id,
        })
        self.recorder.event("ingest.accepted", node=node_id, phase="ingest",
                            payload={"clip_id": clip_id, "formats": formats, "warnings": warnings})
        return IngestResult(accepted=True, clip_id=clip_id, formats=formats, warnings=warnings)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_ingest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/data/ingest.py kernel/ar_kernel/archive/clips.py tests/test_ingest.py
git commit -m "feat(data): ingest pipeline with format eligibility, leakage screen and provenance"
```

---

### Task 9: Data commits and view materialization

**Files:**
- Create: `kernel/ar_kernel/archive/commits.py`
- Test: `tests/test_commits.py`

**Interfaces:**
- Consumes: `ClipStore`, `BlobStore`.
- Produces: `CommitStore(conn, blobs, clips)` with `commit(parent: str | None, datasets: dict[str, dict], message: str, node_id: str, attempt: int = 0) -> str` (validates and returns `commit_id`), `get(commit_id) -> dict`, `manifest(commit_id) -> dict`; `materialize(commit_id, dest: Path) -> dict[str, Path]` building hardlink farms. Dataset entry shape: `{"format": str, "prompt_mode": str | None, "weight": float, "clips": [clip_id, ...]}`. Raises `CommitError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_commits.py
import pytest
from ar_kernel.archive.commits import CommitStore, CommitError
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test fixture"}

@pytest.fixture
def store(tmp_path):
    conn = open_db(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, Recorder(tmp_path))
    ids = []
    # Durations must differ: clip_id is content-addressed, so byte-identical
    # candidates collapse to one clip. Pose count tracks the real frame count (30 fps).
    for name, seconds in (("a", 4.0), ("b", 5.0)):
        stage = tmp_path / "staging" / name
        ids.append(ing.ingest([Candidate(
            video=make_mp4(stage / "v.mp4", seconds=seconds),
            caption=write_caption(stage / "c.json"),
            pose=write_poses(stage / "p.npz", n_frames=int(seconds * 30)),
            camera_motion="moving", provenance=PROV)], node_id="n1")[0].clip_id)
    return CommitStore(conn, ing.blobs, ing.clips), ids, tmp_path

def test_commit_is_content_addressed_and_stable(store):
    cs, ids, _ = store
    datasets = {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                        "weight": 1.0, "clips": ids}}
    first = cs.commit(None, datasets, "initial", node_id="n1")
    second = cs.commit(None, dict(datasets), "initial", node_id="n1")
    assert first == second

def test_clip_ingested_by_another_branch_is_selectable(store):
    cs, ids, _ = store
    commit = cs.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                      "weight": 1.0, "clips": [ids[1]]}}, "cross", node_id="n9")
    assert cs.manifest(commit)["datasets"]["cam"]["clips"] == [ids[1]]

def test_ineligible_format_is_rejected(store):
    cs, ids, _ = store
    with pytest.raises(CommitError, match="not eligible"):
        cs.commit(None, {"static": {"format": "video_caption_static", "prompt_mode": None,
                                    "weight": 1.0, "clips": ids}}, "bad", node_id="n1")

def test_prompt_mode_rules_are_enforced(store):
    cs, ids, _ = store
    with pytest.raises(CommitError, match="prompt_mode"):
        cs.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": "segment",
                                 "weight": 1.0, "clips": ids}}, "bad", node_id="n1")

def test_builtin_dataset_name_is_rejected(store):
    cs, ids, _ = store
    with pytest.raises(CommitError, match="built-in"):
        cs.commit(None, {"spatialvid_hq": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": ids}}, "bad", node_id="n1")

def test_materialize_builds_hardlinked_standard_roots(store):
    cs, ids, tmp_path = store
    commit = cs.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                      "weight": 1.0, "clips": ids}}, "v", node_id="n1")
    roots = cs.materialize(commit, tmp_path / "view")
    root = roots["cam"]
    assert sorted(p.name for p in (root / "videos").iterdir()) == sorted(f"{c}.mp4" for c in ids)
    assert (root / "captions" / f"{ids[0]}.json").exists()
    assert (root / "poses" / f"{ids[0]}.npz").exists()
    assert (root / "videos" / f"{ids[0]}.mp4").stat().st_nlink >= 2  # hardlink, not a copy

def test_static_dataset_view_has_no_poses_dir(tmp_path):
    conn = open_db(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, Recorder(tmp_path))
    stage = tmp_path / "staging" / "s"
    clip_id = ing.ingest([Candidate(video=make_mp4(stage / "v.mp4"),
                                    caption=write_caption(stage / "c.json"), pose=None,
                                    camera_motion="static", provenance=PROV)], node_id="n1")[0].clip_id
    cs = CommitStore(conn, ing.blobs, ing.clips)
    commit = cs.commit(None, {"fixed": {"format": "video_caption_static", "prompt_mode": None,
                                        "weight": 1.0, "clips": [clip_id]}}, "v", node_id="n1")
    root = cs.materialize(commit, tmp_path / "view")["fixed"]
    assert not (root / "poses").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_commits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.archive.commits'`

- [ ] **Step 3: Implement the commit store**

```python
# kernel/ar_kernel/archive/commits.py
from __future__ import annotations
import hashlib, json, os, re, sqlite3, time
from pathlib import Path

FORMATS = {"video_caption_camera", "video_timed_prompts_camera", "video_caption_static"}
PROMPT_MODES = {"segment", "per_chunk"}
BUILTIN_NAMES = {"sekai_real_hq", "spatialvid_hq", "sekai_game_walking", "sekai_real_walking",
                 "mugen_v2", "RealEstate10K", "spatialvid", "veo3", "OpenVid", "mp4_frame_game_3",
                 "sekai_real_mini"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

class CommitError(ValueError):
    """The proposed data commit is invalid."""

class CommitStore:
    def __init__(self, conn: sqlite3.Connection, blobs, clips) -> None:
        self.conn = conn
        self.blobs = blobs
        self.clips = clips

    def _validate(self, datasets: dict[str, dict]) -> dict:
        if not datasets:
            raise CommitError("a commit needs at least one dataset")
        normalized: dict[str, dict] = {}
        usable = 0
        for name, entry in datasets.items():
            if not NAME_RE.match(name):
                raise CommitError(f"dataset name {name!r} must match {NAME_RE.pattern}")
            if name in BUILTIN_NAMES:
                raise CommitError(f"dataset name {name!r} is a built-in source name")
            fmt = entry.get("format")
            if fmt not in FORMATS:
                raise CommitError(f"{name}.format must be one of {sorted(FORMATS)}, got {fmt!r}")
            mode = entry.get("prompt_mode")
            if fmt == "video_timed_prompts_camera":
                if mode not in PROMPT_MODES:
                    raise CommitError(f"{name}.prompt_mode must be one of {sorted(PROMPT_MODES)}")
            elif mode is not None:
                raise CommitError(f"{name}.prompt_mode only applies to video_timed_prompts_camera")
            weight = float(entry.get("weight", 1.0))
            if weight < 0:
                raise CommitError(f"{name}.weight must be >= 0")
            clip_ids = list(entry.get("clips") or [])
            key = fmt if mode is None else f"{fmt}:{mode}"
            for clip_id in clip_ids:
                try:
                    clip = self.clips.get(clip_id)
                except KeyError:
                    raise CommitError(f"{name}: unknown clip {clip_id}") from None
                if key not in clip["formats"]:
                    raise CommitError(f"{name}: clip {clip_id[:12]} is not eligible for {key}")
            if weight > 0 and clip_ids:
                usable += 1
            normalized[name] = {"format": fmt, "prompt_mode": mode, "weight": weight,
                                "clips": sorted(clip_ids)}
        if usable == 0:
            raise CommitError("no dataset has both weight > 0 and clips")
        return {"datasets": dict(sorted(normalized.items()))}

    def commit(self, parent: str | None, datasets: dict[str, dict], message: str,
               node_id: str, attempt: int = 0) -> str:
        manifest = self._validate(datasets)
        blob = json.dumps(manifest, sort_keys=True)
        commit_id = hashlib.sha256(
            f"{parent or ''}|{hashlib.sha256(blob.encode()).hexdigest()}|{message}".encode()
        ).hexdigest()
        self.conn.execute(
            """INSERT OR IGNORE INTO data_commits
               (commit_id, parent_commit, node_id, attempt, manifest, message, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (commit_id, parent, node_id, attempt, blob, message, time.time()),
        )
        return commit_id

    def get(self, commit_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM data_commits WHERE commit_id=?", (commit_id,)).fetchone()
        if row is None:
            raise KeyError(commit_id)
        return dict(row)

    def manifest(self, commit_id: str) -> dict:
        return json.loads(self.get(commit_id)["manifest"])

    def materialize(self, commit_id: str, dest: Path) -> dict[str, Path]:
        dest = Path(dest)
        roots: dict[str, Path] = {}
        for name, entry in self.manifest(commit_id)["datasets"].items():
            if entry["weight"] <= 0 or not entry["clips"]:
                continue
            root = dest / name
            (root / "videos").mkdir(parents=True, exist_ok=True)
            (root / "captions").mkdir(parents=True, exist_ok=True)
            needs_poses = entry["format"] != "video_caption_static"
            if needs_poses:
                (root / "poses").mkdir(parents=True, exist_ok=True)
            for clip_id in entry["clips"]:
                clip = self.clips.get(clip_id)
                links = [(self.blobs.path(clip["video_digest"], "video"), root / "videos" / f"{clip_id}.mp4"),
                         (self.blobs.path(clip["caption_digest"], "caption"), root / "captions" / f"{clip_id}.json")]
                if needs_poses and clip["pose_digest"]:
                    links.append((self.blobs.path(clip["pose_digest"], "pose"),
                                  root / "poses" / f"{clip_id}.npz"))
                for source, target in links:
                    if target.exists():
                        target.unlink()
                    os.link(source, target)
            roots[name] = root
        return roots
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_commits.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/archive/commits.py tests/test_commits.py
git commit -m "feat(archive): content-addressed data commits and hardlinked standard views"
```

---

### Task 10: Resolved config and recipe gate

**Files:**
- Create: `kernel/ar_kernel/train/__init__.py`, `kernel/ar_kernel/train/recipe.py`, `kernel/ar_kernel/train/gate.py`
- Test: `tests/test_recipe.py`, `tests/test_gate.py`

**Interfaces:**
- Consumes: `KernelConfig`, `CommitStore`, `resolve_gpus`, `run_in_env`.
- Produces: `TUNABLE_KEYS: frozenset[str]`; `build_resolved_config(cfg, base_recipe: Path, recipe: dict, roots: dict[str, Path], manifest: dict, node_id: str, node_dir: Path, run_dir: Path) -> dict`; `write_resolved_config(...) -> Path`; `Gate(cfg, commits, recorder)` with `check(recipe: dict, commit_id: str, parent_commit: str | None, node_id: str, node_dir: Path, run_dir: Path, gpus: list[int]) -> GateResult(ok: bool, failures: list[str], resolved_path: Path | None, view_roots: dict[str, Path])`; `steps_per_epoch(manifest, n_gpus, grad_accum) -> tuple[int, int]` returning `(epoch_windows, steps_per_epoch)`.

- [ ] **Step 1: Write the failing recipe test**

```python
# tests/test_recipe.py
import pytest, yaml
from ar_kernel.config import KernelConfig
from ar_kernel.train.recipe import build_resolved_config, steps_per_epoch

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"
MANIFEST = {"datasets": {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                 "weight": 1.0, "clips": ["x"] * 40}}}

def test_resolved_config_injects_datasets_and_kernel_paths(tmp_path):
    resolved = build_resolved_config(
        CFG, BASE, {"optimizer.lr": 1e-4}, {"cam": tmp_path / "view" / "cam"}, MANIFEST,
        node_id="n1", node_dir=tmp_path / "n1", run_dir=tmp_path)
    assert resolved["data"]["sources"] == {}
    assert resolved["data"]["datasets"]["cam"]["root"] == str(tmp_path / "view" / "cam")
    assert resolved["data"]["datasets"]["cam"]["format"] == "video_caption_camera"
    assert resolved["optimizer"]["lr"] == pytest.approx(1e-4)
    assert resolved["run"]["name"] == "node_n1"
    assert resolved["runtime"]["text_embed_cache_dir"] == str(tmp_path / "cache" / "text_embed")
    assert resolved["validation"]["enabled"] is False

def test_prompt_mode_is_emitted_only_for_timed_datasets(tmp_path):
    manifest = {"datasets": {"pencil": {"format": "video_timed_prompts_camera",
                                        "prompt_mode": "per_chunk", "weight": 2.0,
                                        "clips": ["a"] * 8}}}
    resolved = build_resolved_config(CFG, BASE, {}, {"pencil": tmp_path / "p"}, manifest,
                                     node_id="n2", node_dir=tmp_path / "n2", run_dir=tmp_path)
    assert resolved["data"]["datasets"]["pencil"]["prompt_mode"] == "per_chunk"

def test_steps_per_epoch_matches_the_standard_epoch_definition():
    manifest = {"datasets": {
        "a": {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": ["x"] * 6},
        "b": {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": ["y"] * 6}}}
    windows, steps = steps_per_epoch(manifest, n_gpus=4, grad_accum=1)
    assert windows == 12 and steps == 3

def test_steps_per_epoch_is_zero_for_a_tiny_dataset():
    manifest = {"datasets": {"a": {"format": "video_caption_camera", "prompt_mode": None,
                                   "weight": 1.0, "clips": ["x"] * 12}}}
    assert steps_per_epoch(manifest, n_gpus=4, grad_accum=4) == (12, 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_recipe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.train'`

- [ ] **Step 3: Implement `recipe.py`**

```python
# kernel/ar_kernel/train/recipe.py
from __future__ import annotations
import copy, math
from pathlib import Path
import yaml

from ..config import KernelConfig

TUNABLE_KEYS = frozenset({
    "data.overall_caption_prob",
    "optimizer.lr", "optimizer.weight_decay", "optimizer.max_grad_norm",
    "optimizer.warmup_steps", "optimizer.max_steps", "optimizer.epochs",
    "optimizer.grad_accum_steps",
    "sample.height", "sample.width",
    "lora.rank", "lora.alpha",
})

def _assign(tree: dict, dotted: str, value) -> None:
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value

def steps_per_epoch(manifest: dict, n_gpus: int, grad_accum: int) -> tuple[int, int]:
    enabled = {n: d for n, d in manifest["datasets"].items() if d["weight"] > 0 and d["clips"]}
    total_weight = sum(d["weight"] for d in enabled.values())
    epoch_windows = max(math.ceil(len(d["clips"]) / (d["weight"] / total_weight))
                        for d in enabled.values())
    per_rank = epoch_windows // n_gpus
    return epoch_windows, per_rank // grad_accum

def build_resolved_config(cfg: KernelConfig, base_recipe: Path, recipe: dict,
                          roots: dict[str, Path], manifest: dict, node_id: str,
                          node_dir: Path, run_dir: Path) -> dict:
    resolved = yaml.safe_load(Path(base_recipe).read_text(encoding="utf-8"))
    resolved = copy.deepcopy(resolved)
    for dotted, value in recipe.items():
        _assign(resolved, dotted, value)

    datasets = {}
    for name, entry in manifest["datasets"].items():
        if entry["weight"] <= 0 or not entry["clips"]:
            continue
        spec = {"root": str(roots[name]), "format": entry["format"], "weight": entry["weight"]}
        if entry["prompt_mode"] is not None:
            spec["prompt_mode"] = entry["prompt_mode"]
        datasets[name] = spec
    resolved["data"]["sources"] = {}
    resolved["data"]["datasets"] = datasets

    train_dir = Path(node_dir) / "train"
    resolved["run"]["name"] = f"node_{node_id}"
    resolved["run"]["output_dir"] = str(train_dir / "outputs")
    resolved["run"]["log_dir"] = str(train_dir / "logs")
    resolved["runtime"]["text_embed_cache_dir"] = str(Path(run_dir) / "cache" / "text_embed")
    resolved["optimizer"]["checkpoint_steps"] = int(resolved["optimizer"]["max_steps"])
    resolved["optimizer"]["max_checkpoints"] = 1
    resolved["validation"]["enabled"] = False
    first = next(iter(datasets))
    for mode in resolved.get("validation", {}).get("modes", {}).values():
        mode.setdefault("dataset", {})["source"] = first
    return resolved

def write_resolved_config(cfg: KernelConfig, base_recipe: Path, recipe: dict,
                          roots: dict[str, Path], manifest: dict, node_id: str,
                          node_dir: Path, run_dir: Path) -> Path:
    resolved = build_resolved_config(cfg, base_recipe, recipe, roots, manifest, node_id,
                                     node_dir, run_dir)
    target = Path(node_dir) / "train_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    return target
```

- [ ] **Step 4: Run the recipe tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_recipe.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing gate test**

```python
# tests/test_gate.py
import pytest
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import Gate
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test fixture"}

@pytest.fixture
def gate_env(tmp_path):
    conn = open_db(tmp_path)
    rec = Recorder(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, rec)
    clip_ids = []
    for i in range(8):
        stage = tmp_path / "staging" / f"c{i}"
        clip_ids.append(ing.ingest([Candidate(
            video=make_mp4(stage / "v.mp4", seconds=5.0), caption=write_caption(stage / "c.json"),
            pose=write_poses(stage / "p.npz", n_frames=150), camera_motion="moving",
            provenance=PROV)], node_id="n1")[0].clip_id)
    commits = CommitStore(conn, ing.blobs, ing.clips)
    commit = commits.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": clip_ids}}, "v", node_id="n1")
    return Gate(CFG, commits, rec), commit, tmp_path

def test_valid_recipe_passes_every_check(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.max_steps": 2, "optimizer.grad_accum_steps": 1,
                         "optimizer.epochs": 50},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert result.ok, result.failures
    assert result.resolved_path.exists()

def test_non_tunable_key_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"spatial_memory.enabled": False}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("not tunable" in f for f in result.failures)

def test_resolution_outside_the_allowlist_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"sample.height": 544, "sample.width": 960}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("resolution" in f for f in result.failures)

def test_lora_pair_outside_the_allowlist_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"lora.rank": 8, "lora.alpha": 8}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("lora" in f for f in result.failures)

def test_unchanged_data_commit_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({}, commit, commit, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("data commit" in f for f in result.failures)

def test_zero_steps_per_epoch_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.grad_accum_steps": 4, "optimizer.max_steps": 10},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("steps_per_epoch" in f for f in result.failures)

def test_epoch_budget_below_max_steps_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.grad_accum_steps": 1, "optimizer.max_steps": 100,
                         "optimizer.epochs": 2},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("epochs" in f for f in result.failures)

def test_fewer_clips_than_gpus_is_rejected(tmp_path):
    conn = open_db(tmp_path)
    rec = Recorder(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, rec)
    stage = tmp_path / "staging" / "solo"
    clip_id = ing.ingest([Candidate(video=make_mp4(stage / "v.mp4"),
                                    caption=write_caption(stage / "c.json"),
                                    pose=write_poses(stage / "p.npz", n_frames=120),
                                    camera_motion="moving", provenance=PROV)],
                         node_id="n1")[0].clip_id
    commits = CommitStore(conn, ing.blobs, ing.clips)
    commit = commits.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": [clip_id]}}, "v", node_id="n1")
    result = Gate(CFG, commits, rec).check({}, commit, None, "n1", tmp_path / "n1",
                                           tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("clips" in f for f in result.failures)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.train.gate'`

- [ ] **Step 7: Implement `gate.py`**

```python
# kernel/ar_kernel/train/gate.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env
from .recipe import TUNABLE_KEYS, steps_per_epoch, write_resolved_config

@dataclass
class GateResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    resolved_path: Path | None = None
    view_roots: dict[str, Path] = field(default_factory=dict)

class Gate:
    def __init__(self, cfg: KernelConfig, commits, recorder) -> None:
        self.cfg = cfg
        self.commits = commits
        self.recorder = recorder
        self.base_recipe = cfg.repo_root / "configs" / "base_recipe.yaml"

    def check(self, recipe: dict, commit_id: str, parent_commit: str | None, node_id: str,
              node_dir: Path, run_dir: Path, gpus: list[int]) -> GateResult:
        failures: list[str] = []
        for key in recipe:
            if key not in TUNABLE_KEYS:
                failures.append(f"{key} is not tunable; allowed: {sorted(TUNABLE_KEYS)}")

        manifest = self.commits.manifest(commit_id)
        base = __import__("yaml").safe_load(self.base_recipe.read_text(encoding="utf-8"))
        height = recipe.get("sample.height", base["sample"]["height"])
        width = recipe.get("sample.width", base["sample"]["width"])
        if [height, width] not in [list(p) for p in self.cfg.get("train.resolution_allowlist")]:
            failures.append(f"resolution {height}x{width} is not in train.resolution_allowlist")
        rank = recipe.get("lora.rank", base["lora"]["rank"])
        alpha = recipe.get("lora.alpha", base["lora"]["alpha"])
        if [rank, alpha] not in [list(p) for p in self.cfg.get("train.lora_allowlist")]:
            failures.append(f"lora rank/alpha {rank}/{alpha} is not in train.lora_allowlist")

        if parent_commit is not None and commit_id != parent_commit:
            # Compare manifests, not ids: a commit id hashes the message too, so an
            # identical dataset recommitted under a new message would slip through.
            if self.commits.manifest(commit_id) == self.commits.manifest(parent_commit):
                failures.append(
                    "the data commit's manifest is identical to the parent's; every node must change data")
        elif parent_commit is not None:
            failures.append("the data commit is identical to the parent's; every node must change data")

        n_gpus = len(gpus)
        for name, entry in manifest["datasets"].items():
            if entry["weight"] > 0 and 0 < len(entry["clips"]) < n_gpus:
                failures.append(f"dataset {name} has {len(entry['clips'])} clips, fewer than {n_gpus} GPUs")

        grad_accum = int(recipe.get("optimizer.grad_accum_steps", base["optimizer"]["grad_accum_steps"]))
        max_steps = int(recipe.get("optimizer.max_steps", base["optimizer"]["max_steps"]))
        epochs = int(recipe.get("optimizer.epochs", base["optimizer"]["epochs"]))
        windows, per_epoch = steps_per_epoch(manifest, n_gpus, grad_accum)
        if per_epoch < 1:
            failures.append(
                f"steps_per_epoch is 0 (epoch_windows={windows}, n_gpus={n_gpus}, "
                f"grad_accum_steps={grad_accum}); need epoch_windows >= n_gpus * grad_accum_steps")
        elif epochs * per_epoch < max_steps:
            failures.append(
                f"optimizer.epochs={epochs} x steps_per_epoch={per_epoch} < max_steps={max_steps} "
                f"(epoch_windows={windows}); training would stop early")

        if failures:
            self.recorder.event("gate.failed", node=node_id, phase="gate",
                                payload={"failures": failures, "recipe": recipe})
            return GateResult(ok=False, failures=failures)

        roots = self.commits.materialize(commit_id, Path(node_dir) / "view")
        resolved = write_resolved_config(self.cfg, self.base_recipe, recipe, roots, manifest,
                                         node_id, Path(node_dir), Path(run_dir))
        gpu_list = ",".join(str(g) for g in gpus)
        checks = [
            ("check_dataset", ["python", "scripts/tools/check_dataset.py", "--config",
                               str(resolved), "--max-messages", "0"], {}),
            ("precache_dry_run", ["python", "scripts/tools/precache_train_text_embeds.py",
                                  "--config", str(resolved), "--dry-run"], {}),
        ]
        for label, args, extra in checks:
            proc = run_in_env("alayaworld", args, cwd=self.cfg.worldmodel,
                              extra_env={"CUDA_VISIBLE_DEVICES": gpu_list, **extra},
                              timeout=3600, recorder=self.recorder, node=node_id, phase="gate")
            if proc.returncode != 0:
                failures.append(f"{label} failed (rc={proc.returncode}): {proc.stdout[-2000:]}")

        describe = run_in_env(
            "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
            cwd=self.cfg.worldmodel,
            extra_env={"CONFIG_PATH": str(resolved), "CUDA_VISIBLE_DEVICES": gpu_list,
                       "DESCRIBE": "1"},
            timeout=3600, recorder=self.recorder, node=node_id, phase="gate")
        if describe.returncode != 0:
            failures.append(f"describe failed (rc={describe.returncode}): {describe.stdout[-2000:]}")

        ok = not failures
        self.recorder.event("gate.passed" if ok else "gate.failed", node=node_id, phase="gate",
                            payload={"failures": failures, "recipe": recipe,
                                     "resolved": str(resolved)})
        return GateResult(ok=ok, failures=failures, resolved_path=resolved if ok else None,
                          view_roots=roots)
```

- [ ] **Step 8: Run the gate tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_gate.py -v`
Expected: PASS (8 tests)

- [ ] **Step 9: Commit**

```bash
git add kernel/ar_kernel/train tests/test_recipe.py tests/test_gate.py
git commit -m "feat(train): resolved config builder and recipe gate with step-budget checks"
```

---

### Task 11: Training runner and failure classification

**Files:**
- Create: `kernel/ar_kernel/train/runner.py`
- Test: `tests/test_train_runner.py`

**Interfaces:**
- Consumes: `KernelConfig`, `run_in_env`, `Recorder`.
- Produces: `classify_failure(log: str, returncode: int) -> str` returning `"recipe"`, `"infra"` or `"none"`; `parse_train_lines(log: str) -> list[dict]` with keys `step, epoch, loss, grad, lr, time`; `TrainRunner(cfg, recorder)` with `precache(resolved: Path, gpus, node_id) -> None` and `train(resolved: Path, gpus: list[int], node_id: str, node_dir: Path) -> TrainOutcome(checkpoint: Path | None, failure: str, log_path: Path, metrics: list[dict])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_runner.py
from pathlib import Path
from ar_kernel.train.runner import classify_failure, parse_train_lines, newest_checkpoint

CUDA_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
NCCL = "RuntimeError: NCCL communicator was aborted on rank 2"
HOST_OOM = "torchrun ... failed (exitcode: -9)"

def test_cuda_oom_is_a_recipe_failure():
    assert classify_failure(CUDA_OOM, 1) == "recipe"

def test_nan_loss_is_a_recipe_failure():
    assert classify_failure("[Train] step=3 loss=nan grad=inf lr=5.00e-05 time=8s", 1) == "recipe"

def test_host_oom_and_nccl_are_infra_failures():
    assert classify_failure(HOST_OOM, 1) == "infra"
    assert classify_failure(NCCL, 1) == "infra"

def test_clean_run_has_no_failure():
    assert classify_failure("[Train] step=2 loss=0.26 grad=0.05 lr=5.00e-05 time=7.9s", 0) == "none"

def test_parse_train_lines_extracts_metrics():
    log = ("[Train] step=1 epoch=0 source=cam video=abc fs=0 fe=120 K=4 sigma=0.99 "
           "loss=0.261719 grad=0.0579 lr=5.00e-05 time=7.92s\n"
           "[Train] step=2 epoch=0 source=cam video=def fs=8 fe=128 K=4 sigma=0.98 "
           "loss=0.251000 grad=0.0611 lr=5.00e-05 time=7.81s\n")
    rows = parse_train_lines(log)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["loss"] == 0.251
    assert rows[0]["lr"] == 5e-05

def test_newest_checkpoint_picks_the_highest_step(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "lora.safetensors").touch()
    assert newest_checkpoint(tmp_path).name == "checkpoint-300"

def test_newest_checkpoint_ignores_dirs_without_lora(tmp_path):
    (tmp_path / "checkpoint-400").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-100" / "lora.safetensors").touch()
    assert newest_checkpoint(tmp_path).name == "checkpoint-100"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_train_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.train.runner'`

- [ ] **Step 3: Implement `runner.py`**

```python
# kernel/ar_kernel/train/runner.py
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env

RECIPE_SIGNATURES = (
    "CUDA out of memory", "torch.OutOfMemoryError", "loss=nan", "loss=inf",
    "no enabled data sources", "Loaded 0 samples",
)
INFRA_SIGNATURES = (
    "exitcode: -9", "NCCL", "No space left on device", "CUDA driver error",
    "cache miss", "Killed",
)
TRAIN_LINE = re.compile(
    r"\[Train\] step=(?P<step>\d+) epoch=(?P<epoch>\d+).*?"
    r"loss=(?P<loss>[\d.naif]+) grad=(?P<grad>[\d.naif]+) lr=(?P<lr>[\d.e+-]+) time=(?P<time>[\d.]+)s"
)

@dataclass
class TrainOutcome:
    checkpoint: Path | None
    failure: str
    log_path: Path
    metrics: list[dict] = field(default_factory=list)

def classify_failure(log: str, returncode: int) -> str:
    if any(sig in log for sig in RECIPE_SIGNATURES):
        return "recipe"
    if any(sig in log for sig in INFRA_SIGNATURES):
        return "infra"
    return "none" if returncode == 0 else "infra"

def parse_train_lines(log: str) -> list[dict]:
    rows = []
    for match in TRAIN_LINE.finditer(log):
        rows.append({"step": int(match["step"]), "epoch": int(match["epoch"]),
                     "loss": float(match["loss"]), "grad": float(match["grad"]),
                     "lr": float(match["lr"]), "time": float(match["time"])})
    return rows

def newest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in Path(output_dir).glob("checkpoint-*")
                  if (p / "lora.safetensors").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))

class TrainRunner:
    def __init__(self, cfg: KernelConfig, recorder) -> None:
        self.cfg = cfg
        self.recorder = recorder

    def precache(self, resolved: Path, gpus: list[int], node_id: str) -> None:
        two = ",".join(str(g) for g in gpus[:2])
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/precache_train_text_embeds.py", "--config", str(resolved),
             "--device-map", "auto"],
            cwd=self.cfg.worldmodel,
            extra_env={"CUDA_VISIBLE_DEVICES": two, "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"},
            timeout=7200, recorder=self.recorder, node=node_id, phase="precache")
        if proc.returncode != 0:
            raise RuntimeError(f"prompt precache failed (rc={proc.returncode}): {proc.stdout[-4000:]}")

    def train(self, resolved: Path, gpus: list[int], node_id: str, node_dir: Path) -> TrainOutcome:
        import yaml
        config = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        output_dir = Path(config["run"]["output_dir"])
        log_path = Path(node_dir) / "train" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.recorder.span("train", node=node_id, phase="train",
                                payload={"config": str(resolved), "gpus": gpus}):
            proc = run_in_env(
                "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
                cwd=self.cfg.worldmodel,
                extra_env={"CONFIG_PATH": str(resolved),
                           "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                           "LOG_FILTER": "all", "ALAYA_LOG_MEMORY": "1",
                           "ALAYA_DATASET_CACHE_DIR": str(Path(node_dir) / "dataset_cache")},
                timeout=int(48 * 3600), recorder=self.recorder, node=node_id, phase="train")
        log = proc.stdout + proc.stderr
        log_path.write_text(log, encoding="utf-8")
        failure = classify_failure(log, proc.returncode)
        checkpoint = newest_checkpoint(output_dir)
        if failure == "none" and checkpoint is None:
            failure = "recipe"
        return TrainOutcome(checkpoint=checkpoint, failure=failure, log_path=log_path,
                            metrics=parse_train_lines(log))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_train_runner.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/train/runner.py tests/test_train_runner.py
git commit -m "feat(train): training runner with log parsing and failure classification"
```

---

### Task 12: Eval — merge, render, WBench phases, score

**Files:**
- Create: `kernel/ar_kernel/eval/__init__.py`, `kernel/ar_kernel/eval/merge.py`, `kernel/ar_kernel/eval/render.py`, `kernel/ar_kernel/eval/wbench.py`, `kernel/ar_kernel/eval/score.py`
- Test: `tests/test_score.py`, `tests/test_merge_guard.py`

**Interfaces:**
- Consumes: `KernelConfig`, `run_in_env`, `Recorder`.
- Produces:
  - `available_ram_gb() -> float`; `wait_for_ram(cfg, recorder, node_id) -> None`; `merge_lora(cfg, checkpoint: Path, rank: int, alpha: int, run_dir: Path, recorder, node_id) -> Path`.
  - `build_render_config(cfg, merged: Path | None, history_encoder: Path, videos_dir: Path, case_ids: list[str], node_dir: Path) -> Path`; `render_proxy(cfg, render_config: Path, gpus: list[int], node_id: str, recorder, case_ids: list[str]) -> Path` (returns the videos dir).
  - `run_wbench_phases(cfg, work_dir: Path, model: str, gpus, metric_set: list[str], recorder, node_id) -> dict` (the parsed `report.json`).
  - `DIMENSION_METRICS: list[str]`; `resolve_metric_set(cfg, env) -> list[str]`; `score_from_report(report: dict, metric_set: list[str]) -> tuple[float, dict]`; `aggregates(cfg, eval_dir: Path, case_ids: list[str]) -> dict`; `cleanup_eval(work_dir: Path, model: str) -> list[str]`.

- [ ] **Step 1: Write the failing score test**

```python
# tests/test_score.py
import json, pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.score import DIMENSION_METRICS, resolve_metric_set, score_from_report, cleanup_eval

CFG = KernelConfig.load()
REPORT = json.loads((CFG.repo_root / "reference" / "wbench_alayaworld_proxy" / "report.json").read_text())

def test_dimension_metrics_are_the_22_wbench_metrics():
    assert len(DIMENSION_METRICS) == 22
    assert "navigation_trajectory" in DIMENSION_METRICS
    assert "navigation_accuracy" not in DIMENSION_METRICS

def test_metric_set_excludes_vlm_metrics_without_a_key():
    metrics = resolve_metric_set(CFG, {"VLM_API_KEY": ""})
    assert "scene_adherence" not in metrics
    assert "aesthetic_quality" in metrics

def test_metric_set_includes_vlm_metrics_with_a_key():
    metrics = resolve_metric_set(CFG, {"VLM_API_KEY": "abc"})
    assert "causal_fidelity" in metrics

def test_score_is_the_mean_over_the_metric_set():
    metric_set = [m for m in DIMENSION_METRICS if m in REPORT["full"]]
    score, per_metric = score_from_report(REPORT, metric_set)
    expected = sum(REPORT["full"][m]["mean"] for m in metric_set) / len(metric_set)
    assert score == pytest.approx(expected)
    assert per_metric["aesthetic_quality"] == pytest.approx(REPORT["full"]["aesthetic_quality"]["mean"])

def test_missing_metric_raises(tmp_path):
    with pytest.raises(KeyError, match="visual_plausibility"):
        score_from_report(REPORT, ["aesthetic_quality", "visual_plausibility"])

def test_cleanup_removes_regenerable_dirs_only(tmp_path):
    model_dir = tmp_path / "work_dirs" / "m"
    for name in ("videos", "evaluation", "da3_cache", "megasam", "masks", "_navi_videos_tmp"):
        (model_dir / name).mkdir(parents=True)
    removed = cleanup_eval(tmp_path / "work_dirs", "m")
    assert sorted(removed) == ["_navi_videos_tmp", "da3_cache", "masks", "megasam"]
    assert (model_dir / "videos").exists() and (model_dir / "evaluation").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.eval'`

- [ ] **Step 3: Implement `score.py`**

```python
# kernel/ar_kernel/eval/score.py
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Mapping

from ..config import KernelConfig

DIMENSION_METRICS = [
    "aesthetic_quality", "imaging_quality", "temporal_flickering", "dynamic_degree",
    "motion_smoothness", "hpsv3_quality",
    "background_consistency", "segment_continuity", "perspective_consistency",
    "subject_consistency", "geometric_consistency", "photometric_consistency",
    "spatial_consistency", "gated_spatial_consistency",
    "navigation_trajectory", "event_edit_adherence", "subject_action_adherence",
    "perspective_switch_adherence",
    "scene_adherence", "subject_adherence",
    "visual_plausibility", "causal_fidelity",
]
VLM_METRICS = {"scene_adherence", "subject_adherence", "causal_fidelity",
               "event_edit_adherence", "subject_action_adherence", "perspective_switch_adherence"}
REGENERABLE = ("da3_cache", "megasam", "masks", "_navi_videos_tmp")

def resolve_metric_set(cfg: KernelConfig, env: Mapping[str, str]) -> list[str]:
    metrics = list(DIMENSION_METRICS)
    if not env.get("VLM_API_KEY", "").strip():
        metrics = [m for m in metrics if m not in VLM_METRICS]
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        metrics = [m for m in metrics if m != "visual_plausibility"]
    return metrics

def score_from_report(report: dict, metric_set: list[str]) -> tuple[float, dict]:
    full = report["full"]
    missing = [m for m in metric_set if m not in full]
    if missing:
        raise KeyError(f"metrics missing from the report: {missing}")
    per_metric = {m: float(full[m]["mean"]) for m in metric_set}
    return sum(per_metric.values()) / len(per_metric), per_metric

def aggregates(cfg: KernelConfig, eval_dir: Path, case_ids: list[str]) -> dict:
    """Per-metric means plus means by coarse stratum; no case ids leave this function."""
    cases = {}
    for case_id in case_ids:
        path = cfg.wbench / "data" / "cases" / f"case_{case_id}.json"
        case = json.loads(path.read_text())
        settings = case.get("settings") or {}
        cases[case_id] = {
            "types": sorted({i.get("type") for i in case.get("interactions") or []}),
            "category": ((settings.get("scene") or {}).get("category")),
            "perspective": settings.get("perspective"),
        }
    per_case: dict[str, dict] = {}
    for metric_dir in Path(eval_dir).iterdir():
        if not metric_dir.is_dir():
            continue
        for result in metric_dir.glob("case_*.json"):
            case_id = result.stem.replace("case_", "")
            data = json.loads(result.read_text())
            if isinstance(data.get("score"), (int, float)):
                per_case.setdefault(case_id, {})[metric_dir.name] = float(data["score"])
    strata: dict[str, dict[str, list[float]]] = {"interaction_type": {}, "category": {}, "perspective": {}}
    for case_id, metrics in per_case.items():
        mean = sum(metrics.values()) / len(metrics) if metrics else None
        if mean is None or case_id not in cases:
            continue
        facets = cases[case_id]
        for t in facets["types"]:
            strata["interaction_type"].setdefault(str(t), []).append(mean)
        strata["category"].setdefault(str(facets["category"]), []).append(mean)
        strata["perspective"].setdefault(str(facets["perspective"]), []).append(mean)
    return {axis: {key: sum(v) / len(v) for key, v in buckets.items()}
            for axis, buckets in strata.items()}

def cleanup_eval(work_dir: Path, model: str) -> list[str]:
    removed = []
    for name in REGENERABLE:
        target = Path(work_dir) / model / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return sorted(removed)
```

- [ ] **Step 4: Run the score tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_score.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing merge-guard test**

```python
# tests/test_merge_guard.py
import pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.merge import available_ram_gb, wait_for_ram
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()

def test_available_ram_is_plausible():
    assert 1.0 < available_ram_gb() < 4096.0

def test_wait_for_ram_returns_when_threshold_is_met(tmp_path):
    wait_for_ram(CFG, Recorder(tmp_path), "n1", threshold_gb=0.5, poll_s=0.01, alert_after_s=1)

def test_wait_for_ram_alerts_when_never_satisfied(tmp_path):
    rec = Recorder(tmp_path)
    with pytest.raises(TimeoutError):
        wait_for_ram(CFG, rec, "n1", threshold_gb=1e9, poll_s=0.01, alert_after_s=0.05)
    assert any(e["type"] == "eval.ram_wait_alert" for e in rec.read_events("n1"))
```

- [ ] **Step 6: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_merge_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.eval.merge'`

- [ ] **Step 7: Implement `merge.py`**

```python
# kernel/ar_kernel/eval/merge.py
from __future__ import annotations
import shutil, subprocess, time
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env

def available_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")

def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def wait_for_ram(cfg: KernelConfig, recorder, node_id: str, threshold_gb: float | None = None,
                 poll_s: float = 5.0, alert_after_s: float | None = None) -> None:
    threshold = threshold_gb if threshold_gb is not None else float(cfg.get("eval.free_ram_before_render_gb"))
    deadline = alert_after_s if alert_after_s is not None else float(cfg.get("eval.ram_wait_alert_min")) * 60
    subprocess.run(["sync"], check=True)
    started = time.monotonic()
    while available_ram_gb() < threshold:
        if time.monotonic() - started > deadline:
            recorder.event("eval.ram_wait_alert", node=node_id, phase="eval",
                           payload={"available_gb": available_ram_gb(), "threshold_gb": threshold})
            raise TimeoutError(f"host RAM stayed below {threshold} GB for {deadline}s")
        time.sleep(poll_s)

def merge_lora(cfg: KernelConfig, checkpoint: Path, rank: int, alpha: int, run_dir: Path,
               recorder, node_id: str) -> Path:
    slot = Path(run_dir) / "merge_slot"
    if slot.exists():
        shutil.rmtree(slot)
    minimum = float(cfg.get("disk.merge_min_free_gb"))
    if free_disk_gb(Path(run_dir)) < minimum:
        raise RuntimeError(f"less than {minimum} GB free; refusing to merge")
    with recorder.span("merge", node=node_id, phase="eval", payload={"checkpoint": str(checkpoint)}):
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/merge_lora_for_rollout.py",
             "--ckpt_dir", str(checkpoint),
             "--base_transformer", str(cfg.worldmodel / "weights/alaya-world-ar/transformer.pt"),
             "--output", str(slot), "--lora_rank", str(rank), "--lora_alpha", str(alpha)],
            cwd=cfg.worldmodel, timeout=7200, recorder=recorder, node=node_id, phase="eval")
    if proc.returncode != 0:
        raise RuntimeError(f"merge failed (rc={proc.returncode}): {proc.stdout[-4000:]}")
    wait_for_ram(cfg, recorder, node_id)
    return slot
```

- [ ] **Step 8: Run the merge-guard tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_merge_guard.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Implement `render.py` and `wbench.py`**

```python
# kernel/ar_kernel/eval/render.py
from __future__ import annotations
import copy
from pathlib import Path
import yaml

from ..config import KernelConfig
from ..subproc import run_in_env

def build_render_config(cfg: KernelConfig, merged: Path | None, history_encoder: Path,
                        videos_dir: Path, case_ids: list[str], node_dir: Path) -> Path:
    source = yaml.safe_load((cfg.worldmodel / "configs" / "wbench_full.yaml").read_text())
    config = copy.deepcopy(source)
    config["paths"]["resume_checkpoint"] = str(merged or (cfg.worldmodel / "weights/alaya-world-ar"))
    config["paths"]["history_encoder"] = str(history_encoder)
    config["paths"]["dmd_resume"] = str(cfg.worldmodel / "weights/alaya-world-dmd")
    config["run"]["output_dir"] = str(Path(node_dir) / "eval" / "rollout")
    config["run"]["log_dir"] = str(Path(node_dir) / "eval" / "logs")
    config["validation"]["per_sample_seed"] = True
    mode = config["validation"]["modes"]["wbench"]
    mode["dataset"]["root"] = str(cfg.wbench / "data")
    mode["dataset"]["case_ids"] = list(case_ids)
    mode["wbench_output_dir"] = str(videos_dir)
    target = Path(node_dir) / "eval" / "render_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return target

def render_proxy(cfg: KernelConfig, render_config: Path, gpus: list[int], node_id: str,
                 recorder, case_ids: list[str]) -> Path:
    videos_dir = Path(yaml.safe_load(render_config.read_text())
                      ["validation"]["modes"]["wbench"]["wbench_output_dir"])
    videos_dir.mkdir(parents=True, exist_ok=True)
    with recorder.span("render", node=node_id, phase="render", payload={"cases": case_ids}):
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/run_wbench.py", "--config", str(render_config),
             "--gpus", ",".join(str(g) for g in gpus), "--cases", ",".join(case_ids)],
            cwd=cfg.worldmodel, timeout=int(12 * 3600), recorder=recorder, node=node_id,
            phase="render")
    if proc.returncode != 0:
        raise RuntimeError(f"render failed (rc={proc.returncode}): {proc.stdout[-4000:]}")
    rendered = sorted(videos_dir.glob("case_*_combined.mp4"))
    if len(rendered) != len(case_ids):
        raise RuntimeError(f"rendered {len(rendered)} of {len(case_ids)} proxy cases")
    return videos_dir
```

```python
# kernel/ar_kernel/eval/wbench.py
from __future__ import annotations
import json
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env
from .score import VLM_METRICS

def run_wbench_phases(cfg: KernelConfig, work_dir: Path, model: str, gpus: list[int],
                      metric_set: list[str], recorder, node_id: str) -> dict:
    gpu_arg = ",".join(str(g) for g in gpus)
    phases = ["precompute", "gpu"]
    if any(m in VLM_METRICS for m in metric_set):
        phases.append("vlm")
    phases.append("report")
    for phase in phases:
        with recorder.span(f"wbench.{phase}", node=node_id, phase="eval"):
            proc = run_in_env(
                "wbench-main",
                ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                 "--phase", phase, "--gpus", gpu_arg],
                cwd=cfg.wbench, timeout=int(12 * 3600), recorder=recorder, node=node_id,
                phase="eval")
        if proc.returncode != 0:
            raise RuntimeError(f"wbench {phase} failed (rc={proc.returncode}): {proc.stdout[-4000:]}")
    if "visual_plausibility" in metric_set:
        with recorder.span("wbench.visual_plausibility", node=node_id, phase="eval"):
            proc = run_in_env(
                "wbench-vp",
                ["python", "tools/run_visual_plausibility.py", "--model", model,
                 "--work_dir", str(work_dir),
                 "--model_path", str(cfg.wbench / cfg.get("eval.vp_weights"))],
                cwd=cfg.wbench, extra_env={"CUDA_VISIBLE_DEVICES": gpu_arg},
                timeout=int(12 * 3600), recorder=recorder, node=node_id, phase="eval")
        if proc.returncode != 0:
            raise RuntimeError(f"visual_plausibility failed (rc={proc.returncode}): {proc.stdout[-4000:]}")
        with recorder.span("wbench.report2", node=node_id, phase="eval"):
            run_in_env("wbench-main",
                       ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                        "--phase", "report", "--gpus", gpu_arg],
                       cwd=cfg.wbench, timeout=3600, recorder=recorder, node=node_id, phase="eval")
    report_path = Path(work_dir) / model / "evaluation" / "report.json"
    return json.loads(report_path.read_text())
```

- [ ] **Step 10: Run the full unit suite**

Run: `conda run -n autoresearcher python -m pytest tests -v`
Expected: PASS (all tests from Tasks 1-12)

- [ ] **Step 11: Commit**

```bash
git add kernel/ar_kernel/eval tests/test_score.py tests/test_merge_guard.py
git commit -m "feat(eval): merge with RAM guard, proxy render, WBench phases, scoring and cleanup"
```

---

### Task 13: Run bootstrap, metric preflight and the `ar` CLI

**Files:**
- Create: `kernel/ar_kernel/run.py`, `kernel/ar_kernel/cli.py`
- Test: `tests/test_run_bootstrap.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `preflight_metrics(cfg, env) -> tuple[list[str], list[str]]` (metric set, exclusions with reasons); `bootstrap_run(cfg, run_id: str | None, env) -> RunContext(run_dir, conn, recorder, gpus, metric_set, case_ids, versions)`; `score_node(ctx, node_id, checkpoint: Path | None, rank: int, alpha: int) -> tuple[float, dict]` running merge → render → WBench → score → cleanup; `main(argv=None) -> int` implementing `ar init-run`, `ar status`, `ar score-node`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_bootstrap.py
import json
from ar_kernel.config import KernelConfig
from ar_kernel.run import bootstrap_run, preflight_metrics

CFG = KernelConfig.load()

def test_preflight_reports_exclusions_with_reasons():
    metrics, excluded = preflight_metrics(CFG, {"VLM_API_KEY": ""})
    assert "scene_adherence" not in metrics
    assert any("VLM_API_KEY" in reason for reason in excluded)

def test_bootstrap_creates_run_layout_and_records_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))
    ctx = bootstrap_run(CFG, run_id="testrun", env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert (ctx.run_dir / "archive.db").exists()
    assert (ctx.run_dir / "config" / "kernel.yaml").exists()
    versions = json.loads((ctx.run_dir / "config" / "versions.json").read_text())
    assert "worldmodel_sha" in versions and "wbench_sha" in versions
    assert versions["worldmodel_dirty"] in (True, False)
    assert ctx.gpus == [0, 1, 2, 3]
    assert len(ctx.case_ids) == 40
    assert ctx.metric_set

def test_bootstrap_refuses_too_few_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))
    try:
        bootstrap_run(CFG, run_id="bad", env={"CUDA_VISIBLE_DEVICES": "0,1"})
    except Exception as exc:
        assert "at least 4" in str(exc)
    else:
        raise AssertionError("expected a GpuPolicyError")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n autoresearcher python -m pytest tests/test_run_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.run'`

- [ ] **Step 3: Implement `run.py`**

```python
# kernel/ar_kernel/run.py
from __future__ import annotations
import datetime as dt
import json, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .archive.db import open_db
from .config import KernelConfig, resolve_gpus
from .eval.merge import merge_lora
from .eval.render import build_render_config, render_proxy
from .eval.score import (aggregates, cleanup_eval, resolve_metric_set, score_from_report)
from .eval.wbench import run_wbench_phases
from .telemetry.recorder import Recorder

@dataclass
class RunContext:
    run_dir: Path
    conn: object
    recorder: Recorder
    gpus: list[int]
    metric_set: list[str]
    case_ids: list[str]
    versions: dict

def preflight_metrics(cfg: KernelConfig, env: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """The run's metric set plus one human-readable reason per exclusion."""
    metrics = resolve_metric_set(cfg, env)   # single source of truth for the rule
    excluded = []
    if not env.get("VLM_API_KEY", "").strip():
        excluded.append("VLM metrics excluded: VLM_API_KEY is empty")
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        excluded.append(f"visual_plausibility excluded: weights missing at {vp_weights}")
    return metrics, excluded

def _git_state(repo: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty

def bootstrap_run(cfg: KernelConfig, run_id: str | None, env: Mapping[str, str]) -> RunContext:
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_dir / run_id
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache" / "text_embed").mkdir(parents=True, exist_ok=True)
    gpus = resolve_gpus(cfg, env)
    recorder = Recorder(run_dir, redact=[v for k, v in env.items()
                                         if k in {"OPENAI_API_KEY", "VLM_API_KEY", "HF_TOKEN"} and v])
    shutil.copy2(cfg.repo_root / "configs" / "kernel.yaml", run_dir / "config" / "kernel.yaml")
    shutil.copy2(cfg.repo_root / "configs" / "base_recipe.yaml", run_dir / "config" / "base_recipe.yaml")
    wm_sha, wm_dirty = _git_state(cfg.worldmodel)
    wb_sha, wb_dirty = _git_state(cfg.wbench)
    kernel_sha, kernel_dirty = _git_state(cfg.repo_root)
    versions = {"worldmodel_sha": wm_sha, "worldmodel_dirty": wm_dirty,
                "wbench_sha": wb_sha, "wbench_dirty": wb_dirty,
                "kernel_sha": kernel_sha, "kernel_dirty": kernel_dirty}
    (run_dir / "config" / "versions.json").write_text(json.dumps(versions, indent=2))
    metric_set, excluded = preflight_metrics(cfg, env)
    case_ids = (cfg.repo_root / "configs" / "proxy_cases.txt").read_text().strip().split(",")
    recorder.event("run.start", payload={"gpus": gpus, "metric_set": metric_set,
                                         "excluded_metrics": excluded, "versions": versions,
                                         "case_ids": case_ids})
    if wm_dirty or wb_dirty:
        recorder.event("run.warning", payload={"message": "sibling repo has uncommitted changes",
                                               "worldmodel_dirty": wm_dirty, "wbench_dirty": wb_dirty})
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=recorder, gpus=gpus,
                      metric_set=metric_set, case_ids=case_ids, versions=versions)

def score_node(cfg: KernelConfig, ctx: RunContext, node_id: str, checkpoint: Path | None,
               rank: int, alpha: int) -> tuple[float, dict]:
    node_dir = ctx.run_dir / "nodes" / node_id
    work_dir = node_dir / "eval" / "work_dirs"
    model = f"ar_{ctx.run_dir.name}_n{node_id}"
    videos_dir = work_dir / model / "videos"
    if checkpoint is None:
        merged, history = None, cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
    else:
        merged = merge_lora(cfg, checkpoint, rank, alpha, ctx.run_dir, ctx.recorder, node_id)
        history = Path(checkpoint) / "history_encoder.pt"
    render_config = build_render_config(cfg, merged, history, videos_dir, ctx.case_ids, node_dir)
    render_proxy(cfg, render_config, ctx.gpus, node_id, ctx.recorder, ctx.case_ids)
    report = run_wbench_phases(cfg, work_dir, model, ctx.gpus, ctx.metric_set, ctx.recorder, node_id)
    score, per_metric = score_from_report(report, ctx.metric_set)
    strata = aggregates(cfg, work_dir / model / "evaluation", ctx.case_ids)
    ctx.recorder.event("eval.scored", node=node_id, phase="eval",
                       payload={"score": score, "metrics": per_metric, "strata": strata})
    removed = cleanup_eval(work_dir, model)
    if merged is not None and merged.exists():
        shutil.rmtree(merged)
        removed.append("merge_slot")
    ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})
    return score, {"metrics": per_metric, "strata": strata, "report": report}
```

- [ ] **Step 4: Implement `cli.py`**

```python
# kernel/ar_kernel/cli.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

from .archive.nodes import NodeStore
from .config import KernelConfig
from .run import bootstrap_run, score_node

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ar")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-run", help="create a run directory and record versions")
    init.add_argument("--run-id", default=None)
    status = sub.add_parser("status", help="show nodes of a run")
    status.add_argument("--run-id", required=True)
    score = sub.add_parser("score-node", help="render and score one node on the proxy")
    score.add_argument("--run-id", required=True)
    score.add_argument("--node", required=True)
    score.add_argument("--checkpoint", default=None, help="node checkpoint dir; omit for the base model")
    score.add_argument("--lora-rank", type=int, default=64)
    score.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args(argv)

    cfg = KernelConfig.load()
    if args.command == "init-run":
        ctx = bootstrap_run(cfg, args.run_id, os.environ)
        print(ctx.run_dir)
        return 0

    ctx = bootstrap_run(cfg, args.run_id, os.environ)
    nodes = NodeStore(ctx.conn)
    if args.command == "status":
        for node in nodes.all():
            print(f"{node['node_id']:<12} {node['status']:<14} score={node['score']}")
        return 0

    if args.command == "score-node":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        try:
            nodes.get(args.node)
        except KeyError:
            nodes.create(args.node, None, 0)
        score, detail = score_node(cfg, ctx, args.node, checkpoint, args.lora_rank, args.lora_alpha)
        nodes.record_score(args.node, score, ctx.metric_set, detail["metrics"])
        print(json.dumps({"node": args.node, "score": score}, indent=2))
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the bootstrap tests to verify they pass**

Run: `conda run -n autoresearcher python -m pytest tests/test_run_bootstrap.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the whole suite**

Run: `conda run -n autoresearcher python -m pytest tests -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add kernel/ar_kernel/run.py kernel/ar_kernel/cli.py tests/test_run_bootstrap.py
git commit -m "feat(kernel): run bootstrap, metric preflight and the ar CLI"
```

---

### Task 14: Real-hardware verification (GPU, manual)

**Files:**
- Create: `tests/manual/test_real_pipeline.py`, `docs/superpowers/plans/verification-log.md`
- Modify: `configs/kernel.yaml` (prune allowlists that fail preflight)

**Interfaces:**
- Consumes: the whole kernel.
- Produces: a filled-in verification log and, if needed, corrected allowlists in `configs/kernel.yaml`.

These tests use the real GPUs and take hours. Mark them `@pytest.mark.manual` and run them explicitly.

- [ ] **Step 1: Write the manual test module**

```python
# tests/manual/test_real_pipeline.py
"""GPU verification. Run explicitly:
   conda run -n autoresearcher python -m pytest tests/manual -v -s -m manual
"""
import os, pytest
from pathlib import Path
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.run import bootstrap_run, score_node
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import Gate
from ar_kernel.train.runner import TrainRunner

pytestmark = pytest.mark.manual
CFG = KernelConfig.load()
EXAMPLES = CFG.worldmodel / "data" / "examples"

def _ingest_examples(ing, dataset: str, camera_motion: str, node_id="m1"):
    root = EXAMPLES / dataset
    clip_ids = []
    for video in sorted((root / "videos").glob("*.mp4")):
        stem = video.stem
        pose = root / "poses" / f"{stem}.npz"
        result = ing.ingest([Candidate(
            video=video, caption=root / "captions" / f"{stem}.json",
            pose=pose if camera_motion == "moving" and pose.exists() else None,
            camera_motion=camera_motion,
            provenance={"kind": "derived", "from": [], "transform": f"WorldModel example {dataset}"},
        )], node_id=node_id)[0]
        assert result.accepted, result.reasons
        clip_ids.append(result.clip_id)
    return clip_ids

def test_example_datasets_ingest_with_expected_formats(tmp_path):
    ing = Ingestor(CFG, tmp_path, open_db(tmp_path), Recorder(tmp_path))
    cam = _ingest_examples(ing, "video_caption_camera", "moving")
    assert all("video_caption_camera" in ing.clips.get(c)["formats"] for c in cam)
    timed = _ingest_examples(ing, "video_timed_prompts_camera", "moving")
    assert all("video_timed_prompts_camera:per_chunk" in ing.clips.get(c)["formats"] for c in timed)
    static = _ingest_examples(ing, "video_caption_static", "static")
    assert all("video_caption_static" in ing.clips.get(c)["formats"] for c in static)

def test_two_step_training_writes_a_checkpoint(tmp_path):
    ctx = bootstrap_run(CFG, run_id="manual_train", env=os.environ)
    conn = ctx.conn
    ing = Ingestor(CFG, ctx.run_dir, conn, ctx.recorder)
    clips = _ingest_examples(ing, "video_caption_camera", "moving")
    commits = CommitStore(conn, ing.blobs, ing.clips)
    commit = commits.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": clips}}, "manual", node_id="m1")
    node_dir = ctx.run_dir / "nodes" / "m1"
    gate = Gate(CFG, commits, ctx.recorder)
    result = gate.check({"optimizer.max_steps": 2, "optimizer.grad_accum_steps": 1,
                         "optimizer.epochs": 500},
                        commit, None, "m1", node_dir, ctx.run_dir, ctx.gpus)
    assert result.ok, result.failures
    runner = TrainRunner(CFG, ctx.recorder)
    runner.precache(result.resolved_path, ctx.gpus, "m1")
    outcome = runner.train(result.resolved_path, ctx.gpus, "m1", node_dir)
    assert outcome.failure == "none", outcome.log_path.read_text()[-4000:]
    assert (outcome.checkpoint / "lora.safetensors").exists()
    assert (outcome.checkpoint / "history_encoder.pt").exists()

def test_base_model_reproduces_the_recorded_proxy_score():
    ctx = bootstrap_run(CFG, run_id="manual_root", env=os.environ)
    score, detail = score_node(CFG, ctx, "root", None, 64, 64)
    import json
    reference = json.loads(
        (CFG.repo_root / "reference" / "wbench_alayaworld_proxy" / "report.json").read_text())
    for metric, value in detail["metrics"].items():
        if metric in reference["full"]:
            assert abs(value - reference["full"][metric]["mean"]) < 1e-3, metric
    assert 0.0 < score < 1.0
```

- [ ] **Step 2: Register the marker**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["manual: GPU verification, run explicitly"]
addopts = "-q -m 'not manual'"
```

- [ ] **Step 3: Run the ingest verification**

Run: `conda run -n autoresearcher python -m pytest tests/manual -v -s -m manual -k ingest`
Expected: PASS. If a real example clip is rejected, record the checker message in the verification log and fix the ingest rule rather than loosening the checker.

- [ ] **Step 4: Run the training verification**

Run: `conda run -n autoresearcher python -m pytest tests/manual -v -s -m manual -k training`
Expected: PASS in roughly 10 minutes, writing `checkpoint-2`.

- [ ] **Step 5: Run the root-score verification**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4 conda run -n autoresearcher python -m pytest tests/manual -v -s -m manual -k proxy_score`
Expected: PASS in roughly 2 hours (render ~80 min, eval ~25 min); GPU metrics match the recorded report within 1e-3.

- [ ] **Step 6: Preflight the allowlists**

For each `[height, width]` in `train.resolution_allowlist` and the largest LoRA pair, run a 10-step training through the manual path and record peak memory from the `[Mem]` lines. Remove any pair that OOMs from `configs/kernel.yaml`.

- [ ] **Step 7: Write the verification log**

Record, in `docs/superpowers/plans/verification-log.md`: date, GPU list, WorldModel and WBench SHAs, per-check pass/fail, measured timings, peak memory per resolution/rank pair, and the final allowlists.

- [ ] **Step 8: Commit**

```bash
git add tests/manual docs/superpowers/plans/verification-log.md configs/kernel.yaml pyproject.toml
git commit -m "test(kernel): real-hardware verification of ingest, training and proxy scoring"
```

---

## Plan Self-Review

**Spec coverage.** §2 environment → Task 1; §2.3 upstream patches → Task 3; §5.1-5.3 archive and blobs → Tasks 4-5; §5.4-5.5 clips and ingest → Tasks 6-8; §5.6-5.7 commits, views, caches → Tasks 9-10; §6 formats → enforced through Tasks 6-8 by the checker itself; §7.3 training → Task 11; §8 gate → Task 10; §11 eval and scoring → Tasks 12-13; §13 telemetry storage → Task 2 (dashboard is Plan 3); §16.1 unit tests → Tasks 1-13; §16.3 real-component verification items 1, 2, 3, 7, 9, 10 → Task 14.

**Deferred to later plans, by design:** §9 agent layer, §10 kernel tools, §12 selection, §14 retries/stop/resume, §13.4 dashboard, §16.3 items 4, 5, 6 and 8 (sandbox isolation, generator fit, camera annotation backend, merge-then-render soak). Each is a task in Plan 2 or Plan 3.

**Placeholder scan:** no TBDs; every step carries a command or code block.

**Type consistency check:** `Candidate`/`IngestResult` (Task 8) are used unchanged in Tasks 9 and 14; `CommitStore.commit/manifest/materialize` (Task 9) match the calls in Tasks 10 and 14; `GateResult.resolved_path` (Task 10) feeds `TrainRunner.precache/train` (Task 11); `TrainOutcome.checkpoint` (Task 11) feeds `score_node` (Task 13); `resolve_metric_set` and `score_from_report` (Task 12) are called by `preflight_metrics` and `score_node` (Task 13); `Recorder.event/span` (Task 2) is used with the same signature everywhere.
