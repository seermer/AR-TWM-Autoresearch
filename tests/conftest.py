import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

BRIDGES = Path(__file__).resolve().parents[1] / "kernel" / "ar_kernel" / "bridges"
sys.path.insert(0, str(BRIDGES))          # bridges import `protocol` plainly, as when run as scripts


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

def write_poses(path, n_frames, width=736, height=414, moving=True, intrinsics=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    if moving:
        c2w[:, 2, 3] = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    k = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]], dtype=np.float32)
    np.savez(path, cam_c2w=c2w, **({"intrinsics": k} if intrinsics else {}))
    return path

def write_annotation(path, turns=None, scene="A camera moves slowly through a bright room.", character=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scene_prompt": scene, "turns": turns if turns is not None else [{"start_s": 0.0, "perspective": "first_person"}]}
    if character is not None:
        payload["character_prompt"] = character
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def staged_clip(stage, seconds=4.0, fps=30, width=960, height=544, turns=None):
    """(video, annotation, pose) of one ingestable clip under `stage`: 960x544, a moving camera."""
    return (make_mp4(stage / "v.mp4", seconds=seconds, fps=fps, width=width, height=height),
            write_annotation(stage / "a.json", turns),
            write_poses(stage / "p.npz", n_frames=int(seconds * fps), width=width, height=height))

DERIVED = {"kind": "synthetic_method", "name": "unit test fixture"}
NAMED = {"video": "/workspace/staging/v.mp4", "annotation": "/workspace/staging/a.json", "pose": "/workspace/staging/p.npz"}

@pytest.fixture
def clip_dir(tmp_path):
    """A one-clip standard root: 4 s, 30 fps, 16:9, moving camera."""
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4")
    write_caption(root / "captions" / "c1.json")
    write_poses(root / "poses" / "c1.npz", n_frames=120)
    return root


@pytest.fixture(autouse=True)
def _skip_eval_prerequisites(request, monkeypatch):
    """Tests that create runs do not need the 62 GB VP weights; tests marked
    `real_preflight` exercise the actual prerequisite check."""
    if request.node.get_closest_marker("real_preflight"):
        return
    import ar_kernel.run as run
    monkeypatch.setattr(run, "preflight_metrics", lambda cfg, case_ids: list(run.DIMENSION_METRICS))
    monkeypatch.setattr(run, "_uncommitted", lambda repo: False)


@pytest.fixture(autouse=True)
def _all_gpus_free(request, monkeypatch):
    """The GPU choice reads nvidia-smi; outside GPU tests every card counts as free, so a run in
    progress on this machine does not change the outcome."""
    if request.node.get_closest_marker("gpu") or request.node.get_closest_marker("manual"):
        return
    from ar_kernel import guards
    monkeypatch.setattr(guards, "gpu_memory_used", lambda: {g: 1 for g in range(8)})


@pytest.fixture(autouse=True)
def _telemetry_per_test(monkeypatch):
    """No writer, root span or catch-all outlives a test (`ar run` installs them for the process)."""
    import importlib, logging, threading, warnings
    from ar_kernel.telemetry import capture
    monkeypatch.setattr(importlib.import_module("ar_kernel.telemetry.writer"), "_writer", None)
    monkeypatch.setattr(importlib.import_module("ar_kernel.telemetry.api"), "_root", None)
    for name in ("excepthook", "unraisablehook", "stdout", "stderr"):
        monkeypatch.setattr(sys, name, getattr(sys, name))
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    root = logging.getLogger()
    level = root.level
    yield
    for handler in [h for h in root.handlers if isinstance(h, capture._Handler)]:
        root.removeHandler(handler)
    root.setLevel(level)


@pytest.fixture(autouse=True)
def _no_hf_token(monkeypatch):
    """No test uploads to Hugging Face or hands HFK to a sandbox; set first so a .env a test loads is undone."""
    monkeypatch.setenv("HFK", "")
    monkeypatch.delenv("HFK")


RESULT_FILES = {}           # container path of a job's result file -> where it is on the host


@pytest.fixture(autouse=True)
def _remember_result_files(monkeypatch):
    from ar_kernel.tools import jobs
    real = jobs.publish

    def publish(caller, name, result):
        path = real(caller, name, result)
        RESULT_FILES[path] = caller.staging_host / "results" / f"{name}.json"
        return path
    monkeypatch.setattr(jobs, "publish", publish)


def job_result(view: dict) -> dict:
    """The full result of a finished job, read from the file job_wait named."""
    return json.loads(RESULT_FILES[view["result_file"]].read_text(encoding="utf-8"))


def job_folder(run, job_id: str) -> Path:
    """A job's work folder (jobs/<node>/<phase>-<attempt>/<id>); a path that does not exist when it has none."""
    return next((Path(run) / "jobs").glob(f"*/*/{job_id}"), Path(run) / "jobs" / "none" / job_id)


def client_dir(services) -> Path:
    """A short host path to a RunServices' sockets (AF_UNIX paths are capped at 107 bytes)."""
    return Path(f"/proc/{os.getpid()}/fd/{services._fd}")


def git_file(repo, commit, path):
    result = subprocess.run(["git", "--git-dir", str(repo.path), "show", f"{commit}:{path}"],
                            capture_output=True)
    return result.stdout.decode() if result.returncode == 0 else None


def telemetry_records(run) -> list[dict]:
    """Every telemetry record of the run, in file order (one writer's segments in order)."""
    out = []
    for path in sorted((Path(run) / "telemetry" / "records").glob("*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines()]
    return out


def telemetry_payload(run, digest: str):
    import zstandard
    path = Path(run) / "telemetry" / "payloads" / f"{digest}.json.zst"
    return json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))


# ---- `pytest --check-names`: every registered telemetry name is emitted somewhere in the suite ----

# Names only GPU or sandbox tests emit (both left out of the default suite), each with its reason.
UNEMITTED = {}


def pytest_addoption(parser):
    parser.addoption("--check-names", action="store_true",
                     help="fail when a registered span, decision or alert name is never emitted by the tests run")


def _emitted(root: Path) -> set[str]:
    """Names in every record under `root`: the test folders of this session, the workers' and their children's."""
    names = set()
    for path in root.rglob("records/*.jsonl"):
        for line in path.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("kind") in ("start", "decision", "alert") and r.get("name"):
                names.add(r["name"])
    return names


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("--check-names") or hasattr(config, "workerinput"):
        return
    from ar_kernel.telemetry import registry
    names = _emitted(Path(config.option.basetemp))      # never getbasetemp(): its first call empties the folder
    missing = []
    for table in (registry.SPANS, registry.DECISIONS, registry.ALERTS):
        for name in table:
            seen = any(n.startswith(name[:-1]) and n not in table for n in names) if name.endswith(".*") \
                else name in names
            if not seen and name not in UNEMITTED:
                missing.append(name)
    if missing:
        print(f"\nregistered telemetry names never emitted: {sorted(missing)}", file=sys.stderr)
        session.exitstatus = 1
