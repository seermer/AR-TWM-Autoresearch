import json, os, subprocess
from pathlib import Path
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
