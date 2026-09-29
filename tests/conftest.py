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

def write_poses(path, n_frames, width=736, height=414, moving=True, intrinsics=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    if moving:
        c2w[:, 2, 3] = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    k = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]], dtype=np.float32)
    np.savez(path, cam_c2w=c2w, **({"intrinsics": k} if intrinsics else {}))
    return path

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
    monkeypatch.setattr(run, "preflight_metrics", lambda cfg: list(run.DIMENSION_METRICS))
