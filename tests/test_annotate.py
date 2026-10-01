"""annotate_camera: the real backend with a fake subprocess worker, plus one real ViGeo gpu test."""
import copy
import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig, resolve_gpus
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import annotate
from ar_kernel.tools.annotate import AnnotateBackend
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.gpu_jobs import build_gpu_backends
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

REAL = KernelConfig.load()


FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def small_cfg(max_frames=60, env="alayaworld"):
    raw = copy.deepcopy(REAL.raw)
    raw["annotate"].update(max_frames=max_frames, env=env)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The real AnnotateBackend.produce/run_workers, with the ViGeo bridge swapped for the fake
    worker (annotate mode: it honors --max-frames and echoes --repo/--checkpoint)."""
    monkeypatch.setattr(annotate, "BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2, fps=24)           # 48 frames
    make_mp4(ws / "long.mp4", seconds=3, fps=24)        # 72 frames > max_frames 60
    q.register(AnnotateBackend(small_cfg(env="autoresearcher"), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                               gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def run(q, caller, items):
    out = q.wait(caller, q.backends["annotate_camera"].submit(q, caller, {"items": items})["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in out["result"]["items"]}


def test_pose_is_published_and_names_the_agents_clip(env):
    q, caller, staging = env
    job, by = run(q, caller, [{"video": "a.mp4"}])
    item = by[0]
    assert item["video"] == "/workspace/a.mp4"
    assert item["pose"] == f"/workspace/staging/annotations/{job}/0.npz"
    assert item["frames"] == 48 and item["intrinsics"] == [500.0, 500.0, 368.0, 207.0]
    with np.load(staging / "annotations" / job / "0.npz") as z:
        assert z["cam_c2w"].shape == (48, 4, 4) and z["intrinsics"].shape == (3, 3)
    assert not (staging / "annotations" / job / "0.mp4").exists()     # the video is never re-published


def test_a_pose_with_the_wrong_frame_count_is_an_item_error(env):
    q, caller, staging = env
    _, by = run(q, caller, [{"video": "a.mp4", "extra_frames": 1}, {"video": "a.mp4"}])
    assert "49 frames" in by[0]["error"] and "48" in by[0]["error"]
    assert "pose" in by[1]


def test_a_clip_over_max_frames_fails_alone(env):
    q, caller, staging = env
    _, by = run(q, caller, [{"video": "long.mp4"}, {"video": "a.mp4"}, {"video": "a.mp4"}])
    assert by[0]["error"] == "72 frames > max_frames (60)"
    assert "pose" in by[1] and "pose" in by[2]


def test_workers_get_the_annotate_config_and_one_gpu_each(env):
    q, caller, staging = env
    _, by = run(q, caller, [{"video": "a.mp4"} for _ in range(5)])
    wm = REAL.worldmodel
    for i, item in by.items():
        assert item["worker"]["repo"] == str(wm / REAL.get("annotate.repo"))
        assert item["worker"]["checkpoint"] == str(wm / REAL.get("annotate.checkpoint"))
        assert item["worker"]["gpus"] == str([0, 1, 4, 5][i % 4]) and item["worker"]["rank"] == i % 4


def test_non_mp4_is_refused_at_submit(env):
    q, caller, staging = env
    (staging.parent / "ws" / "x.mov").write_bytes(b"x")
    with pytest.raises(ToolError, match="not an .mp4"):
        q.backends["annotate_camera"].submit(q, caller, {"items": [{"video": "x.mov"}]})


def test_limits_come_from_the_annotate_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AnnotateBackend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    assert (b.max_items, b.timeout_s) == (REAL.get("annotate.max_items"), REAL.get("annotate.timeout_s"))


def test_build_gpu_backends_includes_annotate_only_when_enabled(tmp_path):
    rec = Recorder(tmp_path / "run")
    names = lambda cfg: [b.name for b in build_gpu_backends(cfg, tmp_path / "run", [0, 1, 2, 3],
                                                             TokenRegistry(rec), rec)]
    for enabled in (True, False):
        raw = copy.deepcopy(REAL.raw)
        raw["annotate"]["enabled"] = enabled
        assert ("annotate_camera" in names(KernelConfig(raw=raw, repo_root=REAL.repo_root))) is enabled


# ---- real ViGeo against the example poses ----

def _rel0(c2w):
    c2w = np.asarray(c2w, dtype=np.float64)
    return np.linalg.inv(c2w[0]) @ c2w


def _rot_angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def _relative_rotation_errors(pred, gt):
    """Per consecutive-frame step: the angle of dR_pred^T dR_gt, dR = R_i^T R_{i+1} (so a turn
    about the wrong axis or in the wrong direction counts, not only its magnitude)."""
    step = lambda c2w: np.swapaxes(c2w[:-1, :3, :3], 1, 2) @ c2w[1:, :3, :3]
    return _rot_angle(np.swapaxes(step(pred), 1, 2) @ step(gt))


def _umeyama_ate(src, dst):
    """RMS error of src's camera centers after the Sim(3) that best maps them onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    scale = np.trace(np.diag(d) @ s) / xs.var(0).sum()
    aligned = scale * src @ rot.T + (mu_d - scale * rot @ mu_s)
    return float(np.sqrt(((aligned - dst) ** 2).sum(1).mean()))


@pytest.mark.gpu
def test_real_vigeo_matches_the_example_poses(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_annotate.py -m gpu -s"""
    gpus = resolve_gpus(REAL, {"CUDA_VISIBLE_DEVICES": os.environ.get("AR_TEST_GPUS", "0,1,2,3")})
    examples = REAL.worldmodel / "data" / "examples" / "video_caption_camera"
    clips = sorted(p.stem for p in (examples / "videos").glob("*.mp4"))
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    for c in clips:
        shutil.copy(examples / "videos" / f"{c}.mp4", ws / f"{c}.mp4")
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=600)
    q.register(AnnotateBackend(REAL, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    try:
        job = q.backends["annotate_camera"].submit(q, caller, {"items": [{"video": f"{c}.mp4"} for c in clips]})
        out = q.wait(caller, job["job_id"], 600)
        while out["state"] not in ("done", "failed", "cancelled"):
            out = q.wait(caller, job["job_id"], 600)
    finally:
        q.shutdown()
    assert out["state"] == "done", out["error"]
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    rows, failures = [], []
    for item in out["result"]["items"]:
        c = clips[item["index"]]
        assert "error" not in item, item
        pose = staging / Path(item["pose"]).relative_to("/workspace/staging")
        with np.load(pose) as z:
            pred, k = _rel0(z["cam_c2w"]), z["intrinsics"]
        with np.load(examples / "poses" / f"{c}.npz") as z:
            gt, k_gt = _rel0(z["cam_c2w"]), (z["intrinsics"] if "intrinsics" in z.files else None)
        rot = float(np.median(_relative_rotation_errors(pred, gt)))
        path = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
        ate = _umeyama_ate(pred[:, :3, 3], gt[:, :3, 3]) / path
        f_err = [float(k[i, i] / k_gt[i, i] - 1) for i in (0, 1)] if k_gt is not None else None
        # (a) the kernel ingest checker accepts the published pose with the example caption
        stage = staging / "ingest" / c
        stage.mkdir(parents=True)
        shutil.copy(examples / "videos" / f"{c}.mp4", stage / "v.mp4")
        shutil.copy(examples / "captions" / f"{c}.json", stage / "c.json")
        shutil.copy(pose, stage / "p.npz")
        [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=stage / "p.npz",
                                      camera_motion="moving", provenance={"kind": "derived", "from": [],
                                      "transform": "annotate_camera gpu test"})], node_id="gpu")
        row = {"clip": c, "frames": item["frames"], "rot_err_deg": round(rot, 4), "ate_frac": round(ate, 4),
               "f_err": f_err and [round(e, 4) for e in f_err], "intrinsics": item["intrinsics"],
               "seconds": item["worker"].get("seconds"), "peak_mem_mib": item["worker"].get("peak_mem_mib"),
               "ingest_ok": res.accepted and "video_caption_camera" in res.formats}
        rows.append(row)
        if not row["ingest_ok"]:
            failures.append(f"{c}: ingest {res.reasons}")
        if rot >= 1.0:
            failures.append(f"{c}: rotation error {rot:.3f} deg >= 1")
        if ate >= 0.10:
            failures.append(f"{c}: ATE {ate:.1%} of path >= 10%")
        if f_err and max(abs(e) for e in f_err) >= 0.15:
            failures.append(f"{c}: focal error {f_err} >= 15%")
    print(json.dumps({"gpus": gpus, "clips": rows, "gpu_memory_mib": out["result"]["gpu_memory_mib"],
                      "gpu_memory_released": out["result"]["gpu_memory_released"]}, indent=1))
    assert out["result"]["gpu_memory_released"] is True
    assert not failures, failures
