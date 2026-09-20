"""GPU verification. Run explicitly:
   conda run -n autoresearcher python -m pytest tests/manual -v -s -m manual
"""
import os, shutil, pytest
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
    # Stage each clip's files under the run's own directory before ingesting: BlobStore.put()
    # moves/consumes its source, and Ingestor now refuses (by design, see
    # ar_kernel.data.ingest.Ingestor._outside_run_dir) any candidate outside the run dir. Never
    # hand ingest() a path inside the WorldModel checkout directly (this mirrors production,
    # where agents only ever ingest files they produced in their own workspace).
    root = EXAMPLES / dataset
    staging = Path(ing.run_dir) / "staging" / dataset
    clip_ids = []
    for video in sorted((root / "videos").glob("*.mp4")):
        stem = video.stem
        pose_src = root / "poses" / f"{stem}.npz"
        clip_dir = staging / stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        staged_video = Path(shutil.copy2(video, clip_dir / "video.mp4"))
        staged_caption = Path(shutil.copy2(root / "captions" / f"{stem}.json", clip_dir / "caption.json"))
        staged_pose = None
        if camera_motion == "moving" and pose_src.exists():
            staged_pose = Path(shutil.copy2(pose_src, clip_dir / "pose.npz"))
        result = ing.ingest([Candidate(
            video=staged_video, caption=staged_caption, pose=staged_pose,
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
