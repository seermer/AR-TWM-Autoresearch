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

# Two WBench metric inputs are stochastic, so a single tolerance cannot cover all
# metrics. Measured by re-running the 2026-09-13 reference generation end to end
# (see docs/superpowers/plans/verification-log.md):
#
#   deterministic metrics   max |delta| 0.0007   GPU float noise only
#   POSE_DERIVED            max |delta| 0.0084   MegaSAM's pose solver is not
#                                                deterministic: two runs over the
#                                                same video gave cam_c2w differing
#                                                by 6.8e-3 and focal length by 0.92px
#   SUBSAMPLED              max |delta| 0.0015   reconstruction_consistency.py:154
#                                                draws points with an unseeded
#                                                torch.randperm
#
# Tolerances are the measured spread with headroom, NOT a number chosen to make
# the test pass. Keeping the deterministic band tight is the point: it is what
# would catch a real regression in the render path.
POSE_DERIVED = {"spatial_consistency", "gated_spatial_consistency",
                "navigation_trajectory", "navigation_accuracy", "navigation_consistency"}
SUBSAMPLED = {"geometric_consistency", "photometric_consistency"}
TOLERANCE = {"deterministic": 2e-3, "subsampled": 5e-3, "pose": 2e-2}
AGGREGATE_TOLERANCE = 2e-3


def _tolerance(metric: str) -> float:
    if metric in POSE_DERIVED:
        return TOLERANCE["pose"]
    if metric in SUBSAMPLED:
        return TOLERANCE["subsampled"]
    return TOLERANCE["deterministic"]


def test_base_model_reproduces_the_recorded_proxy_score():
    ctx = bootstrap_run(CFG, run_id="manual_root", env=os.environ)
    score, detail = score_node(CFG, ctx, "root", None, 64, 64)
    import json
    import statistics
    reference = json.loads(
        (CFG.repo_root / "reference" / "wbench_alayaworld_proxy" / "report.json").read_text())
    ref_full = reference["full"]

    # Every metric the reference recorded must still be produced. The run that
    # motivated this check lost five of them to a silent MegaSAM failure and
    # still looked healthy.
    shared = sorted(set(detail["metrics"]) & set(ref_full))
    assert not set(ref_full) - set(detail["metrics"]), \
        f"metrics missing vs reference: {sorted(set(ref_full) - set(detail['metrics']))}"

    deltas = {m: detail["metrics"][m] - ref_full[m]["mean"] for m in shared}
    offenders = {m: d for m, d in deltas.items() if abs(d) >= _tolerance(m)}
    assert not offenders, "metrics outside their tolerance: " + ", ".join(
        f"{m} delta={d:+.6f} tol={_tolerance(m):g}" for m, d in sorted(offenders.items()))

    # The aggregate is what parent selection actually consumes, so pin it too:
    # per-metric noise partly cancels, and this is the loop's real signal floor.
    ref_mean = statistics.fmean(ref_full[m]["mean"] for m in shared)
    new_mean = statistics.fmean(detail["metrics"][m] for m in shared)
    assert abs(new_mean - ref_mean) < AGGREGATE_TOLERANCE, \
        f"aggregate {new_mean:.6f} vs reference {ref_mean:.6f}"
    assert 0.0 < score < 1.0
