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
