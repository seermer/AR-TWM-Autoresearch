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

def test_candidate_outside_run_dir_is_rejected(tmp_path, tmp_path_factory):
    # BlobStore.put() moves/consumes its source file; ingest must refuse anything it does not
    # own rather than silently deleting it (this is what consumed WorldModel's real
    # example clips before the guard existed).
    ing = _ingestor(tmp_path)
    outside = tmp_path_factory.mktemp("outside")
    video = make_mp4(outside / "v.mp4")
    caption = write_caption(outside / "c.json")
    pose = write_poses(outside / "p.npz", n_frames=120)
    candidate = Candidate(video=video, caption=caption, pose=pose, camera_motion="moving",
                          provenance=PROV)
    [result] = ing.ingest([candidate], node_id="n1")
    assert result.accepted is False
    assert any("outside the staging directory" in r for r in result.reasons)
    assert video.exists() and caption.exists() and pose.exists()


def test_candidate_inside_run_dir_but_outside_staging_is_rejected(tmp_path):
    """The guard used to accept anything under run_dir, including a
    blob in store/ or a hardlink in another node's view. Only staging/ is fair game."""
    ing = _ingestor(tmp_path)
    elsewhere = tmp_path / "nodes" / "n7" / "view" / "cam"
    video = make_mp4(elsewhere / "v.mp4", seconds=4.0)
    caption = write_caption(elsewhere / "c.json")
    pose = write_poses(elsewhere / "p.npz", n_frames=120)
    result = ing.ingest([Candidate(video=video, caption=caption, pose=pose,
                                   camera_motion="moving", provenance=PROV)], node_id="n1")[0]
    assert not result.accepted
    assert any("staging" in r for r in result.reasons)
    assert video.exists() and caption.exists() and pose.exists()


def test_swapping_the_staged_file_mid_check_cannot_change_what_is_stored(tmp_path):
    """Checks read candidate.video, then blobs.put re-read it later.
    An agent process could swap the staged file after the leakage check and land
    unchecked bytes in the store. Files are now quarantined before any check."""
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "swap")
    original = cand.video.read_bytes()
    staged_path = cand.video
    real_check = ing.leakage.check

    def swap_then_check(video):
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_bytes(b"SWAPPED benchmark frame")   # the agent's swap
        return real_check(video)

    ing.leakage.check = swap_then_check
    result = ing.ingest([cand], node_id="n1")[0]
    assert result.accepted
    clip = ing.clips.get(result.clip_id)
    assert ing.blobs.path(clip["video_digest"], "video").read_bytes() == original


def test_corrupt_video_is_rejected_and_the_batch_continues(tmp_path):
    """A non-video file raised out of ffprobe and aborted the batch,
    with no ingest.rejected event and later candidates never processed."""
    ing = _ingestor(tmp_path)
    bad = _candidate(tmp_path, "bad")
    bad.video.write_bytes(b"this is not an mp4")
    good = _candidate(tmp_path, "good", seconds=5.0)
    results = ing.ingest([bad, good], node_id="n1")
    assert not results[0].accepted
    assert any("could not read" in r for r in results[0].reasons)
    assert results[1].accepted
    events = [e for e in ing.recorder.read_events("n1") if e["type"] == "ingest.rejected"]
    assert events, "a rejected candidate must leave an ingest.rejected event"


def test_a_checker_crash_rejects_only_its_candidate(tmp_path, monkeypatch):
    from ar_kernel.data import ingest
    from ar_kernel.data.checker import CheckerError
    real = ingest.check_clip_formats
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise CheckerError("checker bridge failed (rc=1)")
        return real(*args, **kwargs)
    monkeypatch.setattr(ingest, "check_clip_formats", flaky)
    ing = _ingestor(tmp_path)
    results = ing.ingest([_candidate(tmp_path, "a"), _candidate(tmp_path, "b", seconds=5.0)], node_id="n1")
    assert not results[0].accepted and "checker bridge failed" in results[0].reasons[0]
    assert results[1].accepted


def test_rejected_candidate_files_are_returned_to_staging(tmp_path):
    """Quarantine is only for the duration of the checks; a rejected candidate's
    files go back where the agent staged them so it can inspect or fix them."""
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "narrow", width=640, height=480)
    result = ing.ingest([cand], node_id="n1")[0]
    assert not result.accepted
    assert cand.video.exists() and cand.caption.exists()
    assert not [p for p in (tmp_path / "quarantine").rglob("*") if p.is_file()]


def test_rotated_clip_is_rejected_with_a_fix_hint(tmp_path):
    """WorldModel decodes frames in stored orientation; a rotated clip would train sideways."""
    import subprocess
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "rot")
    rotated = cand.video.with_name("v_rot.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "180",
                    "-i", str(cand.video), "-c", "copy", str(rotated)], check=True)
    rotated.replace(cand.video)
    result = ing.ingest([cand], node_id="n1")[0]
    assert not result.accepted
    assert any("rotation" in r and "re-encode" in r for r in result.reasons)


def test_clip_metadata_records_segments_and_intrinsics(tmp_path):
    """Probed metadata includes has_segments and has_intrinsics."""
    from conftest import write_caption, write_poses
    ing = _ingestor(tmp_path)
    plain = _candidate(tmp_path, "plain")
    timed = _candidate(tmp_path, "timed", seconds=3.0)
    write_caption(timed.caption, segments=[{"time_range_s": [0.0, 3.0], "prompt": "walk forward"}])
    write_poses(timed.pose, n_frames=90, intrinsics=False)
    static = _candidate(tmp_path, "static", seconds=3.5, moving=False)
    results = ing.ingest([plain, timed, static], node_id="n1")
    assert all(r.accepted for r in results), [r.reasons for r in results]
    meta = [ing.clips.get(r.clip_id)["metadata"] for r in results]
    assert [(m["has_segments"], m["has_intrinsics"]) for m in meta] == [
        (False, True), (True, False), (False, False)]

def test_every_ingest_event_names_its_candidate(tmp_path):
    ing = _ingestor(tmp_path)
    good, wide = _candidate(tmp_path, "good"), _candidate(tmp_path, "wide", width=640, height=480)
    bare = _candidate(tmp_path, "bare")
    bare.provenance = {}
    ing.ingest([good, wide, bare], node_id="n1")
    events = [e for e in ing.recorder.read_events("n1") if e["type"].startswith("ingest.")]
    named = [(e["type"], ing.recorder.load_payload(e["payload"])["candidate"]["video"]) for e in events]
    assert ("ingest.leakage", str(good.video)) in named and ("ingest.accepted", str(good.video)) in named
    assert ("ingest.rejected", str(wide.video)) in named and ("ingest.rejected", str(bare.video)) in named
    assert len(named) == 4
    first = ing.recorder.load_payload(events[0]["payload"])["candidate"]
    assert first == {"video": str(good.video), "caption": str(good.caption), "pose": str(good.pose)}


def _path(n=120, step=0.02):
    import numpy as np
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 2, 3] = np.arange(n) * step
    return c2w


def test_pose_jump_names_a_translation_or_rotation_jump_and_nothing_else():
    import numpy as np
    from ar_kernel.data.ingest import pose_jump
    assert pose_jump(_path()) is None
    assert pose_jump(np.tile(np.eye(4), (120, 1, 1))) is None            # a camera that never moves
    moved = _path()
    moved[57:, 2, 3] += 1.0                                              # two renders joined at frame 57
    assert pose_jump(moved).startswith("camera pose jumps between frames 56 and 57: the step is 51.0x")
    turned = _path()
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    turned[30:, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    assert "frames 29 and 30" in pose_jump(turned) and "turns 40.0 degrees" in pose_jump(turned)


def test_a_clip_whose_pose_jumps_is_accepted_with_a_warning(tmp_path):
    import numpy as np
    ing = _ingestor(tmp_path)
    candidate = _candidate(tmp_path)
    poses = _path()
    poses[57:, 2, 3] += 1.0
    np.savez(candidate.pose, cam_c2w=poses.astype(np.float32))
    [result] = ing.ingest([candidate], node_id="n1")
    assert result.accepted
    assert any("camera pose jumps between frames 56 and 57" in w for w in result.warnings)
    assert result.warnings == ing.clips.get(result.clip_id)["warnings"]


import json

from ar_kernel.isolation import EXCLUDED_CLIP

COPIED = "Torch sconces on both walls cast flickering orange light."     # from an evaluation prompt


def _rejected_events(tmp_path):
    rec = Recorder(tmp_path)
    return [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "ingest.rejected"]


@pytest.mark.parametrize("caption,provenance,why", [
    ({"caption": f"A corridor. {COPIED}"}, PROV, "text"),
    ({"caption": "A corridor.", "segments": [{"time_range_s": [0, 4], "prompt": COPIED}]}, PROV, "text"),
    ({"caption": "A corridor."}, {"kind": "hf_dataset", "repo": "x/WBench-mirror", "revision": "a"}, "source"),
])
def test_excluded_candidates_get_one_neutral_reason(tmp_path, caption, provenance, why):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.caption.write_text(json.dumps(caption))
    c = Candidate(video=c.video, caption=c.caption, pose=c.pose, camera_motion="moving", provenance=provenance)
    [result] = ing.ingest([c], node_id="n1")
    assert result.accepted is False and result.reasons == [EXCLUDED_CLIP]
    assert _rejected_events(tmp_path)[-1]["excluded"] == why


def test_a_caption_with_odd_segments_does_not_break_the_text_check(tmp_path):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.caption.write_text(json.dumps({"caption": "A bright room.", "segments": ["not an object", None]}))
    [result] = ing.ingest([c], node_id="n1")          # the format checker's verdict stands, whatever it is
    assert result.reasons != [EXCLUDED_CLIP]


@pytest.mark.parametrize("segments", [5, True, "text"])
def test_segments_that_are_not_a_list_do_not_break_the_batch(tmp_path, segments):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.caption.write_text(json.dumps({"caption": "A bright room.", "segments": segments}))
    [result] = ing.ingest([c], node_id="n1")
    assert result.reasons != [EXCLUDED_CLIP]
