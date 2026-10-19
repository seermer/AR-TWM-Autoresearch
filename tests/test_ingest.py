import pytest
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.annotation import THIRD_PERSON_VIEW, caption_file
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from conftest import DERIVED, make_mp4, staged_clip, write_poses

CFG = KernelConfig.load()
FP, TP = "first_person", "third_person"

def _ingestor(tmp_path):
    return Ingestor(CFG, tmp_path, open_db(tmp_path), Recorder(tmp_path))

def _candidate(tmp_path, name="c1", seconds=4.0, fps=30, width=960, height=544, turns=None):
    video, annotation, pose = staged_clip(tmp_path / "staging" / name, seconds, fps, width, height, turns)
    return Candidate(video=video, annotation=annotation, pose=pose, derived_from=dict(DERIVED))

def test_accepted_clip_is_recorded_and_stores_its_own_copy(tmp_path):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path)
    (cand.video.parent / "keep.mp4").hardlink_to(cand.video)        # the agent keeps a second name for the file
    original = cand.video.read_bytes()
    [result] = ing.ingest([cand], node_id="n1")
    assert result.accepted, result.reasons
    clip = ing.clips.get(result.clip_id)
    assert clip["derived_from"] == DERIVED and clip["ingested_by"] == "n1"
    # the staged files are left alone, and the stored clip is no longer the agent's file
    assert cand.video.exists() and cand.annotation.exists() and cand.pose.exists()
    (cand.video.parent / "keep.mp4").write_bytes(b"rewritten after every check passed")
    blob = ing.blobs.path(clip["video_digest"], "video")
    assert blob.read_bytes() == original and blob.stat().st_nlink == 1
    assert not list((tmp_path / "quarantine").iterdir())


def test_a_staged_folder_swapped_for_a_link_reaches_nothing_outside_staging(tmp_path, tmp_path_factory):
    """The agent's container runs while the kernel checks: it can rename the folder it staged and put
    a link to a host folder in its place. The kernel then read, and on a rejection wrote, through it."""
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "d")
    outside = tmp_path_factory.mktemp("host")
    for f in (cand.video, cand.annotation, cand.pose):
        (outside / f.name).write_bytes(f.read_bytes())
    before = {p.name: p.read_bytes() for p in outside.iterdir()}
    cand.video.parent.rename(tmp_path / "staging" / "d.real")
    (tmp_path / "staging" / "d").symlink_to(outside)                 # staging/d/v.mp4 is now a host file
    # swapped after the up-front path check: the copy into quarantine refuses the file it opened
    with pytest.raises(OSError, match="changed while it was being read"):
        ing._take(cand.video, tmp_path / "taken.mp4")
    # swapped before it: the candidate is rejected
    [result] = ing.ingest([cand], node_id="n1")
    assert not result.accepted and "outside the staging directory" in result.reasons[0]
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == before      # nothing written or removed there
    assert not ing.clips.all() and not (tmp_path / "taken.mp4").exists()

def test_duplicate_content_yields_the_same_clip_id(tmp_path):
    ing = _ingestor(tmp_path)
    [first] = ing.ingest([_candidate(tmp_path, "a")], node_id="n1")
    [second] = ing.ingest([_candidate(tmp_path, "b")], node_id="n2")
    assert first.clip_id == second.clip_id
    assert len(ing.clips.all()) == 1

def test_only_960x544_is_taken(tmp_path):
    ing = _ingestor(tmp_path)
    [result] = ing.ingest([_candidate(tmp_path, width=1280, height=720)], node_id="n1")
    assert result.accepted is False
    assert any("1280x720" in r and "960x544 only" in r and "resize" in r for r in result.reasons)

def test_a_clip_too_short_to_train_is_rejected_in_seconds(tmp_path):
    ing = _ingestor(tmp_path)
    [result] = ing.ingest([_candidate(tmp_path, seconds=1.0)], node_id="n1")
    assert result.accepted is False
    [reason] = result.reasons
    assert reason.startswith("the video lasts 1.00 s at 24 fps (30 frames at 30.00 fps)") and "at least 2.4 s" in reason
    assert "window" not in reason

@pytest.mark.parametrize("derived_from", [{}, {"kind": "rollout", "name": "x"}, {"kind": "model"}, "ltx"])
def test_derived_from_needs_a_listed_kind_and_a_name(tmp_path, derived_from):
    ing = _ingestor(tmp_path)
    candidate = _candidate(tmp_path)
    candidate.derived_from = derived_from
    [result] = ing.ingest([candidate], node_id="n1")
    assert result.accepted is False and any("derived_from" in r for r in result.reasons)

def test_candidate_outside_run_dir_is_rejected(tmp_path, tmp_path_factory):
    ing = _ingestor(tmp_path)
    outside = tmp_path_factory.mktemp("outside")
    video, annotation, pose = staged_clip(outside)
    candidate = Candidate(video=video, annotation=annotation, pose=pose, derived_from=DERIVED)
    [result] = ing.ingest([candidate], node_id="n1")
    assert result.accepted is False
    assert any("outside the staging directory" in r for r in result.reasons)
    assert video.exists() and annotation.exists() and pose.exists()


def test_candidate_inside_run_dir_but_outside_staging_is_rejected(tmp_path):
    """The guard accepts only staging/, not a blob in store/ or a hardlink in another node's view."""
    ing = _ingestor(tmp_path)
    elsewhere = tmp_path / "nodes" / "n7" / "view" / "cam"
    video, annotation, pose = staged_clip(elsewhere)
    result = ing.ingest([Candidate(video=video, annotation=annotation, pose=pose, derived_from=DERIVED)],
                        node_id="n1")[0]
    assert not result.accepted
    assert any("staging" in r for r in result.reasons)
    assert video.exists() and annotation.exists() and pose.exists()


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
    real = ingest.check_clip
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise CheckerError("checker bridge failed (rc=1)")
        return real(*args, **kwargs)
    monkeypatch.setattr(ingest, "check_clip", flaky)
    ing = _ingestor(tmp_path)
    results = ing.ingest([_candidate(tmp_path, "a"), _candidate(tmp_path, "b", seconds=5.0)], node_id="n1")
    [crashed] = [r for r in results if not r.accepted]         # the checks run at the same time: either one
    assert "checker bridge failed" in crashed.reasons[0]


def test_a_full_disk_while_checking_is_raised_not_a_rejection(tmp_path, monkeypatch):
    from ar_kernel.data import ingest

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(ingest, "check_clip", full)
    with pytest.raises(OSError, match="No space left"):
        _ingestor(tmp_path).ingest([_candidate(tmp_path, "a")], node_id="n1")
    assert not list((tmp_path / "quarantine").glob("*/*"))          # its held copy is discarded


def test_every_candidate_hitting_a_checker_crash_marks_them(tmp_path, monkeypatch):
    from ar_kernel.data import ingest
    from ar_kernel.data.checker import CheckerError
    monkeypatch.setattr(ingest, "check_clip", lambda *a, **k: (_ for _ in ()).throw(CheckerError("bridge died")))
    results = _ingestor(tmp_path).ingest([_candidate(tmp_path, "a"), _candidate(tmp_path, "b")], node_id="n1")
    assert all(not r.accepted and r.checker_failed for r in results)


def test_a_clip_whose_frames_cannot_be_extracted_is_rejected_alone(tmp_path, monkeypatch):
    """ffmpeg exits 0 without writing a frame when a clip's stated duration is longer than what
    decodes: opening the missing frame raised FileNotFoundError out of the whole call."""
    ing = _ingestor(tmp_path)
    real = ing.leakage.check

    def no_frame(video):
        if video.stat().st_size == short:
            raise FileNotFoundError(2, "No such file or directory", "/tmp/x/frame.png")
        return real(video)
    cands = [_candidate(tmp_path, "a"), _candidate(tmp_path, "b", seconds=5.0)]
    short = cands[0].video.stat().st_size
    monkeypatch.setattr(ing.leakage, "check", no_frame)
    first, second = ing.ingest(cands, node_id="n1")
    assert not first.accepted and "could not read the candidate" in first.reasons[0]
    assert second.accepted and cands[0].video.exists()


def test_clips_are_checked_several_at_a_time(tmp_path, monkeypatch):
    import threading
    from ar_kernel.data import ingest
    real, together = ingest.check_clip, threading.Barrier(2, timeout=60)

    def both_inside(*args, **kwargs):
        together.wait()                 # times out, and fails the test, unless two checks overlap
        return real(*args, **kwargs)
    monkeypatch.setattr(ingest, "check_clip", both_inside)
    results = _ingestor(tmp_path).ingest(
        [_candidate(tmp_path, "a"), _candidate(tmp_path, "b", seconds=5.0)], node_id="n1")
    assert [r.accepted for r in results] == [True, True]


def test_an_unexpected_error_gives_every_candidates_files_back(tmp_path, monkeypatch):
    ing = _ingestor(tmp_path)
    monkeypatch.setattr(ing.blobs, "put", lambda path, kind: (_ for _ in ()).throw(OSError("disk full")))
    cands = [_candidate(tmp_path, "a"), _candidate(tmp_path, "b", seconds=5.0)]
    with pytest.raises(OSError, match="disk full"):
        ing.ingest(cands, node_id="n1")
    assert all(c.video.exists() and c.annotation.exists() and c.pose.exists() for c in cands)
    assert not [p for p in (tmp_path / "quarantine").rglob("*") if p.is_file()]


def test_a_rejected_candidates_files_stay_where_they_were_staged(tmp_path):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "narrow", width=640, height=480)
    before = cand.video.read_bytes()
    result = ing.ingest([cand], node_id="n1")[0]
    assert not result.accepted
    assert cand.video.read_bytes() == before and cand.annotation.exists()
    assert not list((tmp_path / "quarantine").iterdir())


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


def test_clip_metadata_records_turns_and_intrinsics(tmp_path):
    ing = _ingestor(tmp_path)
    plain = _candidate(tmp_path, "plain")
    timed = _candidate(tmp_path, "timed", seconds=5.0, turns=[
        {"start_s": 0.0, "perspective": FP}, {"start_s": 2.5, "event": "A dog walks in.", "perspective": FP}])
    write_poses(timed.pose, n_frames=150, width=960, height=544, intrinsics=False)
    results = ing.ingest([plain, timed], node_id="n1")
    assert all(r.accepted for r in results), [r.reasons for r in results]
    meta = [ing.clips.get(r.clip_id)["metadata"] for r in results]
    assert [(m["turns"], m["has_intrinsics"]) for m in meta] == [(1, True), (2, False)]


def test_the_training_text_is_made_from_the_annotation_and_both_are_stored(tmp_path):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, seconds=8.0, turns=[
        {"start_s": 0.0, "perspective": FP},
        {"start_s": 2.5, "event": "A camel steps out from behind the tree.", "perspective": FP},
        {"start_s": 5.0, "perspective": TP}])
    annotation = json.loads(cand.annotation.read_text())
    annotation["character_prompt"] = "A hiker in a red jacket."
    cand.annotation.write_text(json.dumps(annotation))
    [result] = ing.ingest([cand], node_id="n1")
    assert result.accepted, result.reasons
    clip = ing.clips.get(result.clip_id)
    scene = "A camera moves slowly through a bright room. A hiker in a red jacket."
    assert json.loads(ing.blobs.path(clip["caption_digest"], "caption").read_text()) == {
        "caption": scene,
        "segments": [
            {"time_range_s": [0.0, 2.5], "prompt": scene},
            {"time_range_s": [2.5, 5.0], "prompt": f"{scene} A camel steps out from behind the tree."},
            {"time_range_s": [5.0, 8.0], "prompt": f"{scene} The view switches from first-person to third-person, "
                                                   f"revealing the character from behind. {THIRD_PERSON_VIEW}"}]}
    assert json.loads(ing.blobs.path(clip["annotation_digest"], "annotation").read_text()) == annotation


def test_third_person_turns_end_with_the_perspective_sentence():
    turns = [{"start_s": 0.0, "event": "A dog walks in.", "perspective": TP}, {"start_s": 2.5, "perspective": TP},
             {"start_s": 5.0, "perspective": FP}]
    caption = caption_file({"scene_prompt": "A park.", "turns": turns}, 8.0)
    assert caption["caption"] == f"A park. {THIRD_PERSON_VIEW}"
    assert [s["prompt"] for s in caption["segments"]] == [
        f"A park. A dog walks in. {THIRD_PERSON_VIEW}", f"A park. {THIRD_PERSON_VIEW}",
        "A park. The view switches back to first-person, returning to the character's own eyes."]


@pytest.mark.parametrize("annotation,message", [
    ("not json", "not valid JSON"),
    ({"turns": [{"start_s": 0}]}, "scene_prompt is required"),
    ({"scene_prompt": "A room."}, "turns is required"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 1.0}]}, "turn 1 must start at 0"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0, "perspective": FP}, {"start_s": 0, "perspective": FP}]},
     "not after turn 1"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0}]}, "perspective is required"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0, "perspective": FP}, {"start_s": 2.5}]}, "perspective is required"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0, "event": "", "perspective": FP}]}, "event must be one sentence"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0, "camera_motion": "W", "perspective": FP}]}, "unknown field 'camera_motion'"),
    ({"scene_prompt": "A room.", "caption": "x", "turns": [{"start_s": 0, "perspective": FP}]}, "unknown field 'caption'"),
    ({"scene_prompt": "A room.", "turns": [{"start_s": 0, "perspective": FP}, {"start_s": 9.0, "perspective": FP}]},
     "at or after the end of the video"),
])
def test_a_malformed_annotation_is_rejected(tmp_path, annotation, message):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path)
    cand.annotation.write_text(annotation if isinstance(annotation, str) else json.dumps(annotation))
    [result] = ing.ingest([cand], node_id="n1")
    assert not result.accepted and any(message in r for r in result.reasons), result.reasons


@pytest.mark.parametrize("starts,reasons", [
    ([0.0, 2.3], ["turn 1 lasts 2.29 s; the first turn must last at least 2.4 s"]),
    ([0.0, 2.5, 3.8], ["turn 2 lasts 1.29 s; a turn after the first must last at least 1.4 s"]),
    ([0.0, 2.5, 7.0], ["turn 3 lasts 1.00 s; a turn after the first must last at least 1.4 s"]),
])
def test_a_turn_too_short_to_train_is_rejected_in_seconds(tmp_path, starts, reasons):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, seconds=8.0, fps=24, turns=[{"start_s": s, "perspective": FP} for s in starts])
    [result] = ing.ingest([cand], node_id="n1")
    assert not result.accepted and result.reasons == reasons


def test_turns_of_exactly_the_shortest_lengths_are_accepted(tmp_path):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, seconds=89 / 24, fps=24, turns=[
        {"start_s": 0.0, "perspective": FP}, {"start_s": 57 / 24, "event": "Rain begins to fall.", "perspective": FP}])
    write_poses(cand.pose, n_frames=89, width=960, height=544)
    [result] = ing.ingest([cand], node_id="n1")
    assert result.accepted, result.reasons


def test_every_ingest_event_names_its_candidate(tmp_path):
    ing = _ingestor(tmp_path)
    good, wide = _candidate(tmp_path, "good"), _candidate(tmp_path, "wide", width=640, height=480)
    bare = _candidate(tmp_path, "bare")
    bare.derived_from = {}
    ing.ingest([good, wide, bare], node_id="n1")
    events = [e for e in ing.recorder.read_events("n1") if e["type"].startswith("ingest.")]
    named = [(e["type"], ing.recorder.load_payload(e["payload"])["candidate"]["video"]) for e in events]
    assert ("ingest.leakage", str(good.video)) in named and ("ingest.accepted", str(good.video)) in named
    assert ("ingest.rejected", str(wide.video)) in named and ("ingest.rejected", str(bare.video)) in named
    assert len(named) == 4
    [accepted] = [ing.recorder.load_payload(e["payload"])["candidate"] for e in events if e["type"] == "ingest.accepted"]
    assert accepted == {"video": str(good.video), "annotation": str(good.annotation), "pose": str(good.pose)}


def test_a_reingest_records_where_a_hidden_clip_moved_from(tmp_path):
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path)
    [first] = ing.ingest([cand], node_id="n1")
    ing.clips.hide("n1", "fail1")
    ing.ingest([cand], node_id="n2")
    ing.ingest([cand], node_id="n3")                                   # visible now: n2 keeps it
    for node, moved_from in (("n2", "n1fail1"), ("n3", None)):
        [event] = [e for e in ing.recorder.read_events(node) if e["type"] == "ingest.accepted"]
        assert ing.recorder.load_payload(event["payload"])["moved_from"] == moved_from
    assert ing.clips.get(first.clip_id)["ingested_by"] == "n2"


def _path(n=120, step=0.02):
    import numpy as np
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 2, 3] = np.arange(n) * step
    return c2w


SMOOTH = "mandelbrot=size=960x544:rate=30"       # a picture that changes evenly from frame to frame


def _joined(path, at=57, n=120):
    """Two shots joined at frame `at`: the same motion, another picture from there on."""
    import subprocess
    filters = (f"[0:v]trim=end_frame={at}[a];[0:v]trim=end_frame={n - at},negate[b];[a][b]concat=n=2:v=1"
               if at else f"[0:v]trim=end_frame={n}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", SMOOTH, "-filter_complex", filters,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_a_cut_is_a_pose_jump_and_a_picture_jump_together(tmp_path):
    import numpy as np
    from ar_kernel.data.ingest import find_cut
    one_shot = _joined(tmp_path / "one.mp4", at=0)
    joined = _joined(tmp_path / "joined.mp4")
    jumped = _path()
    jumped[57:, 2, 3] += 1.0
    turned = _path()
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    turned[57:, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    still = np.tile(np.eye(4), (120, 1, 1))
    assert find_cut(_path(), one_shot) is None
    assert find_cut(jumped, one_shot) is None                 # the camera estimate jumps, the picture does not
    assert find_cut(_path(), joined) is None                  # the picture jumps, the camera does not
    assert find_cut(still, joined) is None                    # a camera that never moves has no jump
    assert find_cut(jumped, joined).startswith("a cut between frames 56 and 57")
    assert find_cut(turned, joined).startswith("a cut between frames 56 and 57")
    late = _path()
    late[61:, 2, 3] += 1.0                                    # four frames after the picture jump
    assert find_cut(late, joined) is None


def test_a_clip_with_a_cut_is_rejected(tmp_path):
    import numpy as np
    ing = _ingestor(tmp_path)
    candidate = _candidate(tmp_path)
    _joined(candidate.video)
    poses = _path()
    poses[57:, 2, 3] += 1.0
    np.savez(candidate.pose, cam_c2w=poses.astype(np.float32))
    [result] = ing.ingest([candidate], node_id="n1")
    assert not result.accepted
    assert result.reasons == ["the clip has a cut between frames 56 and 57: the picture and the measured camera "
                              "both jump there; a training clip is one continuous shot"]


import json

from ar_kernel.isolation import EXCLUDED_CLIP

COPIED = "Torch sconces on both walls cast flickering orange light."     # from an evaluation prompt


def _rejected_events(tmp_path):
    rec = Recorder(tmp_path)
    return [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "ingest.rejected"]


@pytest.mark.parametrize("annotation", [
    {"scene_prompt": f"A corridor. {COPIED}", "turns": [{"start_s": 0, "perspective": FP}]},
    {"scene_prompt": "A corridor.", "character_prompt": COPIED, "turns": [{"start_s": 0, "perspective": FP}]},
    {"scene_prompt": "A corridor.", "turns": [{"start_s": 0, "event": COPIED, "perspective": FP}]},
])
def test_excluded_candidates_get_one_neutral_reason(tmp_path, annotation):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.annotation.write_text(json.dumps(annotation))
    [result] = ing.ingest([c], node_id="n1")
    assert result.accepted is False and result.reasons == [EXCLUDED_CLIP]
    assert _rejected_events(tmp_path)[-1]["excluded"] == "text"


def test_reasons_and_warnings_name_the_candidates_files_not_the_kernels_paths(tmp_path):
    """No message carries the kernel's temp path (runs/<run>/tmp/probe_<hex>/captions/c.json)."""
    ing = _ingestor(tmp_path)
    bad = _candidate(tmp_path, "bad")
    write_poses(bad.pose, n_frames=100, width=960, height=544)
    good = _candidate(tmp_path, "good", seconds=5.0)
    write_poses(good.pose, n_frames=150, width=960, height=544, intrinsics=False)
    rejected, accepted = ing.ingest([bad, good], node_id="n1")
    texts = [*rejected.reasons, *accepted.warnings]
    assert accepted.accepted and accepted.warnings and not rejected.accepted
    assert not any(str(tmp_path) in t or "probe_" in t for t in texts), texts
    assert rejected.reasons[0].startswith("the pose file: 100 poses for 120 video frames")
