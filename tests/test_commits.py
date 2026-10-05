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
    with pytest.raises(CommitError) as refused:
        cs.commit(None, {"static": {"format": "video_caption_static", "prompt_mode": None,
                                    "weight": 1.0, "clips": ids}}, "bad", node_id="n1")
    # every clip by its whole id (what data_query takes), with the formats it can be committed as
    assert f"{len(ids)} of {len(ids)} clips are not eligible for video_caption_static" in str(refused.value)
    assert all(f"{i} (eligible for video_caption_camera" in str(refused.value) for i in ids[:5])

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


def _cam(ids):
    return {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                    "weight": 1.0, "clips": list(ids)}}


def test_rematerializing_a_smaller_commit_removes_stale_clips(store):
    """A gate retry reuses node_dir/view. A clip dropped between
    attempts must not survive in the view, or the node trains on data its
    recorded commit does not contain -- the loader lists videos/ directly."""
    cs, ids, tmp = store
    big = cs.commit(None, _cam(ids), "both clips", node_id="n1")
    small = cs.commit(None, _cam(ids[:1]), "one clip", node_id="n1")
    view = tmp / "node" / "view"
    cs.materialize(big, view)
    assert len(list((view / "cam" / "videos").iterdir())) == 2
    cs.materialize(small, view)
    videos = sorted(p.name for p in (view / "cam" / "videos").iterdir())
    assert videos == [f"{ids[0]}.mp4"]
    assert sorted(p.name for p in (view / "cam" / "captions").iterdir()) == [f"{ids[0]}.json"]


def test_rematerializing_drops_a_dataset_absent_from_the_new_commit(store):
    cs, ids, tmp = store
    two = cs.commit(None, {**_cam(ids), "cam2": _cam(ids[1:])["cam"]}, "two sets", node_id="n1")
    one = cs.commit(None, _cam(ids), "one set", node_id="n1")
    view = tmp / "node" / "view"
    cs.materialize(two, view)
    cs.materialize(one, view)
    assert sorted(p.name for p in view.iterdir() if not p.name.startswith(".")) == ["cam"]


def test_materialize_refuses_to_replace_a_directory_it_did_not_build(store):
    """The replace step deletes the old destination, so it must never touch a
    directory that is not a kernel-built view."""
    cs, ids, tmp = store
    precious = tmp / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("do not delete")
    with pytest.raises(CommitError, match="not a kernel-built view"):
        cs.materialize(cs.commit(None, _cam(ids), "x", node_id="n1"), precious)
    assert (precious / "keep.txt").read_text() == "do not delete"


def test_rematerialize_leaves_blobs_intact(store):
    """Removing the old view unlinks hardlinks only; the blob store keeps its copy."""
    cs, ids, tmp = store
    view = tmp / "node" / "view"
    cs.materialize(cs.commit(None, _cam(ids), "both", node_id="n1"), view)
    cs.materialize(cs.commit(None, _cam(ids[:1]), "one", node_id="n1"), view)
    clip = cs.clips.get(ids[1])
    assert cs.blobs.path(clip["video_digest"], "video").exists()


def test_duplicate_clip_in_a_dataset_is_rejected(store):
    """A clip listed twice would inflate the clips-vs-GPUs count the gate checks
    and silently reweight sampling. Upsampling is what dataset weights are for."""
    cs, ids, _ = store
    with pytest.raises(CommitError, match="more than once"):
        cs.commit(None, _cam([ids[0], ids[0], ids[1]]), "dup", node_id="n1")


@pytest.mark.parametrize("weight", [float("inf"), float("nan"), "heavy"])
def test_non_finite_or_non_numeric_weight_is_rejected(store, weight):
    """Weight inf passed validation, then steps_per_epoch raised
    'cannot convert float NaN to integer' inside the gate."""
    cs, ids, _ = store
    with pytest.raises(CommitError, match="weight"):
        cs.commit(None, {"cam": {**_cam(ids)["cam"], "weight": weight}}, "w", node_id="n1")


def test_commit_with_no_enabled_dataset_is_rejected(store):
    """All-zero weights left steps_per_epoch taking max() of nothing."""
    cs, ids, _ = store
    with pytest.raises(CommitError, match="no dataset"):
        cs.commit(None, {"cam": {**_cam(ids)["cam"], "weight": 0.0}}, "off", node_id="n1")
