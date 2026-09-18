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
    with pytest.raises(CommitError, match="not eligible"):
        cs.commit(None, {"static": {"format": "video_caption_static", "prompt_mode": None,
                                    "weight": 1.0, "clips": ids}}, "bad", node_id="n1")

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
