import pytest
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import Gate
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test fixture"}

@pytest.fixture
def gate_env(tmp_path):
    conn = open_db(tmp_path)
    rec = Recorder(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, rec)
    clip_ids = []
    for i in range(8):
        # Distinct durations: byte-identical inputs collapse to ONE content-addressed
        # clip, which is how this fixture used to hand the gate a single clip listed
        # eight times and still pass the clips-vs-GPUs check.
        seconds = 4.0 + 0.5 * i
        stage = tmp_path / "staging" / f"c{i}"
        clip_ids.append(ing.ingest([Candidate(
            video=make_mp4(stage / "v.mp4", seconds=seconds), caption=write_caption(stage / "c.json"),
            pose=write_poses(stage / "p.npz", n_frames=int(seconds * 30)), camera_motion="moving",
            provenance=PROV)], node_id="n1")[0].clip_id)
    assert len(set(clip_ids)) == 8, "fixture must produce 8 distinct clips"
    commits = CommitStore(conn, ing.blobs, ing.clips)
    commit = commits.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": clip_ids}}, "v", node_id="n1")
    return Gate(CFG, commits, rec), commit, tmp_path

def test_valid_recipe_passes_every_check(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.max_steps": 2, "optimizer.grad_accum_steps": 1,
                         "optimizer.epochs": 50},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert result.ok, result.failures
    assert result.resolved_path.exists()

def test_non_tunable_key_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"spatial_memory.enabled": False}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("not tunable" in f for f in result.failures)

def test_resolution_outside_the_allowlist_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"sample.height": 544, "sample.width": 960}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("resolution" in f for f in result.failures)

def test_lora_pair_outside_the_allowlist_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"lora.rank": 8, "lora.alpha": 8}, commit, None, "n1",
                        tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("lora" in f for f in result.failures)

def test_unchanged_data_commit_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({}, commit, commit, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("data commit" in f for f in result.failures)

def test_recommitted_identical_data_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    manifest = gate.commits.manifest(commit)["datasets"]
    twin = gate.commits.commit(None, manifest, "same data, new message", node_id="n2")
    assert twin != commit
    result = gate.check({}, twin, commit, "n2", tmp_path / "n2", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("identical to the parent" in f for f in result.failures)

def test_zero_steps_per_epoch_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.grad_accum_steps": 4, "optimizer.max_steps": 10},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("steps_per_epoch" in f for f in result.failures)

def test_epoch_budget_below_max_steps_is_rejected(gate_env):
    gate, commit, tmp_path = gate_env
    result = gate.check({"optimizer.grad_accum_steps": 1, "optimizer.max_steps": 100,
                         "optimizer.epochs": 2},
                        commit, None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("epochs" in f for f in result.failures)

def test_fewer_clips_than_gpus_is_rejected(tmp_path):
    conn = open_db(tmp_path)
    rec = Recorder(tmp_path)
    ing = Ingestor(CFG, tmp_path, conn, rec)
    stage = tmp_path / "staging" / "solo"
    clip_id = ing.ingest([Candidate(video=make_mp4(stage / "v.mp4"),
                                    caption=write_caption(stage / "c.json"),
                                    pose=write_poses(stage / "p.npz", n_frames=120),
                                    camera_motion="moving", provenance=PROV)],
                         node_id="n1")[0].clip_id
    commits = CommitStore(conn, ing.blobs, ing.clips)
    commit = commits.commit(None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                           "weight": 1.0, "clips": [clip_id]}}, "v", node_id="n1")
    result = Gate(CFG, commits, rec).check({}, commit, None, "n1", tmp_path / "n1",
                                           tmp_path, [0, 1, 2, 3])
    assert not result.ok and any("clips" in f for f in result.failures)
