from types import SimpleNamespace

import pytest

from ar_kernel.archive.clips import ClipStore, hidden
from ar_kernel.archive.commits import BadClips, CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.data_tools import pool_clips
from ar_kernel.train.gate import Gate

CFG = KernelConfig.load()


def _clip(clip_id: str, node: str) -> dict:
    return {"clip_id": clip_id, "video_digest": "v", "caption_digest": "c", "annotation_digest": "a",
            "pose_digest": "p", "metadata": {}, "warnings": [], "derived_from": {"kind": "model", "name": "x"},
            "ingested_by": node}


def _pool(conn) -> dict:
    return {c["clip_id"]: c["ingested_by"] for c in pool_clips(conn)}


@pytest.mark.parametrize("label, is_hidden", [
    ("node0005", False), ("node0005fail1", True), ("node0005fail12", True), ("node0005quarantine", True),
    ("node0005interrupted", True), ("node0005fail0", False), ("fail1", False), ("node0005failed", False)])
def test_only_the_exact_marks_hide_a_clip(label, is_hidden):
    assert hidden(label) is is_hidden


def test_hide_marks_a_nodes_clips_and_quarantine_also_takes_its_failed_attempts(tmp_path):
    conn = open_db(tmp_path)
    clips = ClipStore(conn)
    for clip_id, node in (("a", "node0005"), ("b", "node0006"), ("c", "node00050")):
        clips.add(_clip(clip_id, node))
    assert clips.hide("node0005", "fail1") == 1
    clips.add(_clip("d", "node0005"))                                  # attempt 2
    assert _pool(conn) == {"b": "node0006", "c": "node00050", "d": "node0005"}
    assert clips.hide("node0005", "quarantine") == 2
    assert {c["clip_id"]: c["ingested_by"] for c in clips.all()}["a"] == "node0005quarantine"
    assert _pool(conn) == {"b": "node0006", "c": "node00050"}


def test_a_reingest_moves_a_hidden_clip_and_leaves_a_visible_one_with_its_owner(tmp_path):
    conn = open_db(tmp_path)
    clips = ClipStore(conn)
    clips.add(_clip("a", "node0005")), clips.add(_clip("b", "node0006"))
    clips.hide("node0005", "fail1")
    assert clips.add(_clip("a", "node0007")) == "node0005fail1"
    assert clips.add(_clip("b", "node0007")) is None
    assert _pool(conn) == {"a": "node0007", "b": "node0006"}


def test_a_data_commit_refuses_clips_not_in_the_pool(tmp_path):
    conn = open_db(tmp_path)
    clips = ClipStore(conn)
    clips.add(_clip("a" * 64, "node0005")), clips.add(_clip("b" * 64, "node0006"))
    clips.hide("node0005", "fail1")
    with pytest.raises(BadClips) as exc:
        CommitStore(conn, None, clips).commit(None, {"d": {"clips": ["a" * 64, "b" * 64]}}, "m", node_id="node0007")
    assert exc.value.bad == {0: "not in the pool"}


def test_the_gate_refuses_a_data_commit_holding_hidden_clips(tmp_path):
    ids = [f"{i}" * 64 for i in range(8)]
    store = {i: {"ingested_by": "node0005fail1" if i == ids[0] else "node0005"} for i in ids}
    commits = SimpleNamespace(manifest=lambda c: {"datasets": {"d": {"weight": 1.0, "clips": ids}}},
                              clips=SimpleNamespace(get=store.__getitem__))
    result = Gate(CFG, commits, Recorder(tmp_path)).check({}, "c", None, "n1", tmp_path / "n1", tmp_path, [0, 1, 2, 3])
    assert not result.ok and "dataset d has 1 clips that are not in the pool" in result.failures
