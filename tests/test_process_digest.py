import json
import sqlite3

import pytest

from ar_kernel.archive.activity import add_activity, add_call
from ar_kernel.archive.db import open_db
from ar_kernel.process_digest import MAX_BYTES, _shape, attempt_process, process_digest


def _act(run, kind, phase="improve_recipe", attempt=1, **row):
    add_activity(run, node="n1", phase=phase, attempt=attempt, kind=kind, **row)


def _call(run, conv, turn, phase="edit_self", attempt=1, role=None, transcript=None, tool=None):
    add_call(run, node="n1", phase=phase, attempt=attempt, role=role, tool=tool, conversation_id=conv, turn=turn,
             transcript=transcript, status=200, tokens=(1, 0, 1), cost_usd=None, mock=False, audit_hits=[])


def test_a_node_without_rows_has_an_empty_digest(tmp_path):
    assert process_digest(open_db(tmp_path), "n9") == {
        "phases": [], "tools": {}, "tool_errors": [], "gpu_jobs": {"run": 0, "failed": 0},
        "ingest": {"accepted": 0, "rejected": 0, "reasons": []}, "gates": {}}


def test_a_database_error_raises(tmp_path):
    conn = open_db(tmp_path)
    conn.execute("DROP TABLE activity")
    with pytest.raises(sqlite3.OperationalError):
        process_digest(conn, "n1")


def test_tool_errors_are_grouped_and_normalised(tmp_path):
    for i in range(2):
        _act(tmp_path, "tool", name="hf_download", ok=False,
             detail=_shape(f"403 (Request ID: Root={i}) job {'a' * 32} at /mnt/x/y/z.mp4 denied"))
    [row] = process_digest(open_db(tmp_path), "n1")["tool_errors"]
    assert row["tool"] == "hf_download" and row["count"] == 2
    assert "<id>" in row["example"] and "<path>" in row["example"] and "Request ID" not in row["example"]
    assert len(row["example"]) <= 1000


def test_each_phase_attempt_lists_its_conversations_snapshots_and_accepted_snapshot(tmp_path):
    for attempt, accepted in ((1, None), (2, "s3")):
        folder = f"transcripts/edit_self-{attempt}"
        for n, (conv, role, turns) in enumerate((("p", "edit_planner", 2), ("c", "coder", 1), ("r", None, 1)), 1):
            for turn in range(turns):
                _call(tmp_path, f"{conv}{attempt}", turn, attempt=attempt, role=role,
                      transcript=f"{folder}/{n:02d}-{role or 'unlabeled'}.jsonl")
            if conv == "c":
                _act(tmp_path, "snapshot", phase="edit_self", attempt=attempt, name=f"s{attempt + 1}")
        _act(tmp_path, "accepted", phase="edit_self", attempt=attempt, name=accepted)
    _call(tmp_path, "smoke", 0, phase="contract")                              # the kernel's own smoke run
    _act(tmp_path, "snapshot", phase="contract", name="smoke")
    conversations = lambda attempt: [
        {"role": "edit_planner", "turns": 2, "transcript": f"transcripts/edit_self-{attempt}/01-edit_planner.jsonl"},
        {"role": "coder", "turns": 1, "transcript": f"transcripts/edit_self-{attempt}/02-coder.jsonl"},
        {"role": "unlabeled", "turns": 1, "transcript": f"transcripts/edit_self-{attempt}/03-unlabeled.jsonl"}]
    conn = open_db(tmp_path)
    assert process_digest(conn, "n1")["phases"] == [
        {"phase": "edit_self", "attempt": 1, "conversations": conversations(1), "snapshots": ["s2"], "accepted": None},
        {"phase": "edit_self", "attempt": 2, "conversations": conversations(2), "snapshots": ["s3"], "accepted": "s3"}]
    assert attempt_process(conn, "n1", "edit_self", 2) == {
        "conversations": [{"role": "edit_planner", "turns": 2}, {"role": "coder", "turns": 1},
                          {"role": "unlabeled", "turns": 1}], "snapshots": ["s3"], "accepted": "s3"}
    assert attempt_process(conn, "n1", "improve_recipe", 1) == {"conversations": [], "snapshots": [], "accepted": None}


def test_conversations_are_in_start_order_past_99(tmp_path):
    folder = "transcripts/improve_recipe-1"
    for n in (100, 9, 10):                                     # ended out of order
        _call(tmp_path, f"c{n}", 0, phase="improve_recipe", transcript=f"{folder}/{n:02d}-unlabeled.jsonl")
    [phase] = process_digest(open_db(tmp_path), "n1")["phases"]
    assert [c["transcript"].split("/")[-1] for c in phase["conversations"]] == [
        "09-unlabeled.jsonl", "10-unlabeled.jsonl", "100-unlabeled.jsonl"]


def test_ask_calls_are_not_conversations(tmp_path):
    _call(tmp_path, "c", 0, phase="improve_recipe", transcript="transcripts/improve_recipe-1/01-unlabeled.jsonl")
    _call(tmp_path, "a", 0, phase="improve_recipe", tool="ask")
    [phase] = process_digest(open_db(tmp_path), "n1")["phases"]
    assert [c["transcript"] for c in phase["conversations"]] == ["transcripts/improve_recipe-1/01-unlabeled.jsonl"]


def test_the_digest_holds_no_seed_conventions_and_no_runtimes(tmp_path):
    _act(tmp_path, "gate", ok=False)
    _act(tmp_path, "gate", ok=True)
    d = process_digest(open_db(tmp_path), "n1")
    assert d["gates"] == {"failed": 1, "passed": 1}
    assert "local_errors" not in d and "compactions" not in json.dumps(d) and "rounds" not in d


def test_the_digest_stays_under_its_size_cap(tmp_path):
    for i in range(40):
        _act(tmp_path, "tool", name=f"tool{i % 3}", ok=False, detail=_shape(f"failure kind {i}: " + "x y " * 600))
    d = process_digest(open_db(tmp_path), "n1")
    assert len(json.dumps(d)) <= MAX_BYTES and 0 < len(d["tool_errors"]) <= 8


def test_tools_jobs_and_ingest(tmp_path):
    for tool, ok in (("data_ingest", True), ("data_ingest", False), ("hf_search", True)):
        _act(tmp_path, "tool", name=tool, ok=ok, detail=None if ok else "bad path")
    _act(tmp_path, "job", name="caption_videos", ok=True, detail="done")
    _act(tmp_path, "job", name="caption_videos", ok=False, detail="failed")
    _act(tmp_path, "job", name="caption_videos", ok=False, detail="canceled")
    _act(tmp_path, "ingest", phase="ingest", attempt=None, name="clip1", ok=True)
    for _ in range(2):
        _act(tmp_path, "ingest", phase="ingest", attempt=None, ok=False,
             detail="moving clips need poses/<id>.npz\nthe clip is too short")
    d = process_digest(open_db(tmp_path), "n1")
    assert d["tools"] == {"data_ingest": {"calls": 2, "errors": 1}, "hf_search": {"calls": 1, "errors": 0}}
    assert d["tool_errors"] == [{"tool": "data_ingest", "count": 1, "example": "bad path"}]
    assert d["gpu_jobs"] == {"run": 3, "failed": 1}
    assert d["ingest"] == {"accepted": 1, "rejected": 2,
                           "reasons": [{"reason": "moving clips need poses/<id>.npz", "count": 2},
                                       {"reason": "the clip is too short", "count": 2}]}
