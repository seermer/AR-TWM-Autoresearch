import os
import subprocess
import sys

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.control import Control, kill_recorded_groups, mark_interrupted
from ar_kernel.run import RunContext
from ar_kernel.subproc import proc_start_time
from ar_kernel.telemetry.recorder import Recorder


def test_claim_ignores_a_recycled_pid(tmp_path):
    c = Control(tmp_path)
    (tmp_path / "control").mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # alive, but its start time is not the recorded one: a recycled pid
        (tmp_path / "control" / "loop.pid").write_text(f"{sleeper.pid} 1")
        c.claim()
        assert c.alive_pid() == os.getpid()
    finally:
        sleeper.kill()


def test_a_live_loop_is_detected_and_blocks_a_second_claim(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        Control(tmp_path).dir.mkdir(parents=True)
        (tmp_path / "control" / "loop.pid").write_text(f"{other.pid} {Control._start_time(other.pid)}")
        assert Control(tmp_path).alive_pid() == other.pid
        with pytest.raises(RuntimeError, match="already running"):
            Control(tmp_path).claim()
    finally:
        other.kill()


def test_stop_request_round_trip(tmp_path):
    c = Control(tmp_path)
    assert not c.stop_requested()
    c.request_stop()
    assert c.stop_requested()
    c.clear_stop()
    assert not c.stop_requested()


def test_recorded_orphan_groups_are_killed_and_recycled_pids_are_not(tmp_path):
    c = Control(tmp_path)
    (c.dir / "pgids").mkdir(parents=True)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (c.dir / "pgids" / str(orphan.pid)).write_text(proc_start_time(orphan.pid))
        (c.dir / "pgids" / str(bystander.pid)).write_text("1")          # recycled pid: start time differs
        assert kill_recorded_groups(c) == [orphan.pid]
        assert orphan.wait(timeout=40) is not None and bystander.poll() is None
        assert not any((c.dir / "pgids").iterdir())
    finally:
        orphan.kill(), bystander.kill()


def test_resume_marks_unfinished_nodes_interrupted_and_keeps_their_files(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr("ar_kernel.control.kill_run_containers",
                        lambda run_id, node=None: killed.append(node) or [f"ar-{run_id}-{node}-x"])
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.7, ["m"], {"m": 0.7})
    nodes.create("n1", "root", 1)                                   # left running by a forced stop
    work = tmp_path / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    work.mkdir(parents=True), (work / "notes.txt").write_text("keep me")
    (tmp_path / "staging" / "n1").mkdir(parents=True)
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "state.json").write_text('{"node": "n1", "phase": "train", "attempt": 1}')
    ctx = RunContext(run_dir=tmp_path, conn=conn, recorder=rec, gpus=[0, 1, 2, 3], metric_set=["m"],
                     case_ids=["1"], versions={})

    assert mark_interrupted(ctx, "resume after forced stop") == ["n1"]

    n1 = nodes.get("n1")
    assert n1["status"] == "interrupted" and "phase train" in n1["error"]
    assert (work / "notes.txt").read_text() == "keep me" and (tmp_path / "staging" / "n1").exists()
    assert killed == ["n1"] and nodes.get("root")["status"] == "scored"
    (event,) = [e for e in rec.read_events() if e["type"] == "node.interrupted"]
    assert rec.load_payload(event["payload"])["phase_reached"] == "train"
