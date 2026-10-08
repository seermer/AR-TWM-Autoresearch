import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.control import mark_interrupted
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    monkeypatch.setattr("ar_kernel.control.kill_run_containers", lambda run_id, node=None: [])


@pytest.fixture
def spawn():
    procs = []

    def start(run, max_nodes, slow):
        # sys.executable is the autoresearcher env's python running this test suite
        procs.append(subprocess.Popen([sys.executable, str(FIX / "fake_loop.py"), str(run), str(max_nodes), slow]))
        return procs[-1]
    yield start
    for proc in procs:                                   # never leave a stray loop behind
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def wait_phase(run, phase, timeout=60, key="phase"):
    state = run / "control" / "state.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.exists() and json.loads(state.read_text()).get(key) == phase:
            return
        time.sleep(0.2)
    raise AssertionError(f"never reached {phase}")


def resume_mark(run):
    ctx = RunContext(run_dir=run, conn=open_db(run), recorder=Recorder(run), gpus=[0, 1, 2, 3],
                     metric_set=["m"], case_ids=["1"], versions={})
    mark_interrupted(ctx, "resume")
    return ctx


def test_multi_node_run_telemetry_is_complete(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 4, "none")
    assert proc.wait(timeout=120) == 0
    events = Recorder(run).read_events()
    created = [e["child"] for e in events if e["type"] == "node.created"]
    ended = [e["child"] for e in events if e["type"] == "node.end"]
    assert created == ["n1", "n2", "n3", "n4"] and ended == ["n1", "n2", "n3"]
    assert [e for e in events if e["type"] == "run.stopped"]
    assert len({e["child"] for e in events if e["type"] == "select"}) == 4
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "n1": "scored", "n2": "invalid_code", "n3": "train_failed",
                        "n4": "running"}                  # its eval failed twice: the run stopped
    failed = sorted(e["message"].split()[0] for e in events if e["type"] == "alert" and e["kind"] == "node_failed")
    assert failed == ["n2", "n3"]


def test_sigterm_mid_phase_stops_within_seconds_and_resume_marks_interrupted(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "improve_recipe")
    wait_phase(run, "improve_recipe")
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0 and time.monotonic() - t0 < 10
    stopped = [e for e in Recorder(run).read_events() if e["type"] == "run.stopped"]
    assert "force stop" in Recorder(run).load_payload(stopped[-1]["payload"])["reason"]
    ctx = resume_mark(run)
    assert {n["node_id"]: n["status"] for n in NodeStore(ctx.conn).all()} == {"root": "scored", "n1": "interrupted"}
    assert (run / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace").exists()   # files kept


def test_kill_9_then_resume_starts_a_fresh_cycle_with_a_new_id(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "train")
    wait_phase(run, "train")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    resume_mark(run)
    proc = spawn(run, 1, "none")                           # max_nodes 1: exactly one more node
    assert proc.wait(timeout=120) == 0
    ids = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert ids == {"root": "scored", "n1": "interrupted", "n2": "scored"}   # n1 kept; ids not reused


def test_graceful_stop_file_finishes_the_current_node(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 5, "pace")                       # ~7 s per node, so the request lands in n1
    wait_phase(run, "n1", key="node")                  # as soon as n1 starts
    (run / "control" / "stop").touch()
    assert proc.wait(timeout=120) == 0
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "n1": "scored"}
