import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.telemetry.recorder import Recorder

FIX = Path(__file__).parent / "fixtures"


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


def test_multi_node_run_telemetry_is_complete(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 4, "none")
    assert proc.wait(timeout=120) == 0
    events = Recorder(run).read_events()
    created = [e["child"] for e in events if e["type"] == "node.created"]
    ended = [e["child"] for e in events if e["type"] == "node.end"]
    assert created == ["node0001", "node0002", "node0003", "node0004"] and ended == ["node0001", "node0002", "node0003"]
    assert [e for e in events if e["type"] == "run.stopped"]
    assert len({e["child"] for e in events if e["type"] == "select"}) == 4
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "node0001": "scored", "node0002": "invalid_code", "node0003": "train_failed",
                        "node0004": "running"}                  # its eval failed twice: the run stopped
    failed = sorted(e["message"].split()[0] for e in events if e["type"] == "alert" and e["kind"] == "node_failed")
    assert failed == ["node0002", "node0003"]


def _resumed(run) -> list[str]:
    return [e["message"] for e in Recorder(run).read_events() if e["type"] == "alert" and e["kind"] == "node_resumed"]


def test_sigterm_mid_phase_stops_within_seconds_and_resume_restarts_that_phase(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "improve_recipe")
    wait_phase(run, "improve_recipe")
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0 and time.monotonic() - t0 < 10
    stopped = [e for e in Recorder(run).read_events() if e["type"] == "run.stopped"]
    assert "force stop" in Recorder(run).load_payload(stopped[-1]["payload"])["reason"]
    assert NodeStore(open_db(run)).get("node0001")["status"] == "running"
    proc = spawn(run, 1, "pace")                           # max_nodes 1: node0001 is the one node
    assert proc.wait(timeout=120) == 0
    assert {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()} == {"root": "scored", "node0001": "scored"}
    assert _resumed(run) == ["node0001 starts again at improve_recipe attempt 1"]


def test_kill_9_in_training_then_resume_trains_the_same_attempt_again(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "train")
    wait_phase(run, "train")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    proc = spawn(run, 1, "pace")
    assert proc.wait(timeout=120) == 0
    nodes = NodeStore(open_db(run))
    assert {n["node_id"]: n["status"] for n in nodes.all()} == {"root": "scored", "node0001": "scored"}
    assert _resumed(run) == ["node0001 starts again at train attempt 1"]
    assert [a["idx"] for a in nodes.attempts("node0001", "improve_recipe")] == [1]     # the attempt was not charged


def test_graceful_stop_file_finishes_the_current_node(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 5, "pace")                       # ~7 s per node, so the request lands in node0001
    wait_phase(run, "node0001", key="node")                  # as soon as node0001 starts
    (run / "control" / "stop").touch()
    assert proc.wait(timeout=120) == 0
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "node0001": "scored"}
