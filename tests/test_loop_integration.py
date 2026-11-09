import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.telemetry import tree

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


def ended(run, code: int) -> list[dict]:
    """The last `ar.run` ended with exit code `code`, and the records hold no telemetry bug; the records."""
    records = telemetry_records(run)
    end = [r for r in records if r["kind"] == "end" and r["name"] == "ar.run"][-1]
    assert (end["status"], telemetry_payload(run, end["output"])) == ("ok" if code == 0 else "error", {"exit_code": code})
    assert [r["name"] for r in records if r["kind"] == "end" and r["writer"] == end["writer"]][-2:] == \
        ["run.shutdown", "ar.run"]
    assert tree.check(tree.load(run)) == []
    return records


def _kids(trace, node, name):
    return [c for c in node.children if isinstance(c, tree.Node) and c.start["name"] == name]


def _decisions(node, name):
    return [c["chosen"] for c in node.children if isinstance(c, dict) and c.get("name") == name]


def test_a_multi_node_run_records_every_node(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 4, "none")
    assert proc.wait(timeout=120) == 1                      # StopRun
    records = ended(run, 1)
    alerts = [(r["name"], r["level"], r["message"]) for r in records if r["kind"] == "alert"]
    failed = [m for n, _, m in alerts if n == "loop.node_failed"]
    assert [m.split(":")[0] for m in failed] == ["node0002 ended invalid_code", "node0003 ended train_failed"]
    assert {n for n, level, _ in alerts if n == "loop.phase_retry" and level == "warning"} == {"loop.phase_retry"}
    [(_, level, message)] = [a for a in alerts if a[0] == "loop.phase_failed"]
    assert level == "error" and message.startswith("node0004 eval: KernelFault") and message.endswith("failed twice")
    trace = tree.load(run)
    [loop] = [n for n in trace.spans.values() if n.start["name"] == "loop"]
    assert trace.payload(loop.end["output"])["reason"].startswith("node0004 eval: KernelFault: wbench gpu failed")
    [root] = _kids(trace, loop, "root.ensure")
    assert _decisions(root, "root.how") == ["evaluate"]
    iterations = _kids(trace, loop, "loop.iteration")
    assert [_decisions(i, "loop.max_nodes") for i in iterations] == [["go on"]] * 4
    nodes = [n for i in iterations for n in _kids(trace, i, "node")]
    statuses = [(trace.payload(n.start["inputs"])["node"], n.end["status"],
                 (trace.payload(n.end.get("output")) or {}).get("status")) for n in nodes]
    assert statuses == [("node0001", "ok", "scored"), ("node0002", "ok", "invalid_code"),
                        ("node0003", "ok", "train_failed"), ("node0004", "error", None)]   # StopRun ends node0004
    attempts = {trace.payload(n.start["inputs"])["node"]: [(a["phase"], a["attempt"]) for a in
                (trace.payload(x.start["inputs"]) for x in _kids(trace, n, "phase.attempt"))] for n in nodes}
    assert attempts["node0001"] == [("improve_recipe", 1), ("improve_recipe", 2), ("train", 2), ("eval", 0)]
    assert attempts["node0002"] == [("edit_self", 1), ("edit_self", 2), ("edit_self", 3)]

    def tries(node, phase):                                # each try is its own span, ended by its failure
        return [(trace.payload(x.start["inputs"])["try"], x.end["status"], _decisions(x, "phase.failure"))
                for x in _kids(trace, node, "phase.attempt") if trace.payload(x.start["inputs"])["phase"] == phase]
    assert tries(nodes[2], "train") == [(1, "error", ["retry"]), (2, "error", ["node fault"])]
    assert tries(nodes[3], "eval") == [(1, "error", ["retry"]), (2, "error", ["stop the run"])]
    finish = {trace.payload(n.start["inputs"])["node"]: _decisions(_kids(trace, n, "node.finish")[0], "node.status")
              for n in nodes[:3]}
    assert finish == {"node0001": ["scored"], "node0002": ["invalid_code"], "node0003": ["train_failed"]}
    assert _decisions(nodes[0], "edit_self.skip") == ["skip"] and _decisions(nodes[1], "edit_self.skip") == ["run edit_self"]
    conn = open_db(run)
    assert len({r[0] for r in conn.execute("SELECT child_id FROM selection_events")}) == 4
    statuses = {n["node_id"]: n["status"] for n in NodeStore(conn).all()}
    assert statuses == {"root": "scored", "node0001": "scored", "node0002": "invalid_code", "node0003": "train_failed",
                        "node0004": "running"}                  # its eval failed twice: the run stopped


def test_sigterm_mid_phase_stops_within_seconds_and_resume_restarts_that_phase(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "improve_recipe")
    wait_phase(run, "improve_recipe")
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 1 and time.monotonic() - t0 < 10
    records = ended(run, 1)
    assert [r["chosen"] for r in records if r["kind"] == "decision" and r["name"] == "run.signal"] == ["force stop"]
    assert NodeStore(open_db(run)).get("node0001")["status"] == "running"
    proc = spawn(run, 1, "pace")                           # max_nodes 1: node0001 is the one node
    assert proc.wait(timeout=120) == 0
    records = ended(run, 0)
    [resumed] = [r for r in records if r["kind"] == "decision" and r["name"] == "run.resume_point"]
    assert [r["message"] for r in records if r["kind"] == "alert" and r["name"] == "loop.node_resumed"] == [
        "node0001 starts again at improve_recipe attempt 1"]
    assert resumed["chosen"] == {"node": "node0001", "phase": "improve_recipe", "attempt": 1}
    nodes = NodeStore(open_db(run))
    assert {n["node_id"]: n["status"] for n in nodes.all()} == {"root": "scored", "node0001": "scored"}
    assert [a["idx"] for a in nodes.attempts("node0001", "improve_recipe")] == [1]     # the attempt was not charged


def test_kill_9_in_training_then_resume_trains_the_same_attempt_again(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 3, "train")
    wait_phase(run, "train")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    proc = spawn(run, 1, "pace")
    assert proc.wait(timeout=120) == 0
    ended(run, 0)
    nodes = NodeStore(open_db(run))
    assert {n["node_id"]: n["status"] for n in nodes.all()} == {"root": "scored", "node0001": "scored"}
    assert [a["idx"] for a in nodes.attempts("node0001", "improve_recipe")] == [1]     # the attempt was not charged


def test_graceful_stop_file_finishes_the_current_node(tmp_path, spawn):
    run = tmp_path / "run"
    proc = spawn(run, 5, "pace")                       # ~7 s per node, so the request lands in node0001
    wait_phase(run, "node0001", key="node")                  # as soon as node0001 starts
    (run / "control" / "stop").touch()
    assert proc.wait(timeout=120) == 1
    records = ended(run, 1)
    assert [r["chosen"] for r in records if r["kind"] == "decision" and r["name"] == "run.stop_file"] == ["graceful"]
    statuses = {n["node_id"]: n["status"] for n in NodeStore(open_db(run)).all()}
    assert statuses == {"root": "scored", "node0001": "scored"}


def test_a_crash_in_loop_code_ends_the_run_with_its_error(tmp_path, spawn):
    run = tmp_path / "run"
    assert spawn(run, 1, "crash").wait(timeout=60) == 1
    [error] = [r for r in ended(run, 1) if r["kind"] == "error"]
    assert error["handled_as"] == "exit code 1"
    assert telemetry_payload(run, error["error"])["message"] == "a bug in the loop"
