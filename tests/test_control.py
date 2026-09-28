import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.control import Control, drive, kill_recorded_groups, mark_interrupted
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
        threading.Thread(target=orphan.wait, daemon=True).start()       # reap it as init would
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


def test_workers_of_a_dead_leader_are_still_killed(tmp_path):
    """Verification-log finding 6: `conda run` (the leader) exits, its workers keep the GPU."""
    c = Control(tmp_path)
    (c.dir / "pgids").mkdir(parents=True)
    code = ("import subprocess, sys\n"
            "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(w.pid, flush=True)\n")
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    started = proc_start_time(leader.pid)
    try:
        worker = int(leader.stdout.readline())
        leader.wait()
        leader.stdout.close()
        (c.dir / "pgids" / str(leader.pid)).write_text(started)
        assert proc_start_time(leader.pid) is None and proc_start_time(worker) is not None
        assert kill_recorded_groups(c) == [leader.pid]
        deadline = time.monotonic() + 5
        while proc_start_time(worker) is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc_start_time(worker) is None
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class _FakeLoop:
    def __init__(self, run_dir, run):
        self.graceful, self.ctx, self.run = threading.Event(), SimpleNamespace(run_dir=run_dir), run


class _Kit:
    def __init__(self, stop=lambda: None):
        self.start, self.stop = (lambda: None), stop


def _stopped(rec):
    return [rec.load_payload(e["payload"])["reason"] for e in rec.read_events() if e["type"] == "run.stopped"]


def test_a_signal_during_cleanup_does_not_abort_it(tmp_path, monkeypatch):
    monkeypatch.setattr("ar_kernel.control.kill_run_containers", lambda run_id, node=None: [])
    rec, c = Recorder(tmp_path), Control(tmp_path)

    def run():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        return "not reached"

    def slow_stop():                             # the operator keeps pressing while cleanup runs
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGINT):
            os.kill(os.getpid(), sig)
        time.sleep(0.3)

    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    assert drive(_FakeLoop(tmp_path, run), _Kit(slow_stop), c, rec) == "force stop (SIGTERM)"
    assert _stopped(rec) == ["force stop (SIGTERM)"]
    assert c.alive_pid() is None and not (c.dir / "loop.pid").exists()
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_a_crash_is_recorded_and_reraised(tmp_path):
    rec, c = Recorder(tmp_path), Control(tmp_path)

    def run():
        raise ValueError("boom")
    with pytest.raises(ValueError):
        drive(_FakeLoop(tmp_path, run), _Kit(), c, rec)
    assert _stopped(rec) == ["crashed: ValueError: boom"] and c.alive_pid() is None


# ---- the process really exits (acceptance_20260928: a force stop left the kernel alive) ----

def test_exit_process_ends_the_process_despite_a_stuck_worker_thread(tmp_path):
    """A tool call ran in an asyncio.to_thread worker (hf_download under HF rate limiting). Python's
    exit joins those workers, so after the stop the kernel lived on with its pid file gone."""
    code = (
        "import asyncio, sys, threading, time\n"
        "from ar_kernel.control import exit_process\n"
        "from ar_kernel.telemetry.recorder import Recorder\n"
        f"rec = Recorder({str(tmp_path)!r})\n"
        "threading.Thread(target=lambda: asyncio.run(asyncio.to_thread(time.sleep, 60)), name='tools').start()\n"
        "time.sleep(0.5)\n"
        "print('exiting', flush=True)\n"
        "exit_process(3, rec)\n")
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
                          cwd=str(Path(__file__).resolve().parents[1]),
                          env={**os.environ, "PYTHONPATH": "kernel:contract:."})
    assert proc.returncode == 3 and "exiting" in proc.stdout, proc.stderr
    assert time.monotonic() - started < 15
    alerts = [e for e in Recorder(tmp_path).read_events() if e["type"] == "alert"]
    assert alerts and alerts[-1]["kind"] == "shutdown" and "tools" in alerts[-1]["message"]
