"""Catch-alls, signal decisions, run.shutdown and the ends of a run."""
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import telemetry
from ar_kernel.control import Control, drive
from ar_kernel.telemetry import capture, registry


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.SPANS, "t.work", ("loop", "test"))
    telemetry.open(tmp_path)
    return tmp_path


def _of(run, kind, name=None):
    return [r for r in telemetry_records(run) if r["kind"] == kind and (name is None or r.get("name") == name)]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_an_exception_in_a_thread_becomes_a_record_under_the_root(run):
    root = telemetry.start("ar.run")
    capture.install()
    t = threading.Thread(target=lambda: 1 / 0, name="worker")
    t.start()
    t.join()
    [err] = _of(run, "error")
    assert err["span"] == root.id and err["handled_as"] == "uncaught: thread worker ends"
    assert telemetry_payload(run, err["error"])["type"] == "builtins.ZeroDivisionError"


def test_library_logging_at_every_level_and_warnings_are_captured(run):
    telemetry.start("ar.run")
    capture.install()
    logging.getLogger("some.lib").debug("connecting to %s", "host")
    warnings.warn("an old api", UserWarning)
    logs = _of(run, "log")
    assert {(r["stream"], r["level"], r["logger"], r["text"]) for r in logs if r["stream"] == "logging"} == {
        ("logging", "DEBUG", "some.lib", "connecting to host")}
    [warned] = [r for r in logs if r["stream"] == "warnings"]
    assert "UserWarning: an old api" in warned["text"] and warned["logger"] == "UserWarning"


def test_printed_lines_are_captured_per_line_and_still_shown(run, capsys):
    telemetry.start("ar.run")
    capture.install()
    print("first")
    print("par", end="")
    print("tial")
    print("to stderr", file=sys.stderr)
    assert [(r["stream"], r["text"]) for r in _of(run, "log")] == [
        ("stdout", "first"), ("stdout", "partial"), ("stderr", "to stderr")]
    shown = capsys.readouterr()
    assert shown.out == "first\npartial\n" and shown.err == "to stderr\n"


class _Loop:
    def __init__(self, run):
        self.graceful, self.run = threading.Event(), run


_KIT = SimpleNamespace(start=lambda: None, stop=lambda: None)


def test_a_signals_decision_lands_before_the_records_after_it(run):
    root = telemetry.start("ar.run")

    def loop_run():
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)
        telemetry.log("stdout", "after the signal")
        return "graceful stop"
    loop = _Loop(loop_run)
    assert drive(loop, _KIT, Control(run)) == "graceful stop" and loop.graceful.is_set()
    records = telemetry_records(run)
    [decision] = [r for r in records if r["kind"] == "decision"]
    after = next(r for r in records if r.get("text") == "after the signal")
    assert decision["seq"] < after["seq"] and decision["span"] == root.id
    assert (decision["name"], decision["chosen"], decision["reason"]) == ("run.signal", "graceful", "first Ctrl-C")
    assert telemetry_payload(run, decision["inputs"])["signal"] == "SIGINT"


def test_a_force_stop_ends_the_spans_it_unwinds_cancelled_with_the_force_stop(run):
    telemetry.start("ar.run")

    def loop_run():
        with telemetry.span("t.work"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
    assert drive(_Loop(loop_run), _KIT, Control(run)) == "force stop (SIGTERM)"
    [end] = _of(run, "end", "t.work")
    assert end["status"] == "cancelled" and telemetry_payload(run, end["error"])["type"] == "ar_kernel.control.ForceStop"
    [decision] = _of(run, "decision")
    assert (decision["chosen"], decision["reason"]) == ("force stop", "SIGTERM")


@pytest.mark.parametrize("ending", ["natural", "force", "crash"])
def test_run_shutdown_is_recorded_for_every_ending(run, ending):
    root = telemetry.start("ar.run")

    def loop_run():
        if ending == "force":
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        if ending == "crash":
            raise ValueError("a bug")
        return "max_nodes reached"
    try:
        drive(_Loop(loop_run), _KIT, Control(run))
    except ValueError:
        assert ending == "crash"
    [start], [end] = _of(run, "start", "run.shutdown"), _of(run, "end", "run.shutdown")
    assert start["parent"] == root.id and "inputs" not in start
    assert end["status"] == "ok" and telemetry_payload(run, end["output"]) == {"groups_killed": []}


def test_a_span_still_working_at_the_end_stays_open_and_its_owner_ends_it_later(run):
    root = telemetry.start("ar.run")
    go, started = threading.Event(), threading.Event()

    def work():
        with telemetry.span("t.work") as sp:
            started.set()
            go.wait()
            sp.output = "late"
    t = telemetry.thread(work, "worker")
    t.start()
    started.wait()
    root.end("ok", {"exit_code": 0})
    assert _of(run, "end", "t.work") == []
    go.set()
    t.join()
    [root_end], [late] = _of(run, "end", "ar.run"), _of(run, "end", "t.work")
    assert late["seq"] > root_end["seq"] and _of(run, "start", "t.work")[0]["parent"] == root.id


_FAILING_RUN = """
import sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from ar_kernel import telemetry
from ar_kernel.control import Control, drive, exit_process
from ar_kernel.loop import NATURAL_END

run, where = Path(sys.argv[1]), sys.argv[2]
w = telemetry.open(run)
telemetry.start("ar.run")

def broken(line):
    raise OSError(28, "No space left on device")

def loop_run():
    w._append = broken
    if where == "main":
        telemetry.log("stdout", "x")
    else:
        t = threading.Thread(target=telemetry.log, args=("stdout", "x"))
        t.start()
        t.join()
    time.sleep(20)
    return NATURAL_END

kit = SimpleNamespace(start=lambda: None, stop=lambda: (run / "cleaned").write_text("yes"))
reason = drive(SimpleNamespace(graceful=threading.Event(), run=loop_run), kit, Control(run))
print(reason, flush=True)
exit_process(0 if reason == NATURAL_END else 1)
"""


@pytest.mark.parametrize("where", ["main", "side"])
def test_a_writer_failure_force_stops_the_run_cleans_up_and_exits_1(tmp_path, where):
    proc = subprocess.run([sys.executable, "-c", _FAILING_RUN, str(tmp_path), where], capture_output=True,
                          text=True, timeout=60, cwd=str(Path(__file__).resolve().parents[1]),
                          env={**os.environ, "PYTHONPATH": "kernel:contract:."})
    assert proc.returncode == 1 and proc.stdout.strip() == "force stop (SIGTERM)", proc.stderr
    assert proc.stderr.count("telemetry: cannot write") == 1 and (tmp_path / "cleaned").exists()
    names = [r.get("name") for r in telemetry_records(tmp_path)]        # Control's files, then quiet after the failure
    assert names == [None, "ar.run", "fs.mkdir", "fs.mkdir", "fs.write", "fs.write"]
