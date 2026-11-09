"""Faults with no other test of their own, each followed by a clean `ar trace --check`."""
import os
import signal

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import telemetry
from ar_kernel.subproc import run_in_env
from ar_kernel.telemetry import tree


@pytest.fixture
def run(tmp_path):
    telemetry.open(tmp_path)
    telemetry.start("ar.run")
    return tmp_path


def _ends(run, name):
    return [r for r in telemetry_records(run) if r["kind"] == "end" and r["name"] == name]


def test_a_failing_subprocess_ends_its_span_with_the_exit_code(run):
    out = run_in_env("autoresearcher", ["bash", "-c", "echo boom >&2; exit 3"], cwd=run)
    assert out.returncode == 3
    [end] = _ends(run, "process")
    assert end["status"] == "error" and telemetry_payload(run, end["output"])["exit_code"] == 3
    assert "boom" in [r.get("text") for r in telemetry_records(run) if r["kind"] == "log"]
    assert tree.check(tree.load(run)) == []


def test_an_exception_during_a_process_wait_kills_it_and_ends_its_span(run):
    class Interrupting:
        def is_set(self):
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_in_env("autoresearcher", ["sleep", "600"], cwd=run, cancel=Interrupting(), poll_s=0.1)
    [end] = _ends(run, "process")
    kills = [r for r in telemetry_records(run) if r["kind"] == "start" and r["name"] == "process.kill"]
    assert end["status"] == "cancelled" and kills
    assert tree.check(tree.load(run)) == []


def test_enospc_at_the_writer_keeps_the_records_before_it_and_a_clean_trace(run, monkeypatch):
    kills = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(sig))
    with telemetry.span("loop") as sp:
        telemetry.log("stdout", "before the disk filled")

        def full(line):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(telemetry.writer(), "_append", full)
        telemetry.log("stdout", "after")
        sp.output = {}
    records = telemetry_records(run)
    assert [r.get("text") for r in records if r["kind"] == "log"] == ["before the disk filled"]
    assert kills == [signal.SIGTERM] and not _ends(run, "loop")         # its end was never written: open
    assert tree.check(tree.load(run)) == []
