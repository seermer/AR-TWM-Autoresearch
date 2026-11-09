"""Spans, the current span, thread hops, record kinds and the error object."""
import asyncio
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import zstandard

from ar_kernel import telemetry
from ar_kernel.control import ForceStop
from ar_kernel.telemetry import registry

api = importlib.import_module("ar_kernel.telemetry.api")
writer_mod = importlib.import_module("ar_kernel.telemetry.writer")


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_mod, "_writer", None)
    monkeypatch.setattr(api, "_root", None)
    monkeypatch.setitem(registry.SPANS, "t.outer", ("loop", "test"))
    monkeypatch.setitem(registry.SPANS, "t.inner", ("tools", "test"))
    monkeypatch.setitem(registry.DECISIONS, "t.pick", ("loop", "test"))
    monkeypatch.setitem(registry.ALERTS, "t.stall", ("monitor", "test"))
    telemetry.open(tmp_path)
    return tmp_path


def _records(run):
    out = []
    for path in sorted((run / "telemetry" / "records").glob("*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines()]
    return out[1:]                                   # after the writer record


def _payload(run, digest):
    path = run / "telemetry" / "payloads" / f"{digest}.json.zst"
    return json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))


def test_spans_nest_by_the_current_span_and_records_go_under_it(run):
    with telemetry.span("ar.run", {"argv": ["run"]}) as root:
        root.output = {"exit_code": 0}
        with telemetry.span("t.outer", {"i": 1}) as outer:
            outer.output = "done"
            with telemetry.span("t.inner") as inner:
                inner.output = 3
                telemetry.log("stdout", "a line")
            telemetry.decision("t.pick", {"n": 2}, ["a", "b"], "a", "first")
            telemetry.alert("t.stall", "warning", "quiet", {"s": 9})
            telemetry.sample(gpu_mem=[1, 2])
    starts = {r["name"]: r for r in _records(run) if r["kind"] == "start"}
    assert starts["ar.run"]["parent"] is None and starts["t.outer"]["parent"] == root.id
    assert starts["t.inner"]["parent"] == outer.id and starts["t.inner"]["component"] == "tools"
    assert "inputs" not in starts["t.inner"] and _payload(run, starts["t.outer"]["inputs"]) == {"i": 1}
    by_kind = {r["kind"]: r for r in _records(run) if r["kind"] not in ("start", "end")}
    assert by_kind["log"]["span"] == inner.id and by_kind["log"]["component"] == "tools"
    assert by_kind["log"]["text"] == "a line" and by_kind["log"]["stream"] == "stdout"
    assert by_kind["decision"]["span"] == outer.id and _payload(run, by_kind["decision"]["inputs"]) == {"n": 2}
    assert (by_kind["decision"]["options"], by_kind["decision"]["chosen"]) == (["a", "b"], "a")
    assert by_kind["alert"]["name"] == "t.stall" and by_kind["alert"]["component"] == "monitor"
    assert by_kind["sample"]["gpu_mem"] == [1, 2] and by_kind["sample"]["span"] == outer.id
    ends = [r for r in _records(run) if r["kind"] == "end"]
    assert [(e["name"], e["status"], _payload(run, e["output"])) for e in ends] == [
        ("t.inner", "ok", 3), ("t.outer", "ok", "done"), ("ar.run", "ok", {"exit_code": 0})]
    assert telemetry.current() is None


def test_an_explicit_parent_wins_over_the_current_span(run):
    with telemetry.span("ar.run"):
        with telemetry.span("t.outer") as outer:
            pass
        with telemetry.span("t.inner", parent=outer):
            pass
    starts = {r["name"]: r for r in _records(run) if r["kind"] == "start"}
    assert starts["t.inner"]["parent"] == outer.id


def test_records_with_no_current_span_go_under_the_root(run):
    root = telemetry.start("ar.run")
    t = threading.Thread(target=lambda: telemetry.log("logging", "from a library thread"))
    t.start()
    t.join()
    [log] = [r for r in _records(run) if r["kind"] == "log"]
    assert log["span"] == root.id


def test_a_status_set_by_the_body_is_kept(run):
    with telemetry.span("ar.run"):
        with telemetry.span("t.outer") as sp:
            sp.status, sp.output = "refused", "not allowed"
        with pytest.raises(TimeoutError):
            with telemetry.span("t.inner") as sp:
                sp.status = "timeout"
                raise TimeoutError("slow")
    ends = {r["name"]: r for r in _records(run) if r["kind"] == "end"}
    assert ends["t.outer"]["status"] == "refused" and ends["t.inner"]["status"] == "timeout"
    assert _payload(run, ends["t.inner"]["error"])["message"] == "slow"


@pytest.mark.parametrize("exc, status", [(ValueError("bad"), "error"), (ForceStop("SIGTERM"), "cancelled"),
                                         (KeyboardInterrupt(), "cancelled"),
                                         (asyncio.CancelledError(), "cancelled")])
def test_each_exception_maps_to_its_status_and_is_carried_and_reraised(run, exc, status):
    with pytest.raises(type(exc)):
        with telemetry.span("ar.run"):
            raise exc
    [end] = [r for r in _records(run) if r["kind"] == "end"]
    error = _payload(run, end["error"])
    assert end["status"] == status and error["type"].endswith(type(exc).__qualname__) and "output" not in end


def test_a_span_started_in_one_thread_ends_in_another(run):
    with telemetry.span("ar.run"):
        job = telemetry.start("t.outer", {"job": 1})
    t = threading.Thread(target=job.end, args=("ok", {"result": 2}))
    t.start()
    t.join()
    end = [r for r in _records(run) if r["kind"] == "end" and r["name"] == "t.outer"][0]
    assert end["thread"] != _records(run)[1]["thread"] and _payload(run, end["output"]) == {"result": 2}


def test_an_unregistered_name_raises_even_without_a_writer(run, monkeypatch):
    monkeypatch.setattr(writer_mod, "_writer", None)
    for call in (lambda: telemetry.start("no.such"), lambda: telemetry.decision("no.such", None, [], None, ""),
                 lambda: telemetry.alert("no.such", "warning", "x")):
        with pytest.raises(ValueError, match="unregistered telemetry name 'no.such'"):
            call()


def test_with_no_writer_spans_still_run_their_body_and_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_mod, "_writer", None)
    monkeypatch.setattr(api, "_root", None)
    with telemetry.span("ar.run") as sp:
        sp.output = 1
        telemetry.log("stdout", "x")
        telemetry.error(ValueError("y"), "ignored")
    assert not (tmp_path / "telemetry").exists()


def test_the_error_object_has_frames_with_redacted_locals_and_the_chain(run):
    telemetry.writer().add_redaction("hunter2")

    def inner(password):
        missing = run / "nope"
        try:
            missing.read_text()
        except FileNotFoundError as exc:
            raise RuntimeError("could not read") from exc
    with telemetry.span("ar.run"):
        try:
            inner("hunter2")
        except RuntimeError as exc:
            telemetry.error(exc, "ignored: a test")
    [err] = [r for r in _records(run) if r["kind"] == "error"]
    error = _payload(run, err["error"])
    assert err["handled_as"] == "ignored: a test" and error["type"] == "builtins.RuntimeError"
    frame = next(f for f in error["frames"] if f["function"] == "inner")
    assert frame["locals"]["password"] == "'[REDACTED]'" and frame["code"].startswith("raise RuntimeError")
    cause = error["cause"]
    assert cause["type"] == "builtins.FileNotFoundError" and cause["errno"] == 2 and cause["filename"] == str(run / "nope")
    assert "context" not in error                                  # the same exception as the cause


def test_thread_and_submit_carry_the_span_and_a_pool_thread_keeps_no_stale_span(run):
    seen = []
    with telemetry.span("ar.run"):
        with telemetry.span("t.outer") as outer:
            t = telemetry.thread(lambda: seen.append(telemetry.current()), "worker")
            t.start()
            t.join()
            with ThreadPoolExecutor(1) as pool:
                assert telemetry.submit(pool, telemetry.current).result() is outer
                assert pool.submit(telemetry.current).result() is None
    assert seen == [outer]
