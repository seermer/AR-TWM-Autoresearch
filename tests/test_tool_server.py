"""Exercise the tool server the way a container does: over a Unix socket, with
the ar_contract client configuration."""
from conftest import client_dir, recorded
import asyncio
import json
import os

import pytest
from mcp.server.mcpserver import Context

from ar_kernel.services import RunServices, socket_dir_for
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.server import ToolError, ToolKit, build_tool_app, new_mcp


def _server(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    kit = ToolKit(reg, rec)
    mcp = new_mcp()

    @mcp.tool(name="echo_caller")
    async def echo_caller(word: str, ctx: Context) -> dict:
        return await kit.call(ctx, "echo_caller", {"word": word},
                              lambda c: {"node": c.node, "word": word})

    @mcp.tool(name="always_fails")
    async def always_fails(ctx: Context) -> dict:
        def boom(_c):
            raise ToolError("the download was too large")
        return await kit.call(ctx, "always_fails", {}, boom)

    @mcp.tool(name="kernel_bug")
    async def kernel_bug(ctx: Context) -> dict:
        return await kit.call(ctx, "kernel_bug", {}, lambda c: 1 / 0)

    return rec, reg, build_tool_app(mcp)


async def _call(sock_dir, token, tool, args):
    from ar_contract.client import mcp_session
    async with mcp_session(sock_dir, token) as session:
        return await session.call_tool(tool, args)


@pytest.fixture
def live(tmp_path):
    rec, reg, tools_app = _server(tmp_path)
    from fastapi import FastAPI
    services = RunServices(socket_dir_for(tmp_path / "run"))
    services.start(FastAPI(), tools_app)
    caller = reg.issue(node="n7", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path, staging_host=tmp_path)
    yield rec, reg, services, caller
    services.stop()


def test_sockets_live_in_the_run_folder_even_past_the_socket_path_limit(tmp_path):
    rec, reg, tools_app = _server(tmp_path)
    from fastapi import FastAPI
    run = tmp_path / ("x" * 150)
    assert socket_dir_for(run) == run / "sockets"
    services = RunServices(socket_dir_for(run))
    services.start(FastAPI(), tools_app)
    try:
        assert len(str(services.socket_dir / "tools.sock")) > 107
        assert (services.socket_dir / "tools.sock").exists()
        caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                           workspace_host=tmp_path, staging_host=tmp_path)
        assert not asyncio.run(_call(client_dir(services), caller.token, "echo_caller", {"word": "x"})).is_error
    finally:
        services.stop()
    assert not (services.socket_dir / "tools.sock").exists()


def test_tool_sees_its_caller_over_the_socket(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "echo_caller", {"word": "hi"}))
    assert not result.is_error
    assert json.loads(result.content[0].text) == {"node": "n7", "word": "hi"}
    kinds = [e["type"] for e in rec.read_events("n7")]
    assert kinds == ["tool.call", "tool.result"]
    assert all(e["component"] == "tools" for e in rec.read_events("n7"))


def test_tool_error_reaches_the_agent_as_an_error_result(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "always_fails", {}))
    assert result.is_error and "too large" in result.content[0].text
    assert [e["type"] for e in rec.read_events("n7")][-1] == "tool.error"


def test_unexpected_kernel_exception_is_contained(live):
    """A bug in a tool must come back as a tool error, never take the server down."""
    rec, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "kernel_bug", {}))
    assert result.is_error and "ZeroDivisionError" in result.content[0].text
    err = [e for e in rec.read_events("n7") if e["type"] == "tool.error"][0]
    assert "Traceback" in rec.load_payload(err["payload"])["traceback"]
    # the server still answers afterwards
    assert not asyncio.run(_call(client_dir(services), caller.token, "echo_caller", {"word": "x"})).is_error


def test_unknown_token_is_refused(live):
    _, _, services, _ = live
    result = asyncio.run(_call(client_dir(services), "ar-forged", "echo_caller", {"word": "hi"}))
    assert result.is_error and "token" in result.content[0].text


def test_sockets_are_owner_only(live):
    _, _, services, _ = live
    assert oct(os.stat(services.socket_dir).st_mode & 0o777) == "0o700"
    # Both sockets must also be owner-only, not just the directory.
    for name in ("gateway.sock", "tools.sock"):
        assert oct(os.stat(services.socket_dir / name).st_mode & 0o777) == "0o600"


def test_stop_leaves_no_live_threads_and_no_sockets(live):
    _, _, services, _ = live
    threads = list(services._threads)
    sock_paths = [services.socket_dir / name for name in ("gateway.sock", "tools.sock")]
    assert all(p.exists() for p in sock_paths)

    services.stop()

    assert not any(thread.is_alive() for thread in threads)
    assert not any(p.exists() for p in sock_paths)


def test_tool_results_and_errors_have_blocked_names_replaced(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec, scrub_names=["wbench"])
    kit = ToolKit(reg, rec)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    from ar_kernel.tools.server import publish, refuse
    publish(caller, "job", {"items": [{"error": "worker log: [run_wbench] died"}]})       # files are scrubbed too
    assert "run_render" in (tmp_path / "results" / "job.json").read_text()
    with pytest.raises(ToolError):
        refuse(caller, "t", ["a"], {0: "WBench case"})
    assert not [p for p in (tmp_path / "results").iterdir() if "wbench" in p.read_text().lower()]
    assert not list(tmp_path.glob("*.tmp"))
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()

    def fail(_caller):
        raise ToolError("worker log: [run_wbench] failed")
    assert asyncio.run(kit.call(ctx, "t", {}, lambda c: {"log": "[run_WBench] ok"})) == {"log": "[run_render] ok"}
    with pytest.raises(ToolError, match=r"\[run_render\] failed"):
        asyncio.run(kit.call(ctx, "t", {}, fail))
    errors = [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "tool.error"]
    assert "run_wbench" in errors[0]["error"]                 # telemetry keeps the real text


def test_an_edit_self_caller_gets_only_the_tools_of_its_phase(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    kit = ToolKit(reg, rec)

    def ctx_for(phase):
        caller = reg.issue(node="n1", phase=phase, attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
        request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
        return type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    edit, data = ctx_for("edit_self"), ctx_for("improve_recipe")
    assert asyncio.run(kit.call(data, "data_query", {}, lambda c: "clips")) == "clips"
    assert asyncio.run(kit.call(edit, "read_knowledge", {}, lambda c: "text")) == "text"
    assert asyncio.run(kit.call(edit, "ask", {}, lambda c: "answer")) == "answer"
    with pytest.raises(ToolError, match="not available in edit_self"):
        asyncio.run(kit.call(edit, "data_query", {}, lambda c: "clips"))


def _direct(tmp_path, alert_after=3):
    """A ToolKit called in-process, as an agent's call would reach it."""
    from ar_kernel.tools.server import ToolHealth
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    kit = ToolKit(reg, rec, ToolHealth(rec, alert_after))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    return rec, kit, caller, lambda name, fn: asyncio.run(kit.call(ctx, name, {}, fn))


def _raise(exc):
    def fn(_caller):
        raise exc
    return fn


def _tool_alerts(rec):
    return [rec.load_payload(e["payload"]) for e in rec.read_events() if e["type"] == "alert"]


def test_a_refusal_is_not_counted_and_a_kernel_bug_is(tmp_path):
    rec, kit, caller, call = _direct(tmp_path, alert_after=2)
    for _ in range(3):
        with pytest.raises(ToolError):
            call("t", _raise(ToolError("bad argument")))
    assert _tool_alerts(rec) == [] and not caller.fault.event.is_set()
    for _ in range(3):
        with pytest.raises(ToolError, match="KeyError"):
            call("t", _raise(KeyError("x")))
    [alert] = _tool_alerts(rec)                                   # once per streak
    assert alert["kind"] == "tool_failing" and alert["tool"] == "t" and not caller.fault.event.is_set()


def test_a_success_ends_the_streak(tmp_path):
    rec, kit, caller, call = _direct(tmp_path, alert_after=2)
    with pytest.raises(ToolError):
        call("t", _raise(RuntimeError("x")))
    assert call("t", lambda c: "ok") == "ok"
    with pytest.raises(ToolError):
        call("t", _raise(RuntimeError("x")))
    with pytest.raises(ToolError):
        call("other", _raise(RuntimeError("x")))
    assert _tool_alerts(rec) == []


@pytest.mark.parametrize("exc", [OSError(28, "No space left on device"), OSError(5, "Input/output error")],
                         ids=["ENOSPC", "EIO"])
def test_a_disk_error_or_kernel_fault_stops_the_phase(tmp_path, exc):
    from ar_kernel.faults import KernelFault
    for error in (exc, KernelFault("GPUs held by another process")):
        rec, kit, caller, call = _direct(tmp_path / type(error).__name__)
        with pytest.raises(ToolError, match="stops this phase"):
            call("t", _raise(error))
        assert caller.fault.event.is_set() and _tool_alerts(rec) == []


def test_other_os_errors_are_counted_not_faults(tmp_path):
    rec, kit, caller, call = _direct(tmp_path, alert_after=1)
    with pytest.raises(ToolError, match="PermissionError"):
        call("t", _raise(PermissionError(13, "Permission denied")))
    assert not caller.fault.event.is_set() and _tool_alerts(rec)[0]["kind"] == "tool_failing"


def test_a_cancelled_call_stays_in_flight_until_its_thread_returns(tmp_path):
    """The MCP server cancels a request whose client went away; the tool's thread still runs."""
    import threading
    import time
    rec, kit, caller, _ = _direct(tmp_path)
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    started, finished = threading.Event(), threading.Event()

    def slow(_caller):
        started.set()
        time.sleep(0.5)
        finished.set()
    call = asyncio.run_coroutine_threadsafe(kit.call(ctx, "data_ingest", {}, slow), loop)
    assert started.wait(5)
    call.cancel()
    kit.registry.revoke(caller.token)
    kit.registry.wait_idle(caller.token)
    assert finished.is_set()
    loop.call_soon_threadsafe(loop.stop)


def test_a_file_recorded_in_a_tool_call_names_that_call(tmp_path):
    from ar_kernel.telemetry import files
    rec, kit, caller, call = _direct(tmp_path)
    call("video_probe", lambda c: files.record(b"x", direction="in", path="/workspace/a.mp4"))
    span = next(e for e in rec.read_events("n1") if e["type"] == "tool.call")["span_id"]
    event = next(e for e in rec.read_events("n1") if e["type"] == "file")
    assert (event["parent_span_id"], event["tool"], event["phase"]) == (span, "video_probe", "improve_recipe")


def test_a_list_given_as_a_file_is_recorded(tmp_path):
    from ar_kernel.tools.server import listed
    rec, kit, caller, call = _direct(tmp_path)
    (tmp_path / "ids.json").write_text('["a", "b"]')
    assert call("data_query", lambda c: listed(c, "ids.json", "clip_ids")) == ["a", "b"]
    assert recorded(rec) == {("in", "/workspace/ids.json"): b'["a", "b"]'}


def test_a_file_swapped_for_a_fifo_or_a_link_after_the_check_is_refused_without_waiting(tmp_path):
    from ar_kernel.tools.context import PathError, TokenRegistry, open_file
    caller = TokenRegistry(Recorder(tmp_path / "run")).issue(node="n1", phase="improve_recipe", attempt=1,
                                                             workspace_host=tmp_path / "ws", staging_host=tmp_path / "st")
    (tmp_path / "ws").mkdir()
    os.mkfifo(tmp_path / "ws" / "pipe.mp4")
    (tmp_path / "secret").write_text("host file")
    (tmp_path / "ws" / "link.mp4").symlink_to(tmp_path / "secret")
    with pytest.raises(PathError, match="is not a file"):    # a FIFO that took the checked file's place
        open_file(caller, tmp_path / "ws" / "pipe.mp4")
    with pytest.raises(OSError):                                 # a link that took its place
        open_file(caller, tmp_path / "ws" / "link.mp4")
