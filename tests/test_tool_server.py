"""Exercise the tool server the way a container does: over a Unix socket, with
the ar_contract client configuration."""
from conftest import client_dir, telemetry_payload, telemetry_records
import asyncio
import json
import os

import pytest
from mcp.server.mcpserver import Context

from ar_kernel import telemetry
from ar_kernel.archive.activity import activity
from ar_kernel.archive.db import open_db
from ar_kernel.services import RunServices, socket_dir_for
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.telemetry import tree
from ar_kernel.tools.server import ToolError, ToolKit, build_tool_app, new_mcp


def _server(tmp_path):
    reg = TokenRegistry()
    kit = ToolKit(reg, tmp_path / "run")
    mcp = new_mcp(reg)

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

    return lambda node: activity(open_db(tmp_path / "run"), node), reg, build_tool_app(mcp)


async def _call(sock_dir, token, tool, args):
    from ar_contract.client import mcp_session
    async with mcp_session(sock_dir, token) as session:
        return await session.call_tool(tool, args)


@pytest.fixture
def live(tmp_path):
    rows, reg, tools_app = _server(tmp_path)
    from fastapi import FastAPI
    services = RunServices(socket_dir_for(tmp_path / "run"))
    services.start(FastAPI(), tools_app)
    caller = reg.issue(node="n7", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path, staging_host=tmp_path)
    yield rows, reg, services, caller
    services.stop()


def test_sockets_live_in_the_run_folder_even_past_the_socket_path_limit(tmp_path):
    _, reg, tools_app = _server(tmp_path)
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
    rows, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "echo_caller", {"word": "hi"}))
    assert not result.is_error
    assert json.loads(result.content[0].text) == {"node": "n7", "word": "hi"}
    assert [(r["kind"], r["name"], r["ok"]) for r in rows("n7")] == [("tool", "echo_caller", 1)]


def test_tool_error_reaches_the_agent_as_an_error_result(live):
    rows, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "always_fails", {}))
    assert result.is_error and "too large" in result.content[0].text
    assert rows("n7")[-1]["ok"] == 0


def test_unexpected_kernel_exception_is_contained(live):
    """A bug in a tool must come back as a tool error, never take the server down."""
    rows, _, services, caller = live
    result = asyncio.run(_call(client_dir(services), caller.token, "kernel_bug", {}))
    assert result.is_error and "ZeroDivisionError" in result.content[0].text
    assert "ZeroDivisionError" in rows("n7")[0]["detail"]
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
    reg = TokenRegistry(scrub_names=["wbench"])
    kit = ToolKit(reg, tmp_path / "run")
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


def test_an_edit_self_caller_gets_only_the_tools_of_its_phase(tmp_path):
    reg = TokenRegistry()
    kit = ToolKit(reg, tmp_path / "run")

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


def _direct(tmp_path):
    """A ToolKit called in-process, as an agent's call would reach it."""
    reg = TokenRegistry()
    kit = ToolKit(reg, tmp_path / "run")
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    return kit, caller, lambda name, fn: asyncio.run(kit.call(ctx, name, {}, fn))


def _raise(exc):
    def fn(_caller):
        raise exc
    return fn


def test_a_refusal_is_not_counted_and_a_kernel_bug_is(tmp_path):
    kit, caller, call = _direct(tmp_path)
    for _ in range(3):
        with pytest.raises(ToolError):
            call("t", _raise(ToolError("bad argument")))
    assert kit.health._streak == {} and not caller.fault.event.is_set()
    for _ in range(3):
        with pytest.raises(ToolError, match="KeyError"):
            call("t", _raise(KeyError("x")))
    assert kit.health._streak == {"t": 3} and not caller.fault.event.is_set()


def test_a_success_ends_the_streak(tmp_path):
    kit, caller, call = _direct(tmp_path)
    with pytest.raises(ToolError):
        call("t", _raise(RuntimeError("x")))
    assert call("t", lambda c: "ok") == "ok"
    with pytest.raises(ToolError):
        call("t", _raise(RuntimeError("x")))
    with pytest.raises(ToolError):
        call("other", _raise(RuntimeError("x")))
    assert kit.health._streak == {"t": 1, "other": 1}


@pytest.mark.parametrize("exc", [OSError(28, "No space left on device"), OSError(5, "Input/output error")],
                         ids=["ENOSPC", "EIO"])
def test_a_disk_error_or_kernel_fault_stops_the_phase(tmp_path, exc):
    from ar_kernel.faults import KernelFault
    for error in (exc, KernelFault("GPUs held by another process")):
        kit, caller, call = _direct(tmp_path / type(error).__name__)
        with pytest.raises(ToolError, match="stops this phase"):
            call("t", _raise(error))
        assert caller.fault.event.is_set() and kit.health._streak == {}


def test_other_os_errors_are_counted_not_faults(tmp_path):
    kit, caller, call = _direct(tmp_path)
    with pytest.raises(ToolError, match="PermissionError"):
        call("t", _raise(PermissionError(13, "Permission denied")))
    assert not caller.fault.event.is_set() and kit.health._streak == {"t": 1}


def test_a_cancelled_call_stays_in_flight_until_its_thread_returns(tmp_path):
    """The MCP server cancels a request whose client went away; the tool's thread still runs."""
    import threading
    import time
    kit, caller, _ = _direct(tmp_path)
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


def test_a_list_given_as_a_file_is_read(tmp_path):
    from ar_kernel.tools.server import listed
    kit, caller, call = _direct(tmp_path)
    (tmp_path / "ids.json").write_text('["a", "b"]')
    assert call("data_query", lambda c: listed(c, "ids.json", "clip_ids")) == ["a", "b"]


def test_a_file_swapped_for_a_fifo_or_a_link_after_the_check_is_refused_without_waiting(tmp_path):
    from ar_kernel.tools.context import PathError, TokenRegistry, open_file
    caller = TokenRegistry().issue(node="n1", phase="improve_recipe", attempt=1,
                                                             workspace_host=tmp_path / "ws", staging_host=tmp_path / "st")
    (tmp_path / "ws").mkdir()
    os.mkfifo(tmp_path / "ws" / "pipe.mp4")
    (tmp_path / "secret").write_text("host file")
    (tmp_path / "ws" / "link.mp4").symlink_to(tmp_path / "secret")
    with pytest.raises(PathError, match="is not a file"):    # a FIFO that took the checked file's place
        open_file(caller, tmp_path / "ws" / "pipe.mp4")
    with pytest.raises(OSError):                                 # a link that took its place
        open_file(caller, tmp_path / "ws" / "link.mp4")


def test_each_finished_call_is_an_activity_row_with_its_error_scrubbed_and_shaped(tmp_path):
    from ar_kernel.archive.activity import activity
    from ar_kernel.archive.db import open_db
    reg = TokenRegistry(scrub_names=["wbench"])
    kit = ToolKit(reg, tmp_path / "run")
    edit = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=2, workspace_host=tmp_path, staging_host=tmp_path)

    def call(who, name, fn):
        request = type("R", (), {"headers": {"authorization": f"Bearer {who.token}"}})()
        return asyncio.run(kit.call(type("C", (), {"request_context": type("RC", (), {"request": request})()})(),
                                    name, {}, fn))
    assert call(caller, "t", lambda c: "ok") == "ok"
    with pytest.raises(ToolError):
        call(caller, "t", _raise(ToolError(f"run_wbench failed on /mnt/x/{'a' * 32}.mp4")))
    with pytest.raises(ToolError):
        call(caller, "u", lambda c: 1 / 0)
    with pytest.raises(ToolError, match="not available"):
        call(edit, "data_query", lambda c: "clips")                         # refused before it ran: no row
    rows = [(r["phase"], r["attempt"], r["kind"], r["name"], r["ok"], r["detail"])
            for r in activity(open_db(tmp_path / "run"), "n1")]
    assert rows == [("improve_recipe", 2, "tool", "t", 1, None),
                    ("improve_recipe", 2, "tool", "t", 0, "run_render failed on <path>"),
                    ("improve_recipe", 2, "tool", "u", 0, "ZeroDivisionError: division by zero")]


def test_error_details_are_shaped_before_the_host_folders_are_named(tmp_path):
    from ar_kernel.archive.activity import activity
    from ar_kernel.archive.db import open_db
    reg = TokenRegistry(scrub_names=["wbench"], host_roots=(("/mnt/proj/runs", "<host>/runs"),))
    kit = ToolKit(reg, tmp_path / "run")
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    with pytest.raises(ToolError, match="<host>/runs/r1/x.mp4"):          # the agent's own error names the folder
        asyncio.run(kit.call(ctx, "t", {}, _raise(ToolError("cannot open /mnt/proj/runs/r1/x.mp4 for WBench"))))
    [row] = activity(open_db(tmp_path / "run"), "n1")
    assert row["detail"] == "cannot open <path> for render"


def test_a_row_that_cannot_be_written_stops_the_callers_phase(tmp_path, monkeypatch):
    from ar_kernel.faults import KernelFault
    from ar_kernel.tools import server

    def broken(*a, **k):
        raise KernelFault("the archive could not record activity: disk I/O error")
    monkeypatch.setattr(server, "add_activity", broken)
    kit, caller, call = _direct(tmp_path)
    assert call("t", lambda c: "ok") == "ok"
    assert caller.fault.event.is_set() and caller.fault.reason == "t: the archive could not record activity: disk I/O error"


def _traced(tmp_path, phase="improve_recipe", token=None):
    """_direct, with the caller issued inside a phase attempt span as the loop issues it."""
    run = tmp_path / "trun"
    telemetry.open(run)
    telemetry.start("ar.run")
    reg = TokenRegistry()
    with telemetry.span("phase.attempt", {"node": "n1", "phase": phase, "attempt": 1}) as attempt:
        caller = reg.issue(node="n1", phase=phase, attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
        attempt.output = {}
    kit = ToolKit(reg, tmp_path / "run")
    request = type("R", (), {"headers": {"authorization": f"Bearer {token or caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()

    def spans():
        """(tool.request start, its end or None, [(child start, child end or None)])."""
        records = telemetry_records(run)
        ends = {r["span"]: r for r in records if r["kind"] == "end"}
        out = []
        for start in (r for r in records if r["kind"] == "start" and r["name"] == "tool.request"):
            out.append((start, ends.get(start["span"]),
                        [(r, ends.get(r["span"])) for r in records if r["kind"] == "start" and r["parent"] == start["span"]]))
        return out
    return kit, caller, ctx, attempt, run, spans


def test_a_call_is_a_request_span_under_the_attempt_and_a_work_span_under_it(tmp_path):
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    assert asyncio.run(kit.call(ctx, "t", {"x": 1}, lambda c: {"done": 1})) == {"done": 1}
    [(start, end, children)] = spans()
    assert start["parent"] == attempt.id and telemetry_payload(run, start["inputs"]) == {"tool": "t", "args": {"x": 1}}
    assert end["status"] == "ok" and telemetry_payload(run, end["output"]) == {"done": 1}
    [(work, work_end)] = [c for c in children if c[0]["name"] == "tool.t"]
    assert work_end["status"] == "ok" and telemetry_payload(run, work_end["output"]) == {"done": 1}
    assert "db.write" in [c[0]["name"] for c in children]                  # the activity row
    assert tree.check(tree.load(run)) == []


def test_a_call_the_server_refuses_before_the_tool_runs_is_a_refused_request(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    mcp = new_mcp(kit.registry)

    @mcp.tool(name="echo")
    async def echo(word: str, ctx: Context) -> dict:
        return await kit.call(ctx, "echo", {"word": word}, lambda c: {"word": word})

    for name, args in (("echo", {"word": {"not": "text"}}), ("no_such_tool", {})):
        with pytest.raises(MCPToolError):
            asyncio.run(mcp.call_tool(name, args, ctx))
    got = [(s["parent"], telemetry_payload(run, s["inputs"]), e["status"], telemetry_payload(run, e["output"])["error"])
           for s, e, _ in spans()]
    assert [g[:3] for g in got] == [(attempt.id, {"tool": "echo", "args": {"word": {"not": "text"}}}, "refused"),
                                    (attempt.id, {"tool": "no_such_tool", "args": {}}, "refused")]
    assert "word" in got[0][3] and "no_such_tool" in got[1][3]
    assert tree.check(tree.load(run)) == []


@pytest.mark.parametrize("why", ["token", "edit_self"])
def test_a_refused_call_ends_refused_without_work(tmp_path, why):
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path, phase="edit_self" if why == "edit_self" else "improve_recipe",
                                                    token="ar-nope" if why == "token" else None)
    with pytest.raises(ToolError):
        asyncio.run(kit.call(ctx, "data_query", {}, lambda c: 1))
    [(start, end, children)] = spans()
    assert end["status"] == "refused" and children == []
    assert (start["parent"] == attempt.id) == (why == "edit_self")
    assert tree.check(tree.load(run)) == []


def test_a_tool_error_ends_both_spans_error_and_a_kernel_bug_is_counted_then_alerted(tmp_path):
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    with pytest.raises(ToolError):
        asyncio.run(kit.call(ctx, "t", {}, _raise(ToolError("bad argument"))))
    for _ in range(3):
        with pytest.raises(ToolError):
            asyncio.run(kit.call(ctx, "t", {}, lambda c: 1 / 0))
    first, *bugs = spans()
    assert first[1]["status"] == "error" and [e["status"] for s, e in first[2] if s["name"] == "tool.t"] == ["error"]
    records = telemetry_records(run)
    assert [d["chosen"] for d in records if d["kind"] == "decision" and d["name"] == "tool.fault"] == ["count"] * 3
    [alert] = [r for r in records if r["kind"] == "alert"]
    assert alert["name"] == "tool.failing" and alert["span"] == bugs[-1][0]["span"]
    assert tree.check(tree.load(run)) == []


def test_a_kernel_fault_stops_the_phase_inside_the_request(tmp_path):
    from ar_kernel.faults import KernelFault
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    with pytest.raises(ToolError):
        asyncio.run(kit.call(ctx, "t", {}, _raise(KernelFault("GPUs held"))))
    [(start, end, children)] = spans()
    assert "caller.fail" in [c[0]["name"] for c in children] and caller.fault.event.is_set()
    [d] = [r for r in telemetry_records(run) if r["kind"] == "decision" and r["name"] == "tool.fault"]
    assert d["chosen"] == "stop the phase" and d["span"] == start["span"]


def _loop_thread():
    import threading
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def test_a_cancelled_request_ends_cancelled_and_its_work_ends_from_the_work(tmp_path):
    import threading
    import time
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    loop, started = _loop_thread(), threading.Event()

    def slow(_caller):
        started.set()
        time.sleep(0.5)
        return "late"
    call = asyncio.run_coroutine_threadsafe(kit.call(ctx, "t", {}, slow), loop)
    assert started.wait(5)
    call.cancel()
    kit.registry.revoke(caller.token)
    kit.registry.wait_idle(caller.token)
    loop.call_soon_threadsafe(loop.stop)
    [(start, end, children)] = spans()
    [(work, work_end)] = [c for c in children if c[0]["name"] == "tool.t"]
    assert end["status"] == "cancelled" and work_end["status"] == "ok"
    assert telemetry_payload(run, work_end["output"]) == "late"


def test_work_that_returns_after_the_event_loop_stopped_still_ends_its_span(tmp_path):
    import threading
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    loop, started, release = _loop_thread(), threading.Event(), threading.Event()

    def held(_caller):
        started.set()
        release.wait(5)
        return "done"
    asyncio.run_coroutine_threadsafe(kit.call(ctx, "t", {}, held), loop)
    assert started.wait(5)
    loop.call_soon_threadsafe(loop.stop)
    while loop.is_running():
        pass
    release.set()
    for _ in range(100):
        [(start, end, children)] = spans()
        work_end = [e for s, e in children if s["name"] == "tool.t"][0]
        if work_end is not None:
            break
        threading.Event().wait(0.05)
    assert end is None and work_end["status"] == "ok"                  # the request stays open: its loop stopped


def test_a_published_result_is_kept_as_a_file_handed_to_the_agent(tmp_path):
    from ar_kernel.tools.server import publish
    kit, caller, ctx, attempt, run, spans = _traced(tmp_path)
    path = asyncio.run(kit.call(ctx, "data_query", {}, lambda c: publish(c, "data_query-x", [1, 2])))
    [(start, end, children)] = spans()
    [(work, _)] = [c for c in children if c[0]["name"] == "tool.data_query"]
    records = telemetry_records(run)
    [kept] = [r for r in records if r["kind"] == "start" and r["name"] == "file.kept" and r["parent"] == work["span"]]
    assert telemetry_payload(run, kept["inputs"]) == {"path": path, "how": "handed"}
    [kept_end] = [r for r in records if r["kind"] == "end" and r["span"] == kept["span"]]
    digest = telemetry_payload(run, kept_end["output"])["sha256"]
    assert (run / "telemetry" / "files" / digest).read_bytes() == (tmp_path / "results" / "data_query-x.json").read_bytes()
