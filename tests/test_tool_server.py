"""Exercise the tool server the way a container does: over a Unix socket, with
the ar_contract client configuration."""
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


def test_socket_dir_is_short_and_private(tmp_path):
    d = socket_dir_for(tmp_path / ("x" * 150))
    assert len(str(d / "gateway.sock")) < 100
    assert socket_dir_for(tmp_path / ("x" * 150)) == d          # deterministic per run


def test_tool_sees_its_caller_over_the_socket(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "echo_caller", {"word": "hi"}))
    assert not result.is_error
    assert json.loads(result.content[0].text) == {"node": "n7", "word": "hi"}
    kinds = [e["type"] for e in rec.read_events("n7")]
    assert kinds == ["tool.call", "tool.result"]
    assert all(e["component"] == "tools" for e in rec.read_events("n7"))


def test_tool_error_reaches_the_agent_as_an_error_result(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "always_fails", {}))
    assert result.is_error and "too large" in result.content[0].text
    assert [e["type"] for e in rec.read_events("n7")][-1] == "tool.error"


def test_unexpected_kernel_exception_is_contained(live):
    """A bug in a tool must come back as a tool error, never take the server down."""
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "kernel_bug", {}))
    assert result.is_error and "ZeroDivisionError" in result.content[0].text
    err = [e for e in rec.read_events("n7") if e["type"] == "tool.error"][0]
    assert "Traceback" in rec.load_payload(err["payload"])["traceback"]
    # the server still answers afterwards
    assert not asyncio.run(_call(services.socket_dir, caller.token, "echo_caller", {"word": "x"})).is_error


def test_unknown_token_is_refused(live):
    _, _, services, _ = live
    result = asyncio.run(_call(services.socket_dir, "ar-forged", "echo_caller", {"word": "hi"}))
    assert result.is_error and "token" in result.content[0].text


def test_sockets_are_owner_only(live):
    _, _, services, _ = live
    assert oct(os.stat(services.socket_dir).st_mode & 0o777) == "0o700"
    # Controller ruling: both sockets must also be owner-only, not just the directory.
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
