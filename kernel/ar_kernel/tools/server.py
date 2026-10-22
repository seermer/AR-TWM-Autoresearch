"""MCP tool server core: caller resolution, telemetry and error containment for
every kernel tool."""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from mcp.server.transport_security import TransportSecuritySettings

from ..faults import KernelFault, disk_fault
from ..guards import alert
from ..isolation import scrub
from .context import STAGING, WORKSPACE, PathError, bearer, to_host


class ToolError(_MCPToolError):
    """A failure the agent should see and may act on.

    Subclasses the SDK's own `ToolError` (not merely `Exception`): the SDK
    treats that class, and only that class, as an anticipated failure whose
    message reaches the client. Any other exception type is masked to a
    generic "Error executing tool <name>" (see `Tool.run`), which would hide
    the message this class exists to deliver.
    """


FROM_FILE = "or the path of a .json file under /workspace that holds the list as a JSON array"
SHOWN = 5                   # bad items named in a refusal, and errors listed in a summary: the rest are in the file


def listed(caller, value, what: str) -> list:
    """A list argument: the list itself, or the .json file under /workspace that holds it (FROM_FILE)."""
    if not isinstance(value, str):
        return list(value or [])
    if not value.endswith(".json"):
        raise ToolError(f"{what}: {value} is not a .json file: a list comes from a .json file holding a JSON array")
    try:
        host = to_host(caller, value if value.startswith("/") else str(WORKSPACE / value))
        text = host.read_text(encoding="utf-8")
    except (PathError, OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) else exc      # an OSError's text names the host path
        raise ToolError(f"{what}: cannot read the list file {value}: {reason}") from exc
    try:
        items = json.loads(text)
    except ValueError as exc:
        raise ToolError(f"{what}: {value} is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ToolError(f"{what}: {value} holds a JSON {type(items).__name__}, not an array")
    return items


def refuse(caller, tool: str, items: list, bad: dict[int, str]) -> None:
    """Refuse a call for every bad item at once (`bad`: index -> error). The message shows the first few;
    all of them go to a file, as a result does, so a script can drop them from the list and call again."""
    if not bad:
        return
    rows = [{"index": n, "item": items[n], "error": error} for n, error in sorted(bad.items())]
    path = publish(caller, f"{tool}-refused-{uuid.uuid4().hex[:8]}", rows)
    first = "\n".join(f"item {row['index']}: {row['error']}" for row in rows[:SHOWN])
    raise ToolError(f"{len(rows)} of {len(items)} items are bad and nothing was submitted. Every bad item, with "
                    f"its index and error, is in {path}. The first {min(SHOWN, len(rows))}:\n{first}")


def publish(caller, name: str, result) -> str:
    """Write a tool's full result to /workspace/staging/results/<name>.json; returns that path. Results
    go to a file, never into the conversation: a role reads them with a script, it does not retype them."""
    from .hf_tools import move_into            # imports this module
    with tempfile.NamedTemporaryFile("w", dir=caller.staging_host, suffix=".tmp", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(scrub(result, caller.scrub_names, caller.host_roots), handle, ensure_ascii=False, indent=1)
    rel = f"results/{name}.json"
    try:
        move_into(Path(handle.name), caller.staging_host, rel)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return str(STAGING / rel)


def new_mcp() -> MCPServer:
    return MCPServer("ar-kernel-tools",
                      instructions="Privileged AutoResearcher kernel tools. Paths are container "
                                   "paths under /workspace; write candidates under /workspace/staging.")


def build_tool_app(mcp: MCPServer):
    return mcp.streamable_http_app(
        # Over a Unix socket the Host header carries no port; without this the
        # rebinding guard answers 421.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["localhost"], allowed_origins=[]),
        # improve_recipe can go a long time between tool calls.
        session_idle_timeout=None,
    )


EDIT_SELF_TOOLS = {"ask", "read_knowledge", "snapshot"}


class ToolHealth:
    """Failures in a row per tool (a GPU job counts as "<tool> job"), in memory: `alert_after` in a row
    record one alert per streak; a success ends the streak."""

    def __init__(self, recorder, alert_after: int = 3) -> None:
        self.recorder, self.alert_after = recorder, int(alert_after)
        self._streak: dict[str, int] = {}
        self._lock = threading.Lock()

    def failed(self, tool: str, error: str) -> None:
        with self._lock:
            n = self._streak[tool] = self._streak.get(tool, 0) + 1
        if n == self.alert_after:
            alert(self.recorder, "tool_failing", f"{tool} failed {n} times in a row; the last: {error}", tool=tool)

    def ok(self, tool: str) -> None:
        with self._lock:
            self._streak.pop(tool, None)


def fault_or_count(caller, health: ToolHealth, tool: str, exc: Exception) -> str:
    """For an exception that is not a ToolError: a KernelFault or a disk error stops the caller's phase,
    anything else counts toward `tool`'s failures. Returns the error text."""
    error = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, KernelFault) or disk_fault(exc):
        caller.fail(f"{tool}: {error}")
        return f"{error}; the kernel stops this phase and runs it again"
    health.failed(tool, error)
    return error


class ToolKit:
    def __init__(self, registry, recorder, health: ToolHealth | None = None) -> None:
        self.registry = registry
        self.recorder = recorder
        self.health = health or ToolHealth(recorder)

    async def call(self, ctx: Context, name: str, args: dict, fn: Callable[[Any], Any], executor=None) -> Any:
        """`fn(caller)` in a worker thread: asyncio's shared pool, or `executor`. The call is in flight until
        that thread returns, even when the request is cancelled."""
        request = getattr(ctx.request_context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        caller = self.registry.begin(bearer(header))
        if caller is None:
            raise ToolError("unknown or revoked token")
        work = None

        def start():
            nonlocal work
            work = asyncio.get_running_loop().run_in_executor(executor, fn, caller)
            work.add_done_callback(lambda _: self.registry.end(caller.token))
            return asyncio.shield(work)
        try:
            return await self._call(caller, name, args, start)
        finally:
            if work is None:
                self.registry.end(caller.token)

    async def _call(self, caller, name: str, args: dict, start: Callable[[], Any]) -> Any:
        if caller.phase == "edit_self" and name not in EDIT_SELF_TOOLS:
            raise ToolError(f"{name} is not available in edit_self")
        base = dict(node=caller.node, phase=caller.phase, attempt=caller.attempt, component="tools")
        span = self.recorder.event("tool.call", payload={"tool": name, "args": args}, tool=name, **base)
        started = time.monotonic()
        try:
            result = await start()
        except ToolError as exc:
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": str(exc)}, **base)
            raise ToolError(scrub(str(exc), caller.scrub_names, caller.host_roots)) from None
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}, **base)
            error = fault_or_count(caller, self.health, name, exc)
            raise ToolError(scrub(error, caller.scrub_names, caller.host_roots)) from exc
        self.health.ok(name)
        self.recorder.event("tool.result", parent_span_id=span, tool=name,
                             duration_s=time.monotonic() - started,
                             payload={"tool": name, "result": result}, **base)
        return scrub(result, caller.scrub_names, caller.host_roots)
