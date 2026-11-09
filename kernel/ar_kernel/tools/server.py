"""MCP tool server core: caller resolution, activity rows and error containment for
every kernel tool."""
from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from mcp.server.transport_security import TransportSecuritySettings

from .. import fs, telemetry
from ..archive.activity import add_activity
from ..faults import KernelFault, disk_fault
from ..isolation import scrub
from ..process_digest import _shape
from .context import STAGING, WORKSPACE, PathError, bearer, open_file, to_host


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
    named = value if value.startswith("/") else str(WORKSPACE / value)
    try:
        with open(open_file(caller, to_host(caller, named)), "rb") as handle:
            data = handle.read()
        telemetry.keep(named, data, "read")
        text = data.decode("utf-8")
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
    data = json.dumps(scrub(result, caller.scrub_names, caller.host_roots), ensure_ascii=False, indent=1).encode()
    rel = f"results/{name}.json"
    fs.place_bytes(data, caller.staging_host, rel)
    telemetry.keep(str(STAGING / rel), data, "handed")
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
    """Failures in a row per tool (a GPU job counts as "<tool> job"), in memory; a success ends the streak."""

    def __init__(self, alert_after: int = 3) -> None:
        self.alert_after = int(alert_after)
        self._streak: dict[str, int] = {}
        self._lock = threading.Lock()

    def failed(self, tool: str, error: str) -> None:
        with self._lock:
            self._streak[tool] = streak = self._streak.get(tool, 0) + 1
        if streak == self.alert_after:
            telemetry.alert("tool.failing", "warning", f"{tool} failed {streak} times in a row", {"error": error})

    def ok(self, tool: str) -> None:
        with self._lock:
            self._streak.pop(tool, None)


def fault_or_count(caller, health: ToolHealth, tool: str, exc: Exception) -> str:
    """For an exception that is not a ToolError: a KernelFault or a disk error stops the caller's phase,
    anything else counts toward `tool`'s failures. Returns the error text."""
    error = f"{type(exc).__name__}: {exc}"
    fault = isinstance(exc, KernelFault) or disk_fault(exc)
    telemetry.decision("tool.fault", {"tool": tool, "error": error}, ["stop the phase", "count"],
                       "stop the phase" if fault else "count",
                       "a kernel fault or a disk error" if fault else "a tool failure: counted toward its alert")
    if fault:
        caller.fail(f"{tool}: {error}")
        return f"{error}; the kernel stops this phase and runs it again"
    health.failed(tool, error)
    return error


class ToolKit:
    def __init__(self, registry, run_dir: Path, health: ToolHealth | None = None,
                 touch: Callable[[], None] = lambda: None) -> None:
        """`touch`: called as each call starts and ends."""
        self.registry = registry
        self.run_dir = Path(run_dir)
        self.touch = touch
        self.health = health or ToolHealth()

    async def call(self, ctx: Context, name: str, args: dict, fn: Callable[[Any], Any], executor=None) -> Any:
        """`fn(caller)` in a worker thread: asyncio's shared pool, or `executor`. The call is in flight until
        that thread returns, even when the request is cancelled."""
        request = getattr(ctx.request_context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        caller = self.registry.begin(bearer(header))
        with telemetry.span("tool.request", {"tool": name, "args": args},
                            parent=caller.span if caller is not None else None) as sp:
            if caller is None:
                sp.status = "refused"
                raise ToolError("unknown or revoked token")
            work = None

            def run(caller):
                with telemetry.span(f"tool.{name}") as w:      # ends with the work, whether or not the request waits
                    w.output = fn(caller)
                    return w.output

            def start():
                nonlocal work
                work = asyncio.get_running_loop().run_in_executor(executor, contextvars.copy_context().run, run, caller)
                work.add_done_callback(lambda _: self.registry.end(caller.token))
                return asyncio.shield(work)
            try:
                sp.output = await self._call(sp, caller, name, args, start)
                return sp.output
            finally:
                if work is None:
                    self.registry.end(caller.token)

    async def _call(self, sp, caller, name: str, args: dict, start: Callable[[], Any]) -> Any:
        if caller.phase == "edit_self" and name not in EDIT_SELF_TOOLS:
            sp.status = "refused"
            raise ToolError(f"{name} is not available in edit_self")
        self.touch()
        try:
            return await self._run(caller, name, args, start)
        finally:
            self.touch()

    async def _run(self, caller, name: str, args: dict, start: Callable[[], Any]) -> Any:
        try:
            result = await start()
        except ToolError as exc:
            self._record(caller, name, str(exc))
            raise ToolError(scrub(str(exc), caller.scrub_names, caller.host_roots)) from None
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self._record(caller, name, f"{type(exc).__name__}: {exc}")
            error = fault_or_count(caller, self.health, name, exc)
            raise ToolError(scrub(error, caller.scrub_names, caller.host_roots)) from exc
        self.health.ok(name)
        self._record(caller, name, None)
        return scrub(result, caller.scrub_names, caller.host_roots)

    def _record(self, caller, name: str, error: str | None) -> None:
        """The call's activity row (`error`: None when it succeeded); a failed write stops the caller's phase."""
        detail = None if error is None else scrub(_shape(error), caller.scrub_names, caller.host_roots)
        try:
            add_activity(self.run_dir, node=caller.node, phase=caller.phase, attempt=caller.attempt, kind="tool",
                         name=name, ok=error is None, detail=detail)
        except KernelFault as exc:
            telemetry.error(exc, "the phase stops")
            caller.fail(f"{name}: {exc}")
