"""MCP tool server core: caller resolution, telemetry and error containment for
every kernel tool."""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from mcp.server.transport_security import TransportSecuritySettings

from ..isolation import scrub
from .context import bearer


class ToolError(_MCPToolError):
    """A failure the agent should see and may act on.

    Subclasses the SDK's own `ToolError` (not merely `Exception`): the SDK
    treats that class, and only that class, as an anticipated failure whose
    message reaches the client. Any other exception type is masked to a
    generic "Error executing tool <name>" (see `Tool.run`), which would hide
    the message this class exists to deliver.
    """


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


EDIT_SELF_TOOLS = {"ask", "read_skill"}       # edit_self improves the agent: no data tools, no scores


class ToolKit:
    def __init__(self, registry, recorder, scrub_names: list[str] = ()) -> None:
        self.registry = registry
        self.recorder = recorder
        self.scrub_names = list(scrub_names)      # replaced in everything a tool returns to an agent

    async def call(self, ctx: Context, name: str, args: dict, fn: Callable[[Any], Any]) -> Any:
        request = getattr(ctx.request_context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        caller = self.registry.lookup(bearer(header))
        if caller is None:
            raise ToolError("unknown or revoked token")
        if caller.phase == "edit_self" and name not in EDIT_SELF_TOOLS:
            raise ToolError(f"{name} is not available in edit_self")
        base = dict(node=caller.node, phase=caller.phase, attempt=caller.attempt, component="tools")
        span = self.recorder.event("tool.call", payload={"tool": name, "args": args}, tool=name, **base)
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(fn, caller)
        except ToolError as exc:
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": str(exc)}, **base)
            raise ToolError(scrub(str(exc), self.scrub_names)) from None
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}, **base)
            raise ToolError(scrub(f"{type(exc).__name__}: {exc}", self.scrub_names)) from exc
        self.recorder.event("tool.result", parent_span_id=span, tool=name,
                             duration_s=time.monotonic() - started,
                             payload={"tool": name, "result": result}, **base)
        return scrub(result, self.scrub_names)
