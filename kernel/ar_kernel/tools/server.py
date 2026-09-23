"""MCP tool server core: caller resolution, telemetry and error containment for
every kernel tool (spec 10, 13.3 row "Tool calls")."""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from mcp.server.transport_security import TransportSecuritySettings

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
        # rebinding guard answers 421 (verified fact 3).
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["localhost"], allowed_origins=[]),
        # improve_recipe can go a long time between tool calls (fact 4).
        session_idle_timeout=None,
    )


class ToolKit:
    def __init__(self, registry, recorder) -> None:
        self.registry = registry
        self.recorder = recorder

    async def call(self, ctx: Context, name: str, args: dict, fn: Callable[[Any], Any]) -> Any:
        request = getattr(ctx.request_context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        caller = self.registry.lookup(bearer(header))
        if caller is None:
            raise ToolError("unknown or revoked token")
        base = dict(node=caller.node, phase=caller.phase, attempt=caller.attempt, component="tools")
        span = self.recorder.event("tool.call", payload={"tool": name, "args": args}, tool=name, **base)
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(fn, caller)
        except ToolError as exc:
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": str(exc)}, **base)
            raise
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}, **base)
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        self.recorder.event("tool.result", parent_span_id=span, tool=name,
                             duration_s=time.monotonic() - started,
                             payload={"tool": name, "result": result}, **base)
        return result
