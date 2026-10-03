"""Single-agent inner loop: an explicit ReAct graph, plus Claude-Code-style auto-compaction.

Reproduces langchain.agents.create_agent (langchain 1.4.2, factory.py @ 4af7ab8)
with no middleware, no response_format and async tools:
  START -> model; model -> END if the last AI message has no tool calls,
  else one Send("tools", [call]) per tool call (parallel); tools -> model.
Tool execution follows langgraph.prebuilt.ToolNode's default messages.

Four deliberate differences from create_agent's default:
  1. A tool that raises is reported to the model as an error ToolMessage
     (create_agent re-raises and ends the run).
  2. Auto-compact: call_model estimates the context size before every model call
     (the last reply's reported usage plus ~4 chars/token for what came after, with
     each image counted as a fixed IMAGE_TOKENS), on the merged state -- there is no
     per-Send routing decision, so parallel tool results are never seen in isolation.
     Below compact_at * context_window nothing happens and messages append linearly.
     At or above it, a dedicated summarizer call branches off the SAME bound model as the
     normal call (system prompt, tools, tool_choice=None) with one instruction message
     (prompts/compact.md) appended -- byte-identical to the agent's normal request up to
     that append, maximizing prompt-cache hits. If the model calls a tool instead of
     writing the summary, the call is retried once with tool_choice="none" to force text.
     The history is then replaced by one user message -- a continuation preamble plus the
     summary -- before the real model call runs on it. An empty summary raises.
  3. A text result longer than TOOL_RESULT_LIMIT is saved to /workspace/tool_output and the model gets
     its head and tail with the file's path.
  4. A `return_direct` tool ends the run only when it succeeds: after all parallel tool results are in
     (node "tools_done"), the run ends if one of them is a successful return_direct result, and goes back
     to the model otherwise. So a rejected submission returns to the model to be corrected, and an
     accepted one is not followed by another model call.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage, RemoveMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Send
from pydantic import ValidationError

RECURSION_LIMIT = 9_999
COMPACT_PROMPT = Path(__file__).resolve().parent / "prompts" / "compact.md"
INVALID_TOOL = "Error: {name} is not a valid tool, try one of [{names}]."
INVALID_ARGS = ("Error invoking tool '{name}' with kwargs {args} with error:\n"
                " {error}\n Please fix the error and try again.")
TOOL_ERROR = "Error: {error}\n Please fix your mistakes."   # ToolNode's handle_tool_errors=True text
TOOL_BLOCK_TYPES = {"text", "image_url", "image", "json", "search_result", "custom_tool_call_output",
                    "document", "file"}

# chars. A safety cap: kernel tools write large results to files and return a summary.
TOOL_RESULT_LIMIT, TOOL_RESULT_SHOWN = 200_000, 20_000
CHARS_PER_TOKEN = 4          # no tokenizer offline (tiktoken downloads its encodings)
# An image costs a bounded number of vision tokens however long its base64 is, so it is
# counted as a fixed, generous estimate instead of by characters.
IMAGE_TOKENS = 1_500
IMAGE_BLOCK_TYPES = {"image_url", "image", "input_image"}
CONTINUATION = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
    "<summary>\n{summary}\n</summary>\n\n"
    "Continue the work from where it left off without asking any further questions. "
    "Resume directly: do not acknowledge the summary or recap what was happening."
)


_saved = itertools.count(1)


def cap_result(name: str, text: str) -> str:
    """A result past TOOL_RESULT_LIMIT: saved whole to a file; the model sees its head and tail."""
    if len(text) <= TOOL_RESULT_LIMIT:
        return text
    path = (Path(os.environ.get("AR_WORKSPACE", "/workspace")) / "tool_output"
            / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{next(_saved):04d}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    half = TOOL_RESULT_SHOWN // 2
    return (f"[Long result: {len(text)} chars. Only the first and last {half} are shown; the full result is in "
            f"{path}.]\n{text[:half]}\n[...]\n{text[-half:]}")


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _content(output: Any) -> str | list:
    if isinstance(output, str) or (isinstance(output, list) and all(
            isinstance(x, dict) and x.get("type") in TOOL_BLOCK_TYPES for x in output)):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(output)


async def run_tool(tools: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(INVALID_TOOL.format(name=call["name"], names=", ".join(tools)),
                           name=call["name"], tool_call_id=call["id"], status="error")
    try:
        message = await tool.ainvoke({**call, "type": "tool_call"})
    except ValidationError as exc:
        error = "\n".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'Unknown error')}"
                          for e in exc.errors())
        return ToolMessage(INVALID_ARGS.format(name=call["name"], args=call["args"], error=error),
                           name=call["name"], tool_call_id=call["id"], status="error")
    except Exception as exc:  # noqa: BLE001 -- difference 1: report, do not crash
        return ToolMessage(TOOL_ERROR.format(error=repr(exc)), name=call["name"],
                           tool_call_id=call["id"], status="error")
    message.content = _content(message.content)
    if isinstance(message.content, str):
        message.content = cap_result(call["name"], message.content)
    return message


def _chars(message: AnyMessage) -> int:
    if isinstance(message.content, str):
        size = len(message.content)
    else:
        size = sum(IMAGE_TOKENS * CHARS_PER_TOKEN
                   if isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES
                   else len(block if isinstance(block, str) else json.dumps(block, default=str))
                   for block in message.content)
    calls = getattr(message, "tool_calls", None) or []
    return size + (len(json.dumps(calls, default=str)) if calls else 0)


def estimate_tokens(messages: list[AnyMessage], system_prompt: str | None = None) -> int:
    """The last reply's reported usage (input + output) plus an estimate for what came after.
    Without a reported usage (first call, or right after compaction), estimate everything."""
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AIMessage) and message.usage_metadata:
            tail = sum(_chars(m) for m in messages[i + 1:])
            return message.usage_metadata["total_tokens"] + tail // CHARS_PER_TOKEN
    head = len(system_prompt or "")
    return (head + sum(_chars(m) for m in messages)) // CHARS_PER_TOKEN


def needs_compaction(messages: list[AnyMessage], system_prompt: str | None, context_window: int,
                     compact_at: float) -> bool:
    # One message cannot be condensed further; let the model call fail loudly instead of looping.
    return len(messages) > 1 and estimate_tokens(messages, system_prompt) >= compact_at * context_window


def build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str | None = None, *,
                      context_window: int, compact_at: float,
                      compact_prompt: str | None = None):
    by_name = {t.name: t for t in tools}
    system = [SystemMessage(content=system_prompt)] if system_prompt is not None else []

    async def call_model(state: ReactState) -> dict:
        messages, reset = state["messages"], []
        bound = model.bind_tools(tools, tool_choice=None) if tools else model.bind()
        if needs_compaction(messages, system_prompt, context_window, compact_at):
            # The summarizer call branches off the ORIGINAL conversation using the SAME
            # bound model as the normal call, plus one appended instruction message: the
            # request is byte-identical to the agent's normal request up to that append,
            # maximizing prompt-cache hits. The instruction tells the model not to call
            # tools and to answer with the summary text; if it calls one anyway (no text),
            # retry once forcing tool_choice="none".
            instruction = compact_prompt if compact_prompt is not None else COMPACT_PROMPT.read_text()
            prompt = [*system, *messages, HumanMessage(content=instruction)]
            reply = await bound.ainvoke(prompt)
            if reply.tool_calls and not reply.text.strip():
                fallback = model.bind_tools(tools, tool_choice="none") if tools else model.bind()
                reply = await fallback.ainvoke(prompt)
            if not reply.text.strip():
                raise RuntimeError("compaction produced no summary")
            messages = [HumanMessage(content=CONTINUATION.format(summary=reply.text.strip()))]
            reset = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
        return {"messages": [*reset, await bound.ainvoke([*system, *messages])]}

    async def call_tool(calls: list[ToolCall]) -> dict:
        return {"messages": [await run_tool(by_name, call) for call in calls]}

    def after_model(state: ReactState):
        last = state["messages"][-1]
        if not last.tool_calls:
            return END
        return [Send("tools", [call]) for call in last.tool_calls]

    direct = {t.name for t in tools if t.return_direct}

    def after_tools(state: ReactState):
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            if message.name in direct and message.status != "error":
                return END
        return "model"

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    if tools:
        graph.add_node("tools", call_tool)
        graph.add_node("tools_done", lambda state: {})         # runs once, on the merged tool results
        graph.add_conditional_edges("model", after_model, ["tools", END])
        graph.add_edge("tools", "tools_done")
        graph.add_conditional_edges("tools_done", after_tools, ["model", END])
    else:
        graph.add_edge("model", END)
    return graph.compile().with_config({"recursion_limit": RECURSION_LIMIT})
