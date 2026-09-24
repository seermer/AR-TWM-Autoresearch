"""Single-agent inner loop: an explicit ReAct graph.

Reproduces langchain.agents.create_agent (langchain 1.4.2, factory.py @ 4af7ab8)
with no middleware, no response_format and async tools:
  START -> model; model -> END if the last AI message has no tool calls,
  else one Send("tools", [call]) per tool call (parallel); tools -> model.
Tool execution follows langgraph.prebuilt.ToolNode's default error handling.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import ValidationError

RECURSION_LIMIT = 9_999
INVALID_TOOL = "Error: {name} is not a valid tool, try one of [{names}]."
INVALID_ARGS = ("Error invoking tool '{name}' with kwargs {args} with error:\n"
                " {error}\n Please fix the error and try again.")
TOOL_BLOCK_TYPES = {"text", "image_url", "image", "json", "search_result", "custom_tool_call_output",
                    "document", "file"}


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
    message.content = _content(message.content)
    return message


def build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str | None = None):
    by_name = {t.name: t for t in tools}
    system = [SystemMessage(content=system_prompt)] if system_prompt is not None else []

    async def call_model(state: ReactState) -> dict:
        bound = model.bind_tools(tools, tool_choice=None) if tools else model.bind()
        return {"messages": [await bound.ainvoke([*system, *state["messages"]])]}

    async def call_tool(calls: list[ToolCall]) -> dict:
        return {"messages": [await run_tool(by_name, call) for call in calls]}

    def after_model(state: ReactState):
        last = state["messages"][-1]
        if not tools or not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        return [Send("tools", [call]) for call in last.tool_calls]

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    if tools:
        graph.add_node("tools", call_tool)
        graph.add_conditional_edges("model", after_model, ["tools", END])
        graph.add_edge("tools", "model")
    else:
        graph.add_edge("model", END)
    return graph.compile().with_config({"recursion_limit": RECURSION_LIMIT})
