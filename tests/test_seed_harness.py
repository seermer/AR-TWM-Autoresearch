"""The seed harness (agent.harness) against langchain.agents.create_agent.

create_agent (langchain 1.4.2, factory.py @ 4af7ab8) is the reference: both must
produce identical message lists AND show the model identical prompts and tool bindings."""
import asyncio
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

SEED = Path(__file__).resolve().parents[1] / "seed_agent"
sys.path.insert(0, str(SEED))

from agent.harness import build_react_agent  # noqa: E402


class Scripted(BaseChatModel):
    """Replays AIMessages in order; records every prompt and every bind_tools call."""
    script: list
    log: dict

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.log["bind"].append(([convert_to_openai_tool(t) for t in tools], kwargs))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log["prompts"].append([_norm(m) for m in messages])
        reply = self.script[len(self.log["prompts"]) - 1].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _norm(m: BaseMessage) -> dict:
    d = m.model_dump()
    d.pop("id", None)                    # message ids are random per run
    d.pop("response_metadata", None)     # carries the model run id
    return d


def _ai(text="", calls=(), total=None):
    m = AIMessage(content=text, tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"}
                                            for n, a, i in calls])
    if total is not None:
        m.usage_metadata = {"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total}
    return m


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
async def slow_echo(text: str, delay: float) -> str:
    """Echo text after a delay."""
    await asyncio.sleep(delay)
    return text


@tool
async def as_dict(key: str) -> dict:
    """Return a dict."""
    return {"key": key, "n": [1, 2]}


@tool
async def strings(n: int) -> list:
    """Return a list of strings (content the tool node re-serializes)."""
    return ["plain", f"n={n}"]


@tool
async def boom(x: int) -> str:
    """Always fails."""
    raise RuntimeError(f"boom {x}")


TOOLS = [add, slow_echo, as_dict, strings, boom]

SCENARIOS = {
    "plain_answer": [_ai("hello")],
    "one_call": [_ai("", [("add", {"a": 1, "b": 2}, "c1")]), _ai("3")],
    "parallel_calls_finish_out_of_order": [
        _ai("", [("slow_echo", {"text": "a", "delay": 0.3}, "c1"),
                 ("slow_echo", {"text": "b", "delay": 0.0}, "c2"),
                 ("add", {"a": 5, "b": 6}, "c3")]), _ai("done")],
    "unknown_tool": [_ai("", [("nope", {"q": 1}, "c1")]), _ai("sorry")],
    "bad_args": [_ai("", [("add", {"a": "x"}, "c1")]), _ai("fixed")],
    "dict_output": [_ai("", [("as_dict", {"key": "k"}, "c1")]), _ai("ok")],
    "list_output": [_ai("", [("strings", {"n": 3}, "c1")]), _ai("ok")],
    "multi_step": [_ai("", [("add", {"a": 1, "b": 1}, "c1")]),
                   _ai("thinking", [("add", {"a": 2, "b": 2}, "c2"), ("nope", {}, "c3")]),
                   _ai("", [("as_dict", {"key": "z"}, "c4")]), _ai("final")],
}


async def _run(builder, script, tools, system, history):
    log = {"bind": [], "prompts": []}
    agent = builder(Scripted(script=script, log=log), tools, system)
    out = await agent.ainvoke({"messages": history})
    return [_norm(m) for m in out["messages"]], log


def _ours(model, tools, system):
    return build_react_agent(model, tools, system)


def _theirs(model, tools, system):
    return create_agent(model, tools, system_prompt=system)


@pytest.mark.parametrize("system", [None, "You are careful."])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_create_agent(name, system):
    history = [HumanMessage("earlier"), AIMessage("earlier reply"), HumanMessage("go")]
    ours = asyncio.run(_run(_ours, SCENARIOS[name], TOOLS, system, history))
    theirs = asyncio.run(_run(_theirs, SCENARIOS[name], TOOLS, system, history))
    assert ours[0] == theirs[0]          # the final message list
    assert ours[1] == theirs[1]          # every prompt the model saw, and every tool binding


def test_no_tools_matches_create_agent():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")])]    # a tool call with no tools ends the loop
    assert (asyncio.run(_run(_ours, script, [], "s", [HumanMessage("x")]))
            == asyncio.run(_run(_theirs, script, [], "s", [HumanMessage("x")])))


def test_a_raising_tool_propagates_in_both():
    script = [_ai("", [("boom", {"x": 1}, "c1")]), _ai("unreachable")]
    for builder in (_ours, _theirs):
        with pytest.raises(RuntimeError, match="boom 1"):
            asyncio.run(_run(builder, script, TOOLS, None, [HumanMessage("x")]))
