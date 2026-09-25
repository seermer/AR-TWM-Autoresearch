"""The seed harness (agent.harness) against langchain.agents.create_agent.

create_agent (langchain 1.4.2, factory.py @ 4af7ab8) is the reference: both must
produce identical message lists AND show the model identical prompts and tool bindings."""
import asyncio
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

SEED = Path(__file__).resolve().parents[1] / "seed_agent"
sys.path.insert(0, str(SEED))

from agent.harness import CONTINUATION, estimate_tokens, needs_compaction  # noqa: E402
from agent.harness import build_react_agent  # noqa: E402

BIG = 10 ** 9      # a context window compaction never reaches


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
    return build_react_agent(model, tools, system, context_window=BIG, compact_at=0.85)


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


def test_tool_exception_is_reported_where_create_agent_raises():
    script = [_ai("", [("boom", {"x": 1}, "c1")]), _ai("recovered")]
    with pytest.raises(RuntimeError, match="boom 1"):
        asyncio.run(_run(_theirs, script, TOOLS, None, [HumanMessage("x")]))
    messages, _ = asyncio.run(_run(_ours, script, TOOLS, None, [HumanMessage("x")]))
    err = messages[2]
    assert err["type"] == "tool" and err["status"] == "error" and err["tool_call_id"] == "c1"
    assert err["content"] == "Error: RuntimeError('boom 1')\n Please fix your mistakes."
    assert messages[-1]["content"] == "recovered"


def test_estimate_uses_last_reported_usage_plus_tail():
    msgs = [HumanMessage("x" * 400), _ai("hi", total=1000), ToolMessage("y" * 80, tool_call_id="c")]
    assert estimate_tokens(msgs) == 1000 + 20
    assert estimate_tokens([HumanMessage("x" * 400)], system_prompt="s" * 40) == 110


def test_images_count_as_a_fixed_estimate_not_their_base64_length():
    from agent.harness import IMAGE_TOKENS
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 400_000}}
    msgs = [HumanMessage(content=[{"type": "text", "text": "x" * 40}, image])]
    assert IMAGE_TOKENS <= estimate_tokens(msgs) < IMAGE_TOKENS + 100      # not 400_000 / 4


def test_a_single_message_is_never_compacted():
    assert not needs_compaction([HumanMessage("x" * 10 ** 6)], None, 1000, 0.85)


def test_compaction_replaces_history_and_continues():
    # call 1 asks for a tool and reports 900 tokens >= 0.85 * 1000, so the harness compacts
    # before call 2 (the summarizer); call 3 continues from the summary alone.
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900), _ai("SUMMARY TEXT"),
              _ai("done", total=50)]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_at=0.85, compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    summarizer = log["prompts"][1]
    assert [m["type"] for m in summarizer] == ["system", "human", "ai", "tool", "human"]
    assert summarizer[0]["content"] == "sys" and summarizer[-1]["content"] == "COMPACT NOW"
    # The summarizer reuses the SAME bind as the normal model call (cache-friendly): one
    # bind per call_model invocation, not a separate tool_choice="none" bind for it.
    assert len(log["bind"]) == 2
    assert log["bind"][1][1] == {"tool_choice": None}
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and log["prompts"][2][1]["content"] == continuation
    assert out["messages"][-1].content == "done"


def test_below_the_threshold_messages_append_linearly():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=800), _ai("done", total=820)]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_at=0.85)
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"] and len(log["prompts"]) == 2


def test_compaction_triggers_once_when_parallel_outputs_only_cross_together():
    # Neither slow_echo output alone reaches the threshold (700 + 350/4 = 787 < 850); combined
    # they do (700 + 700/4 = 875 >= 850). The compaction check must see both tool results
    # merged in one call_model invocation, not one call per Send branch, or this due
    # compaction would be skipped.
    script = [_ai("", [("slow_echo", {"text": "A" * 350, "delay": 0.0}, "c1"),
                       ("slow_echo", {"text": "B" * 350, "delay": 0.0}, "c2")], total=700),
              _ai("SUMMARY TEXT"), _ai("done")]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_at=0.85, compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(log["prompts"]) == 3                                       # no call on the old history
    # One bind per call_model invocation; the summarizer reuses it (no separate summarizer bind).
    assert len(log["bind"]) == 2 and log["bind"][1][1] == {"tool_choice": None}
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and out["messages"][-1].content == "done"


def test_compaction_triggers_once_even_when_one_parallel_output_alone_crosses():
    # slow_echo's first output alone already crosses (700 + 700/4 = 875 >= 850); the second does
    # not (700 + 10/4 = 702 < 850). Per-Send routing would send the first branch to "compact"
    # and the second to "model" in the same step, producing a stray reply on the stale merged
    # history alongside the summary; call_model must instead see them merged and compact once.
    script = [_ai("", [("slow_echo", {"text": "C" * 700, "delay": 0.0}, "c1"),
                       ("slow_echo", {"text": "D" * 10, "delay": 0.0}, "c2")], total=700),
              _ai("SUMMARY TEXT"), _ai("done")]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_at=0.85, compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(log["prompts"]) == 3                                       # no model call besides the summarizer
    assert len(log["bind"]) == 2 and log["bind"][1][1] == {"tool_choice": None}
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and out["messages"][-1].content == "done"


def test_compaction_falls_back_to_tool_choice_none_when_the_model_calls_a_tool_instead():
    # The summarizer's first attempt uses the normal bind (tool_choice=None), so a model
    # that ignores the "do not call tools" instruction can still call one, with no text.
    # The harness then retries the SAME prompt once, forcing tool_choice="none".
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900),
              _ai("", [("add", {"a": 9, "b": 9}, "cX")]),   # summarizer attempt: tool call, no text
              _ai("SUMMARY TEXT"),                          # fallback retry: plain text
              _ai("done", total=50)]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_at=0.85, compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(log["prompts"]) == 4                          # +1 for the fallback retry
    assert log["prompts"][1] == log["prompts"][2]            # the identical prompt, retried verbatim
    assert log["bind"][1][1] == {"tool_choice": None}        # first attempt: the normal bind
    assert log["bind"][2][1] == {"tool_choice": "none"}      # fallback: forces text
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert out["messages"][0].content == continuation
    assert out["messages"][-1].content == "done"


def test_the_default_compaction_prompt_is_the_seed_prompt_file():
    from agent.harness import COMPACT_PROMPT
    assert COMPACT_PROMPT.is_file() and "Next step" in COMPACT_PROMPT.read_text()
