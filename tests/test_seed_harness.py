"""The seed harness and the seed's own tools."""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

SEED = Path(__file__).resolve().parents[1] / "seed_agent"
sys.path.insert(0, str(SEED))

from agent import harness  # noqa: E402
from agent.harness import CONTINUATION, Guard, error_text, estimate_tokens, needs_compaction, run_loop  # noqa: E402

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
    "dict_output": [_ai("", [("as_dict", {"key": "k"}, "c1")]), _ai("ok")],
    "list_output": [_ai("", [("strings", {"n": 3}, "c1")]), _ai("ok")],
    "multi_step": [_ai("", [("add", {"a": 1, "b": 1}, "c1")]),
                   _ai("thinking", [("add", {"a": 2, "b": 2}, "c2"), ("add", {"a": 3, "b": 3}, "c3")]),
                   _ai("", [("as_dict", {"key": "z"}, "c4")]), _ai("final")],
}


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_WORKSPACE", str(tmp_path / "ws"))
    return tmp_path / "ws"


class _Ours:
    def __init__(self, model, tools, system, **kw):
        self.args, self.kw = (model, tools, system), {"role": "r", "pinned": "PINNED", "compact_at": 0.85,
                                                     "context_window": BIG, **kw}

    async def ainvoke(self, state):
        return {"messages": await run_loop(*self.args, state["messages"], **self.kw)}


async def _run(builder, script, tools, system, history, **kw):
    log = {"bind": [], "prompts": []}
    agent = builder(Scripted(script=script, log=log), tools, system, **kw)
    out = await agent.ainvoke({"messages": history})
    return [_norm(m) for m in out["messages"]], log


def _theirs(model, tools, system):
    return create_agent(model, tools, system_prompt=system)


@pytest.mark.parametrize("system", [None, "You are careful."])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_create_agent(name, system):
    history = [HumanMessage("earlier"), AIMessage("earlier reply"), HumanMessage("go")]
    ours = asyncio.run(_run(_Ours, SCENARIOS[name], TOOLS, system, list(history)))
    theirs = asyncio.run(_run(_theirs, SCENARIOS[name], TOOLS, system, list(history)))
    assert ours[0] == theirs[0]          # the final message list
    assert ours[1] == theirs[1]          # every prompt the model saw, and every tool binding


def test_no_tools_matches_create_agent():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")])]    # a tool call with no tools ends the loop
    assert (asyncio.run(_run(_Ours, script, [], "s", [HumanMessage("x")]))
            == asyncio.run(_run(_theirs, script, [], "s", [HumanMessage("x")])))


# ---- errors ----

def _tool_results(script, tools):
    messages, _ = asyncio.run(_run(_Ours, script, tools, None, [HumanMessage("x")]))
    return [m for m in messages if m["type"] == "tool"]


def test_a_failed_call_is_reported_and_the_run_goes_on():
    script = [_ai("", [("boom", {"x": 1}, "c1"), ("nope", {"q": 1}, "c2"), ("add", {"a": "x"}, "c3")]),
              _ai("recovered")]
    errors = _tool_results(script, TOOLS)
    assert [e["status"] for e in errors] == ["error"] * 3
    assert errors[0]["content"] == "Error: RuntimeError: boom 1"
    assert errors[1]["content"] == "Error: nope is not a valid tool, try one of [add, slow_echo, as_dict, strings, boom]."
    assert errors[2]["content"].startswith("Error: invalid arguments for add:\na: ")


def test_a_kernel_tool_error_reads_as_its_message_not_as_a_repr():
    schema = ToolException("Error executing tool data_commit: 1 validation error for data_commitArguments\ndatasets\n"
                           "  Input should be a valid dictionary [type=dict_type, input_value='/workspace/r.json', "
                           "input_type=str]\n    For further information visit https://errors.pydantic.dev/2.13/v/dict_type")
    assert error_text("data_commit", schema) == ("Error: 1 validation error for data_commitArguments\ndatasets\n"
                                                 "  Input should be a valid dictionary")
    with pytest.raises(FileNotFoundError) as missing:
        open("/no/such/file.txt")
    assert error_text("x", missing.value).startswith("Error: FileNotFoundError: ") and "'/no/such/file.txt'" in error_text("x", missing.value)


def test_a_result_json_cannot_encode_still_reaches_the_model():
    @tool
    async def where() -> dict:
        """Where."""
        return {"path": Path("/workspace/x")}
    [result] = _tool_results([_ai("", [("where", {}, "c1")]), _ai("ok")], [where])
    assert result["content"] == '{"path": "/workspace/x"}' and result["status"] == "success"


# ---- the execution rule ----

def _timed(name, log, parallel):
    async def run(tag: str) -> str:
        log.append(("start", tag))
        await asyncio.sleep(0.2)
        log.append(("end", tag))
        return tag
    return StructuredTool.from_function(coroutine=run, name=name, description=name,
                                        metadata={"parallel": True} if parallel else None)


def test_a_batch_of_kernel_tools_runs_in_parallel_and_any_other_batch_in_order():
    log = []
    k1, k2, local = _timed("k1", log, True), _timed("k2", log, True), _timed("local", log, False)
    _tool_results([_ai("", [("k1", {"tag": "a"}, "c1"), ("k2", {"tag": "b"}, "c2")]), _ai("ok")], [k1, k2, local])
    assert [e for e, _ in log] == ["start", "start", "end", "end"]
    log.clear()
    results = _tool_results([_ai("", [("k1", {"tag": "a"}, "c1"), ("local", {"tag": "b"}, "c2"),
                                      ("k2", {"tag": "c"}, "c3")]), _ai("ok")], [k1, k2, local])
    assert log == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b"), ("start", "c"), ("end", "c")]
    assert [r["content"] for r in results] == ["a", "b", "c"]
    log.clear()
    _tool_results([_ai("", [("k1", {"tag": "a"}, "c1"), ("nope", {}, "c2")]), _ai("ok")], [k1])
    assert log == [("start", "a"), ("end", "a")]                        # an unknown name is no kernel tool


def _submit(accept):
    async def submit(answer: str) -> str:
        if not accept(answer):
            raise ToolException("rejected")
        return "submitted"
    return StructuredTool.from_function(coroutine=submit, name="submit", description="Submit.", return_direct=True)


def test_an_accepted_submit_ends_the_run_and_later_calls_are_not_run():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1"), ("submit", {"answer": "x"}, "c2"),
                       ("add", {"a": 3, "b": 4}, "c3")]), _ai("never asked")]
    messages, log = asyncio.run(_run(_Ours, script, [add, _submit(lambda a: True)], None, [HumanMessage("go")]))
    assert len(log["prompts"]) == 1
    assert [m["content"] for m in messages if m["type"] == "tool"] == [
        "3", "submitted", "Not run: the turn ended when submit was accepted."]
    assert messages[-1]["status"] == "success" and messages[-1]["tool_call_id"] == "c3"


def test_a_rejected_submit_goes_back_to_the_model_and_later_calls_run():
    script = [_ai("", [("submit", {"answer": "bad"}, "c1"), ("add", {"a": 1, "b": 1}, "c2")]),
              _ai("", [("submit", {"answer": "good"}, "c3")])]
    messages, log = asyncio.run(_run(_Ours, script, [add, _submit(lambda a: a == "good")], None, [HumanMessage("go")]))
    assert len(log["prompts"]) == 2
    assert [(m["status"], m["content"]) for m in messages if m["type"] == "tool"] == [
        ("error", "Error: rejected"), ("success", "2"), ("success", "submitted")]


def test_a_delegate_call_ends_its_batch_but_not_the_run():
    async def builder(briefing: str) -> str:
        return f"built {briefing}"
    delegate = StructuredTool.from_function(coroutine=builder, name="coder", description="Brief the coder.",
                                            metadata={"ends_batch": True})
    script = [_ai("", [("coder", {"briefing": "x"}, "c1"), ("coder", {"briefing": "y"}, "c2")]), _ai("read it")]
    messages, log = asyncio.run(_run(_Ours, script, [delegate], None, [HumanMessage("go")]))
    assert [m["content"] for m in messages if m["type"] == "tool"] == [
        "built x", "Not run: coder ran in this turn; read its result first."]
    assert len(log["prompts"]) == 2 and messages[-1]["content"] == "read it"


# ---- compaction ----

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


def _compacting(script, monkeypatch, history=None):
    monkeypatch.setattr(harness, "COMPACT_PROMPT", type("P", (), {"read_text": lambda self: "COMPACT NOW"})())
    return asyncio.run(_run(_Ours, script, TOOLS, "sys", history or [HumanMessage("CONTEXT\n\nBRIEFING")],
                            context_window=1000, pinned="CONTEXT\n\nBRIEFING"))


def test_compaction_keeps_the_context_and_briefing_verbatim_with_the_summary_in_the_first_user_message(monkeypatch):
    # call 1 asks for a tool and reports 900 tokens >= 0.85 * 1000, so the harness compacts
    # before call 2 (the summarizer); call 3 continues from context, briefing and summary.
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900), _ai("SUMMARY TEXT"), _ai("done", total=50)]
    messages, log = _compacting(script, monkeypatch)
    summarizer = log["prompts"][1]
    assert [m["type"] for m in summarizer] == ["system", "human", "ai", "tool", "human"]
    assert summarizer[0]["content"] == "sys" and summarizer[-1]["content"] == "COMPACT NOW"
    assert len(log["bind"]) == 2 and log["bind"][1][1] == {"tool_choice": None}     # the summarizer reuses the bind
    first = "CONTEXT\n\nBRIEFING\n\n" + CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m["type"] for m in messages] == ["human", "ai"]
    assert messages[0]["content"] == first and log["prompts"][2][1]["content"] == first
    assert messages[-1]["content"] == "done"


def test_below_the_threshold_messages_append_linearly(monkeypatch):
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=800), _ai("done", total=820)]
    messages, log = _compacting(script, monkeypatch)
    assert [m["type"] for m in messages] == ["human", "ai", "tool", "ai"] and len(log["prompts"]) == 2


def test_compaction_sees_every_result_of_a_batch_together(monkeypatch):
    # Neither output alone reaches the threshold (700 + 350/4 = 787 < 850); together they do.
    script = [_ai("", [("slow_echo", {"text": "A" * 350, "delay": 0.0}, "c1"),
                       ("slow_echo", {"text": "B" * 350, "delay": 0.0}, "c2")], total=700),
              _ai("SUMMARY TEXT"), _ai("done")]
    messages, log = _compacting(script, monkeypatch)
    assert len(log["prompts"]) == 3 and [m["type"] for m in messages] == ["human", "ai"]


def test_compaction_falls_back_to_tool_choice_none_when_the_model_calls_a_tool_instead(monkeypatch):
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900),
              _ai("", [("add", {"a": 9, "b": 9}, "cX")]),   # summarizer attempt: tool call, no text
              _ai("SUMMARY TEXT"),                          # fallback retry: plain text
              _ai("done", total=50)]
    messages, log = _compacting(script, monkeypatch)
    assert len(log["prompts"]) == 4 and log["prompts"][1] == log["prompts"][2]
    assert log["bind"][1][1] == {"tool_choice": None} and log["bind"][2][1] == {"tool_choice": "none"}
    assert messages[0]["content"].endswith(CONTINUATION.format(summary="SUMMARY TEXT"))


def test_compaction_on_delegation_runs_before_the_delegate_and_its_results_come_as_a_user_message(monkeypatch):
    async def builder(briefing: str) -> str:
        return f"built {briefing}"
    delegate = StructuredTool.from_function(coroutine=builder, name="coder", description="Brief the coder.",
                                            metadata={"ends_batch": True, "delegates": True})
    monkeypatch.setattr(harness, "COMPACT_PROMPT", type("P", (), {"read_text": lambda self: "COMPACT NOW"})())
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1"), ("coder", {"briefing": "x"}, "c2")]),
              _ai("SUMMARY TEXT"), _ai("done")]
    messages, log = asyncio.run(_run(_Ours, script, [add, delegate], "sys", [HumanMessage("PINNED")],
                                     compact_on_delegation=True))
    summarizer = log["prompts"][1]
    assert [m["type"] for m in summarizer] == ["system", "human", "ai", "tool", "tool", "human"]
    assert summarizer[3]["content"] == harness.PENDING.format(name="add")
    assert [m["type"] for m in messages] == ["human", "human", "ai"]
    assert messages[0]["content"] == "PINNED\n\n" + CONTINUATION.format(summary="SUMMARY TEXT")
    assert [b["text"] for b in messages[1]["content"]] == ["add returned:", "3", "coder returned:", "built x"]


def test_without_compaction_on_delegation_a_delegate_call_is_not_compacted():
    async def builder(briefing: str) -> str:
        return "built"
    delegate = StructuredTool.from_function(coroutine=builder, name="coder", description="Brief the coder.",
                                            metadata={"ends_batch": True, "delegates": True})
    script = [_ai("", [("coder", {"briefing": "x"}, "c1")]), _ai("done")]
    messages, log = asyncio.run(_run(_Ours, script, [delegate], None, [HumanMessage("go")]))
    assert [m["type"] for m in messages] == ["human", "ai", "tool", "ai"] and len(log["prompts"]) == 2


def test_an_empty_summary_raises(monkeypatch):
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900), _ai("  ")]
    with pytest.raises(RuntimeError, match="no summary"):
        _compacting(script, monkeypatch)


def test_the_compaction_prompt_is_the_seed_prompt_file():
    from agent.harness import COMPACT_PROMPT
    assert COMPACT_PROMPT.is_file() and "Next step" in COMPACT_PROMPT.read_text()


# ---- every result goes to a file ----

def test_every_result_is_saved_and_a_long_one_is_shown_as_head_and_tail(workspace):
    from agent.harness import SHOW_WHOLE, SHOWN, spill
    assert spill("coder", "add", {"a": 1}, "3") == "3"
    [saved] = (workspace / "tool_output").glob("coder-add-*-*.txt")
    assert saved.read_text() == '$ add {"a": 1}\n3'
    text = "".join(f"{i}\n" for i in range(SHOW_WHOLE))
    out = spill("planner", "data_query", {}, text)
    [long] = (workspace / "tool_output").glob("planner-data_query-*.txt")
    assert long.read_text() == f"$ data_query {{}}\n{text}" and str(long) in out.split("\n", 1)[0]
    assert out.split("\n")[1] == "0" and out.endswith(text[-SHOWN:]) and len(out) - len(str(long)) < 2 * SHOWN + 300
    assert spill("r", "x", {}, "y" * SHOW_WHOLE) == "y" * SHOW_WHOLE


def test_run_commands_exit_line_stays_first(workspace):
    from agent.harness import SHOW_WHOLE, spill
    out = spill("coder", "run_command", {"command": "seq 9"}, "exit 0\n" + "1\n" * SHOW_WHOLE)
    assert out.startswith("exit 0\n[Long result: ")
    [saved] = (workspace / "tool_output").glob("coder-run_command-*.txt")
    assert saved.read_text().startswith("$ seq 9\nexit 0\n1\n")


def test_a_result_that_cannot_be_saved_says_so(workspace):
    from agent.harness import spill
    workspace.mkdir()
    (workspace / "tool_output").write_text("a file, not a folder")
    assert spill("r", "add", {}, "3").startswith("3\n[Error: the full result could not be saved")
    blocks = spill("r", "look", {}, [{"type": "image_url", "image_url": {"url": "data:,"}}])
    assert blocks[0]["type"] == "image_url" and blocks[1]["text"].startswith("[Error: the full result could not be saved")


def test_the_loop_names_files_by_role_and_tool(workspace):
    asyncio.run(_run(_Ours, SCENARIOS["one_call"], TOOLS, None, [HumanMessage("go")], role="edit_planner"))
    assert [p.name.split("-")[:2] for p in (workspace / "tool_output").iterdir()] == [["edit_planner", "add"]]


# ---- the write guard ----

@pytest.fixture
def guarded(tmp_path, workspace):
    tree = tmp_path / "agent"
    (tree / "agent").mkdir(parents=True)
    (tree / "agent" / "entry.py").write_text("v1\n")
    return tree, Guard(str(tree), "coder", str(tmp_path / "missing" / "guard.git"))


def _shell(cwd):
    from agent.tools import make_file_tools
    return {t.name: t for t in make_file_tools(str(cwd))}


def test_a_non_owners_change_is_reverted_and_reported(guarded):
    tree, guard = guarded
    tools = _shell(tree)
    script = [_ai("", [("run_command", {"command": "echo v2 > agent/entry.py; touch agent/new.py; mkdir -p d/e; "
                                                   "touch d/e/f"}, "c1")]), _ai("ok")]
    messages, _ = asyncio.run(_run(_Ours, script, [tools["run_command"]], None, [HumanMessage("go")],
                                   role="edit_planner", guard=guard))
    result = messages[2]
    assert result["status"] == "error" and result["content"].startswith(
        f"Error: run_command changed files in {tree}, which belongs to coder: ")
    for path in ("agent/entry.py (reverted)", "agent/new.py (reverted)", "d/e/f (reverted)"):
        assert path in result["content"]
    assert result["content"].endswith("What run_command returned:\nexit 0\n")
    assert (tree / "agent" / "entry.py").read_text() == "v1\n" and not (tree / "agent" / "new.py").exists()
    assert not (tree / "d").exists()


def test_the_owner_and_unguarded_tools_are_not_checked(guarded):
    tree, guard = guarded
    tools = _shell(tree)
    script = [_ai("", [("write_file", {"path": "agent/x.py", "content": "x"}, "c1")]), _ai("ok")]
    messages, _ = asyncio.run(_run(_Ours, script, [tools["write_file"]], None, [HumanMessage("go")],
                                   role="coder", guard=guard))
    assert messages[2]["status"] == "success" and (tree / "agent" / "x.py").exists() and guard.stale

    async def coder(briefing: str) -> str:            # a delegate call runs the owner
        (tree / "agent" / "y.py").write_text("y")
        guard.stale = True
        return "built"
    delegate = StructuredTool.from_function(coroutine=coder, name="coder", description="d", metadata={"ends_batch": True})
    script = [_ai("", [("run_command", {"command": "true"}, "c1")]), _ai("", [("coder", {"briefing": "b"}, "c2")]),
              _ai("", [("run_command", {"command": "cat agent/y.py"}, "c3")]), _ai("ok")]
    messages, _ = asyncio.run(_run(_Ours, script, [tools["run_command"], delegate], None, [HumanMessage("go")],
                                   role="edit_planner", guard=guard))
    assert [m["status"] for m in messages if m["type"] == "tool"] == ["success"] * 3
    assert (tree / "agent" / "y.py").read_text() == "y"            # the baseline was retaken after the owner


def test_a_guard_that_cannot_take_its_baseline_runs_nothing_and_says_so(guarded, monkeypatch):
    tree, guard = guarded
    tools = _shell(tree)
    monkeypatch.setattr(guard, "_git", lambda *a: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git", stderr="boom")))
    script = [_ai("", [("run_command", {"command": "touch agent/z.py"}, "c1")]), _ai("ok")]
    messages, _ = asyncio.run(_run(_Ours, script, [tools["run_command"]], None, [HumanMessage("go")],
                                   role="edit_planner", guard=guard))
    assert messages[2]["status"] == "error" and messages[2]["content"].startswith(
        f"Error: the write guard of {tree} failed: CalledProcessError: boom, so run_command was not run")
    assert not (tree / "agent" / "z.py").exists()


def test_a_guard_that_cannot_clear_its_git_dir_raises(tmp_path):
    locked = tmp_path / "guard.git" / "objects" / "zz"
    locked.mkdir(parents=True)
    (locked / "f").write_text("x")
    locked.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            Guard(str(tmp_path), "coder", str(tmp_path / "guard.git"))
    finally:
        locked.chmod(0o700)


# ---- the seed's own tools ----

def test_edit_file_needs_exactly_one_match():
    from agent.tools import replace_once
    assert replace_once("a b a", "b", "c") == "a c a"
    with pytest.raises(ValueError, match="found 2"):
        replace_once("a b a", "a", "c")
    with pytest.raises(ValueError, match="found 0"):
        replace_once("a", "z", "c")


def test_edit_file_refuses_non_utf8_files_and_leaves_them_untouched(tmp_path):
    raw = b"caption: caf\xe9\n"                              # Latin-1, not UTF-8
    (tmp_path / "c.txt").write_bytes(raw)
    with pytest.raises(ValueError, match="not UTF-8"):
        _shell(tmp_path)["edit_file"].invoke({"path": "c.txt", "old_string": "caption", "new_string": "title"})
    assert (tmp_path / "c.txt").read_bytes() == raw


def test_file_tools_work_relative_to_their_folder_and_take_absolute_paths(tmp_path):
    tools = _shell(tmp_path / "scratch")
    assert "relative to " + str(tmp_path / "scratch") in tools["write_file"].description
    assert f"in {tmp_path / 'scratch'}" in tools["run_command"].description
    tools["write_file"].invoke({"path": "a.txt", "content": "x"})
    tools["write_file"].invoke({"path": str(tmp_path / "b.txt"), "content": "y"})
    assert (tmp_path / "scratch" / "a.txt").read_text() == "x" and (tmp_path / "b.txt").read_text() == "y"
    assert tools["run_command"].invoke({"command": "cat a.txt; exit 3"}) == "exit 3\nx"


def test_run_command_timeout_kills_what_the_shell_started(tmp_path):
    run = _shell(tmp_path)["run_command"]
    out = run.invoke({"command": "(sleep 3; touch late.txt) & wait", "timeout_s": 1})
    assert out.startswith("timed out after 1s")
    time.sleep(3)
    assert not (tmp_path / "late.txt").exists()


def test_kernel_tools_run_in_parallel_and_leave_out_snapshot():
    from types import SimpleNamespace
    from agent.tools import kernel_tools
    specs = [SimpleNamespace(name=n, description=n, input_schema={"type": "object", "properties": {}})
             for n in ("data_query", "snapshot", "ask")]

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=specs)
    tools = asyncio.run(kernel_tools(Session()))
    assert [t.name for t in tools] == ["data_query", "ask"] and all(t.metadata == {"parallel": True} for t in tools)
