"""ar_contract runs inside the container; these tests run it in-process with
temp directories standing in for /agent, /context and /workspace."""
import json
import textwrap

import pytest

from ar_contract import models
from ar_contract.run import main


def _edit_ctx(**over):
    base = {"nodes_remaining": 5, "attempt": 1, "max_attempts": 3}
    return {**base, **over}


def _layout(tmp_path, entry_src, ctx):
    agent = tmp_path / "agent_repo"
    (agent / "agent").mkdir(parents=True)
    (agent / "agent" / "__init__.py").write_text("")
    (agent / "agent" / "entry.py").write_text(textwrap.dedent(entry_src))
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "context.json").write_text(json.dumps(ctx))
    (tmp_path / "workspace").mkdir()
    return agent


@pytest.fixture
def env(tmp_path, monkeypatch):
    def setup(entry_src, ctx):
        agent = _layout(tmp_path, entry_src, ctx)
        monkeypatch.setenv("AR_AGENT_DIR", str(agent))
        monkeypatch.setenv("AR_CONTEXT_DIR", str(tmp_path / "context"))
        monkeypatch.setenv("AR_WORKSPACE", str(tmp_path / "workspace"))
        # ar_contract.run imports `agent.entry` from AR_AGENT_DIR via sys.path;
        # prepend through monkeypatch so it is undone automatically, and drop any
        # cached `agent`/`agent.*` modules so each test imports its own entry.py.
        monkeypatch.syspath_prepend(str(agent))
        import sys
        for name in [n for n in sys.modules if n == "agent" or n.startswith("agent.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        return tmp_path / "workspace" / "result.json"
    yield setup
    import sys
    for name in [n for n in sys.modules if n == "agent" or n.startswith("agent.")]:
        del sys.modules[name]


GOOD = """
from ar_contract.models import EditResult, RecipeResult
def edit_self(ctx):
    return EditResult(summary=f"attempt {ctx.attempt}")
def improve_recipe(ctx):
    return RecipeResult(data_commit="c1", recipe={"optimizer.lr": 1e-5}, rationale="why")
"""


def test_good_edit_self_writes_ok_result(env):
    out = env(GOOD, _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text()) == {"ok": True, "result": {"summary": "attempt 1"}}


def test_the_running_code_comes_from_the_code_dir_when_one_is_set(env, tmp_path, monkeypatch):
    out = env("raise RuntimeError('the tree being edited is broken')\n", _edit_ctx())
    code = tmp_path / "code"
    (code / "agent").mkdir(parents=True)
    (code / "agent" / "__init__.py").write_text("")
    (code / "agent" / "entry.py").write_text(GOOD)
    monkeypatch.setenv("AR_CODE_DIR", str(code))
    assert main(["edit_self"]) == 0 and json.loads(out.read_text())["ok"] is True


def test_async_entry_points_are_supported(env):
    out = env("""
from ar_contract.models import EditResult
async def edit_self(ctx):
    return EditResult(summary="async ok")
def improve_recipe(ctx):
    raise NotImplementedError
""", _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text())["result"]["summary"] == "async ok"


def test_raising_entry_point_writes_error_and_nonzero_exit(env):
    out = env("""
def edit_self(ctx):
    raise RuntimeError("boom")
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 1
    body = json.loads(out.read_text())
    assert body["ok"] is False and "boom" in body["error"] and "Traceback" in body["traceback"]


def test_a_failure_inside_a_task_group_is_named_not_the_group(env):
    """live-10-02 n1: the retry was told only 'ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)'."""
    out = env("""
import asyncio
async def edit_self(ctx):
    async def fail():
        raise ValueError("the provider returned an empty response")
    async with asyncio.TaskGroup() as outer:
        async def inner():
            async with asyncio.TaskGroup() as group:
                group.create_task(fail())
        outer.create_task(inner())
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 1
    assert json.loads(out.read_text())["error"] == "ValueError: the provider returned an empty response"


def test_invalid_result_is_a_schema_failure(env):
    out = env("""
def edit_self(ctx):
    return {"summary": ""}          # empty summary violates min_length=1
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 1
    assert "summary" in json.loads(out.read_text())["error"]


def test_dict_results_are_validated_and_accepted(env):
    out = env("""
def edit_self(ctx):
    return {"summary": "plain dict is fine"}
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text())["result"]["summary"] == "plain dict is fine"


def test_unknown_kind_is_rejected(env):
    env(GOOD, _edit_ctx())
    assert main(["drop_tables"]) == 2


def test_recipe_result_requires_commit_recipe_and_rationale():
    with pytest.raises(Exception):
        models.RecipeResult(data_commit="", recipe={}, rationale="x")
    ok = models.RecipeResult(data_commit="c", recipe={"optimizer.lr": 1e-5}, rationale="r")
    assert ok.recipe["optimizer.lr"] == 1e-5




def test_clients_speak_over_the_socket_directory(monkeypatch, tmp_path):
    """The chat model talks to the gateway socket (Chat Completions, no client-side retries); the MCP
    session is an async context manager. Built without connecting."""
    monkeypatch.setenv("AR_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("AR_TOKEN", "tok-abc")
    monkeypatch.setenv("AR_DEFAULT_MODEL", "gpt-x")
    import inspect
    import httpx
    from ar_contract import client
    model = client.chat_model()
    assert model.model_name == "gpt-x" and model.openai_api_base == "http://localhost/v1"
    assert model.openai_api_key.get_secret_value() == "tok-abc"
    assert isinstance(model, client.ReasoningChatOpenAI)
    assert model.use_responses_api is False and model.max_retries == 0
    assert isinstance(model.http_client, httpx.Client)             # sync invoke() also uses the socket
    assert isinstance(model.http_async_client, httpx.AsyncClient)
    assert inspect.isasyncgenfunction(client.mcp_session.__wrapped__)


def _stub_model(replies, seen):
    """The client class on a stub transport: records request bodies, returns `replies`."""
    import httpx
    from ar_contract.client import ReasoningChatOpenAI
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(seen) - 1])
    return ReasoningChatOpenAI(model="m", base_url="http://up/v1", api_key="k",
                               use_responses_api=False, max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _chat_reply(message):
    return {"id": "cc1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}


def test_reasoning_round_trips_through_the_chat_model():
    """A reply's `reasoning` (OpenAI's field) is kept on the AIMessage and sent back with that
    assistant message; stock ChatOpenAI drops it both ways. Messages that never had it are sent
    unchanged."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    seen = []
    model = _stub_model([
        _chat_reply({"role": "assistant", "content": "", "reasoning": "think first",
                     "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "echo", "arguments": '{"x": 1}'}}]}),
        _chat_reply({"role": "assistant", "content": "done", "reasoning": ""})], seen)
    first = model.invoke([HumanMessage("go")])
    assert first.additional_kwargs["reasoning"] == "think first"
    assert first.tool_calls[0]["name"] == "echo"
    plain = AIMessage("earlier reply without reasoning")
    second = model.invoke([HumanMessage("go"), plain, HumanMessage("again"), first,
                           ToolMessage("1", tool_call_id="c1")])
    assert second.text == "done" and second.additional_kwargs["reasoning"] == ""
    assistants = [m for m in seen[1]["messages"] if m["role"] == "assistant"]
    assert "reasoning" not in assistants[0]
    assert assistants[1]["reasoning"] == "think first" and "reasoning_content" not in assistants[1]
    assert assistants[1]["tool_calls"][0]["id"] == "c1"


@pytest.mark.parametrize("message,expected", [
    ({"reasoning_content": "backup"}, "backup"),                        # only the other name
    ({"reasoning": "", "reasoning_content": "backup"}, "backup"),       # reasoning empty
    ({"reasoning": None, "reasoning_content": "backup"}, "backup"),
    ({"reasoning": "main", "reasoning_content": "other"}, "main"),      # reasoning wins
])
def test_reasoning_content_is_only_the_fallback(message, expected):
    from langchain_core.messages import HumanMessage
    seen = []
    model = _stub_model([_chat_reply({"role": "assistant", "content": "hi", **message})] * 2, seen)
    first = model.invoke([HumanMessage("go")])
    assert first.additional_kwargs["reasoning"] == expected and "reasoning_content" not in first.additional_kwargs
    model.invoke([HumanMessage("go"), first, HumanMessage("more")])
    assert seen[1]["messages"][1] == {"role": "assistant", "content": "hi", "reasoning": expected}


def test_reasoning_is_never_invented():
    from langchain_core.messages import HumanMessage
    seen = []
    model = _stub_model([_chat_reply({"role": "assistant", "content": "hi"})] * 2, seen)
    first = model.invoke([HumanMessage("go")])
    assert "reasoning" not in first.additional_kwargs and "reasoning_content" not in first.additional_kwargs
    model.invoke([HumanMessage("go"), first, HumanMessage("more")])
    assert seen[1]["messages"][1] == {"role": "assistant", "content": "hi"}
