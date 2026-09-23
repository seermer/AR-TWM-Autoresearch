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
    assert json.loads(out.read_text()) == {"ok": True, "result": {"summary": "attempt 1", "component": None}}


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


def test_edit_result_component_is_optional_and_checked():
    assert models.EDIT_COMPONENTS == ("prompts", "tools", "harness", "orchestration", "knowledge")
    assert models.EditResult(summary="s").component is None
    assert models.EditResult(summary="s", component="harness").component == "harness"
    with pytest.raises(Exception):
        models.EditResult(summary="s", component="everything")


def test_clients_speak_over_the_socket_directory(monkeypatch, tmp_path):
    """Facts 2-6: the chat model talks to the gateway socket (Responses API, no client-side
    retries); the MCP session is an async context manager. Built without connecting; Task 6
    and Task 17 exercise both over real sockets."""
    monkeypatch.setenv("AR_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("AR_TOKEN", "tok-abc")
    monkeypatch.setenv("AR_DEFAULT_MODEL", "gpt-x")
    import inspect
    import httpx
    from ar_contract import client
    model = client.chat_model()
    assert model.model_name == "gpt-x" and model.openai_api_base == "http://localhost/v1"
    assert model.openai_api_key.get_secret_value() == "tok-abc"
    assert model.use_responses_api is True and model.max_retries == 0
    assert isinstance(model.http_client, httpx.Client)             # sync invoke() also uses the socket
    assert isinstance(model.http_async_client, httpx.AsyncClient)
    assert inspect.isasyncgenfunction(client.mcp_session.__wrapped__)
