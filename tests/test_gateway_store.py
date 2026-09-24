import pytest

from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry.recorder import Recorder, TelemetryError
from ar_kernel.tools.context import TokenRegistry


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path)
    caller = TokenRegistry(rec).issue(node="n1", phase="improve_recipe", attempt=2,
                                      workspace_host=tmp_path, staging_host=tmp_path)
    return rec, CallStore(rec), caller


def _resp(rid, output):
    return {"id": rid, "output": output, "usage": {"input_tokens": 3, "output_tokens": 2}}


FIRST_INPUT = [{"role": "user", "content": "build a dataset"}]
FIRST_OUTPUT = [{"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "data_query",
                 "arguments": "{}", "status": "completed"}]


def test_request_and_response_are_recorded_with_attribution(env):
    rec, store, caller = env
    meta = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(meta, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0.5, attempts=1)
    events = rec.read_events("n1")
    kinds = [e["type"] for e in events]
    assert kinds == ["llm.request", "llm.response"]
    assert all(e["component"] == "gateway" and e["attempt"] == 2 for e in events)
    assert events[0]["phase"] == "improve_recipe"
    assert rec.load_payload(events[1]["payload"])["body"]["id"] == "r1"


def test_persistence_failure_raises_so_the_call_is_never_forwarded(env, monkeypatch):
    rec, store, caller = env
    def broken(*a, **k):
        raise TelemetryError("disk full")
    monkeypatch.setattr(rec, "event", broken)
    with pytest.raises(TelemetryError):
        store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})


def test_previous_response_id_links_the_conversation(env):
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(caller, "/v1/responses",
                     {"model": "m", "previous_response_id": "r1",
                      "input": [{"type": "function_call_output", "call_id": "c1", "output": "[]"}]})
    assert m2["conversation_id"] == m1["conversation_id"]
    assert m2["parent_call_id"] == m1["call_id"] and m2["turn_index"] == 1


def test_resent_history_links_by_prefix(env):
    """How a Responses client continues a run (verified for ChatOpenAI, fact 6): it resends
    the prior input and output items, plus the tool output."""
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    resent = FIRST_INPUT + [dict(FIRST_OUTPUT[0], status=None)] + [
        {"type": "function_call_output", "call_id": "c1", "output": "[]"}]
    m2 = store.begin(caller, "/v1/responses", {"model": "m", "input": resent})
    assert m2["conversation_id"] == m1["conversation_id"] and m2["turn_index"] == 1


def test_unrelated_request_starts_a_new_conversation(env):
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(caller, "/v1/responses",
                     {"model": "m", "input": [{"role": "user", "content": "something else"}]})
    assert m2["conversation_id"] != m1["conversation_id"] and m2["turn_index"] == 0


def test_chat_completions_link_by_message_prefix(env):
    _, store, caller = env
    msgs = [{"role": "user", "content": "hi"}]
    m1 = store.begin(caller, "/v1/chat/completions", {"model": "m", "messages": msgs})
    store.end(m1, caller, status=200, latency_s=0, attempts=1,
              body={"id": "cc1", "choices": [{"message": {"role": "assistant", "content": "yo"}}]})
    m2 = store.begin(caller, "/v1/chat/completions", {"model": "m", "messages": msgs + [
        {"role": "assistant", "content": "yo"}, {"role": "user", "content": "more"}]})
    assert m2["conversation_id"] == m1["conversation_id"]


def test_prefix_matching_never_crosses_containers(tmp_path):
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
    a = reg.issue(node="n1", phase="p", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    b = reg.issue(node="n2", phase="p", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    m1 = store.begin(a, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, a, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(b, "/v1/responses", {"model": "m", "input": FIRST_INPUT + FIRST_OUTPUT})
    assert m2["conversation_id"] != m1["conversation_id"]


# --- Controller ruling: memory bound for CallStore (forget()/on_revoke wiring) ---

def test_forget_starts_a_new_conversation_for_a_later_request(env):
    """After forget(), a later request from the same token must not link to the
    forgotten call's conversation: it starts a fresh conversation at turn 0."""
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)

    store.forget(caller.token)

    resent = FIRST_INPUT + [dict(FIRST_OUTPUT[0], status=None)] + [
        {"type": "function_call_output", "call_id": "c1", "output": "[]"}]
    m2 = store.begin(caller, "/v1/responses", {"model": "m", "input": resent})
    assert m2["conversation_id"] != m1["conversation_id"]
    assert m2["turn_index"] == 0


def test_forget_removes_linking_state(env):
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)

    store.forget(caller.token)

    assert m1["call_id"] not in store._calls
    assert "r1" not in store._by_response
    assert store._last_in_conv.get(m1["conversation_id"]) is None


def test_end_after_forget_records_response_without_error(env):
    """A call still in flight for a revoked token must not crash end(): the
    llm.response event is still recorded, linking-state updates are just skipped."""
    rec, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})

    store.forget(caller.token)

    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0.1, attempts=1)

    events = rec.read_events("n1")
    kinds = [e["type"] for e in events]
    assert kinds == ["llm.request", "llm.response"]
    assert rec.load_payload(events[1]["payload"])["body"]["id"] == "r1"
    # linking state was not resurrected by end()
    assert m1["call_id"] not in store._calls
    assert "r1" not in store._by_response


def test_on_revoke_wired_to_forget_via_registry(tmp_path):
    """Sanity check that TokenRegistry.on_revoke + CallStore.forget compose the way
    Task 5's gateway app factory will wire them: registry.on_revoke(store.forget)."""
    rec = Recorder(tmp_path)
    reg = TokenRegistry(rec)
    store = CallStore(rec)
    reg.on_revoke(store.forget)

    caller = reg.issue(node="n1", phase="p", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)

    reg.revoke(caller.token)

    assert m1["call_id"] not in store._calls


def test_chat_model_resent_history_links_to_a_real_style_chat_response(env):
    """Task 19: the agent's ChatOpenAI (Chat Completions) resends the prior request plus the
    assistant message. Upstream messages carry fields the client never sends back (OpenAI's
    `refusal`/`annotations`, a tool-call `index`, empty content next to tool calls)
    and may space the arguments JSON differently; linking must survive all of that."""
    import json

    import httpx
    from langchain_core.messages import HumanMessage, ToolMessage
    from ar_contract.client import ReasoningChatOpenAI
    reply = {"id": "cc1", "object": "chat.completion", "created": 1, "model": "m",
             "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                 "role": "assistant", "content": "", "refusal": None, "annotations": [],
                 "reasoning_content": "plan", "tool_calls": [
                     {"index": 0, "id": "c1", "type": "function",
                      "function": {"name": "data_query", "arguments": '{"q":1,"a":"b"}'}}]}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply)
    model = ReasoningChatOpenAI(model="m", base_url="http://up/v1", api_key="k", max_retries=0,
                                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    first = model.invoke([HumanMessage("go")])
    model.invoke([HumanMessage("go"), first, ToolMessage("[]", tool_call_id="c1")])
    _, store, caller = env
    m1 = store.begin(caller, "/v1/chat/completions", sent[0])
    store.end(m1, caller, status=200, body=reply, latency_s=0, attempts=1)
    m2 = store.begin(caller, "/v1/chat/completions", sent[1])
    assert m2["conversation_id"] == m1["conversation_id"] and m2["turn_index"] == 1
