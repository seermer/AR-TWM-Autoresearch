import json

import httpx
import pytest
from conftest import telemetry_payload, telemetry_records
from fastapi.testclient import TestClient

from ar_kernel import telemetry
from ar_kernel.archive.activity import calls
from ar_kernel.archive.db import open_db
from ar_kernel.budget import Budget
from ar_kernel.gateway.app import PROBE_PROMPT, Upstream, create_gateway_app
from ar_kernel.gateway.mock import MockBook, function_call, message
from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry import tree
from ar_kernel.tools.context import TokenRegistry


def _upstream_that(handler, seen):
    def wrapped(request: httpx.Request):
        seen.append(json.loads(request.content))
        return handler(request)
    async def no_sleep(_s):
        pass
    return Upstream("https://api.example/v1", "sk-REALKEY", timeout_s=5, wait_s=7,     # sleeps 1+2+4: four tries
                    transport=httpx.MockTransport(wrapped), sleep=no_sleep)


def _ok(request):
    return httpx.Response(200, json={"id": "resp_up", "object": "response", "output": [],
                                     "usage": {"input_tokens": 1, "output_tokens": 1}})


@pytest.fixture
def make(tmp_path):
    def build(handler=_ok, mock_script=None, allowed=frozenset({"gpt-x"})):
        reg, store, seen = TokenRegistry(), CallStore(tmp_path), []
        caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                           staging_host=tmp_path, mock_script=mock_script)
        app = create_gateway_app(registry=reg, store=store, allowed_models=set(allowed),
                                 upstream=_upstream_that(handler, seen), mocks=MockBook.default())
        return TestClient(app), caller, lambda: calls(open_db(tmp_path), "n1"), seen
    return build


def _post(client, token, body, path="/v1/responses"):
    return client.post(path, json=body, headers={"Authorization": f"Bearer {token}"})


def test_unknown_token_is_401(make):
    client, _, _, seen = make()
    assert _post(client, "ar-nope", {"model": "gpt-x", "input": "hi"}).status_code == 401
    assert seen == []


def test_disallowed_model_is_403_and_never_forwarded(make):
    client, caller, _, seen = make()
    assert _post(client, caller.token, {"model": "gpt-other", "input": "hi"}).status_code == 403
    assert seen == []


def test_streaming_is_refused(make):
    client, caller, _, _ = make()
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi", "stream": True}).status_code == 400



@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe"], ids=["malformed", "array", "binary"])
def test_a_body_that_is_not_a_json_object_is_400_and_never_forwarded(make, content):
    client, caller, _, seen = make()
    r = client.post("/v1/chat/completions", content=content,
                    headers={"Authorization": f"Bearer {caller.token}", "Content-Type": "application/json"})
    assert r.status_code == 400 and "JSON object" in r.json()["error"]["message"]
    assert seen == []


@pytest.mark.parametrize("content", [
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
    {"type": "video", "source": {"url": "https://example.com/clip.mp4"}},
    {"type": "input_video", "video_url": "https://example.com/clip.mp4"},
    {"type": "file", "file": {"file_data": "data:video/mp4;base64,AAAA", "filename": "clip.mp4"}},
    {"type": "input_file", "mime_type": "video/mp4", "file_data": "AAAA"},
], ids=["video_url", "video", "input_video", "file_mime", "input_file_mime"])
def test_chat_completions_rejects_video_content_parts(make, content):
    client, caller, _, seen = make()
    body = {"model": "gpt-x", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look at this"}, content]}]}
    r = _post(client, caller.token, body, path="/v1/chat/completions")
    assert r.status_code == 400 and "caption_videos" in r.json()["error"]["message"]
    assert seen == []


def test_chat_completions_rejects_images(make):
    """Images reach the model only through the ask tool, which caps them."""
    client, caller, _, seen = make()
    body = {"model": "gpt-x", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]}
    r = _post(client, caller.token, body, path="/v1/chat/completions")
    assert r.status_code == 400 and "ask kernel tool" in r.json()["error"]["message"]
    assert seen == []


@pytest.mark.parametrize("item", [
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
    {"type": "input_video", "video_url": "https://example.com/clip.mp4"},
    {"type": "message", "role": "user", "content": [
        {"type": "input_file", "mime_type": "video/mp4", "file_data": "AAAA"}]},
], ids=["video_url_item", "input_video_item", "nested_video_file"])
def test_responses_rejects_video_input_items(make, item):
    client, caller, _, seen = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": [item]})
    assert r.status_code == 400 and "caption_videos" in r.json()["error"]["message"]
    assert seen == []


def test_responses_rejects_images(make):
    client, caller, _, seen = make()
    body = {"model": "gpt-x", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"}]}]}
    r = _post(client, caller.token, body)
    assert r.status_code == 400 and "ask kernel tool" in r.json()["error"]["message"]
    assert seen == []


def test_forwarded_call_is_recorded(make):
    client, caller, rows, seen = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and r.json()["id"] == "resp_up"
    assert len(seen) == 1
    assert [row["status"] for row in rows()] == [200]


@pytest.mark.parametrize("header,role", [("coder", "coder"), ("../x y", "___x_y"), (None, None), ("  ", None)])
def test_the_role_header_labels_the_call(make, header, role):
    client, caller, rows, _ = make()
    headers = {"Authorization": f"Bearer {caller.token}", **({"X-AR-Role": header} if header is not None else {})}
    assert client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"}, headers=headers).status_code == 200
    assert [row["role"] for row in rows()] == [role]


def test_upstream_429_is_retried_then_succeeds(make):
    calls = {"n": 0}
    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if calls["n"] < 3 else _ok(request)
    client, caller, _, _ = make(handler=flaky)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
    assert calls["n"] == 3


def _is_probe(request) -> bool:
    return PROBE_PROMPT in request.content.decode()


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_a_4xx_the_provider_also_gives_the_probe_is_waited_out(make, status):
    """A rejected key, spent credits, a field the provider rejects: the agent cannot fix any of them."""
    calls = {"n": 0}
    def flaky(request):
        calls["n"] += 1
        return httpx.Response(status, json={"error": "no"}) if calls["n"] < 3 else _ok(request)
    client, caller, _, seen = make(handler=flaky)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
    assert calls["n"] == 3 and PROBE_PROMPT in json.dumps(seen[1])          # the call, its probe, the call
    assert not caller.fault.event.is_set()


@pytest.mark.parametrize("status", [400, 403, 404, 413, 422])
def test_a_4xx_only_the_agents_request_gets_goes_back_to_the_agent_at_once(make, status):
    client, caller, _, seen = make(handler=lambda r: _ok(r) if _is_probe(r) else
                                     httpx.Response(status, json={"error": {"message": "context too long"}}))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == status and "context too long" in r.json()["error"]["message"]
    assert len(seen) == 2 and PROBE_PROMPT in json.dumps(seen[1]) and not caller.fault.event.is_set()


def test_a_probe_verdict_is_reused_for_a_while(make):
    client, caller, _, seen = make(handler=lambda r: _ok(r) if _is_probe(r) else httpx.Response(400, json={"error": "bad"}))
    for _ in range(3):
        assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 400
    assert len(seen) == 4                     # three calls, one probe


def test_persistent_upstream_failure_returns_its_status_and_stops_the_phase(make):
    client, caller, _, seen = make(handler=lambda r: httpx.Response(503, json={"error": "down"}))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 503
    assert len(seen) == 4                     # waited out: 1 + 2 + 4 s of the 7 s wait
    assert caller.fault.event.is_set() and "HTTP 503" in caller.fault.reason


def test_a_persistent_4xx_the_probe_also_gets_stops_the_phase(make):
    client, caller, _, seen = make(handler=lambda r: httpx.Response(401, json={"error": "bad key"}))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 401
    assert caller.fault.event.is_set()


def test_non_json_upstream_error_is_retried_recorded_and_returned(make):
    """Proxies answer 502/503/504 with HTML; that must not crash the gateway."""
    client, caller, rows, seen = make(handler=lambda r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "non-JSON" in r.json()["error"]["message"]
    assert len(seen) == 4                     # retried like any 5xx
    assert [row["status"] for row in rows()] == [502]


def test_non_json_upstream_error_body_is_returned_without_truncation(make):
    """A long non-JSON error body (e.g. a verbose HTML error page from a proxy) comes back in full, not clipped."""
    long_body = "<html>" + ("x" * 3000) + "</html>"
    client, caller, _, seen = make(handler=lambda r: httpx.Response(502, text=long_body))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502
    assert r.json()["error"]["body"] == long_body


def test_non_json_success_body_becomes_a_502(make):
    client, caller, _, _ = make(handler=lambda r: httpx.Response(200, text="not json"))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 502


def test_unexpected_gateway_exception_is_still_recorded(make, monkeypatch):
    async def bug(self, path, body):
        raise RuntimeError("gateway bug")
    monkeypatch.setattr(Upstream, "post", bug)
    client, caller, rows, _ = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "gateway bug" in r.json()["error"]["message"]
    assert [row["status"] for row in rows()] == [502]


def test_upstream_base_url_defaults_to_openai_v1_and_is_not_rewritten():
    import asyncio
    seen = []
    def record(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={})
    for base in (None, "", "https://proxy.example/api/v3/"):
        up = Upstream(base, "k", timeout_s=5, wait_s=0, transport=httpx.MockTransport(record))
        asyncio.run(up.post("/responses", {}))
    assert seen == ["https://api.openai.com/v1/responses", "https://api.openai.com/v1/responses",
                    "https://proxy.example/api/v3/responses"]


def test_mock_mode_serves_scripted_output_without_upstream(make):
    client, caller, rows, seen = make(mock_script="smoke")
    body = _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).json()
    assert body["output"][0]["content"][0]["text"] == "ok"
    assert seen == []
    assert [(row["status"], row["mock"]) for row in rows()] == [(200, 1)]


def test_mock_script_is_consumed_in_order_then_repeats_its_last_step(tmp_path):
    reg, store = TokenRegistry(), CallStore(tmp_path)
    book = MockBook()
    book.add("two_steps", [[function_call("data_query", {}, "c1")], [message("done")]])
    caller = reg.issue(node="n", phase="p", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path, mock_script="two_steps")
    client = TestClient(create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"},
                                           upstream=None, mocks=book))
    kinds = [_post(client, caller.token, {"model": "gpt-x", "input": "x"}).json()["output"][0]["type"]
             for _ in range(3)]
    assert kinds == ["function_call", "message", "message"]


def test_chat_completions_is_forwarded_to_the_matching_path(make):
    seen_paths = []
    def handler(request):
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"id": "cc", "choices": [{"message": {"role": "assistant", "content": "k"}}]})
    client, caller, _, _ = make(handler=handler)
    r = _post(client, caller.token, {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]},
              path="/v1/chat/completions")
    assert r.status_code == 200 and seen_paths == ["/v1/chat/completions"]


def test_revoking_a_token_drops_its_linking_state_in_the_store(tmp_path):
    """Create_gateway_app must wire registry.on_revoke(store.forget) so the
    store's memory is bound to live containers -- once a container's token is revoked, its
    linking state (requests, response ids, conversation tails) is dropped."""
    reg, store = TokenRegistry(), CallStore(tmp_path)
    caller = reg.issue(node="n1", phase="p", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    client = TestClient(create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"},
                                           upstream=None, mocks=MockBook.default()))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200
    assert store._calls        # linking state exists after a completed call

    reg.revoke(caller.token)

    assert store._calls == {}  # dropped by the on_revoke -> forget wiring
    assert reg.lookup(caller.token) is None


def _make_with_effort(tmp_path, effort, handler=_ok):
    reg, store, seen = TokenRegistry(), CallStore(tmp_path), []
    caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    up = _upstream_that(handler, seen)
    up.effort = effort
    app = create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"}, upstream=up,
                             mocks=MockBook.default())
    return TestClient(app), caller, seen


def test_gateway_enforces_reasoning_effort_on_chat_completions(tmp_path):
    client, caller, seen = _make_with_effort(tmp_path, "low")
    r = _post(client, caller.token, {"model": "gpt-x", "reasoning_effort": "high",
                                     "messages": [{"role": "user", "content": "hi"}]},
              path="/v1/chat/completions")
    assert r.status_code == 200 and seen[0]["reasoning_effort"] == "low"


def test_gateway_enforces_reasoning_effort_on_responses(tmp_path):
    client, caller, seen = _make_with_effort(tmp_path, "low")
    _post(client, caller.token, {"model": "gpt-x", "input": "hi",
                                 "reasoning": {"effort": "high", "summary": "auto"}})
    _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert [s["reasoning"] for s in seen] == [{"effort": "low", "summary": "auto"}, {"effort": "low"}]


def test_gateway_forwards_the_agent_effort_when_none_is_configured(tmp_path):
    client, caller, seen = _make_with_effort(tmp_path, None)
    _post(client, caller.token, {"model": "gpt-x", "reasoning_effort": "high",
                                 "messages": [{"role": "user", "content": "hi"}]},
          path="/v1/chat/completions")
    assert seen[0]["reasoning_effort"] == "high"



def test_a_non_object_reasoning_field_is_replaced_not_a_crash(tmp_path):
    client, caller, seen = _make_with_effort(tmp_path, "low")
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi", "reasoning": "high"})
    assert r.status_code == 200 and seen[0]["reasoning"] == {"effort": "low"}

def test_upstream_from_env():
    up = Upstream.from_env({"OPENAI_BASE_URL": "https://llm.example", "OPENAI_API_KEY": "sk-x",
                            "OPENAI_EFFORT": "low"}, timeout_s=5, wait_s=1)
    assert (up._base, up._headers, up.effort) == ("https://llm.example",
                                                  {"Authorization": "Bearer sk-x"}, "low")
    up = Upstream.from_env({"OPENAI_EFFORT": ""}, timeout_s=5, wait_s=1)
    assert (up._base, up.effort) == ("https://api.openai.com/v1", None)


def test_mock_mode_serves_chat_completions(tmp_path):
    reg, store = TokenRegistry(), CallStore(tmp_path)
    book = MockBook()
    book.add("two_steps", [[function_call("data_query", {"q": 1}, "c1")], [message("done")]])
    caller = reg.issue(node="n", phase="p", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path, mock_script="two_steps")
    client = TestClient(create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"},
                                           upstream=None, mocks=book))
    body = {"model": "gpt-x", "messages": [{"role": "user", "content": "x"}]}
    first = _post(client, caller.token, body, path="/v1/chat/completions").json()
    second = _post(client, caller.token, body, path="/v1/chat/completions").json()
    assert first["object"] == "chat.completion" and first["model"] == "gpt-x"
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert first["choices"][0]["message"] == {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "data_query", "arguments": '{"q": 1}'}}]}
    assert second["choices"][0]["finish_reason"] == "stop"
    assert second["choices"][0]["message"] == {"role": "assistant", "content": "done"}
    assert set(first["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_budget_cap_blocks_further_real_calls_but_not_mock_calls(tmp_path):
    """The stub's usage (1 input token + 1 output token, priced at $1/Mtok each) costs
    more than the $0.000001 cap, so the cap is reached after the very first real call."""
    seen = []
    reg, store = TokenRegistry(), CallStore(tmp_path)
    real = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                     staging_host=tmp_path)
    mocked = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path, mock_script="smoke")
    upstream = _upstream_that(_ok, seen)
    budget = Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})
    app = create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"}, upstream=upstream,
                             mocks=MockBook.default(), budget=budget)
    client = TestClient(app)

    r1 = _post(client, real.token, {"model": "gpt-x", "input": "hi"})
    assert r1.status_code == 200
    rows = lambda: calls(open_db(tmp_path), "n1")
    assert rows()[-1]["cost_usd"] > 0 and rows()[-1]["mock"] == 0
    assert len(seen) == 1

    r2 = _post(client, real.token, {"model": "gpt-x", "input": "hi again"})
    assert r2.status_code == 402 and "budget" in r2.json()["error"]["message"]
    assert len(seen) == 1                                           # upstream never saw the 2nd call
    assert len(rows()) == 1                                         # nor was it recorded

    r3 = _post(client, mocked.token, {"model": "gpt-x", "input": "hi"})
    assert r3.status_code == 200
    assert rows()[-1]["mock"] == 1


def test_the_gateway_censors_tool_results_before_forwarding_and_recording(tmp_path):
    reg, store, seen = TokenRegistry(), CallStore(tmp_path), []
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    app = create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"}, censor=["wbench"],
                             upstream=_upstream_that(_ok, seen), mocks=MockBook.default())
    body = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"},
                                           {"role": "tool", "tool_call_id": "c", "content": "ranked on WBench. 3 clips ok."}]}
    assert _post(TestClient(app), caller.token, body, "/v1/chat/completions").status_code == 200
    assert seen[0]["messages"][1]["content"] == "3 clips ok."
    assert "wbench" not in "".join(p.read_text() for p in tmp_path.rglob("*.jsonl")).lower()


def test_an_error_body_sent_with_http_200_is_retried_like_its_code(make):
    def handler(request):                     # a router in front of the provider answers 200 with the provider's error
        if len(seen) < 3:
            return httpx.Response(200, json={"error": {"code": 502, "message": "Provider returned an empty response"}})
        return _ok(request)
    client, caller, _, seen = make(handler=handler)
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and "error" not in r.json() and len(seen) == 3


def test_a_persistent_error_body_sent_with_http_200_is_returned_as_its_code(make):
    client, caller, _, seen = make(handler=lambda request: httpx.Response(200, json={"error": {"code": 502, "message": "empty"}}))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 502 and len(seen) == 4


def test_an_error_inside_the_choice_sent_with_http_200_is_retried_too(make):
    """HTTP 200, choices[0].finish_reason 'error', choices[0].error.code 502."""
    def handler(request):
        if len(seen) < 2:
            return httpx.Response(200, json={"choices": [{"finish_reason": "error", "message": {"content": None},
                                                          "error": {"code": 502, "message": "Network connection lost."}}]})
        return _ok(request)
    client, caller, _, seen = make(handler=handler)
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and len(seen) == 2


def test_a_reply_with_no_content_and_no_tool_call_is_retried_too(make):
    """HTTP 200, finish_reason 'stop', content null, no tool call, twice in one role."""
    def handler(request):
        if len(seen) < 2:
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": None, "reasoning": "thinking only"}}]})
        return _ok(request)
    client, caller, _, seen = make(handler=handler)
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and len(seen) == 2


def _traced(tmp_path, handler=_ok, mock_script=None, budget=None):
    """A gateway whose caller was issued inside a phase attempt span, as the loop issues it."""
    run = tmp_path / "trun"
    telemetry.open(run)
    telemetry.start("ar.run")
    reg, seen = TokenRegistry(), []
    with telemetry.span("phase.attempt", {"node": "n1", "phase": "edit_self", "attempt": 1}) as attempt:
        caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path, staging_host=tmp_path,
                           mock_script=mock_script)
        attempt.output = {}
    # requests come from the server's thread, outside the attempt span
    app = create_gateway_app(registry=reg, store=CallStore(tmp_path), allowed_models={"gpt-x"},
                             upstream=_upstream_that(handler, seen), mocks=MockBook.default(), budget=budget)
    client = TestClient(app)

    def requests():
        records = telemetry_records(run)
        out = []
        for start in (r for r in records if r["kind"] == "start" and r["name"] == "llm.request"):
            mine = [r for r in records if r.get("span") == start["span"] and r["kind"] != "start"
                    or r["kind"] == "start" and r["parent"] == start["span"]]
            [end] = [r for r in mine if r["kind"] == "end" and r["span"] == start["span"]]
            out.append({"start": start, "end": end, "output": telemetry_payload(run, end["output"]),
                        "children": [r["name"] for r in mine if r["kind"] == "start"],
                        "decisions": [(r["name"], r["chosen"]) for r in mine if r["kind"] == "decision"]})
        return out
    yield client, caller, requests, attempt, run
    assert tree.check(tree.load(run)) == []


def test_an_ok_call_is_one_request_span_under_the_phase_attempt(tmp_path):
    for client, caller, requests, attempt, run in _traced(tmp_path):
        assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
        [req] = requests()
        assert req["start"]["parent"] == attempt.id and req["end"]["status"] == "ok"
        assert json.loads(telemetry_payload(run, req["start"]["inputs"])["body"]) == {"model": "gpt-x", "input": "hi"}
        assert req["children"] == ["llm.enforce", "llm.censor", "llm.try", "db.write", "fs.mkdir", "fs.write"]
        assert [d for d in req["decisions"] if d[0] != "llm.check"] == [("llm.link", "new"), ("llm.upstream", "return")]
        assert {c for n, c in req["decisions"] if n == "llm.check"} == {"pass"}
        assert req["output"]["status"] == 200 and req["output"]["body"]["id"] == "resp_up"


@pytest.mark.parametrize("check, body, status", [
    ("token", None, 401), ("json object", b"[1]", 400), ("stream", {"stream": True}, 400),
    ("video", {"input": [{"type": "input_video", "video_url": "v.mp4"}]}, 400),
    ("image", {"input": [{"type": "input_image", "image_url": "i.png"}]}, 400),
    ("model", {"model": "gpt-other"}, 403), ("budget", {}, 402)])
def test_each_refusal_ends_the_request_refused_with_the_response(tmp_path, check, body, status):
    budget = Budget(max_usd=0, prices={"input": 1, "cached_input": 1, "output": 1})
    for client, caller, requests, attempt, run in _traced(tmp_path, budget=budget):
        headers = {"Authorization": f"Bearer {'ar-nope' if check == 'token' else caller.token}"}
        content = body if isinstance(body, bytes) else json.dumps({"model": "gpt-x", "input": "hi", **(body or {})})
        assert client.post("/v1/responses", content=content, headers=headers).status_code == status
        [req] = requests()
        assert req["end"]["status"] == "refused" and req["output"]["status"] == status
        assert req["decisions"][-1] == ("llm.check", "refuse") and req["children"] == []
        [refusal] = [r for r in telemetry_records(run) if r["kind"] == "decision" and r["chosen"] == "refuse"]
        assert telemetry_payload(run, refusal["inputs"])["check"] == check
        assert (req["start"]["parent"] == attempt.id) == (check != "token")


def test_a_retry_then_success_records_each_try_and_wait(tmp_path):
    n = {"calls": 0}

    def flaky(request):
        n["calls"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if n["calls"] < 3 else _ok(request)
    for client, caller, requests, attempt, run in _traced(tmp_path, handler=flaky):
        assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
        [req] = requests()
        assert [c for c in req["children"] if c.startswith("llm.")] == [
            "llm.enforce", "llm.censor", "llm.try", "llm.wait", "llm.try", "llm.wait", "llm.try"]
        assert [c for n_, c in req["decisions"] if n_ == "llm.upstream"] == ["wait", "wait", "return"]


def test_a_give_up_fails_the_caller_inside_the_request(tmp_path):
    for client, caller, requests, attempt, run in _traced(tmp_path, handler=lambda r: httpx.Response(401, json={})):
        assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 401
        [req] = requests()
        assert "llm.probe" in req["children"] and "caller.fail" in req["children"]
        assert ("llm.upstream", "give up") in req["decisions"]
        assert req["end"]["status"] == "error" and caller.fault.event.is_set()


def test_a_mock_call_is_answered_without_enforce_or_tries(tmp_path):
    for client, caller, requests, attempt, run in _traced(tmp_path, mock_script="smoke"):
        assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
        [req] = requests()
        assert [c for c in req["children"] if c.startswith("llm.")] == ["llm.censor"]
        assert req["output"]["mock"] is True and req["end"]["status"] == "ok"


@pytest.mark.parametrize("path, effort", [("/v1/chat/completions", "high"), ("/v1/responses", "low"),
                                          ("/v1/chat/completions", None)])
def test_the_forwarded_request_is_rebuilt_from_the_raw_one_and_each_steps_changes(tmp_path, path, effort):
    """The record holds the agent's request once; the effort and censor steps record only what they changed."""
    from panel.records import FORWARDED, Records
    from panel.runfiles import RunFiles
    run = tmp_path / "trun"
    telemetry.open(run)
    telemetry.start("ar.run")
    reg, seen = TokenRegistry(), []
    caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    upstream = _upstream_that(_ok, seen)
    upstream.effort = effort
    app = create_gateway_app(registry=reg, store=CallStore(run), allowed_models={"gpt-x"}, censor=["wbench"],
                             upstream=upstream, mocks=MockBook.default())
    said = "Read the wbench cases. Then train."
    body = ({"model": "gpt-x", "messages": [{"role": "user", "content": said}]} if "chat" in path else
            {"model": "gpt-x", "input": [{"role": "user", "content": said}], "reasoning": {"summary": "auto"}})
    assert _post(TestClient(app), caller.token, body, path).status_code == 200
    [sent] = seen
    assert "wbench" not in json.dumps(sent) and (effort is None or effort in json.dumps(sent))
    (run / "config").mkdir()
    (run / "config" / "run.json").write_text("{}")
    records = Records(RunFiles(run))
    [span] = records.named("llm.request")
    assert records.payload(FORWARDED + span.id)["body"] == sent
    steps = {c.start["name"]: records.payload(c.end["output"]) for c in span.children if hasattr(c, "start")
             and c.start["name"] in ("llm.enforce", "llm.censor")}
    assert len(json.dumps(steps)) < len(json.dumps(body)) + 200            # changes, never a second whole body
