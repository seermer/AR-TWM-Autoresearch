import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ar_kernel.budget import Budget
from ar_kernel.gateway.app import Upstream, create_gateway_app
from ar_kernel.gateway.mock import MockBook, function_call, message
from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry.recorder import Recorder, TelemetryError
from ar_kernel.tools.context import TokenRegistry


def _upstream_that(handler, seen):
    def wrapped(request: httpx.Request):
        seen.append(json.loads(request.content))
        return handler(request)
    async def no_sleep(_s):
        pass
    return Upstream("https://api.example/v1", "sk-REALKEY", timeout_s=5, retries=3,
                    transport=httpx.MockTransport(wrapped), sleep=no_sleep)


def _ok(request):
    return httpx.Response(200, json={"id": "resp_up", "object": "response", "output": [],
                                     "usage": {"input_tokens": 1, "output_tokens": 1}})


@pytest.fixture
def make(tmp_path):
    def build(handler=_ok, mock_script=None, allowed=frozenset({"gpt-x"})):
        rec = Recorder(tmp_path)
        reg, store, seen = TokenRegistry(rec), CallStore(rec), []
        caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                           staging_host=tmp_path, mock_script=mock_script)
        app = create_gateway_app(registry=reg, store=store, allowed_models=set(allowed),
                                 upstream=_upstream_that(handler, seen), mocks=MockBook.default())
        return TestClient(app), caller, rec, seen
    return build


def _post(client, token, body, path="/v1/responses"):
    return client.post(path, json=body, headers={"Authorization": f"Bearer {token}"})


def _telemetry_contains(tmp_path, rec: Recorder, needle: str) -> bool:
    """Scan telemetry for `needle`, decompressing payloads (Controller ruling: payloads are
    zstd-compressed as `payloads/<sha256>.json.zst`, so a raw-byte scan can never find a leak
    that redaction failed to catch -- it would pass whether or not the leak was actually
    redacted)."""
    telem = tmp_path / "telemetry"
    for events_file in (telem / "events").glob("*.jsonl"):
        text = events_file.read_text()
        if needle in text:
            return True
        for line in text.splitlines():
            if not line.strip():
                continue
            digest = json.loads(line).get("payload")
            if digest and needle in json.dumps(rec.load_payload(digest)):
                return True
    return False


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


def test_chat_completions_allows_images(make):
    client, caller, _, seen = make(handler=lambda r: httpx.Response(
        200, json={"id": "cc", "choices": [{"message": {"role": "assistant", "content": "k"}}]}))
    body = {"model": "gpt-x", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]}
    assert _post(client, caller.token, body, path="/v1/chat/completions").status_code == 200
    assert len(seen) == 1


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


def test_responses_allows_images(make):
    client, caller, _, seen = make()
    body = {"model": "gpt-x", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"}]}]}
    assert _post(client, caller.token, body).status_code == 200
    assert len(seen) == 1


def test_forwarded_call_is_recorded_and_upstream_key_never_reaches_telemetry(make, tmp_path):
    client, caller, rec, seen = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and r.json()["id"] == "resp_up"
    assert len(seen) == 1
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert kinds == ["llm.request", "llm.response"]
    assert not _telemetry_contains(tmp_path, rec, "sk-REALKEY")


def test_request_is_persisted_before_forwarding(make, monkeypatch):
    """Fail-closed (spec 13.1.2): if the request cannot be recorded, nothing is sent."""
    client, caller, rec, seen = make()
    def broken(*a, **k):
        raise TelemetryError("disk full")
    monkeypatch.setattr(rec, "event", broken)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 500
    assert seen == []


def test_upstream_429_is_retried_then_succeeds(make):
    calls = {"n": 0}
    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if calls["n"] < 3 else _ok(request)
    client, caller, rec, _ = make(handler=flaky)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
    response_event = [e for e in rec.read_events("n1") if e["type"] == "llm.response"][0]
    assert response_event["attempts"] == 3


def test_persistent_upstream_failure_returns_its_status(make):
    client, caller, _, seen = make(handler=lambda r: httpx.Response(503, json={"error": "down"}))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 503
    assert len(seen) == 4                     # 1 try + 3 retries


def test_non_json_upstream_error_is_retried_recorded_and_returned(make):
    """Proxies answer 502/503/504 with HTML; that must not crash the gateway."""
    client, caller, rec, seen = make(handler=lambda r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "non-JSON" in r.json()["error"]["message"]
    assert len(seen) == 4                     # retried like any 5xx
    response_event = [e for e in rec.read_events("n1") if e["type"] == "llm.response"][0]
    assert response_event["attempts"] == 4 and response_event["status"] == 502


def test_non_json_upstream_error_body_is_recorded_without_truncation(make):
    """No truncation or sampling of telemetry (spec 13.1.3): a long non-JSON error body
    (e.g. a verbose HTML error page from a proxy) must be recorded in full, not clipped."""
    long_body = "<html>" + ("x" * 3000) + "</html>"
    client, caller, rec, seen = make(handler=lambda r: httpx.Response(502, text=long_body))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502
    assert r.json()["error"]["body"] == long_body
    response_event = [e for e in rec.read_events("n1") if e["type"] == "llm.response"][0]
    recorded = rec.load_payload(response_event["payload"])
    assert recorded["body"]["error"]["body"] == long_body
    assert len(recorded["body"]["error"]["body"]) == len(long_body) > 2000


def test_non_json_success_body_becomes_a_502(make):
    client, caller, _, _ = make(handler=lambda r: httpx.Response(200, text="not json"))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 502


def test_unexpected_gateway_exception_is_still_recorded(make, monkeypatch):
    async def bug(self, path, body):
        raise RuntimeError("gateway bug")
    monkeypatch.setattr(Upstream, "post", bug)
    client, caller, rec, _ = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "gateway bug" in r.json()["error"]["message"]
    assert [e["type"] for e in rec.read_events("n1")] == ["llm.request", "llm.response"]


def test_upstream_base_url_defaults_to_openai_v1_and_is_not_rewritten():
    import asyncio
    seen = []
    def record(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={})
    for base in (None, "", "https://proxy.example/api/v3/"):
        up = Upstream(base, "k", timeout_s=5, retries=0, transport=httpx.MockTransport(record))
        asyncio.run(up.post("/responses", {}))
    assert seen == ["https://api.openai.com/v1/responses", "https://api.openai.com/v1/responses",
                    "https://proxy.example/api/v3/responses"]


def test_mock_mode_serves_scripted_output_without_upstream(make):
    client, caller, rec, seen = make(mock_script="smoke")
    body = _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).json()
    assert body["output"][0]["content"][0]["text"] == "ok"
    assert seen == []
    assert [e["type"] for e in rec.read_events("n1")] == ["llm.request", "llm.response"]


def test_mock_script_is_consumed_in_order_then_repeats_its_last_step(tmp_path):
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
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
    """Controller ruling: create_gateway_app must wire registry.on_revoke(store.forget) so the
    store's memory is bound to live containers -- once a container's token is revoked, its
    linking state (requests, response ids, conversation tails) is dropped."""
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
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
    rec = Recorder(tmp_path)
    reg, store, seen = TokenRegistry(rec), CallStore(rec), []
    caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    up = _upstream_that(handler, seen)
    up.effort = effort
    app = create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"}, upstream=up,
                             mocks=MockBook.default())
    return TestClient(app), caller, rec, seen


def test_gateway_enforces_reasoning_effort_on_chat_completions(tmp_path):
    client, caller, rec, seen = _make_with_effort(tmp_path, "low")
    r = _post(client, caller.token, {"model": "gpt-x", "reasoning_effort": "high",
                                     "messages": [{"role": "user", "content": "hi"}]},
              path="/v1/chat/completions")
    assert r.status_code == 200 and seen[0]["reasoning_effort"] == "low"
    request = [e for e in rec.read_events("n1") if e["type"] == "llm.request"][0]
    assert rec.load_payload(request["payload"])["body"]["reasoning_effort"] == "low"   # as forwarded


def test_gateway_enforces_reasoning_effort_on_responses(tmp_path):
    client, caller, _, seen = _make_with_effort(tmp_path, "low")
    _post(client, caller.token, {"model": "gpt-x", "input": "hi",
                                 "reasoning": {"effort": "high", "summary": "auto"}})
    _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert [s["reasoning"] for s in seen] == [{"effort": "low", "summary": "auto"}, {"effort": "low"}]


def test_gateway_forwards_the_agent_effort_when_none_is_configured(tmp_path):
    client, caller, _, seen = _make_with_effort(tmp_path, None)
    _post(client, caller.token, {"model": "gpt-x", "reasoning_effort": "high",
                                 "messages": [{"role": "user", "content": "hi"}]},
          path="/v1/chat/completions")
    assert seen[0]["reasoning_effort"] == "high"



def test_a_non_object_reasoning_field_is_replaced_not_a_crash(tmp_path):
    client, caller, _, seen = _make_with_effort(tmp_path, "low")
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi", "reasoning": "high"})
    assert r.status_code == 200 and seen[0]["reasoning"] == {"effort": "low"}

def test_upstream_from_env():
    up = Upstream.from_env({"OPENAI_BASE_URL": "https://llm.example", "OPENAI_API_KEY": "sk-x",
                            "OPENAI_EFFORT": "low"}, timeout_s=5, retries=1)
    assert (up._base, up._headers, up.effort) == ("https://llm.example",
                                                  {"Authorization": "Bearer sk-x"}, "low")
    up = Upstream.from_env({"OPENAI_EFFORT": ""}, timeout_s=5, retries=1)
    assert (up._base, up.effort) == ("https://api.openai.com/v1", None)


def test_mock_mode_serves_chat_completions(tmp_path):
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
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
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
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
    responses = [e for e in rec.read_events("n1") if e["type"] == "llm.response"]
    assert responses[-1]["cost_usd"] > 0 and responses[-1]["mock"] is False
    assert len(seen) == 1

    r2 = _post(client, real.token, {"model": "gpt-x", "input": "hi again"})
    assert r2.status_code == 402 and "budget" in r2.json()["error"]["message"]
    assert len(seen) == 1                                           # upstream never saw the 2nd call
    requests = [e for e in rec.read_events("n1") if e["type"] == "llm.request"]
    assert len(requests) == 1                                       # no 2nd llm.request event

    r3 = _post(client, mocked.token, {"model": "gpt-x", "input": "hi"})
    assert r3.status_code == 200
    responses = [e for e in rec.read_events("n1") if e["type"] == "llm.response"]
    assert responses[-1]["mock"] is True
