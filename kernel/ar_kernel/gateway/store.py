"""Fail-closed LLM call records with conversation linking (spec 13.3, row "LLM calls").

Linking, in order: an explicit previous_response_id; a Responses API conversation
id; otherwise the request's history begins with a prior call's request + response
(how ChatOpenAI continues a run, fact 6). Prefix matching only looks at calls from the
same container token.

CallStore keeps every call's full normalized history for the whole run, so memory
grows with calls x history. forget() bounds that growth: once a caller token is
revoked, its linking state (requests, response ids, conversation tails) is dropped.
A response for a call that arrived after its token was forgotten is still recorded
(telemetry is never lost) -- only the linking-state bookkeeping is skipped.
"""
from __future__ import annotations

import json
import threading
import uuid
from typing import Any

_VOLATILE = {"id", "status"}          # differ between an output item and its resent copy


def _items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return list(body.get("messages") or [])
    raw = body.get("input")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    return list(raw or [])


def _out_items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return [c.get("message") for c in body.get("choices") or [] if c.get("message")]
    return list(body.get("output") or [])


def _norm(item: Any) -> str:
    if isinstance(item, dict):
        item = {k: v for k, v in item.items() if k not in _VOLATILE and v is not None}
    return json.dumps(item, sort_keys=True, default=str)


class CallStore:
    def __init__(self, recorder) -> None:
        self._rec = recorder
        self._lock = threading.Lock()
        self._calls: dict[str, dict] = {}          # call_id -> linking state
        self._by_response: dict[str, str] = {}     # response id -> call_id
        self._last_in_conv: dict[str, str] = {}    # conversation id -> latest call_id

    def _link(self, caller, endpoint: str, body: dict) -> tuple[str, int, str | None]:
        prev = body.get("previous_response_id")
        if prev and prev in self._by_response:
            parent = self._calls[self._by_response[prev]]
            return parent["conversation_id"], parent["turn_index"] + 1, parent["call_id"]
        conv = body.get("conversation")
        if isinstance(conv, dict):
            conv = conv.get("id")
        if conv:
            cid = f"oai:{conv}"
            parent_id = self._last_in_conv.get(cid)
            turn = self._calls[parent_id]["turn_index"] + 1 if parent_id else 0
            return cid, turn, parent_id
        current = [_norm(i) for i in _items(endpoint, body)]
        for call in reversed(list(self._calls.values())):
            if call["token"] != caller.token or call["endpoint"] != endpoint or not call["done"]:
                continue
            prefix = call["prefix"]
            if prefix and len(prefix) <= len(current) and current[:len(prefix)] == prefix:
                return call["conversation_id"], call["turn_index"] + 1, call["call_id"]
        return uuid.uuid4().hex, 0, None

    def begin(self, caller, endpoint: str, body: dict) -> dict:
        with self._lock:
            conv, turn, parent = self._link(caller, endpoint, body)
            meta = {"call_id": uuid.uuid4().hex, "conversation_id": conv,
                    "turn_index": turn, "parent_call_id": parent}
            # Persist BEFORE the caller forwards anything; a TelemetryError propagates.
            self._rec.event("llm.request", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="gateway",
                            payload={"endpoint": endpoint, "body": body, **meta},
                            call_id=meta["call_id"], conversation_id=conv,
                            turn_index=turn, parent_call_id=parent, model=body.get("model"))
            self._calls[meta["call_id"]] = {**meta, "token": caller.token, "endpoint": endpoint,
                                            "request": [_norm(i) for i in _items(endpoint, body)],
                                            "prefix": None, "done": False}
            self._last_in_conv[conv] = meta["call_id"]
            return meta

    def end(self, meta: dict, caller, *, status: int, body: dict, latency_s: float,
            attempts: int) -> None:
        with self._lock:
            self._rec.event("llm.response", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="gateway",
                            payload={"status": status, "body": body, "latency_s": latency_s,
                                     "attempts": attempts, **meta},
                            call_id=meta["call_id"], conversation_id=meta["conversation_id"],
                            status=status, latency_s=latency_s, attempts=attempts,
                            usage=body.get("usage") if isinstance(body, dict) else None)
            call = self._calls.get(meta["call_id"])
            if call is None:
                # The caller's token was forgotten (revoked) while this call was in
                # flight. The response is recorded above; there is no linking state
                # left to update.
                return
            endpoint = call["endpoint"]
            call["prefix"] = call["request"] + [_norm(i) for i in _out_items(endpoint, body or {})]
            call["request"] = None  # prefix subsumes it; drop the duplicate copy
            call["done"] = True
            if isinstance(body, dict) and body.get("id"):
                self._by_response[body["id"]] = meta["call_id"]

    def forget(self, token: str) -> None:
        """Drop all linking state for `token`'s calls.

        Bounds CallStore's memory growth: once a caller token is revoked, calls
        made with it can no longer be linked to (as a previous_response_id parent
        or a prefix match) and are removed from `_calls`, `_by_response` and
        `_last_in_conv`. Any response that still arrives for one of these calls is
        recorded by `end()` without error; linking-state updates are just skipped.
        """
        with self._lock:
            dead = {call_id for call_id, call in self._calls.items() if call["token"] == token}
            if not dead:
                return
            for call_id in dead:
                del self._calls[call_id]
            for response_id, call_id in list(self._by_response.items()):
                if call_id in dead:
                    del self._by_response[response_id]
            for conv_id, call_id in list(self._last_in_conv.items()):
                if call_id in dead:
                    del self._last_in_conv[conv_id]
