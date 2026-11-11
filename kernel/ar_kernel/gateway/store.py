"""Record LLM calls and redacted transcripts; link by response parent, conversation ID, then token-local history.
Normalize client-visible assistant fields; drop revoked tokens' linking state while recording late replies."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .. import fs, telemetry
from ..archive.activity import add_call
from ..budget import usage_tokens
from ..faults import KernelFault
from ..isolation import _found, strings
from ..process_digest import AGENT_PHASES

_VOLATILE = {"id", "status"}          # differ between an output item and its resent copy
_NUMBERED = re.compile(r"^(\d+)-")


def _assistant(message: dict) -> dict:
    def args(raw: Any) -> Any:
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    calls = [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
              "arguments": args((c.get("function") or {}).get("arguments"))}
             for c in message.get("tool_calls") or []]
    return {"role": "assistant", "content": message.get("content") or None,
            "tool_calls": calls or None,     # reasoning: the ar_contract.client.reasoning_of rule
            "reasoning": message.get("reasoning") or message.get("reasoning_content") or None}


def _chat(messages: list[Any]) -> list[Any]:
    return [_assistant(m) if isinstance(m, dict) and m.get("role") == "assistant" else m
            for m in messages]


def _items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return _chat(list(body.get("messages") or []))
    raw = body.get("input")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    return list(raw or [])


def _out_items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return _chat([c.get("message") for c in body.get("choices") or [] if c.get("message")])
    return list(body.get("output") or [])


def _norm(item: Any) -> str:
    if isinstance(item, dict):
        item = {k: v for k, v in item.items() if k not in _VOLATILE and v is not None}
    return json.dumps(item, sort_keys=True, default=str)


def _write(path: Path, messages: list, secrets: list[str]) -> None:
    text = "".join(json.dumps(m, ensure_ascii=False, sort_keys=True) + "\n" for m in messages)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    fs.mkdir(path.parent)
    # a unique temporary name: two calls of one conversation may end at once. Each version is the call's request
    # messages and reply, already in its llm.request span: the record holds the hash, not the text again
    fs.write_text(path, text, atomic=True, record_text=False)


class CallStore:
    def __init__(self, run_dir: Path, secrets=(), audit_patterns=()) -> None:
        self._run_dir, self._secrets, self._audit = Path(run_dir), [s for s in secrets if s], list(audit_patterns)
        self._lock = threading.Lock()
        self._numbers: dict[Path, int] = {}        # transcript folder -> highest number given
        self._messages: dict[str, list] = {}       # call_id -> request messages, until the call ends
        self._calls: dict[str, dict] = {}          # call_id -> linking state
        self._by_response: dict[str, str] = {}     # response id -> call_id
        self._last_in_conv: dict[str, str] = {}    # conversation id -> latest call_id

    def _link(self, caller, endpoint: str, body: dict) -> tuple[str, int, str | None, str]:
        """(conversation, turn, parent call, how)."""
        prev = body.get("previous_response_id")
        if prev and prev in self._by_response:
            parent = self._calls[self._by_response[prev]]
            return parent["conversation_id"], parent["turn_index"] + 1, parent["call_id"], "previous response"
        conv = body.get("conversation")
        if isinstance(conv, dict):
            conv = conv.get("id")
        if conv:
            cid = f"oai:{conv}"
            parent_id = self._last_in_conv.get(cid)
            turn = self._calls[parent_id]["turn_index"] + 1 if parent_id else 0
            return cid, turn, parent_id, "conversation"
        current = [_norm(i) for i in _items(endpoint, body)]
        for call in reversed(list(self._calls.values())):
            if call["token"] != caller.token or call["endpoint"] != endpoint or not call["done"]:
                continue
            prefix = call["prefix"]
            if prefix and len(prefix) <= len(current) and current[:len(prefix)] == prefix:
                return call["conversation_id"], call["turn_index"] + 1, call["call_id"], "prefix"
        return uuid.uuid4().hex, 0, None, "new"

    def begin(self, caller, endpoint: str, body: dict, tool: str | None = None, role: str | None = None) -> dict:
        """`tool`: the kernel tool making the call (ask), else None. `role`: the X-AR-Role header."""
        with self._lock:
            conv, turn, parent, how = self._link(caller, endpoint, body)
            transcript = None
            if tool is None and caller.phase in AGENT_PHASES:
                transcript = (parent and self._calls[parent]["transcript"]) or self._number(caller, role)
            meta = {"call_id": uuid.uuid4().hex, "conversation_id": conv,
                    "turn_index": turn, "parent_call_id": parent, **({"tool": tool} if tool else {}),
                    **({"role": role} if role else {}), **({"transcript": transcript} if transcript else {})}
            telemetry.decision("llm.link", {**meta, "previous_response_id": body.get("previous_response_id"),
                                            "conversation": body.get("conversation")},
                               ["previous response", "conversation", "prefix", "new"], how,
                               f"continues call {parent}" if parent else "the first call of its conversation"
                               if how == "conversation" else "no earlier call of this caller matches")
            self._calls[meta["call_id"]] = {**meta, "token": caller.token, "endpoint": endpoint,
                                            "request": [_norm(i) for i in _items(endpoint, body)],
                                            "prefix": None, "done": False, "transcript": transcript}
            self._last_in_conv[conv] = meta["call_id"]
            if transcript:
                self._messages[meta["call_id"]] = list(body.get("messages") or [])
            return meta

    def _number(self, caller, role: str | None) -> str:
        """A new conversation's transcript: one more than the highest number in its folder or given here."""
        folder = f"transcripts/{caller.phase}-{caller.attempt}"
        host = self._run_dir / "nodes" / caller.node / folder
        names = os.listdir(host) if host.is_dir() else []
        n = max([self._numbers.get(host, 0), *(int(m[1]) for m in map(_NUMBERED.match, names) if m)]) + 1
        self._numbers[host] = n
        return f"{folder}/{n:02d}-{role or 'unlabeled'}.jsonl"

    def end(self, meta: dict, caller, *, status: int, body: dict, cost_usd: float | None = None,
            mock: bool = False) -> None:
        with self._lock:
            messages = self._messages.pop(meta["call_id"], None)
            # Revoked tokens have no linking state; in-flight responses are still recorded.
            call = self._calls.get(meta["call_id"])
            if call is not None:
                endpoint = call["endpoint"]
                call["prefix"] = call["request"] + [_norm(i) for i in _out_items(endpoint, body or {})]
                call["request"] = None  # prefix subsumes it; drop the duplicate copy
                call["done"] = True
                if isinstance(body, dict) and body.get("id"):
                    self._by_response[body["id"]] = meta["call_id"]
        self._keep(meta, caller, status, body, cost_usd, mock, messages)

    def _keep(self, meta, caller, status, body, cost_usd, mock, messages) -> None:
        """The call's row and its conversation's transcript; a failure is a KernelFault."""
        body = body if isinstance(body, dict) else {}
        tool, transcript = meta.get("tool"), meta.get("transcript")
        add_call(self._run_dir, node=caller.node, phase=caller.phase, attempt=caller.attempt, role=meta.get("role"),
                 tool=tool, conversation_id=meta["conversation_id"], turn=meta["turn_index"], transcript=transcript,
                 status=status, tokens=usage_tokens(body.get("usage")), cost_usd=cost_usd, mock=mock,
                 audit_hits=[] if tool else sorted(set(_found(self._audit, " ".join(strings(body))))))
        if transcript:
            choices = body.get("choices")
            if choices:
                messages = [*messages, choices[0].get("message") or {}]
            try:
                _write(self._run_dir / "nodes" / caller.node / transcript, messages, [*self._secrets, caller.token])
            except OSError as exc:
                raise KernelFault(f"the transcript {transcript} could not be written: {exc}") from exc

    def forget(self, token: str) -> None:
        """Drop a token's linking state; record in-flight replies without recreating that state."""
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
