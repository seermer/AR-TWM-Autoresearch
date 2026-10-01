"""Agent conversations from the gateway's llm.request / llm.response events.

The gateway links a request to an earlier call when it resends that call's history, so a
conversation id holds one uninterrupted history. Compaction replaces the history, so the
continuation arrives under a new id; `chains` stitches it back for display only.
A chat is a list of items {"kind", "title", "text"} that the UI turns into chat bubbles."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

Load = Callable[[str | None], dict | None]

# How the seed harness and run_command report a failure to the model (agent code, so a self-edit
# may change it: anything found this way is labelled inferred).
ERROR_TEXT = re.compile(r"(Error\b|exit [1-9]\d*\b|timed out after\b)")


def is_tool_error(text: str | None) -> bool:
    return bool(ERROR_TEXT.match((text or "").lstrip()))


@dataclass
class Call:
    call_id: str
    conversation_id: str
    turn_index: int
    ts: float
    request: str | None
    response: str | None = None
    status: int | None = None
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float | None = None


@dataclass
class Segment:
    conversation_id: str
    calls: list[Call]

    @property
    def first_ts(self) -> float:
        return self.calls[0].ts

    @property
    def last_ts(self) -> float:
        return self.calls[-1].ts


def item(kind: str, title: str | None, text: str) -> dict:
    return {"kind": kind, "title": title, "text": text}


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def request_messages(payload: dict | None) -> list[dict]:
    return list(((payload or {}).get("body") or {}).get("messages") or [])


def request_tools(payload: dict | None) -> list[dict]:
    return list(((payload or {}).get("body") or {}).get("tools") or [])


def response_message(payload: dict | None) -> dict | None:
    choices = ((payload or {}).get("body") or {}).get("choices") or []
    return choices[0].get("message") if choices else None


def _pretty(arguments) -> str:
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def segments(events: list[dict], node: str, phase: str, attempt: int) -> list[Segment]:
    calls: dict[str, Call] = {}
    for e in events:
        if (e.get("node"), e.get("phase"), e.get("attempt")) != (node, phase, attempt):
            continue
        if e.get("type") == "llm.request" and e.get("call_id"):
            calls[e["call_id"]] = Call(e["call_id"], e.get("conversation_id") or e["call_id"],
                                       e.get("turn_index") or 0, e.get("ts_wall", 0.0), e.get("payload"))
        elif e.get("type") == "llm.response" and e.get("call_id") in calls:
            c = calls[e["call_id"]]
            c.response, c.status, c.usage = e.get("payload"), e.get("status"), e.get("usage") or {}
            c.cost_usd, c.latency_s = e.get("cost_usd"), e.get("latency_s")
    grouped: dict[str, list[Call]] = {}
    for c in sorted(calls.values(), key=lambda c: c.ts):
        grouped.setdefault(c.conversation_id, []).append(c)
    return sorted((Segment(k, v) for k, v in grouped.items()), key=lambda s: s.first_ts)


def chains(segs: list[Segment], load: Load) -> list[list[Segment]]:
    """B continues A when B starts after A's last call and B's first user message contains
    A's last reply (whitespace-stripped; the harness pastes `reply.text.strip()`). The
    wording of the compaction prompt is never relied on."""
    last_reply, first_user = {}, {}
    for s in segs:
        last = s.calls[-1]
        reply = response_message(load(last.response)) if last.response else None
        last_reply[s.conversation_id] = text_of((reply or {}).get("content")).strip()
        messages = request_messages(load(s.calls[0].request))
        first_user[s.conversation_id] = next((text_of(m.get("content")) for m in messages
                                              if m.get("role") == "user"), "")
    prev: dict[str, Segment] = {}
    taken: set[str] = set()
    for b in segs:
        options = [a for a in segs if a is not b and a.last_ts < b.first_ts
                   and a.conversation_id not in taken and last_reply[a.conversation_id]
                   and last_reply[a.conversation_id] in first_user[b.conversation_id]]
        if options:
            a = max(options, key=lambda s: s.last_ts)
            prev[b.conversation_id] = a
            taken.add(a.conversation_id)
    by_id = {s.conversation_id: s for s in segs}
    following = {a.conversation_id: by_id[b_id] for b_id, a in prev.items()}
    out = []
    for s in segs:
        if s.conversation_id in prev:
            continue
        chain = [s]
        while chain[-1].conversation_id in following:
            chain.append(following[chain[-1].conversation_id])
        out.append(chain)
    return out


def compaction_calls(seg: Segment, load: Load) -> list[Call]:
    """The summarizer call ends the segment; a forced retry (tools disabled) sends the same
    messages again, so trailing calls with identical requests are all compaction calls."""
    last = request_messages(load(seg.calls[-1].request))
    out = [seg.calls[-1]]
    for c in reversed(seg.calls[:-1]):
        if request_messages(load(c.request)) != last:
            break
        out.insert(0, c)
    return out


def message_items(m: dict, names: dict[str, str]) -> list[dict]:
    role = m.get("role")
    if role == "system":
        return [item("system", "System prompt", text_of(m.get("content")))]
    if role == "user":
        return [item("user", None, text_of(m.get("content")))]
    if role == "tool":
        name, text = names.get(m.get("tool_call_id"), "tool"), text_of(m.get("content"))
        if is_tool_error(text):
            return [item("tool_error", f"⚠ Tool error: {name}", text)]
        return [item("tool_output", f"Tool output: {name}", text)]
    out = []
    reasoning = m.get("reasoning") or m.get("reasoning_content")     # reasoning_content: fallback only
    if reasoning:
        out.append(item("reasoning", "Reasoning", text_of(reasoning)))
    if text_of(m.get("content")).strip():
        out.append(item("assistant", None, text_of(m.get("content"))))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        names[call.get("id")] = fn.get("name", "?")
        out.append(item("tool_call", f"Tool call: {fn.get('name', '?')}", _pretty(fn.get("arguments"))))
    return out


def _usage(call: Call, turn: int) -> dict:
    u = call.usage or {}
    parts = [f"turn {turn}: {u.get('prompt_tokens', '?')} in / {u.get('completion_tokens', '?')} out"]
    if call.cost_usd is not None:
        parts.append(f"${call.cost_usd:.4f}")
    if call.latency_s is not None:
        parts.append(f"{call.latency_s:.1f} s")
    if call.status not in (None, 200):
        parts.append(f"status {call.status}")
    return item("usage", None, " · ".join(parts))


def chat_items(chain: list[Segment], load: Load, awaiting: bool) -> list[dict]:
    items: list[dict] = []
    names: dict[str, str] = {}
    turn = 0
    for k, seg in enumerate(chain):
        final = k == len(chain) - 1
        comp = [] if final else compaction_calls(seg, load)
        normal = seg.calls[:len(seg.calls) - len(comp)]
        base = comp[0] if comp else seg.calls[-1]
        messages = request_messages(load(base.request))
        if not messages:                       # the payload file is missing or unreadable
            items.append(item("note", None, "request payload missing: this segment's history cannot be shown"))
        if comp:
            messages = messages[:-1]               # the compaction instruction: shown in its own turn
        if k == 0:
            tools = request_tools(load(base.request))
            if tools:
                lines = [f"- {(t.get('function') or {}).get('name', '?')}: "
                         f"{(t.get('function') or {}).get('description', '')}" for t in tools]
                items.append(item("tools", f"Tools offered ({len(tools)})", "\n".join(lines)))
            system_at = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
            if system_at is not None:           # the system prompt leads the chat
                items.insert(0, item("system", "System prompt", text_of(messages[system_at].get("content"))))
        first_user_seen = False
        answered = 0
        for m in messages:
            if m.get("role") == "system":
                continue
            if k > 0 and m.get("role") == "user" and not first_user_seen:
                first_user_seen = True
                items.append(item("context", "Context after compaction", text_of(m.get("content"))))
                continue
            items += message_items(m, names)
            if m.get("role") == "assistant" and answered < len(normal):
                turn += 1
                items.append(_usage(normal[answered], turn))
                answered += 1
        if final:
            last = seg.calls[-1]
            reply = response_message(load(last.response)) if last.response else None
            if reply is not None:
                items += message_items(reply, names)
                turn += 1
                items.append(_usage(last, turn))
            else:
                items.append(item("note", None, "awaiting response" if awaiting else "no response recorded"))
            continue
        last_request = request_messages(load(comp[-1].request))
        instruction = text_of(last_request[-1].get("content")) if last_request else "(request payload missing)"
        before = comp[-1].usage.get("prompt_tokens", "?")
        after = chain[k + 1].calls[0].usage.get("prompt_tokens", "?")
        items.append(item("compaction_request", f"Compaction #{k + 1}: ~{before} → ~{after} tokens", instruction))
        for r, c in enumerate(comp):
            reply = response_message(load(c.response)) if c.response else None
            text = text_of((reply or {}).get("content")).strip() or "(no text: the model called a tool instead)"
            title = f"Compaction #{k + 1} summary" + (", retry without tools" if r else "")
            items.append(item("compaction_summary", title, text))
    return items


def infer_role(system_text: str | None, prompt_files: dict[str, str]) -> str:
    """The prompt file (in the agent code that ran) the system prompt starts with; else the
    system prompt's first line. Only a label: roles are the agent's own convention."""
    if not (system_text or "").strip():
        return "(no system prompt)"
    matches = [name for name, text in prompt_files.items() if text and system_text.startswith(text)]
    if matches:
        return max(matches, key=lambda n: len(prompt_files[n]))
    return system_text.strip().splitlines()[0][:80]
