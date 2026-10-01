"""Readable transcripts of a node's agent conversations, written when the node ends so that later
agents can read them under /nodes/<node>/transcripts. Built from the gateway's telemetry."""
from __future__ import annotations

from pathlib import Path

from .process_digest import _payload, _text
from .telemetry.recorder import Recorder

AGENT_PHASES = ("edit_self", "improve_recipe")


def _role(body: dict) -> str:
    """Named after the role's submit_<x> tool: the only role label the kernel can see."""
    names = [(t.get("function") or {}).get("name", "") for t in body.get("tools") or []]
    return next((n.removeprefix("submit_") for n in names if n.startswith("submit_")), "conversation")


def render(messages: list[dict]) -> str:
    names: dict[str, str] = {}
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            parts.append(f"## tool result: {names.get(m.get('tool_call_id'), 'tool')}\n\n{_text(m.get('content'))}")
            continue
        parts.append(f"## {role}")
        if m.get("reasoning"):
            parts.append(f"### reasoning\n\n{_text(m['reasoning'])}")
        if _text(m.get("content")).strip():
            parts.append(_text(m.get("content")))
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            names[call.get("id")] = fn.get("name", "?")
            parts.append(f"### tool call: {fn.get('name', '?')}\n\n{fn.get('arguments')}")
    return "\n\n".join(parts) + "\n"


def write_transcripts(run_dir: Path, node_id: str) -> None:
    """One markdown file per conversation, in start order, under nodes/<node>/transcripts/<phase>-<attempt>/.
    A compacted conversation continues in the next file. Never raises: transcripts are a convenience."""
    try:
        rec = Recorder(run_dir)
        convs: dict[str, dict] = {}
        for e in rec.read_events(node_id):
            if e.get("phase") not in AGENT_PHASES or e.get("tool"):      # ask calls: in the caller's transcript
                continue
            if e["type"] == "llm.request":
                convs.setdefault(e.get("conversation_id"), {"dir": f"{e['phase']}-{e.get('attempt')}"})["request"] = e
            elif e["type"] == "llm.response" and e.get("conversation_id") in convs:
                convs[e["conversation_id"]]["response"] = e
        out = Path(run_dir) / "nodes" / node_id / "transcripts"
        counts: dict[str, int] = {}
        for conv in convs.values():
            i = counts[conv["dir"]] = counts.get(conv["dir"], 0) + 1
            body = _payload(rec, conv["request"]).get("body") or {}
            messages = list(body.get("messages") or [])
            choices = (_payload(rec, conv["response"]).get("body") or {}).get("choices") if "response" in conv else None
            if choices:
                messages.append(choices[0].get("message") or {})
            path = out / conv["dir"] / f"{i:02d}-{_role(body)}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(messages), encoding="utf-8")
    except Exception:                                   # noqa: BLE001 -- never fail a node over transcripts
        pass
