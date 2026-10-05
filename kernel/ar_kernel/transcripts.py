"""Transcripts of a node's agent conversations, written when the node ends so that later
agents can read them under /nodes/<node>/transcripts. Built from the gateway's telemetry."""
from __future__ import annotations

import json
from pathlib import Path

from .process_digest import AGENT_PHASES, _payload, role_of
from .telemetry.recorder import Recorder


def write_transcripts(run_dir: Path, node_id: str) -> None:
    """One file per conversation, in start order, under nodes/<node>/transcripts/<phase>-<attempt>/: JSON
    Lines, one message per line (role, content, reasoning, tool_calls, tool_call_id), so a script counts
    turns and calls exactly and grep still works. A compacted conversation continues in the next file.
    Never raises: transcripts are a convenience."""
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
            path = out / conv["dir"] / f"{i:02d}-{role_of(body)}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages), encoding="utf-8")
    except Exception:                                   # noqa: BLE001 -- never fail a node over transcripts
        pass
