"""Transcripts of a node's agent conversations, written when the node ends so that later
agents can read them under /nodes/<node>/transcripts. Built from the gateway's telemetry."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from .guards import alert
from .process_digest import _payload, conversations
from .telemetry.recorder import Recorder


def write_transcripts(run_dir: Path, node_id: str, recorder) -> None:
    """One file per conversation, in start order, under nodes/<node>/transcripts/<phase>-<attempt>/: JSON
    Lines, one message per line (role, content, reasoning, tool_calls, tool_call_id), so a script counts
    turns and calls exactly and grep still works. A compacted conversation continues in the next file.
    Never raises: transcripts are a convenience, so a failure is a warning alert on `recorder`."""
    try:
        rec = Recorder(run_dir)
        out = Path(run_dir) / "nodes" / node_id
        for conv in conversations(rec.read_events(node_id)):
            body = _payload(rec, conv["requests"][-1]).get("body") or {}
            messages = list(body.get("messages") or [])
            choices = (_payload(rec, conv["response"]).get("body") or {}).get("choices") if "response" in conv else None
            if choices:
                messages.append(choices[0].get("message") or {})
            path = out / conv["transcript"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages), encoding="utf-8")
    except Exception as exc:                            # noqa: BLE001 -- never fail a node over transcripts
        alert(recorder, "transcripts_failed", f"{node_id}: transcripts not written: {type(exc).__name__}: {exc}",
              level="warning", node_id=node_id, traceback=traceback.format_exc())
