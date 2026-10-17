"""How a node's run went, from what the kernel records."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .telemetry.recorder import Recorder

# Safety caps: the longest real error message so far is 469 characters.
MAX_ERRORS, EXAMPLE_CHARS, MAX_BYTES = 8, 1000, 10_000
_HEX = re.compile(r"\b[0-9a-f]{16,}\b")
_PATH = re.compile(r"/(?:mnt|home|tmp|workspace)/[^\s'\"]*")
_REQUEST_ID = re.compile(r"\(Request ID: [^)]*\)")


def _shape(message: str) -> str:
    """One line with ids and paths removed, so the same mistake counts as one error."""
    message = _REQUEST_ID.sub("", message)
    return " ".join(_PATH.sub("<path>", _HEX.sub("<id>", message)).split())[:EXAMPLE_CHARS]


def _payload(rec: Recorder, event: dict) -> dict:
    ref = event.get("payload")
    try:
        return rec.load_payload(ref) if isinstance(ref, str) else (ref or {})
    except (OSError, ValueError):
        return {}


MAX_REASONS = 5
AGENT_PHASES = ("edit_self", "improve_recipe")


def role_of(event: dict) -> str:
    """The X-AR-Role label, or `unlabeled`."""
    return event.get("role") or "unlabeled"


def conversations(events: list[dict]) -> list[dict]:
    """The agent conversations in start order."""
    convs: dict[str, dict] = {}
    for e in events:
        if e.get("phase") not in AGENT_PHASES or e.get("tool"):      # ask calls: in the caller's conversation
            continue
        if e["type"] == "llm.request":
            convs.setdefault(e.get("conversation_id"), {"phase": e["phase"], "attempt": e.get("attempt"),
                                                        "role": role_of(e), "requests": []})["requests"].append(e)
        elif e["type"] == "llm.response" and e.get("conversation_id") in convs:
            convs[e["conversation_id"]]["response"] = e
    counts: Counter = Counter()
    for c in convs.values():
        folder = f"{c['phase']}-{c['attempt']}"
        counts[folder] += 1
        c["transcript"] = f"transcripts/{folder}/{counts[folder]:02d}-{c['role']}.jsonl"
    return list(convs.values())


def phases(rec: Recorder, events: list[dict]) -> list[dict]:
    """Per phase attempt: its conversations, its snapshots and the snapshot its result named."""
    out: dict[tuple, dict] = {}

    def row(phase, attempt) -> dict:
        return out.setdefault((phase, attempt), {"phase": phase, "attempt": attempt, "conversations": [],
                                                 "snapshots": [], "accepted": None})
    starts = {c["requests"][0]["span_id"]: c for c in conversations(events)}
    for e in events:
        phase, attempt = e.get("phase"), e.get("attempt")
        if phase not in AGENT_PHASES:
            continue
        if e["type"] == "llm.request" and e.get("span_id") in starts:
            c = starts[e["span_id"]]
            row(phase, attempt)["conversations"].append(
                {"role": c["role"], "turns": len(c["requests"]), "transcript": c["transcript"]})
        elif e["type"] == "tool.result" and e.get("tool") == "snapshot":
            snapshot_id = (_payload(rec, e).get("result") or {}).get("snapshot_id")
            if snapshot_id:
                row(phase, attempt)["snapshots"].append(snapshot_id)
        elif e["type"] == "phase.end":
            row(phase, attempt)["accepted"] = (_payload(rec, e).get("result") or {}).get("snapshot_id")
    return list(out.values())


def attempt_process(run_dir: Path, node_id: str, phase: str, attempt: int) -> dict:
    """One phase attempt as `phases` gives it, without transcripts."""
    rec = Recorder(run_dir)
    row = next((r for r in phases(rec, rec.read_events(node_id)) if (r["phase"], r["attempt"]) == (phase, attempt)),
               {"conversations": [], "snapshots": [], "accepted": None})
    return {"conversations": [{"role": c["role"], "turns": c["turns"]} for c in row["conversations"]],
            "snapshots": row["snapshots"], "accepted": row["accepted"]}


def process_digest(run_dir: Path, node_id: str) -> dict:
    try:
        return _digest(Path(run_dir), node_id)
    except Exception:                                       # noqa: BLE001 -- context building must not fail
        return {}


def _digest(run_dir: Path, node_id: str) -> dict:
    if not (run_dir / "telemetry" / "events" / f"{node_id}.jsonl").exists():
        return {}
    rec = Recorder(run_dir)
    events = rec.read_events(node_id)
    errors: dict[tuple, list] = defaultdict(list)
    tools: dict[str, dict] = {}
    gates, jobs, ingest, reasons = Counter(), Counter(), Counter(), Counter()
    for e in events:
        kind = e["type"]
        if kind == "tool.call":
            tools.setdefault(e.get("tool"), {"calls": 0, "errors": 0})["calls"] += 1
        elif kind == "tool.error":
            tools.setdefault(e.get("tool"), {"calls": 0, "errors": 0})["errors"] += 1
            errors[(e.get("tool"), _shape(str(_payload(rec, e).get("error", ""))))].append(e)
        elif kind in ("gate.failed", "gate.passed"):
            gates["failed" if kind == "gate.failed" else "passed"] += 1
        elif kind == "job.finished":
            jobs["run"] += 1
            jobs["failed"] += e.get("state") == "failed"
        elif kind == "ingest.accepted":
            ingest["accepted"] += 1
        elif kind == "ingest.rejected":
            ingest["rejected"] += 1
            reasons.update(_shape(str(r)) for r in _payload(rec, e).get("reasons") or [])
    out = {"phases": phases(rec, events), "tools": tools,
           "tool_errors": [{"tool": tool, "count": len(evs), "example": example}
                           for (tool, example), evs in sorted(errors.items(), key=lambda kv: -len(kv[1]))[:MAX_ERRORS]],
           "gpu_jobs": {"run": jobs["run"], "failed": jobs["failed"]},
           "ingest": {"accepted": ingest["accepted"], "rejected": ingest["rejected"],
                      "reasons": [{"reason": r, "count": n} for r, n in reasons.most_common(MAX_REASONS)]},
           "gates": dict(gates)}
    while len(json.dumps(out)) > MAX_BYTES and out["tool_errors"]:
        out["tool_errors"].pop()
    return out
