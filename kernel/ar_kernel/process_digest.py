"""How a node's run went, from the archive's `llm_calls` and `activity` rows."""
from __future__ import annotations

import json
import re
from collections import Counter

from .archive.activity import activity, calls

# Safety caps: the longest real error message so far is 469 characters.
MAX_ERRORS, EXAMPLE_CHARS, MAX_BYTES = 8, 1000, 10_000
_HEX = re.compile(r"\b[0-9a-f]{16,}\b")
_PATH = re.compile(r"/(?:mnt|home|tmp|workspace)/[^\s'\"]*")
_REQUEST_ID = re.compile(r"\(Request ID: [^)]*\)")


def _shape(message: str) -> str:
    """One line with ids and paths removed, so the same mistake counts as one error."""
    message = _REQUEST_ID.sub("", message)
    return " ".join(_PATH.sub("<path>", _HEX.sub("<id>", message)).split())[:EXAMPLE_CHARS]


MAX_REASONS = 5
AGENT_PHASES = ("edit_self", "improve_recipe")


def _number(conv: dict) -> int:
    return int(conv["transcript"].rsplit("/", 1)[1].split("-", 1)[0])


def phases(llm: list[dict], acts: list[dict]) -> list[dict]:
    """Per agent phase attempt: its conversations in start order, its snapshots and the snapshot its result named."""
    out: dict[tuple, dict] = {}
    convs: dict[str, dict] = {}
    facts = [c for c in llm if c["transcript"]] + [a for a in acts if a["kind"] in ("snapshot", "accepted")
                                                    and a["phase"] in AGENT_PHASES]
    for f in sorted(facts, key=lambda f: f["created_at"]):
        row = out.setdefault((f["phase"], f["attempt"]), {"phase": f["phase"], "attempt": f["attempt"],
                                                           "conversations": [], "snapshots": [], "accepted": None})
        if "transcript" in f:
            if f["conversation_id"] not in convs:
                convs[f["conversation_id"]] = {"role": f["role"] or "unlabeled", "turns": 0,
                                               "transcript": f["transcript"]}
                row["conversations"].append(convs[f["conversation_id"]])
            convs[f["conversation_id"]]["turns"] += 1
        elif f["kind"] == "snapshot":
            row["snapshots"].append(f["name"])
        else:
            row["accepted"] = f["name"]
    for row in out.values():
        row["conversations"].sort(key=_number)
    return list(out.values())


def attempt_process(conn, node_id: str, phase: str, attempt: int) -> dict:
    """One phase attempt as `phases` gives it, without transcripts."""
    row = next((r for r in phases(calls(conn, node_id), activity(conn, node_id))
                if (r["phase"], r["attempt"]) == (phase, attempt)),
               {"conversations": [], "snapshots": [], "accepted": None})
    return {"conversations": [{"role": c["role"], "turns": c["turns"]} for c in row["conversations"]],
            "snapshots": row["snapshots"], "accepted": row["accepted"]}


def process_digest(conn, node_id: str) -> dict:
    acts = activity(conn, node_id)
    tools: dict[str, dict] = {}
    errors, gates, jobs, ingest, reasons = Counter(), Counter(), Counter(), Counter(), Counter()
    for a in acts:
        kind = a["kind"]
        if kind == "tool":
            counts = tools.setdefault(a["name"], {"calls": 0, "errors": 0})
            counts["calls"] += 1
            if not a["ok"]:
                counts["errors"] += 1
                errors[(a["name"], a["detail"])] += 1
        elif kind == "gate":
            gates["passed" if a["ok"] else "failed"] += 1
        elif kind == "job":
            jobs["run"] += 1
            jobs["failed"] += a["detail"] == "failed"
        elif kind == "ingest":
            ingest["accepted" if a["ok"] else "rejected"] += 1
            reasons.update((a["detail"] or "").splitlines())
    out = {"phases": phases(calls(conn, node_id), acts), "tools": tools,
           "tool_errors": [{"tool": tool, "count": n, "example": example}
                           for (tool, example), n in sorted(errors.items(), key=lambda kv: -kv[1])[:MAX_ERRORS]],
           "gpu_jobs": {"run": jobs["run"], "failed": jobs["failed"]},
           "ingest": {"accepted": ingest["accepted"], "rejected": ingest["rejected"],
                      "reasons": [{"reason": r, "count": n} for r, n in reasons.most_common(MAX_REASONS)]},
           "gates": dict(gates)}
    while len(json.dumps(out)) > MAX_BYTES and out["tool_errors"]:
        out["tool_errors"].pop()
    return out
