"""A short digest of how a node's run went, for the edit planner: phase times, LLM
turns and compactions, repeated tool errors, gate results. Built from telemetry; never raises."""
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


def _local_errors(rec: Recorder, requests: dict) -> dict:
    """Failed tool results the agent saw, counted from the last request of each conversation
    (which carries the whole conversation): `Error: ...` results and run_command's `exit N`."""
    counts = Counter()
    for events in requests.values():
        for message in _payload(rec, events[-1]).get("body", {}).get("messages", []):
            if message.get("role") != "tool":
                continue
            text = message.get("content")
            text = text if isinstance(text, str) else json.dumps(text)
            if text.startswith("Error:"):
                counts["tool_error_messages"] += 1
            elif re.match(r"exit [1-9]", text):
                counts["run_command_nonzero"] += 1
    return dict(counts)


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def _continuations(rec: Recorder, requests: dict, responses: dict) -> list[tuple]:
    """Conversations that continue a compacted one: compaction starts a new conversation whose
    first user message carries the previous conversation's last reply (the summary)."""
    replies = {}
    for key, e in responses.items():
        choices = (_payload(rec, e).get("body") or {}).get("choices") or [{}]
        replies[key] = _text((choices[0].get("message") or {}).get("content")).strip()
    out = []
    for key, evs in requests.items():
        messages = (_payload(rec, evs[0]).get("body") or {}).get("messages") or []
        first_user = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
        if any(other != key and other[0] == key[0] and reply and reply in first_user
               for other, reply in replies.items()):
            out.append(key)
    return out


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
    started, phases, durations = {}, {}, Counter()
    requests: dict[tuple, list] = defaultdict(list)
    responses: dict[tuple, dict] = {}             # the last response of each conversation
    errors: dict[tuple, list] = defaultdict(list)
    gates, gpu_job_s = Counter(), 0.0
    for e in events:
        kind, phase = e["type"], e.get("phase")
        if kind == "phase.start":
            started[(phase, e.get("attempt"))] = e["ts_wall"]
        elif kind == "phase.end" and (phase, e.get("attempt")) in started:
            phases[phase] = phases.get(phase, 0) + round(e["ts_wall"] - started[(phase, e.get("attempt"))])
        elif kind in ("llm.request", "llm.response") and e.get("tool"):
            pass                                  # a kernel tool's own model call (ask), not a role's conversation
        elif kind == "llm.request":
            requests[(phase, e.get("conversation_id"))].append(e)
        elif kind == "llm.response":
            responses[(phase, e.get("conversation_id"))] = e
        elif kind == "tool.error":
            errors[(e.get("tool"), _shape(str(_payload(rec, e).get("error", ""))))].append(e)
        elif kind in ("gate.failed", "gate.passed"):
            gates["failed" if kind == "gate.failed" else "passed"] += 1
        elif kind == "job.finished":
            gpu_job_s += e.get("gpu_seconds") or 0.0
        elif kind.endswith(".end") and e.get("duration_s") is not None:
            if kind == "train.end":
                durations["train_s"] += e["duration_s"]
            elif kind == "render.end":
                durations["render_s"] += e["duration_s"]
            elif kind.startswith("wbench."):
                durations["eval_s"] += e["duration_s"]
    llm = {}
    for (phase, _), evs in requests.items():
        row = llm.setdefault(phase, {"conversations": 0, "turns": 0, "compactions": 0})
        row["conversations"] += 1
        row["turns"] += len(evs)
    for phase, _ in _continuations(rec, requests, responses):
        llm[phase]["compactions"] += 1
    out = {"phases": {**phases, **{k: round(v) for k, v in durations.items()}}, "llm": llm,
           "tool_errors": [{"tool": tool, "count": len(evs), "example": example}
                           for (tool, example), evs in sorted(errors.items(), key=lambda kv: -len(kv[1]))[:MAX_ERRORS]],
           "local_errors": _local_errors(rec, requests), "gates": dict(gates), "gpu_job_s": round(gpu_job_s)}
    while len(json.dumps(out)) > MAX_BYTES and out["tool_errors"]:
        out["tool_errors"].pop()
    return out
