"""Spans and marks (single records) for the timeline view."""
from __future__ import annotations

import time

from .problems import FINE
from .views import Run

EVAL_LANES = {"eval": "eval", "eval.concat_lora": "concat_lora", "eval.render": "render", "eval.wbench": "wbench",
              "eval.judge": "judge"}
OWN = {"sandbox": "sandbox", "process": "subprocess", "train": "train", "train.precache": "train",
       "hf.upload": "upload", "ingest.candidate": "ingest", "job.item": "job items", **EVAL_LANES}
INFO_MARKS = {"node.select", "node.status", "run.signal", "run.stop_file", "run.resume_point"}


def _command_label(command: list) -> str:
    args = [str(a) for a in command or []]
    if "-m" in args[:-1]:
        return args[args.index("-m") + 1]
    return next((a.rsplit("/", 1)[-1] for a in args if a.endswith(".py")), " ".join(a.rsplit("/", 1)[-1] for a in args[:2]))


def _lane(run: Run, n) -> tuple[str, str, dict] | None:
    """(lane, label, extra fields) of a span shown on the timeline, or None."""
    name = n.start["name"]
    if name == "phase.attempt":
        _, phase, attempt = run.records.where(n.id)
        t = (run.records.payload(n.start.get("inputs")) or {}).get("try")
        return "phase", f"{phase}-{attempt}" + (f" try {t}" if t and t > 1 else ""), {}
    if name == "llm.request":
        call = run.records.call(n)
        if call is None:
            return None
        lane = f"llm · {call.role or (f'kernel tool: {call.tool}' if call.tool else 'unlabeled')}"
        return lane, f"turn {call.turn_index}", {"conversation_id": call.conversation_id,
                                                 "tokens": call.usage.get("total_tokens")}
    if name.startswith("tool.") and name != "tool.request":
        return "kernel tools", name.removeprefix("tool."), {}
    if name.startswith("job.") and name not in OWN and name not in ("job.cancel", "job.gpu_lock"):
        inputs = run.records.payload(n.start.get("inputs")) or {}
        return "gpu jobs", f"{inputs.get('backend') or name.removeprefix('job.')} {str(inputs.get('job', ''))[:8]}", {}
    if name == "process":
        return "subprocess", _command_label((run.records.payload(n.start.get("inputs")) or {}).get("command")), {}
    if name in OWN:
        return OWN[name], name, {}
    return None


def timeline(run: Run, node: str | None = None) -> dict:
    records = run.records.all()
    alive = run.files.loop_pid() is not None
    stamp = (len(records), alive)
    cached = run.cache.get(("timeline", node))
    if cached and cached[0] == stamp:
        return cached[1]
    index = {(r["writer"], r["seq"]): i for i, r in enumerate(records)}
    last = max((r["ts"] for r in records), default=time.time())
    spans, marks = [], []
    for n in run.records.spans():
        where = run.records.where(n.id)
        if node and where[0] != node:
            continue
        shown = _lane(run, n)
        if shown is None:
            continue
        lane, label, extra = shown
        running = run.records.running(n)
        status = n.end.get("status") if n.end else "running" if running else "no end recorded"
        end = n.end["ts"] if n.end else time.time() if running else last
        spans.append({"lane": lane, "label": label, "start": n.start["ts"], "end": end, "status": status,
                      "node": where[0], "phase": where[1], "attempt": where[2],
                      "seq": index.get((n.start["writer"], n.start["seq"])),
                      "end_seq": index.get((n.end["writer"], n.end["seq"])) if n.end else None,
                      "inferred": False, **extra})
    for i, r in enumerate(records):
        failing = r["kind"] == "decision" and r["name"] in FINE and r.get("chosen") not in FINE[r["name"]]
        if not (r["kind"] in ("alert", "error") or failing or (r["kind"] == "decision" and r["name"] in INFO_MARKS)):
            continue
        where = run.records.place(r)
        if node and where[0] != node:
            continue
        level = r.get("level") if r["kind"] == "alert" else "error" if r["kind"] == "error" or failing else None
        label = r.get("message") if r["kind"] == "alert" else r.get("handled_as") if r["kind"] == "error" \
            else f"{r.get('chosen')}: {r.get('reason') or ''}"
        marks.append({"type": f"{r['kind']} {r.get('name') or ''}".strip(), "ts": r["ts"], "seq": i,
                      "node": where[0], "phase": where[1], "attempt": where[2], "label": label, "level": level})
    out = {"spans": sorted(spans, key=lambda s: s["start"]), "marks": marks, "alive": alive,
           "first": min((r["ts"] for r in records), default=None), "last": last}
    run.cache[("timeline", node)] = (stamp, out)
    return out
