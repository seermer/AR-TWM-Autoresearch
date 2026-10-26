"""Spans (start/end pairs) and marks (single events) for the timeline view."""
from __future__ import annotations

import time

from .views import Run

MARKS = {"alert", "gate.passed", "gate.failed", "eval.scored", "node.created", "select", "ingest.rejected", "hf_upload",
         "contract.report", "run.start", "node.end", "tool.error", "eval.cleanup", "judge.server_ready",
         "caption.server_ready", "caption.server_stopped", "locate.server_stopped"}
OWN_PAIRS = {"phase", "sandbox", "subproc", "job", "tool", "llm", "run"}   # run.start has no end event


def _span(lane, label, start_e, end_e, status, **extra) -> dict:
    return {"lane": lane, "label": label, "start": start_e["ts_wall"], "end": end_e["ts_wall"] if end_e else None,
            "status": status, "node": start_e.get("node"), "phase": start_e.get("phase"),
            "attempt": start_e.get("attempt"), "seq": start_e["_seq"], "end_seq": end_e["_seq"] if end_e else None,
            "inferred": False, **extra}


def _subproc_label(payload: dict | None) -> str:
    args = [str(a) for a in (payload or {}).get("args") or []]
    if "-m" in args[:-1]:
        name = args[args.index("-m") + 1]
    else:
        name = next((a.rsplit("/", 1)[-1] for a in args if a.endswith(".py")), " ".join(args[:2]))
    return f"{name} ({(payload or {}).get('env')})"


def timeline(run: Run, node: str | None = None) -> dict:
    events = run.log.events(files=[node] if node else None)
    stamp = (len(events), events[-1]["_seq"] if events else None, run.files.loop_pid())
    cached = run.cache.get(("timeline", node))
    if cached and cached[0] == stamp:
        return cached[1]
    load = run.log.payload
    spans, marks = [], []
    open_phase, open_box, open_job, open_tool, open_llm, open_span = {}, {}, {}, {}, {}, {}
    open_sub: dict[tuple, list] = {}
    for e in events:
        t = e.get("type", "")
        where = (e.get("node"), e.get("phase"), e.get("attempt"))
        if t == "phase.start":
            open_phase[where] = e
        elif t == "phase.end" and where in open_phase:
            spans.append(_span("phase", f"{e['phase']}-{e['attempt']}", open_phase.pop(where), e,
                               "ok" if e.get("ok") else "error", exit_code=e.get("exit_code")))
        elif t == "sandbox.start":
            open_box[e.get("container")] = e
        elif t in ("sandbox.end", "sandbox.error") and e.get("container") in open_box:
            ok = t == "sandbox.end" and e.get("exit_code") in (0, None)
            spans.append(_span("sandbox", e.get("container") or "sandbox", open_box.pop(e.get("container")), e,
                               "cancelled" if e.get("cancelled") else "ok" if ok else "error",
                               exit_code=e.get("exit_code")))
        elif t == "job.submitted":
            open_job[e.get("job_id")] = (e, (load(e.get("payload")) or {}).get("backend"))
        elif t == "job.finished" and e.get("job_id") in open_job:
            start, backend = open_job.pop(e["job_id"])
            body = load(e.get("payload")) or {}
            spans.append(_span("gpu jobs", f"{backend or 'job'} {e['job_id'][:8]}", start, e,
                               "error" if body.get("error") else e.get("state") or "ok",
                               gpu_seconds=e.get("gpu_seconds")))
        elif t == "subproc.start":
            p = load(e.get("payload")) or {}
            open_sub.setdefault((e.get("node"), e.get("phase")), []).append((e, _subproc_label(p), p.get("log_path")))
        elif t in ("subproc.end", "subproc.cancelled", "subproc.error"):
            queue = open_sub.get((e.get("node"), e.get("phase"))) or []
            if not queue:
                continue
            log = (load(e.get("payload")) or {}).get("log_path")
            index = next((i for i, q in enumerate(queue) if log and q[2] == log), None)
            inferred = index is None and len(queue) > 1        # several open: paired by order
            start, label, _ = queue.pop(index if index is not None else 0)
            status = ("cancelled" if t == "subproc.cancelled" else
                      "ok" if t == "subproc.end" and e.get("returncode") == 0 else "error")
            spans.append(_span("subprocess", label, start, e, status, returncode=e.get("returncode"),
                               inferred=inferred))
        elif t == "tool.call":
            open_tool[e["span_id"]] = e
        elif t in ("tool.result", "tool.error") and e.get("parent_span_id") in open_tool:
            spans.append(_span("kernel tools", e.get("tool") or "tool", open_tool.pop(e["parent_span_id"]), e,
                               "ok" if t == "tool.result" else "error"))
        elif t == "llm.request" and not e.get("tool"):
            open_llm[e.get("call_id")] = e
        elif t == "llm.response" and e.get("call_id") in open_llm:
            start = open_llm.pop(e["call_id"])
            spans.append(_span(f"llm · {start.get('role') or 'unlabeled'}", f"turn {start.get('turn_index')}",
                               start, e, "ok" if e.get("status") == 200 else "error",
                               conversation_id=start.get("conversation_id"), tokens=(e.get("usage") or {}).get("total_tokens")))
        elif t.endswith(".start") and t.split(".", 1)[0] not in OWN_PAIRS:
            open_span[e["span_id"]] = e
        elif (t.endswith(".end") or t.endswith(".error")) and e.get("span_id") in open_span:
            start = open_span.pop(e["span_id"])
            name = start["type"].removesuffix(".start")
            spans.append(_span(name.split(".", 1)[0], name, start, e, "ok" if t.endswith(".end") else "error",
                               duration_s=e.get("duration_s")))
        if t in MARKS or t.endswith(".warning"):
            marks.append({"type": t, "ts": e["ts_wall"], "seq": e["_seq"], "node": e.get("node"),
                          "phase": e.get("phase"), "attempt": e.get("attempt"),
                          "label": e.get("message") or e.get("tool") or e.get("kind") or t,
                          "level": e.get("level") or ("error" if t in ("gate.failed", "tool.error", "ingest.rejected")
                                                      or (t == "contract.report" and e.get("ok") is False) else None)})
    alive = run.files.loop_pid() is not None
    last = events[-1]["ts_wall"] if events else time.time()
    unfinished = [(e, lane, label) for e, lane, label in
                  [(e, "phase", f"{e['phase']}-{e['attempt']}") for e in open_phase.values()]
                  + [(e, "sandbox", e.get("container") or "sandbox") for e in open_box.values()]
                  + [(e, "gpu jobs", f"{b or 'job'} {e['job_id'][:8]}") for e, b in open_job.values()]
                  + [(q[0], "subprocess", q[1]) for qs in open_sub.values() for q in qs]
                  + [(e, "kernel tools", e.get("tool") or "tool") for e in open_tool.values()]
                  + [(e, f"llm · {e.get('role') or 'unlabeled'}", f"turn {e.get('turn_index')}") for e in open_llm.values()]
                  + [(e, e["type"].split(".", 1)[0], e["type"].removesuffix(".start")) for e in open_span.values()]]
    for e, lane, label in unfinished:
        s = _span(lane, label, e, None, "running" if alive else "no end recorded")
        s["end"] = time.time() if alive else last
        spans.append(s)
    out = {"spans": sorted(spans, key=lambda s: s["start"]), "marks": marks, "alive": alive,
           "first": events[0]["ts_wall"] if events else None, "last": last}
    run.cache[("timeline", node)] = (stamp, out)
    return out
