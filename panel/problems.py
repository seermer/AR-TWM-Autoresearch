"""Every failure in a run, newest first: "recorded" by the kernel, or "inferred" from the seed's tool error text."""
from __future__ import annotations

from ar_kernel.telemetry import tree

from .chat import chains, is_tool_error, request_messages, segments, text_of
from .runfiles import fmt_ts
from .views import Run, agent_attempts

# A decision is a problem when its choice is not one of these.
FINE = {"contract.step": {"pass"}, "recipe.gate": {"pass"}, "phase.result": {"accept"}, "eval.score": {"score"},
        "train.outcome": {"checkpoint"}, "train.nonfinite": {"keep"}, "phase.audit": {"go on"},
        "phase.failure": set(), "eval.failure": set(), "node.fault": set(), "tool.fault": set()}


def _row(ts, source, kind, where, summary, seq=None, detail=None) -> dict:
    node, phase, attempt = where if isinstance(where, tuple) else (None, None, None)
    return {"ts": ts, "time": fmt_ts(ts), "source": source, "kind": kind, "node": node, "phase": phase,
            "attempt": attempt, "where": where if isinstance(where, str) else "", "summary": summary[:300],
            "seq": seq, "detail": detail}


def _said(error: dict | None) -> str:
    return f"{(error or {}).get('type', '').rsplit('.', 1)[-1]}: {(error or {}).get('message', '')}"


def recorded(run: Run) -> list[dict]:
    """Spans that ended in an error (the innermost of one exception) or refused, handled errors, alerts, failing
    decisions, rejected ingest candidates, and spans with no end whose writer is gone."""
    records = run.records.all()
    load = run.records.payload
    index = {(r["writer"], r["seq"]): i for i, r in enumerate(records)}
    out = []
    for i, r in enumerate(records):
        where = run.records.place(r)
        add = lambda kind, summary, detail=None: out.append(_row(r["ts"], "recorded", kind, where, summary, i, detail))
        if r["kind"] == "end" and r.get("status") in ("error", "refused") \
                and (span := run.records.span(r["span"])) is not None:
            name = span.start["name"]
            error, output = load(r.get("error")), load(r.get("output")) or {}
            if error is not None and any(isinstance(c, tree.Node) and c.end and c.end.get("status") == "error"
                                         and _said(load(c.end.get("error"))) == _said(error) for c in span.children):
                continue                              # the same exception, already listed at its inner span
            body = output.get("body") if isinstance(output, dict) else None
            message = (body.get("error") or {}).get("message") if isinstance(body, dict) \
                and isinstance(body.get("error"), dict) else None
            if error:
                said = _said(error)
            elif "exit_code" in output:
                stderr = output.get("stderr") or "\n".join(      # captured, or streamed as log records
                    c["text"] for c in span.children if isinstance(c, dict) and c.get("stream") == "stderr")
                said = f"exit code {output['exit_code']}: {stderr.strip()[-300:]}".strip(": ")
            else:
                said = f"status {output.get('status')}: {message or ''}".strip()
            add("refused" if r["status"] == "refused" else name, f"{name}: {said}")
        elif r["kind"] == "error":
            add("handled error", f"{_said(load(r.get('error')))} → {r.get('handled_as')}")
        elif r["kind"] == "alert":
            add("alert", f"{r.get('level')} {r.get('name')}: {r.get('message')}")
        elif r["kind"] == "decision" and r["name"] in FINE and r.get("chosen") not in FINE[r["name"]]:
            add(r["name"], f"{r['chosen']}: {r.get('reason') or ''}")
    for n in run.records.spans():
        if n.start["name"] == "ingest.candidate" and n.end and n.end.get("status") == "ok":
            output = load(n.end.get("output")) or {}
            if output.get("accepted") is False:
                video = ((load(n.start.get("inputs")) or {}).get("named") or {}).get("video") or ""
                out.append(_row(n.end["ts"], "recorded", "ingest_rejected", run.records.where(n.id),
                                f"{video.rsplit('/', 1)[-1]}: " + "; ".join(output.get("reasons") or []),
                                index.get((n.end["writer"], n.end["seq"]))))
        elif n.end is None and not run.records.running(n):
            out.append(_row(n.start["ts"], "recorded", "no end recorded", run.records.where(n.id),
                            f"{n.start['name']}: {run.records.open_facts(n)}",
                            index.get((n.start["writer"], n.start["seq"]))))
    return out


def _inferred_attempt(run: Run, node: str, phase: str, attempt: int, kernel_tools: set[str]) -> list[dict]:
    """Error tool outputs of the agent's own tools. A kernel tool's error is recorded on its span already, so it
    is left out here. Each output is timed by the first request that carried it."""
    load = run.records.payload
    out = []
    for index, chain in enumerate(chains(segments(run.records.calls(node, phase, attempt)), load)):
        where = f"{phase}-{attempt} · conversation {index}"
        for seg in chain:
            messages = request_messages(load(seg.calls[-1].request))
            names, assistants = {}, 0
            for m in messages:
                if m.get("role") == "assistant":
                    assistants += 1
                    for call in m.get("tool_calls") or []:
                        names[call.get("id")] = (call.get("function") or {}).get("name", "?")
                elif m.get("role") == "tool":
                    name, text = names.get(m.get("tool_call_id"), "tool"), text_of(m.get("content"))
                    if name in kernel_tools or not is_tool_error(text):
                        continue
                    call = seg.calls[min(assistants, len(seg.calls) - 1)]
                    row = _row(call.ts, "inferred", "agent_tool", where, f"{name}: {text.strip()}", detail=text)
                    row.update(node=node, phase=phase, attempt=attempt)
                    out.append(row)
    return out


def inferred(run: Run) -> list[dict]:
    out = []
    kernel_tools = {n.start["name"].removeprefix("tool.") for n in run.records.spans()
                    if n.start["name"].startswith("tool.") and n.start["name"] != "tool.request"}
    for n in run.nodes():
        node = n["node_id"]
        for phase, attempt in agent_attempts(run, node):
            calls = run.records.calls(node, phase, attempt)
            last = tuple((c.call_id, c.done_ts) for c in calls)
            key = ("inferred", node, phase, attempt)
            stamp, cached = run.cache.get(key, (None, None))
            if stamp != last:                          # the attempt had LLM traffic since: recompute
                cached = _inferred_attempt(run, node, phase, attempt, kernel_tools)
                run.cache[key] = (last, cached)
            out += cached
    return out


def problems(run: Run) -> list[dict]:
    return sorted(recorded(run) + inferred(run), key=lambda p: p["ts"], reverse=True)


def problem_counts(run: Run, hours: float = 1.0) -> list[dict]:
    """Problems per kind, in the run's last `hours` (ending at its latest record) and in total."""
    rows = problems(run)
    records = run.records.all()
    since = max((r["ts"] for r in records), default=0.0) - hours * 3600
    counts: dict[tuple, dict] = {}
    for p in rows:
        c = counts.setdefault((p["kind"], p["source"]), {"kind": p["kind"], "source": p["source"],
                                                         "last hour": 0, "total": 0})
        c["total"] += 1
        c["last hour"] += p["ts"] >= since
    return sorted(counts.values(), key=lambda c: (-c["total"], c["kind"]))
