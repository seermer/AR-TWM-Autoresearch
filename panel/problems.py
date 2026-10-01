"""Every failure in a run, in one list (newest first).

"recorded": failures the kernel wrote down as such -- failed LLM calls, kernel tool errors,
per-item errors inside GPU job results, gate/ingest/contract failures, failed subprocesses and
containers, error spans, warnings and alerts. "inferred": the agent's own in-container tools
(run_command, read_file, ...), whose failures exist only as tool outputs inside the model
conversation, recognised by the seed harness's error text (see chat.is_tool_error)."""
from __future__ import annotations

from .chat import chains, is_tool_error, request_messages, segments, text_of
from .runfiles import fmt_ts
from .views import Run, agent_attempts

HANDLED_ERRORS = {"tool.error", "sandbox.error", "subproc.error"}


def _row(ts, source, kind, e_or_where, summary, seq=None, detail=None) -> dict:
    where = e_or_where if isinstance(e_or_where, dict) else {}
    return {"ts": ts, "time": fmt_ts(ts), "source": source, "kind": kind,
            "node": where.get("node"), "phase": where.get("phase"), "attempt": where.get("attempt"),
            "where": e_or_where if isinstance(e_or_where, str) else "", "summary": summary[:300],
            "seq": seq, "detail": detail}


def _item_errors(result, prefix="") -> list[tuple[str, str]]:
    """(item, error) pairs anywhere in a job result: {"clips": {path: {"error": ...}}} and alike."""
    out = []
    if isinstance(result, dict):
        if result.get("error") and prefix:
            out.append((prefix, str(result["error"])))
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                out += _item_errors(value, key if isinstance(value, dict) and "error" in value else prefix)
    elif isinstance(result, list):
        for i, value in enumerate(result):
            out += _item_errors(value, f"{prefix}[{i}]" if isinstance(value, dict) and "error" in value else prefix)
    return out


def recorded(run: Run) -> list[dict]:
    events = run.log.events()
    load = run.log.payload
    alive = run.files.loop_pid() is not None
    ended = {(e.get("node"), e.get("phase"), e.get("attempt")) for e in events if e.get("type") == "phase.end"}
    answered = {e.get("call_id") for e in events if e.get("type") == "llm.response"}
    out = []
    for e in events:
        kind, ts, seq = e.get("type", ""), e.get("ts_wall", 0.0), e["_seq"]
        add = lambda k, summary, detail=None: out.append(_row(ts, "recorded", k, e, summary, seq, detail))
        if kind == "llm.response" and e.get("status") != 200:
            body = (load(e.get("payload")) or {}).get("body") or {}
            message = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            add("llm_call", f"status {e.get('status')}: {message or ''}".strip())
        elif kind == "llm.request" and e.get("call_id") not in answered:
            key = (e.get("node"), e.get("phase"), e.get("attempt"))
            if key in ended or not alive:
                add("llm_call", "no response recorded (the call never returned)")
        elif kind == "tool.error":
            add("tool_error", f"{e.get('tool')}: {(load(e.get('payload')) or {}).get('error', '')}")
        elif kind == "job.finished":
            body = load(e.get("payload")) or {}
            if body.get("error"):
                add("job_failed", f"job {e.get('job_id')}: {body['error']}")
            for item_name, error in _item_errors(body.get("result")):
                add("job_item", f"job {e.get('job_id')}: {item_name}: {error}")
        elif kind == "gate.failed":
            add("gate_failed", "; ".join(map(str, (load(e.get("payload")) or {}).get("failures") or [])))
        elif kind == "ingest.rejected":
            p = load(e.get("payload")) or {}
            video = (p.get("candidate") or {}).get("video") or ""
            add("ingest_rejected", f"{video.rsplit('/', 1)[-1]}: " + "; ".join(p.get("reasons") or []))
        elif kind == "contract.report":
            p = load(e.get("payload")) or {}
            if p.get("ok") is False:
                add("contract_failed", f"{p.get('failed_step')}: {p.get('detail') or ''}")
        elif kind in ("subproc.end", "subproc.error") and (kind == "subproc.error" or e.get("returncode") != 0):
            p = load(e.get("payload")) or {}
            add("subprocess", f"{e.get('phase')}: rc={e.get('returncode')} " + (p.get("stderr") or p.get("error") or "")[-200:])
        elif kind in ("sandbox.end", "sandbox.error") and (kind == "sandbox.error" or e.get("exit_code") not in (0, None)
                                                           or e.get("timed_out")):
            add("sandbox", f"container exit={e.get('exit_code')} timed_out={e.get('timed_out')}")
        elif kind.endswith(".error") and kind not in HANDLED_ERRORS:
            add("error_event", f"{kind}: {(load(e.get('payload')) or {}).get('error', '')}")
        elif kind.endswith(".warning"):
            add("warning", e.get("message") or kind)
        elif kind == "alert":
            add("alert", f"{e.get('level')} {e.get('kind')}: {e.get('message')}")
    return out


def _inferred_attempt(run: Run, node: str, phase: str, attempt: int, kernel_tools: set[str]) -> list[dict]:
    """Error tool outputs of the agent's own tools. A kernel tool's error is recorded as tool.error
    already, so it is left out here. Each output is timed by the first request that carried it."""
    load = run.log.payload
    out = []
    events = run.log.events(files=[node])
    for index, chain in enumerate(chains(segments(events, node, phase, attempt), load)):
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
    for n in run.nodes():
        node = n["node_id"]
        events = run.log.events(files=[node])
        kernel_tools = {e.get("tool") for e in events if e.get("type") in ("tool.call", "tool.error")}
        last = tuple(e.get("call_id") for e in events if e.get("type") in ("llm.request", "llm.response"))
        for phase, attempt in agent_attempts(run, node):
            key = ("inferred", node, phase, attempt)
            stamp, cached = run.cache.get(key, (None, None))
            if stamp != last:                          # the node had LLM traffic since: recompute
                cached = _inferred_attempt(run, node, phase, attempt, kernel_tools)
                run.cache[key] = (last, cached)
            out += cached
    return out


def problems(run: Run) -> list[dict]:
    return sorted(recorded(run) + inferred(run), key=lambda p: p["ts"], reverse=True)


def problem_counts(run: Run, hours: float = 1.0) -> list[dict]:
    """Problems per kind, in the run's last `hours` (ending at its latest event) and in total."""
    rows = problems(run)
    events = run.log.events()
    since = (events[-1].get("ts_wall", 0.0) if events else 0.0) - hours * 3600
    counts: dict[tuple, dict] = {}
    for p in rows:
        c = counts.setdefault((p["kind"], p["source"]), {"kind": p["kind"], "source": p["source"],
                                                         "last hour": 0, "total": 0})
        c["total"] += 1
        c["last hour"] += p["ts"] >= since
    return sorted(counts.values(), key=lambda c: (-c["total"], c["kind"]))
