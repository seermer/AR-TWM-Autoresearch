"""Per-tab data for the run panel: plain dicts and lists, no Gradio."""
from __future__ import annotations

import difflib
import heapq
import math
import re
import time
from pathlib import Path

from .chat import (_usage, call_turns, chains, chat_items, is_tool_error, item, message_items, request_messages,
                   request_tools, tool_lines, tool_names, response_message, segments, text_of)
from .events import EventLog, event_row, filter_events
from .runfiles import RunFiles, fmt_ts, loads

HOST_PATH = re.compile(r"(?<![\w.:/-])/(?:workspace|agent)/[\w.\-/]+")
LOSS = re.compile(r"\[Train\] step=(\d+)\b.*?\bloss=([-+0-9.eE]+).*?\blr=([-+0-9.eE]+)")


class Run:
    def __init__(self, run_dir: Path | str) -> None:
        self.files = RunFiles(run_dir)
        self.log = EventLog(self.files)
        self.cache: dict = {}                  # derived results keyed by what they were derived from

    @property
    def name(self) -> str:
        return self.files.root.name

    def nodes(self) -> list[dict]:
        return self.files.query("SELECT * FROM nodes ORDER BY created_at")

    def node(self, node_id: str | None) -> dict | None:
        if not node_id:
            return None
        rows = self.files.query("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
        return rows[0] if rows else None


def unified(a: str, b: str, name_a: str, name_b: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), name_a, name_b))


# ---- Overview ----

def overview(run: Run) -> dict:
    events = run.log.events()
    nodes = run.nodes()
    by_id = {n["node_id"]: n for n in nodes}
    rows = []
    for n in nodes:
        parent = by_id.get(n["parent_id"]) if n["parent_id"] else None
        delta = (n["score"] - parent["score"] if n["score"] is not None and parent
                 and parent["score"] is not None else None)
        total = (loads(n["phase_timings"], {}) or {}).get("total_s")
        rows.append({"node": n["node_id"], "parent": n["parent_id"] or "", "status": n["status"],
                     "score": n["score"], "vs parent": delta,
                     "attempts": n["attempt_counts"] or "",
                     "duration_min": round(total / 60, 1) if total else None, "error": n["error"] or "",
                     "depth": n["depth"], "created": fmt_ts(n["created_at"]), "created_at": n["created_at"],
                     "subtree_value": n["subtree_value"], "lora_rank": n["lora_rank"], "lora_alpha": n["lora_alpha"],
                     "metrics": loads(n["metrics"], {}) or {}})
    alerts = [e for e in events if e.get("type") == "alert"][-50:]
    return {
        "nodes": rows,
        "alerts": [{"time": fmt_ts(a["ts_wall"]), "level": a.get("level"), "kind": a.get("kind"),
                    "message": a.get("message")} for a in reversed(alerts)],
        "recent": [event_row(e) for e in reversed(events[-50:])],
    }


def _gpu_summary(run: Run, lo: float, hi: float) -> list[dict]:
    acc: dict[str, dict] = {}
    for e in run.log.events(files=["gpu"]):
        if e.get("type") != "gpu.sample" or not lo <= e.get("ts_wall", 0) <= hi:
            continue
        for gpu, s in (e.get("gpus") or {}).items():
            a = acc.setdefault(str(gpu), {"n": 0, "util": 0.0, "peak": 0.0})
            a["n"] += 1
            a["util"] += s.get("util") or 0
            a["peak"] = max(a["peak"], (s.get("memory_mib") or 0) / 1024)
    return [{"gpu": g, "mean_util": round(a["util"] / a["n"], 1), "peak_memory_gib": round(a["peak"], 2)}
            for g, a in sorted(acc.items())]


# ---- Trace ----

def trace_choices(run: Run) -> dict:
    events = run.log.events(include_gpu=True)
    pick = lambda key: sorted({str(e.get(key)) for e in events if e.get(key) is not None})
    return {"node": pick("node"), "phase": pick("phase"), "type": pick("type"), "component": pick("component"),
            "attempt": sorted({str(e["attempt"]) for e in events if e.get("attempt") is not None}, key=int)}


def trace(run: Run, *, node=None, phase=None, attempt=None, types=None, component=None, text=None,
          include_gpu=False, offset=0, limit=500) -> dict:
    found = filter_events(run.log.events(include_gpu=include_gpu), node=node or None, phase=phase or None,
                          attempt=attempt, types=types or None, component=component or None, text=text)
    rows = []
    for e in found[offset:offset + limit]:
        rows.append({**event_row(e), "ts": e.get("ts_wall")})
    return {"rows": rows, "total": len(found), "offset": offset}


def trace_detail(run: Run, seq: int) -> dict | None:
    event = run.log.by_seq(int(seq))
    if event is None:
        return None
    payload = run.log.payload(event.get("payload"))
    chat: list[dict] = []
    names: dict[str, str] = {}
    if event.get("type") == "llm.request":
        for m in request_messages(payload):
            chat += message_items(m, names)
        tools = request_tools(payload)
        if tools:
            chat.append({"kind": "tools", "title": f"Tools offered ({len(tools)})",
                         "text": tool_lines(tools)})
    elif event.get("type") == "llm.response":
        reply = response_message(payload)
        chat = message_items(reply, names) if reply else []
    return {"event": {k: v for k, v in event.items() if not k.startswith("_")}, "payload": payload, "chat": chat}


# ---- Node ----

def node_detail(run: Run, node_id: str) -> dict | None:
    n = run.node(node_id)
    if n is None:
        return None
    lineage, cur = [], n
    while cur is not None:
        lineage.append(cur["node_id"])
        cur = run.node(cur["parent_id"])
    parent, root = run.node(n["parent_id"]), run.node("root")
    base = f"nodes/{node_id}"
    recipe = run.files.read_text(f"{base}/recipe.yaml")
    parent_recipe = run.files.read_text(f"nodes/{parent['node_id']}/recipe.yaml") if parent else None
    metrics, pm = loads(n["metrics"], {}) or {}, (loads(parent["metrics"], {}) or {}) if parent else {}
    rm = (loads(root["metrics"], {}) or {}) if root else {}
    commit = run.files.query("SELECT * FROM data_commits WHERE commit_id = ?", (n["data_commit"],)) \
        if n["data_commit"] else []
    manifest = loads(commit[0]["manifest"], {}) if commit else {}
    return {
        "node": n, "baseline": n["parent_id"] is None, "lineage": " → ".join(reversed(lineage)),
        "edit": run.files.read_json(f"{base}/edit.json"), "rationale": run.files.read_text(f"{base}/rationale.md"),
        "summary": run.files.read_text(f"{base}/summary.md"),
        "recipe": recipe, "parent_recipe": parent_recipe,
        "recipe_diff": unified(parent_recipe or "", recipe, f"{parent['node_id'] if parent else 'none'}/recipe.yaml",
                               f"{node_id}/recipe.yaml") if recipe else None,
        "data_commit": {"commit_id": n["data_commit"], "message": commit[0]["message"] if commit else None,
                        "datasets": {k: len(v.get("clips", [])) for k, v in (manifest.get("datasets") or {}).items()}}
        if n["data_commit"] else None,
        "metrics": [{"metric": m, "node": metrics.get(m), "parent": pm.get(m), "root": rm.get(m)}
                    for m in sorted(set(metrics) | set(pm) | set(rm))],
        "timings": loads(n["phase_timings"], {}),
        "attempts": [{"phase": a["phase"], "attempt": a["idx"], "outcome": a["outcome"],
                      "time": fmt_ts(a["created_at"]), "detail": a["detail"]}
                     for a in run.files.query("SELECT * FROM attempts WHERE node_id = ? ORDER BY id", (node_id,))],
    }


# ---- Conversations ----

def agent_attempts(run: Run, node_id: str) -> list[tuple[str, int]]:
    pairs = {(e.get("phase"), e.get("attempt")) for e in run.log.events(files=[node_id])
             if e.get("type") in ("llm.request", "phase.start")}
    return sorted(p for p in pairs if p[0] and p[1] is not None)


def conversations(run: Run, node_id: str, phase: str, attempt: int) -> list[dict]:
    """Compactions and tool errors are inferred (chat.chains, chat.is_tool_error)."""
    events = run.log.events(files=[node_id])
    load = run.log.payload
    out = []
    for i, chain in enumerate(chains(segments(events, node_id, phase, attempt), load)):
        calls = [c for s in chain for c in s.calls]
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        errors = sum(is_tool_error(text_of(m.get("content"))) for s in chain
                     for m in request_messages(load(s.calls[-1].request)) if m.get("role") == "tool")
        out.append({"index": i, "role": chain[0].calls[0].role or "unlabeled", "started": fmt_ts(chain[0].first_ts),
                    "calls": len(calls), "compactions": len(chain) - 1, "tool_errors": errors,
                    "tokens": sum(c.usage.get("total_tokens") or 0 for c in calls),
                    "usd": round(sum(costs), 4) if costs else None, "chain": chain})
    return out


def _awaiting(run: Run, node_id: str, phase: str, attempt: int) -> bool:
    ended = any(e.get("type") == "phase.end" and e.get("phase") == phase and e.get("attempt") == attempt
                for e in run.log.events(files=[node_id]))
    return run.files.loop_pid() is not None and not ended


def _annotate(run: Run, node_id: str, phase: str, attempt: int, items: list[dict]) -> list[dict]:
    """Each item gets the turn it leads to (none for earlier context), and links for its host paths."""
    turn = next((it["turn"] for it in items if it["kind"] == "usage" and it["turn"]), 1)
    for it in items:
        if it["kind"] == "usage" and it["turn"]:
            turn = it["turn"]
        if not it.get("prior"):
            it["at"] = turn
        if it["kind"] == "usage" and it["turn"]:
            turn += 1
        links = {}
        for found in dict.fromkeys(HOST_PATH.findall(it.get("text") or "")):
            rel = run.files.host_path(node_id, phase, attempt, found.rstrip(".,:;"))
            if rel and len(links) < 50 and run.files.path(rel).exists():
                links[found.rstrip(".,:;")] = rel
        if links:
            it["links"] = links
    return items


def conversation_chat(run: Run, node_id: str, phase: str, attempt: int, index: int) -> list[dict]:
    convs = conversations(run, node_id, phase, attempt)
    if not 0 <= index < len(convs):
        return []
    items = chat_items(convs[index]["chain"], run.log.payload, _awaiting(run, node_id, phase, attempt))
    return _annotate(run, node_id, phase, attempt, items)


def _tasks(run: Run, node_id: str, phase: str, attempt: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Delegated tasks, from timing alone, whatever the orchestration: a conversation's calls form one task until
    it resumes after another conversation made a whole call while it was idle; that starts its next task. Each task
    comes after the task whose call was the last to end before its first call; tasks with a call still in flight
    then are listed as alongside."""
    load = run.log.payload
    convs = conversations(run, node_id, phase, attempt)
    conv_of = {s.conversation_id: i for i, conv in enumerate(convs) for s in conv["chain"]}
    turns = [call_turns(conv["chain"], load) for conv in convs]
    end_of = lambda c: c.done_ts or c.ts
    calls = sorted((c for conv in convs for s in conv["chain"] for c in s.calls), key=lambda c: c.ts)
    tasks: list[dict] = []
    edges: list[dict] = []
    task_of: dict[str, int] = {}               # call id -> task
    open_task: dict[int, int] = {}             # conversation -> its current task
    pending: list = []                         # (end, order, call) of calls not yet ended
    last = None
    for k, c in enumerate(calls):
        while pending and pending[0][0] <= c.ts:
            done = heapq.heappop(pending)
            if last is None or done[0] >= last[0]:
                last = done
        j = conv_of[c.conversation_id]
        mine = open_task.get(j)
        prev = last[2] if last else None
        resumed = mine is not None and (prev is None or task_of[prev.call_id] == mine
                                        or end_of(tasks[mine]["calls"][-1]) > prev.ts)
        if resumed:
            tasks[mine]["calls"].append(c)
        else:
            t = len(tasks)
            alongside = sorted({task_of[x.call_id] for _, _, x in pending})
            tasks.append({"index": t, "role": convs[j]["role"], "conversation": j, "calls": [c], "alongside": alongside})
            if prev is not None:
                src = task_of[prev.call_id]
                edges.append({"source": src, "target": t, "tools": tool_names(prev, load)})
            open_task[j] = mine = t
        task_of[c.call_id] = mine
        heapq.heappush(pending, (end_of(c), k, c))
    counts: dict[str, int] = {}
    for t in tasks:
        cs = t["calls"]
        counts[t["role"]] = counts.get(t["role"], 0) + 1
        costs = [c.cost_usd for c in cs if c.cost_usd is not None]
        t.update(of_role=counts[t["role"]], start=cs[0].ts, end=end_of(cs[-1]), started=fmt_ts(cs[0].ts),
                 ended=fmt_ts(end_of(cs[-1])), turns=[turns[t["conversation"]].get(c.call_id) for c in (cs[0], cs[-1])],
                 tokens=sum(c.usage.get("total_tokens") or 0 for c in cs),
                 usd=round(sum(costs), 4) if costs else None)
    for t in tasks:
        t["role_tasks"] = counts[t["role"]]
    return tasks, edges, convs


def tasks(run: Run, node_id: str, phase: str, attempt: int) -> dict:
    found, edges, _ = _tasks(run, node_id, phase, attempt)
    return {"tasks": [{**t, "calls": len(t["calls"])} for t in found], "edges": edges}


def task_chat(run: Run, node_id: str, phase: str, attempt: int, index: int) -> list[dict]:
    """The context the model really had in this task: its last call's request as sent, then the reply. Messages that
    were already there at the task's first call are marked `prior`."""
    found, _, convs = _tasks(run, node_id, phase, attempt)
    if not 0 <= index < len(found):
        return []
    load = run.log.payload
    calls = found[index]["calls"]
    turns = call_turns(convs[found[index]["conversation"]]["chain"], load)
    first, final = request_messages(load(calls[0].request)), calls[-1]
    messages = request_messages(load(final.request))
    if not messages:
        return [item("note", None, "request payload missing: this task's context cannot be shown")]
    cut = max((i + 1 for i, m in enumerate(first) if m.get("role") == "assistant"), default=0)
    items, names, answered = [], {}, 0
    tools = request_tools(load(final.request))
    if tools:
        items.append(item("tools", f"Tools offered ({len(tools)})", tool_lines(tools), prior=cut > 0))
    for i, m in enumerate(messages):
        new = message_items(m, names)
        for it in new:
            it["prior"] = i < cut
        items += new
        if i >= cut and m.get("role") == "assistant" and answered < len(calls) - 1:
            items.append(_usage(calls[answered], turns.get(calls[answered].call_id)))
            answered += 1
    if answered < len(calls) - 1:
        items.append(item("note", None, f"{len(calls) - 1 - answered} of this task's calls are not in the last "
                                        "request's history; shown: the last request as sent"))
    reply = response_message(load(final.response)) if final.response else None
    if reply is not None:
        items += message_items(reply, names)
        items.append(_usage(final, turns.get(final.call_id)))
    else:
        items.append(item("note", None, "awaiting response" if _awaiting(run, node_id, phase, attempt)
                                        else "no response recorded"))
    return _annotate(run, node_id, phase, attempt, items)


def tool_logs(run: Run, node_id: str, phase: str, attempt: int) -> list[str]:
    folder = run.files.path(f"nodes/{node_id}/attempts/{phase}-{attempt}/workspace/tool_output")
    if not folder.is_dir():
        return []
    return [str(p.relative_to(run.files.root)) for p in sorted(folder.iterdir()) if p.is_file()]


# ---- Code edits ----

def code_edits(run: Run, node_id: str) -> dict:
    n = run.node(node_id)
    parent = run.node(n["parent_id"]) if n else None
    node_diff = (run.files.git("diff", parent["agent_commit"], n["agent_commit"])
                 if n and parent and n["agent_commit"] and parent["agent_commit"] else None)
    committed = {}
    for line in (run.files.git("for-each-ref", "--format=%(refname) %(objectname)",
                               f"refs/attempts/{node_id}/") or "").splitlines():
        ref, sha = line.split()
        m = re.search(r"/edit_self-(\d+)$", ref)
        if m:
            committed[int(m.group(1))] = sha
    reports = {e.get("attempt"): run.log.payload(e.get("payload")) for e in run.log.events(files=[node_id])
               if e.get("type") == "contract.report"}
    outcomes = {a["idx"]: a for a in run.files.query(
        "SELECT * FROM attempts WHERE node_id = ? AND phase = 'edit_self'", (node_id,))}
    attempts = []
    for k in sorted(set(committed) | set(outcomes) | set(reports)):
        sha = committed.get(k)
        base = (run.files.git("rev-parse", f"{sha}^") or "").strip() if sha else ""
        outcome = outcomes.get(k)
        attempts.append({"attempt": k, "commit": sha, "base": base or None,
                         "outcome": outcome["outcome"] if outcome else "running",
                         "detail": outcome["detail"] if outcome else None,
                         "diff": run.files.git("diff", base, sha) if sha and base else None,
                         "contract": reports.get(k)})
    return {"baseline": bool(n) and n["parent_id"] is None, "node_diff": node_diff,
            "parent_commit": parent["agent_commit"] if parent else None,
            "agent_commit": n["agent_commit"] if n else None, "attempts": attempts}


COMMIT = re.compile(r"[0-9a-f]{4,64}")


def agent_tree(run: Run, commit: str) -> list[str]:
    """`commit` comes from a textbox: only a hex id reaches git, never an option like --output."""
    if not COMMIT.fullmatch(commit or ""):
        return []
    return (run.files.git("ls-tree", "-r", "--name-only", commit) or "").splitlines()


def agent_file(run: Run, commit: str, path: str) -> str | None:
    if not COMMIT.fullmatch(commit or ""):
        return None
    return run.files.git("show", f"{commit}:{path}")


def changed_files(run: Run, base: str, head: str) -> list[dict]:
    """Files that differ between two agent commits, with git's status letter (A, M, D, R...)."""
    if not (COMMIT.fullmatch(base or "") and COMMIT.fullmatch(head or "")):
        return []
    fields = (run.files.git("diff", "--name-status", "-M", "-z", base, head) or "").split("\0")
    out, i = [], 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        paths = fields[i + 1:i + (3 if status[0] in "RC" else 2)]
        out.append({"status": status, "path": paths[-1], "old_path": paths[0]})
        i += 1 + len(paths)
    return out


def commit_choices(run: Run, node_id: str) -> list[dict]:
    """Every agent version worth browsing for a node: its own, its parent's, each edit attempt's."""
    d = code_edits(run, node_id)
    n = run.node(node_id)
    parent = run.node(n["parent_id"]) if n and n["parent_id"] else None
    out = []
    if d["agent_commit"]:
        out.append({"label": f"{node_id} · this node" if parent else f"{node_id} · seed agent", "commit": d["agent_commit"]})
    if parent and parent["agent_commit"]:
        out.append({"label": f"{parent['node_id']} · parent", "commit": parent["agent_commit"]})
    out += [{"label": f"edit_self attempt {a['attempt']} · {a['outcome']} · {a['commit'][:8]}", "commit": a["commit"]}
            for a in d["attempts"] if a["commit"]]
    return out


# ---- Training ----

def loss_curve(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = LOSS.search(line)
            if m:
                out.append({"step": int(m.group(1)), "loss": float(m.group(2)), "lr": float(m.group(3))})
    return out


def _window(events: list[dict], node: str, phase: str, attempt: int) -> tuple[float, float]:
    """From the attempt's phase.start to the next attempt's phase.start (the kernel's own
    gate and training for an attempt run after its agent phase ends)."""
    starts = sorted((e["ts_wall"], e.get("attempt")) for e in events if e.get("type") == "phase.start"
                    and e.get("node") == node and e.get("phase") == phase)
    begin = next((t for t, k in starts if k == attempt), None)
    if begin is None:
        return 0.0, -1.0
    return begin, next((t for t, k in starts if k > attempt), math.inf)


def training_attempts(run: Run, node_id: str) -> list[int]:
    return sorted({e.get("attempt") for e in run.log.events(files=[node_id])
                   if e.get("type") == "phase.start" and e.get("phase") == "improve_recipe"})


def training(run: Run, node_id: str, attempt: int) -> dict:
    base = f"nodes/{node_id}/attempts/improve_recipe-{attempt}"
    events = run.log.events(files=[node_id])
    lo, hi = _window(events, node_id, "improve_recipe", attempt)
    inside = [e for e in events if lo <= e.get("ts_wall", 0) < hi]
    start = next((e for e in inside if e.get("type") == "train.start"), None)
    end = next((e for e in inside if e.get("type") == "train.end"), None)
    outcome = run.files.query("SELECT * FROM attempts WHERE node_id = ? AND phase = 'improve_recipe' AND idx = ?",
                              (node_id, attempt))
    return {
        "config": run.files.read_text(f"{base}/train_config.yaml"),
        "loss": loss_curve(run.files.path(f"{base}/train/train.log")),
        "gates": [{"time": fmt_ts(e["ts_wall"]), "result": e["type"].split(".", 1)[1],
                   "failures": (run.log.payload(e.get("payload")) or {}).get("failures")}
                  for e in inside if e.get("type") in ("gate.passed", "gate.failed")],
        "reached_training": start is not None,
        "started": fmt_ts(start["ts_wall"]) if start else None,
        "duration_min": round((end["ts_wall"] - start["ts_wall"]) / 60, 1) if start and end else None,
        "outcome": outcome[0] if outcome else None,
        "gpu": _gpu_summary(run, start["ts_wall"], end["ts_wall"] if end else time.time()) if start else [],
    }


# ---- Selection and cost ----

def selection(run: Run) -> list[dict]:
    return [{"time": fmt_ts(r["created_at"]), "child": r["child_id"], "chosen": r["chosen"], "seed": r["seed"],
             "candidates": loads(r["candidates"], [])}
            for r in run.files.query("SELECT * FROM selection_events ORDER BY id")]


def cost(run: Run) -> dict:
    by_phase: dict[tuple, dict] = {}
    errors = []
    for e in run.log.events():
        if e.get("type") != "llm.response":
            continue
        key = (e.get("node"), e.get("phase"))
        r = by_phase.setdefault(key, {"node": key[0], "phase": key[1], "calls": 0, "tokens": 0, "usd": 0.0,
                                      "latency_s": 0.0, "errors": 0})
        r["calls"] += 1
        r["tokens"] += (e.get("usage") or {}).get("total_tokens") or 0
        r["usd"] += e.get("cost_usd") or 0.0
        r["latency_s"] += e.get("latency_s") or 0.0
        if e.get("status") != 200:
            r["errors"] += 1
            errors.append({"time": fmt_ts(e["ts_wall"]), "node": key[0], "phase": key[1], "status": e.get("status")})
    roles: dict[str, dict] = {}
    for n in run.nodes():
        for phase, attempt in agent_attempts(run, n["node_id"]):
            for c in conversations(run, n["node_id"], phase, attempt):
                r = roles.setdefault(c["role"], {"role": c["role"], "conversations": 0, "calls": 0,
                                                 "tokens": 0, "usd": 0.0})
                r["conversations"] += 1
                r["calls"] += c["calls"]
                r["tokens"] += c["tokens"]
                r["usd"] += c["usd"] or 0.0
    for r in [*by_phase.values(), *roles.values()]:
        r["usd"] = round(r["usd"], 4)
        if "latency_s" in r:
            r["latency_s"] = round(r["latency_s"], 1)
    return {"by_phase": list(by_phase.values()), "by_role": list(roles.values()), "errors": errors}


# ---- Run-wide series ----

def gpu_full(run: Run, buckets: int = 1000) -> dict:
    """Every GPU's utilization and memory over the whole run: per time bucket its lowest and highest
    sample (in time order), so peaks and dips survive; all samples when there are few."""
    samples = [e for e in run.log.events(files=["gpu"]) if e.get("type") == "gpu.sample"]
    gpus = sorted({str(g) for e in samples for g in (e.get("gpus") or {})}, key=lambda g: (len(g), g))
    out = {"samples": len(samples), "buckets": 0, "first": samples[0]["ts_wall"] if samples else None,
           "last": samples[-1]["ts_wall"] if samples else None,
           "gpus": {g: {"util": [], "memory_gib": []} for g in gpus}}
    if not samples:
        return out
    t0, span = samples[0]["ts_wall"], max(samples[-1]["ts_wall"] - samples[0]["ts_wall"], 1e-9)
    n = buckets if len(samples) > 2 * buckets else len(samples)
    out["buckets"] = n
    groups: dict[int, list[dict]] = {}
    for k, e in enumerate(samples):
        groups.setdefault(min(int((e["ts_wall"] - t0) / span * n), n - 1) if n < len(samples) else k, []).append(e)
    for g in gpus:
        for metric, read in (("util", lambda s: s.get("util")),
                             ("memory_gib", lambda s: round(s["memory_mib"] / 1024, 2) if s.get("memory_mib") is not None else None)):
            series = out["gpus"][g][metric]
            for group in groups.values():
                points = [(round(e["ts_wall"], 1), read((e.get("gpus") or {}).get(g) or {}))
                          for e in group]
                points = [p for p in points if p[1] is not None]
                if not points:
                    continue
                low, high = min(points, key=lambda p: p[1]), max(points, key=lambda p: p[1])
                series.extend(sorted({low, high}))
    return out


def llm_calls(run: Run) -> list[dict]:
    """Every answered model call, in time order, for the cost and context charts."""
    requests = {e.get("call_id"): e for e in run.log.events() if e.get("type") == "llm.request"}
    out = []
    for e in run.log.events():
        if e.get("type") != "llm.response":
            continue
        req = requests.get(e.get("call_id")) or {}
        u = e.get("usage") or {}
        out.append({"ts": e["ts_wall"], "node": e.get("node"), "phase": e.get("phase"), "attempt": e.get("attempt"),
                    "role": e.get("role") or req.get("role") or ("kernel tool: " + req["tool"] if req.get("tool") else "unlabeled"),
                    "conversation_id": e.get("conversation_id"), "turn_index": req.get("turn_index"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "total_tokens": u.get("total_tokens"), "usd": e.get("cost_usd"), "latency_s": e.get("latency_s"),
                    "status": e.get("status"), "model": req.get("model"), "seq": e["_seq"]})
    return out


def training_all(run: Run, node_id: str) -> list[dict]:
    return [{"attempt": a, **training(run, node_id, a)} for a in training_attempts(run, node_id)]


def training_compare(run: Run) -> list[dict]:
    """Each node's loss curve: every improve_recipe attempt that logged any."""
    out = []
    for n in run.nodes():
        for a in training_attempts(run, n["node_id"]):
            loss = loss_curve(run.files.path(f"nodes/{n['node_id']}/attempts/improve_recipe-{a}/train/train.log"))
            if loss:
                out.append({"node": n["node_id"], "attempt": a, "score": n["score"], "loss": loss})
    return out


def eval_matrix(run: Run) -> dict:
    """Every scored node's metrics, in the run's metric order."""
    order = (run.files.read_json("config/run.json") or {}).get("metric_set") or []
    nodes = [{"node": n["node_id"], "parent": n["parent_id"], "score": n["score"],
              "metrics": loads(n["metrics"], {}) or {},
              "aggregates": run.files.read_json(f"nodes/{n['node_id']}/eval/aggregates.json")}
             for n in run.nodes() if n["score"] is not None or n["metrics"]]
    extra = sorted({m for n in nodes for m in n["metrics"]} - set(order))
    return {"metrics": [*order, *extra], "nodes": nodes}
