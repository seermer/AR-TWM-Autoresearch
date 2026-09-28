"""Per-tab data for the run panel (spec section 5): plain dicts and lists, no Gradio."""
from __future__ import annotations

import difflib
import math
import re
import shutil
import threading
import time
from pathlib import Path

from .chat import (chains, chat_items, infer_role, message_items, request_messages, request_tools,
                   response_message, segments, text_of)
from .events import EventLog, event_row, filter_events
from .runfiles import RunFiles, fmt_ts, loads

LOSS = re.compile(r"\[Train\] step=(\d+)\b.*?\bloss=([-+0-9.eE]+).*?\blr=([-+0-9.eE]+)")


class Run:
    def __init__(self, run_dir: Path | str) -> None:
        self.files = RunFiles(run_dir)
        self.log = EventLog(self.files)
        self._prompts: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

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

    def prompt_files(self, commit: str | None) -> dict[str, str]:
        """agent/prompts/*.md of an agent commit, cached (a commit never changes)."""
        if not commit:
            return {}
        with self._lock:
            if commit in self._prompts:
                return self._prompts[commit]
        names = (self.files.git("ls-tree", "--name-only", f"{commit}:agent/prompts") or "").split()
        found = {n[:-3]: self.files.git("show", f"{commit}:agent/prompts/{n}") or ""
                 for n in names if n.endswith(".md")}
        with self._lock:
            self._prompts[commit] = found
        return found


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
                     "score": n["score"], "vs parent": delta, "component": n["edit_component"] or "",
                     "attempts": n["attempt_counts"] or "",
                     "duration_min": round(total / 60, 1) if total else None, "error": n["error"] or ""})
    pid = run.files.loop_pid()
    responses = [e for e in events if e.get("type") == "llm.response"]
    costs = [e["cost_usd"] for e in responses if e.get("cost_usd") is not None]
    alerts = [e for e in events if e.get("type") == "alert"][-50:]
    return {
        "loop": {"alive": pid is not None, "pid": pid, "state": run.files.read_json("control/state.json"),
                 "state_label": "current" if pid else "last recorded",
                 "max_nodes": (run.files.read_json("control/run_args.json") or {}).get("max_nodes")},
        "spend": {"calls": len(responses),
                  "tokens": sum((e.get("usage") or {}).get("total_tokens") or 0 for e in responses),
                  "usd": round(sum(costs), 4) if costs else None},
        "nodes": rows,
        "scores": [{"node": n["node_id"], "score": n["score"]} for n in nodes
                   if n["status"] == "scored" and n["score"] is not None],
        "alerts": [{"time": fmt_ts(a["ts_wall"]), "level": a.get("level"), "kind": a.get("kind"),
                    "message": a.get("message")} for a in reversed(alerts)],
        "disk_free_gb": round(shutil.disk_usage(run.files.root).free / 1e9, 1),
        "recent": [event_row(e) for e in reversed(events[-50:])],
    }


def gpu_series(run: Run, hours: float = 6.0) -> list[dict]:
    """The last `hours` of GPU samples, ending at the run's latest sample (a stopped run's
    samples are all older than the clock's last hours)."""
    samples = [e for e in run.log.events(files=["gpu"]) if e.get("type") == "gpu.sample"]
    if not samples:
        return []
    since = max(e.get("ts_wall", 0) for e in samples) - hours * 3600
    out = []
    for e in samples:
        if e.get("ts_wall", 0) < since:
            continue
        for gpu, s in (e.get("gpus") or {}).items():
            out.append({"time": fmt_ts(e["ts_wall"]), "gpu": str(gpu), "util": s.get("util"),
                        "memory_gib": round((s.get("memory_mib") or 0) / 1024, 2)})
    return out


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


def trace(run: Run, *, node, phase, attempt, types, component, text, include_gpu, page, page_size=100) -> dict:
    found = filter_events(run.log.events(include_gpu=include_gpu), node=node or None, phase=phase or None,
                          attempt=attempt, types=types or None, component=component or None, text=text)
    pages = max(1, math.ceil(len(found) / page_size))
    page = min(max(0, int(page or 0)), pages - 1)
    return {"rows": [event_row(e) for e in found[page * page_size:(page + 1) * page_size]],
            "total": len(found), "pages": pages}


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
                         "text": "\n".join((t.get("function") or {}).get("name", "?") for t in tools)})
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
        "recipe": recipe,
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


def _phase_start(events: list[dict], phase: str, attempt: int) -> dict | None:
    return next((e for e in events if e.get("type") == "phase.start" and e.get("phase") == phase
                 and e.get("attempt") == attempt), None)


def conversations(run: Run, node_id: str, phase: str, attempt: int) -> list[dict]:
    events = run.log.events(files=[node_id])
    load = run.log.payload
    start = _phase_start(events, phase, attempt)
    prompts = run.prompt_files((load(start.get("payload")) or {}).get("code_commit") if start else None)
    out = []
    for i, chain in enumerate(chains(segments(events, node_id, phase, attempt), load)):
        system = next((text_of(m.get("content")) for m in request_messages(load(chain[0].calls[0].request))
                       if m.get("role") == "system"), None)
        calls = [c for s in chain for c in s.calls]
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        out.append({"index": i, "role": infer_role(system, prompts), "started": fmt_ts(chain[0].first_ts),
                    "calls": len(calls), "compactions": len(chain) - 1,
                    "tokens": sum(c.usage.get("total_tokens") or 0 for c in calls),
                    "usd": round(sum(costs), 4) if costs else None, "chain": chain})
    return out


def conversation_chat(run: Run, node_id: str, phase: str, attempt: int, index: int) -> list[dict]:
    convs = conversations(run, node_id, phase, attempt)
    if not 0 <= index < len(convs):
        return []
    events = run.log.events(files=[node_id])
    ended = any(e.get("type") == "phase.end" and e.get("phase") == phase and e.get("attempt") == attempt
                for e in events)
    awaiting = run.files.loop_pid() is not None and not ended
    return chat_items(convs[index]["chain"], run.log.payload, awaiting)


def tool_logs(run: Run, node_id: str, phase: str, attempt: int) -> list[str]:
    folder = run.files.path(f"nodes/{node_id}/attempts/{phase}-{attempt}/workspace/tool_output")
    if not folder.is_dir():
        return []
    return [str(p.relative_to(run.files.root)) for p in sorted(folder.glob("*.log"))]


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
        attempts.append({"attempt": k, "commit": sha, "outcome": outcome["outcome"] if outcome else "running",
                         "detail": outcome["detail"] if outcome else None,
                         "diff": run.files.git("diff", base, sha) if sha and base else None,
                         "contract": reports.get(k)})
    return {"baseline": bool(n) and n["parent_id"] is None, "node_diff": node_diff,
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
                r = roles.setdefault(c["role"], {"role (inferred)": c["role"], "conversations": 0, "calls": 0,
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
