"""`ar status`: one plain-JSON snapshot of a run. Any future UI reads this."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .archive.db import open_db_readonly
from .archive.nodes import NodeStore
from .budget import Budget
from .config import KernelConfig
from .control import Control
from .selection import candidates
from .telemetry import tree


def run_status(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    control = Control(run_dir)
    cfg = KernelConfig.for_run(run_dir)
    budget = Budget.from_config(cfg)
    conn = open_db_readonly(run_dir, writer_alive=control.alive_pid() is not None)
    try:
        nodes = NodeStore(conn).all()
        budget.load(conn)
    finally:
        conn.close()
    try:      # the probabilities the NEXT draw would use, computed live from the current tree
        probs = {c["node_id"]: c["P"] for c in candidates(nodes, cfg)}
    except ValueError:                                         # no scored node yet
        probs = {}
    state_file = control.dir / "state.json"
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    meta_path = run_dir / "config" / "run.json"
    judge = json.loads(meta_path.read_text()).get("judge") if meta_path.exists() else None
    trace = tree.load(run_dir)
    alerts = [r for r in trace.records if r["kind"] == "alert"][-20:]
    open_spans = [n for n in trace.spans.values() if n.end is None]
    return {
        "run_id": run_dir.name, "judge": judge, "loop_pid": control.alive_pid(),
        "state": json.loads(state_file.read_text()) if state_file.exists() else None,
        "max_nodes": control.args().get("max_nodes"),
        "nodes": [{"node_id": n["node_id"], "parent_id": n["parent_id"], "depth": n["depth"],
                   "status": n["status"], "score": n["score"], "value": n["subtree_value"],
                   "P": probs.get(n["node_id"]), "error": n["error"]}
                  for n in nodes],
        "best": {"node_id": best["node_id"], "score": best["score"]} if best else None,
        "spend": budget.snapshot(),
        "alerts": [{"ts": a["ts"], "name": a["name"], "level": a["level"], "message": a["message"]} for a in alerts],
        "open_spans": [{"ts": n.start["ts"], "name": n.start["name"], "facts": tree.open_span_facts(trace, n)}
                       for n in open_spans],
    }


def _time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def format_status(d: dict) -> str:
    lines = [f"run {d['run_id']}: loop {'running (pid %s)' % d['loop_pid'] if d['loop_pid'] else 'not running'}"]
    if d.get("judge"):
        lines.append(f"judge: {d['judge']['kind']} {d['judge']['model']}")
    if d["state"]:
        s = d["state"]
        lines.append(f"now: {s.get('node')} {s.get('phase')} attempt {s.get('attempt')}")
    spend = d["spend"]
    usd = "n/a (no prices)" if spend["usd"] is None else f"${spend['usd']:.4f}"
    cap = "none" if spend["max_usd"] is None else f"${spend['max_usd']}"
    lines.append(f"spend: {usd}, {spend['tokens']} tokens, {spend['calls']} calls; cap: {cap}")
    done = sum(1 for n in d["nodes"] if n["parent_id"] is not None)
    lines.append(f"nodes: {done} of {d['max_nodes']}; best: {d['best']}")
    children: dict[str | None, list[dict]] = {}
    for n in d["nodes"]:
        children.setdefault(n["parent_id"], []).append(n)
    todo = children.get(None, [])[::-1]
    while todo:                                 # depth first: a node is listed under its own parent
        n = todo.pop()
        todo += children.get(n["node_id"], [])[::-1]
        score = "-" if n["score"] is None else f"{n['score']:.4f}"
        p = "" if n["P"] is None else f" P={n['P']:.2f}"
        err = f"  ! {n['error'][:120]}" if n["error"] else ""
        lines.append(f"  {'  ' * n['depth']}{n['node_id']:<8} {n['status']:<14} {score}{p}{err}")
    if d["alerts"]:
        lines.append("alerts (latest last):")
        lines += [f"  {_time(a['ts'])} [{a['level']}] {a['name']}: {a['message']}" for a in d["alerts"]]
    if d["open_spans"]:
        lines.append("open spans:")
        lines += [f"  {_time(o['ts'])} {o['name']}: {o['facts']}" for o in d["open_spans"]]
    return "\n".join(lines)
