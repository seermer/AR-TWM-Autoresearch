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


def _events(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:                    # a line torn by kill -9 or a full disk
            continue
    return [e for e in events if e.get("type") == kind]


def run_status(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    control = Control(run_dir)
    conn = open_db_readonly(run_dir, writer_alive=control.alive_pid() is not None)
    try:
        nodes = NodeStore(conn).all()
    finally:
        conn.close()
    cfg = KernelConfig.for_run(run_dir)
    try:      # the probabilities the NEXT draw would use, computed live from the current tree
        probs = {c["node_id"]: c["P"] for c in candidates(nodes, cfg)}
    except ValueError:                                         # no scored node yet
        probs = {}
    state_file = control.dir / "state.json"
    budget = Budget.from_config(cfg)
    budget.load(run_dir)
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    alerts = _events(run_dir / "telemetry" / "events" / "run.jsonl", "alert")[-20:]
    meta_path = run_dir / "config" / "run.json"
    judge = json.loads(meta_path.read_text()).get("judge") if meta_path.exists() else None
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
        "alerts": [{"ts": a["ts_wall"], "kind": a.get("kind"), "level": a.get("level"),
                    "message": a.get("message")} for a in alerts],
    }


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
    done = sum(1 for n in d["nodes"] if n["parent_id"] is not None and n["status"] != "interrupted")
    lines.append(f"nodes: {done} of {d['max_nodes']}; best: {d['best']}")
    for n in d["nodes"]:
        score = "-" if n["score"] is None else f"{n['score']:.4f}"
        p = "" if n["P"] is None else f" P={n['P']:.2f}"
        err = f"  ! {n['error'][:120]}" if n["error"] else ""
        lines.append(f"  {'  ' * n['depth']}{n['node_id']:<6} {n['status']:<14} {score}{p}{err}")
    if d["alerts"]:
        lines.append("alerts (latest last):")
        lines += [f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a['ts']))} [{a['level']}] {a['kind']}: {a['message']}" for a in d["alerts"]]
    return "\n".join(lines)
