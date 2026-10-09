"""Parent selection (user decision 2026-09-27): a softmax
over each scored node's value -- mostly its own score, partly its subtree mean shrunk toward that
score -- with a temperature set by the spread of values (floored at proxy noise), a penalty for
large subtrees, and a uniform floor. Continuous in the scores."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time

from .archive.nodes import NodeStore


def candidates(nodes: list[dict], cfg) -> list[dict]:
    s = cfg.get("selection")
    decay, prior, share = float(s["decay"]), float(s["prior_weight"]), float(s["subtree_share"])
    kids: dict[str | None, list[dict]] = {}
    for n in nodes:
        kids.setdefault(n["parent_id"], []).append(n)
    rows = []
    for n in nodes:
        if n["status"] != "scored" or n["score"] is None:
            continue
        num, den, size = prior * n["score"], prior, 0.0
        frontier, k = [n["node_id"]], 0
        while frontier:
            k += 1
            nxt = []
            for pid in frontier:
                for child in kids.get(pid, []):
                    if child["status"] == "interrupted":    # it and anything under it count nowhere
                        continue
                    nxt.append(child["node_id"])
                    size += decay ** (k - 1)
                    if child["status"] == "scored" and child["score"] is not None:
                        num += decay ** k * child["score"]
                        den += decay ** k
            frontier = nxt
        mean = num / den
        rows.append({"node_id": n["node_id"], "score": n["score"], "subtree_mean": mean,
                     "value": (1 - share) * n["score"] + share * mean, "size": size,
                     "penalty": 1 / (1 + size / float(s["size_scale"]))})
    if not rows:
        raise ValueError("no scored node to select a parent from")
    if len(rows) == 1:
        rows[0].update(w=1.0, P=1.0)
        return rows
    values = [r["value"] for r in rows]
    # the 1e-12 guard keeps tau > 0 even if noise_floor is configured as 0 and all values tie
    tau = float(s["temperature"]) * max(statistics.pstdev(values), float(s["noise_floor"]), 1e-12)
    top = max(values)
    for r in rows:
        r["w"] = math.exp((r["value"] - top) / tau) * r["penalty"]
    total, eps = sum(r["w"] for r in rows), float(s["epsilon"])
    for r in rows:
        r["P"] = (1 - eps) * r["w"] / total + eps / len(rows)
    return rows


def selection_seed(run_id: str, child_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_id}:{child_id}".encode()).digest()[:4], "big")


def select_parent(conn, cfg, child_id: str, seed: int, recorder) -> str:
    rows = candidates(NodeStore(conn).all(), cfg)
    chosen = random.Random(seed).choices([r["node_id"] for r in rows], weights=[r["P"] for r in rows])[0]
    conn.execute("INSERT INTO selection_events (child_id, chosen, seed, candidates, created_at) "
                 "VALUES (?,?,?,?,?)", (child_id, chosen, seed, json.dumps(rows), time.time()))
    recorder.event("select", payload={"child": child_id, "chosen": chosen, "seed": seed,
                                      "candidates": rows}, chosen=chosen, child=child_id)
    return chosen


def update_values(conn, cfg) -> None:
    nodes = NodeStore(conn)
    for row in candidates(nodes.all(), cfg):
        nodes.set_fields(row["node_id"], subtree_value=row["value"])
