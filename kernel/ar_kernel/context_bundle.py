"""What an agent sees: built from the archive, written to /context. Each phase gets what its mission
needs. improve_recipe: how earlier nodes scored, on what data, with which recipe. edit_self: how the
agent code changed and how each run went, and never a score."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from ar_contract.models import EditContext, RecipeContext

from .archive.clips import ClipStore
from .archive.nodes import NOT_SHOWN, NodeStore
from .config import run_config_path
from .eval.score import agent_aggregates, agent_metrics, metric_guide
from .process_digest import process_digest
from .tools.data_tools import clip_record, dataset_stats, pool_clips, scores_by_clip
from .train.recipe import RECIPE_RULES, TUNABLE_KEYS
from .train.recipe_guide import recipe_guide

# A failure inside the kernel is not the agent's doing, and its raw text can name what agents never see.
KERNEL_FAILURES = {"eval_failed": "the kernel's evaluation of this node failed",
                   "crashed": "the kernel failed while running this node"}


def _node_file(run_dir: Path, node_id: str, rel: str) -> Path:
    return Path(run_dir) / "nodes" / node_id / rel


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _data_stats(conn, commit_id: str | None) -> dict:
    if not commit_id:
        return {}
    row = conn.execute("SELECT manifest FROM data_commits WHERE commit_id=?", (commit_id,)).fetchone()
    if row is None:
        return {}
    return dataset_stats(json.loads(row["manifest"]), ClipStore(conn).all())      # an existing commit, whole


def _entry(conn, run_dir: Path, repo, node: dict, parent_commit: str | None, phase: str) -> dict:
    nid = node["node_id"]
    entry = {"node_id": nid, "status": node["status"],
             "error": KERNEL_FAILURES.get(node["status"], node["error"])}
    if phase == "improve_recipe":
        recipe_path, rationale = _node_file(run_dir, nid, "recipe.yaml"), _node_file(run_dir, nid, "rationale.md")
        return {**entry, "score": node["score"], "metrics": agent_metrics(node["metrics"]),
                "aggregates": agent_aggregates(_read_json(_node_file(run_dir, nid, "eval/aggregates.json"))),
                "data": _data_stats(conn, node["data_commit"]),
                "recipe": yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None,
                "rationale": rationale.read_text() if rationale.exists() else None}
    attempts = [{"phase": p, "attempt": a["idx"], "outcome": a["outcome"]}
                for p in ("edit_self", "improve_recipe") for a in NodeStore(conn).attempts(nid, p)]
    return {**entry, "edit": _read_json(_node_file(run_dir, nid, "edit.json")),
            "code_diff_stats": (repo.diff_stats(parent_commit, node["agent_commit"])
                                if parent_commit and node["agent_commit"] else []),
            "process": {**process_digest(run_dir, nid), "attempts": attempts}}


def lineage(conn, run_dir: Path, repo, node_id: str, phase: str) -> list[dict]:
    nodes = NodeStore(conn)
    chain = []
    current = node_id
    while current:
        chain.append(nodes.get(current))
        current = chain[-1]["parent_id"]
    chain.reverse()
    return [_entry(conn, run_dir, repo, node, chain[i - 1]["agent_commit"] if i else None, phase)
            for i, node in enumerate(chain)]


def siblings(conn, run_dir: Path, repo, parent_id: str, phase: str) -> list[dict]:
    """The finished children of the parent: what was already tried from the same starting point."""
    parent_commit = NodeStore(conn).get(parent_id)["agent_commit"]
    return [_entry(conn, run_dir, repo, node, parent_commit, phase) for node in NodeStore(conn).all()
            if node["parent_id"] == parent_id and node["status"] not in NOT_SHOWN]


def archive_summary(conn, phase: str) -> dict:
    nodes = [n for n in NodeStore(conn).all() if n["status"] != "quarantined"]
    rows = [{"node_id": n["node_id"], "parent_id": n["parent_id"], "status": n["status"], "depth": n["depth"]}
            for n in nodes]
    if phase == "edit_self":
        return {"nodes": rows}
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    return {"nodes": [{**row, "score": n["score"], "subtree_value": n["subtree_value"]}
                      for row, n in zip(rows, nodes)],
            "n_scored": len(scored),
            "best": {"node_id": best["node_id"], "score": best["score"]} if best else None}


def clip_pool_summary(conn, cap: int = 2000) -> list[dict]:
    usage = scores_by_clip(conn)
    return [clip_record(c, usage) for c in pool_clips(conn)[-cap:]]


def build_edit_context(*, conn, run_dir: Path, repo, parent_id: str, attempt: int, max_attempts: int,
                       retry: dict | None, nodes_remaining: int, dry_run: bool = False) -> EditContext:
    return EditContext(lineage=lineage(conn, run_dir, repo, parent_id, "edit_self"),
                       siblings=siblings(conn, run_dir, repo, parent_id, "edit_self"),
                       archive=archive_summary(conn, "edit_self"),
                       nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts,
                       retry=retry, dry_run=dry_run)


def build_recipe_context(*, cfg, conn, run_dir: Path, repo, node_id: str, parent_id: str, attempt: int,
                         max_attempts: int, retry: dict | None, nodes_remaining: int, n_gpus: int,
                         tools: list[str], dry_run: bool = False) -> RecipeContext:
    parent = NodeStore(conn).get(parent_id)
    recipe_path = _node_file(run_dir, parent_id, "recipe.yaml")
    base = yaml.safe_load(run_config_path(cfg, run_dir, "base_recipe.yaml").read_text())
    return RecipeContext(
        lineage=lineage(conn, run_dir, repo, parent_id, "improve_recipe"),
        siblings=siblings(conn, run_dir, repo, parent_id, "improve_recipe"),
        archive=archive_summary(conn, "improve_recipe"),
        nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts, retry=retry,
        dry_run=dry_run, clip_pool=clip_pool_summary(conn), clip_pool_size=len(pool_clips(conn)),
        parent_data_commit=parent["data_commit"],
        parent_recipe=yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {},
        base_recipe=base, recipe_guide=recipe_guide(base),
        metric_guide=metric_guide(cfg.get("eval.score_weights")),
        tunable_rules={k: {"type": RECIPE_RULES[k][0], "min": RECIPE_RULES[k][1], "max": RECIPE_RULES[k][2]}
                       for k in sorted(TUNABLE_KEYS)},
        resolution_allowlist=[list(p) for p in cfg.get("train.resolution_allowlist")],
        lora_allowlist=[list(p) for p in cfg.get("train.lora_allowlist")],
        n_gpus=n_gpus, tools=list(tools))


def write_bundle(ctx, dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "context.json"
    path.write_text(ctx.model_dump_json(indent=1))
    if ctx.retry:
        (dest / "retry.json").write_text(json.dumps(ctx.retry, indent=1))
    return path
