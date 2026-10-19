"""What an agent sees: built from the archive, written to /context. Each phase gets what its mission
needs. improve_recipe: how earlier nodes scored, on what data, with which recipe. edit_self: how the
agent code changed and how each run went, and never a score."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml
from ar_contract.models import EditContext, RecipeContext

from .archive.clips import ClipStore
from .archive.nodes import NOT_SHOWN, NodeStore
from .config import run_config_path
from .eval.score import agent_aggregates, agent_metrics, metric_guide
from .isolation import scrub
from .process_digest import process_digest
from .tools.data_tools import clip_source, dataset_stats, pool_clips
from .train.recipe import RECIPE_RULES, TUNABLE_KEYS
from .train.recipe_guide import recipe_guide

# A failed eval's raw error can name what agents never see.
KERNEL_FAILURES = {"eval_failed": "the evaluation of this node's model failed"}


# What each folder of the container is for, per phase: a role is told this first, so it need not find out.
_SHARED = {
    "/context/context.json": "the exact data this message was built from",
    "/nodes/<node>/": "read-only: what each finished node left behind (layout: skill node_files). Kernel tools "
                      "cannot read it; copy a file into /workspace to pass it to one",
    "/tmp": "scratch in memory, gone when the phase ends"}
FOLDERS = {
    "improve_recipe": {
        "/workspace": "writable; later nodes see it under /nodes/<this node>/attempts/",
        "/workspace/staging": "what goes to and comes from kernel tools: downloads (hf/), GPU job outputs "
                              "(annotations/, rollouts/, images/), result files (results/), clips copied from the pool "
                              "(pool_clips/) and the candidates "
                              "you stage for data_ingest, which copies them into the archive. A separate mount: a hard link "
                              "between it and the rest of /workspace fails, so copy or move. Not kept for later "
                              "nodes",
        "/agent": "read-only: this agent's code",
        **_SHARED,
        "kernel tool paths": "regular files under /workspace only; a symbolic link is refused unless it is "
                             "relative and stays inside /workspace or /workspace/staging, so copy the file instead"},
    "edit_self": {
        "/agent": "the agent's code, writable in this phase",
        "/code": "read-only: the parent's code, which is the code running you",
        "/workspace": "scratch for this phase, writable",
        **_SHARED}}


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
        summary = _node_file(run_dir, nid, "summary.md")
        return {**entry, "score": node["score"], "metrics": agent_metrics(node["metrics"]),
                "aggregates": agent_aggregates(_read_json(_node_file(run_dir, nid, "eval/aggregates.json"))),
                "data_commit": node["data_commit"], "data": _data_stats(conn, node["data_commit"]),
                "recipe": yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None,
                "rationale": rationale.read_text() if rationale.exists() else None,
                "summary": summary.read_text() if summary.exists() else None}
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
    nodes = [n for n in NodeStore(conn).all() if n["status"] not in NOT_SHOWN]
    rows = [{"node_id": n["node_id"], "parent_id": n["parent_id"], "status": n["status"]} for n in nodes]
    if phase == "edit_self":
        return {"nodes": rows}
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    return {"nodes": [{**row, "score": n["score"], "data_commit": n["data_commit"]}
                      for row, n in zip(rows, nodes)],
            "n_scored": len(scored),
            "best": {"node_id": best["node_id"], "score": best["score"]} if best else None}


def clip_pool_stats(clips: list[dict]) -> dict:
    """How many pool clips each node ingested and each source gave. The clips
    themselves are listed by data_query: a record per clip made the context grow by megabytes."""
    counts = {"by_node": Counter(c.get("ingested_by") for c in clips),
              "by_source": Counter(clip_source(c) for c in clips)}
    return {key: dict(count.most_common()) for key, count in counts.items()}


def build_edit_context(*, conn, run_dir: Path, repo, parent_id: str, dry_run: bool = False) -> EditContext:
    return EditContext(lineage=lineage(conn, run_dir, repo, parent_id, "edit_self"),
                       siblings=siblings(conn, run_dir, repo, parent_id, "edit_self"),
                       archive=archive_summary(conn, "edit_self"),
                       dry_run=dry_run, folders=FOLDERS["edit_self"])


def build_recipe_context(*, cfg, conn, run_dir: Path, repo, node_id: str, parent_id: str, n_gpus: int,
                         tools: list[str], dry_run: bool = False) -> RecipeContext:
    parent = NodeStore(conn).get(parent_id)
    pool = pool_clips(conn)
    recipe_path = _node_file(run_dir, parent_id, "recipe.yaml")
    base = yaml.safe_load(run_config_path(cfg, run_dir, "base_recipe.yaml").read_text())
    return RecipeContext(
        lineage=lineage(conn, run_dir, repo, parent_id, "improve_recipe"),
        siblings=siblings(conn, run_dir, repo, parent_id, "improve_recipe"),
        archive=archive_summary(conn, "improve_recipe"),
        dry_run=dry_run, folders=FOLDERS["improve_recipe"], node_id=node_id,
        clip_pool_size=len(pool), clip_pool_stats=clip_pool_stats(pool),
        parent_data_commit=parent["data_commit"],
        parent_recipe=yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {},
        base_recipe=base, recipe_guide=recipe_guide(base),
        metric_guide=metric_guide(NodeStore(conn).get("root")["metric_set"],     # the run's: the root is scored first
                                  cfg.get("eval.score_weights")),
        target_skill=cfg.get("eval.target_skill") or {},
        tunable_rules={k: {"type": RECIPE_RULES[k][0], "min": RECIPE_RULES[k][1], "max": RECIPE_RULES[k][2]}
                       for k in sorted(TUNABLE_KEYS)},
        lora_allowlist=[list(p) for p in cfg.get("train.lora_allowlist")],
        n_gpus=n_gpus, tools=list(tools))


def write_bundle(ctx, dest: Path, censor: list[str] = (), roots=()) -> Path:
    """`censor`: names replaced in everything written, as are the host folders `roots`. The context carries node and tool error
    texts, and it reaches the model as a user message, which the gateway does not censor."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data = scrub(ctx.model_dump(mode="json"), list(censor), roots)
    path = dest / "context.json"
    path.write_text(json.dumps(data, indent=1))
    return path
