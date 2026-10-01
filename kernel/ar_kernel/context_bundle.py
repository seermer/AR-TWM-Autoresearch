"""What an agent sees: built from the archive, written to /context."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from ar_contract.models import EditContext, RecipeContext

from .archive.clips import ClipStore
from .archive.nodes import UNFINISHED, NodeStore
from .config import run_config_path
from .process_digest import process_digest
from .tools.data_tools import clip_record, dataset_stats, scores_by_clip
from .train.recipe import RECIPE_RULES, TUNABLE_KEYS
from .train.recipe_guide import recipe_guide

FORMAT_RULES = """\
Standard data formats (the only ones accepted):
- Layout per dataset root: videos/<id>.mp4, captions/<id>.json, poses/<id>.npz (camera formats only).
- video_caption_camera: video + caption + per-frame camera poses.
- video_timed_prompts_camera: as above + caption "segments": [{"time_range_s": [start, end), "prompt"}].
  per_chunk mode: every boundary inside the clip on a round boundary 25/24 + k*32/24 s (within half a frame).
  segment mode: segments shorter than 2.375 s are never trained.
- video_caption_static: video + caption; truly fixed camera; no poses (identity is used).
- Training window: 57 frames at 24 fps (25 history + 32 target). Rollout rounds are 32 frames.
- Video: .mp4, fps >= 24 (higher is subsampled), duration >= 2.375 s, DISPLAY aspect within 2% of 16:9,
  no display rotation (re-encode with rotation applied). Frames are resized, never cropped.
- Caption: non-empty "caption" string.
- Poses: cam_c2w [N,4,4] with N = the mp4's frame count, camera-to-world, OpenCV convention, finite,
  bottom row [0,0,0,1], orthonormal rotation with det +1. Optional intrinsics [3,3] or [N,3,3] in pixels.
- Each enabled dataset needs at least as many clips as training GPUs.
- Clips are immutable: a re-captioned, cropped or trimmed clip is a new clip with derived_from set.
"""


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
    return dataset_stats(json.loads(row["manifest"]), ClipStore(conn).all())


def _entry(conn, run_dir: Path, repo, node: dict, parent_commit: str | None) -> dict:
    recipe_path = _node_file(run_dir, node["node_id"], "recipe.yaml")
    rationale = _node_file(run_dir, node["node_id"], "rationale.md")
    return {
        "node_id": node["node_id"], "status": node["status"],
        "error": node["error"],
        "score": node["score"],
        "metrics": node["metrics"], "data": _data_stats(conn, node["data_commit"]),
        "recipe": yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None,
        "rationale": rationale.read_text() if rationale.exists() else None,
        "edit": _read_json(_node_file(run_dir, node["node_id"], "edit.json")),
        "aggregates": _read_json(_node_file(run_dir, node["node_id"], "eval/aggregates.json")),
        "process": process_digest(run_dir, node["node_id"]),
        "code_diff_stats": (repo.diff_stats(parent_commit, node["agent_commit"])
                            if parent_commit and node["agent_commit"] else []),
    }


def lineage(conn, run_dir: Path, repo, node_id: str) -> list[dict]:
    nodes = NodeStore(conn)
    chain = []
    current = node_id
    while current:
        chain.append(nodes.get(current))
        current = chain[-1]["parent_id"]
    chain.reverse()
    return [_entry(conn, run_dir, repo, node, chain[i - 1]["agent_commit"] if i else None)
            for i, node in enumerate(chain)]


def siblings(conn, run_dir: Path, repo, parent_id: str) -> list[dict]:
    """The finished children of the parent: what was already tried from the same starting point."""
    parent_commit = NodeStore(conn).get(parent_id)["agent_commit"]
    return [_entry(conn, run_dir, repo, node, parent_commit) for node in NodeStore(conn).all()
            if node["parent_id"] == parent_id and node["status"] not in UNFINISHED]


def archive_summary(conn) -> dict:
    nodes = NodeStore(conn).all()
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    return {"nodes": [{"node_id": n["node_id"], "parent_id": n["parent_id"], "status": n["status"],
                       "score": n["score"], "subtree_value": n["subtree_value"], "depth": n["depth"],
                       "error": n["error"]}
                      for n in nodes],
            "n_scored": len(scored),
            "best": {"node_id": best["node_id"], "score": best["score"]} if best else None}


def clip_pool_summary(conn, cap: int = 2000) -> list[dict]:
    usage = scores_by_clip(conn)
    clips = ClipStore(conn).all()[-cap:]
    return [clip_record(c, usage) for c in clips]


def build_edit_context(*, conn, run_dir: Path, repo, parent_id: str, attempt: int, max_attempts: int,
                       retry: dict | None, nodes_remaining: int, dry_run: bool = False) -> EditContext:
    return EditContext(lineage=lineage(conn, run_dir, repo, parent_id),
                       siblings=siblings(conn, run_dir, repo, parent_id), archive=archive_summary(conn),
                       nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts,
                       retry=retry, dry_run=dry_run)


def build_recipe_context(*, cfg, conn, run_dir: Path, repo, node_id: str, parent_id: str, attempt: int,
                         max_attempts: int, retry: dict | None, nodes_remaining: int, n_gpus: int,
                         tools: list[str], dry_run: bool = False) -> RecipeContext:
    parent = NodeStore(conn).get(parent_id)
    recipe_path = _node_file(run_dir, parent_id, "recipe.yaml")
    base = yaml.safe_load(run_config_path(cfg, run_dir, "base_recipe.yaml").read_text())
    return RecipeContext(
        lineage=lineage(conn, run_dir, repo, parent_id),
        siblings=siblings(conn, run_dir, repo, parent_id), archive=archive_summary(conn),
        nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts, retry=retry,
        dry_run=dry_run, clip_pool=clip_pool_summary(conn),
        clip_pool_size=conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
        parent_data_commit=parent["data_commit"],
        parent_recipe=yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {},
        base_recipe=base, recipe_guide=recipe_guide(base),
        tunable_rules={k: {"type": RECIPE_RULES[k][0], "min": RECIPE_RULES[k][1], "max": RECIPE_RULES[k][2]}
                       for k in sorted(TUNABLE_KEYS)},
        resolution_allowlist=[list(p) for p in cfg.get("train.resolution_allowlist")],
        lora_allowlist=[list(p) for p in cfg.get("train.lora_allowlist")],
        format_rules=FORMAT_RULES, n_gpus=n_gpus, tools=list(tools))


def write_bundle(ctx, dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "context.json"
    path.write_text(ctx.model_dump_json(indent=1))
    if ctx.retry:
        (dest / "retry.json").write_text(json.dumps(ctx.retry, indent=1))
    return path
