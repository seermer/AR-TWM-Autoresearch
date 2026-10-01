"""Data tools: video_probe, data_ingest, data_query, data_commit, recipe_check."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..archive.commits import CommitError, CommitStore
from ..archive.db import open_db
from ..archive.nodes import NodeStore
from ..data.ingest import Candidate, Ingestor
from ..data.leakage import LeakageChecker
from ..data.probe import probe_video
from ..train.gate import Gate
from .context import PathError, to_host
from .server import ToolError

QUERY_LIMIT = 100           # clips per page: a clip record is about 1,200 characters


class DataTools:
    def __init__(self, cfg, run_dir: Path, recorder, gpus: list[int], gpu_lock: threading.Lock) -> None:
        self.cfg, self.run_dir, self.recorder = cfg, Path(run_dir), recorder
        self.gpus, self.gpu_lock = list(gpus), gpu_lock
        self._leakage: LeakageChecker | None = None
        self._leakage_lock = threading.Lock()

    def _leakage_checker(self) -> LeakageChecker:
        """Built on first ingest and shared: it hashes every WBench case image (7.5 s warm,
        72 s cold). check() only reads it, so concurrent ingests may share it."""
        with self._leakage_lock:
            if self._leakage is None:
                self._leakage = LeakageChecker(self.cfg)
            return self._leakage

    def _host(self, caller, path: str) -> Path:
        try:
            return to_host(caller, path)
        except PathError as exc:
            raise ToolError(str(exc)) from exc

    def probe(self, caller, path: str) -> dict:
        info = probe_video(self._host(caller, path))
        return {"frames": info.frames, "fps": info.fps, "width": info.width, "height": info.height,
                "duration": info.duration, "rotation": info.rotation, "sar": info.sar,
                "display_aspect": info.display_aspect}

    def ingest(self, caller, candidates: list[dict]) -> list[dict]:
        built = []
        for i, c in enumerate(candidates):
            if not c.get("provenance"):
                raise ToolError(f"candidate {i}: provenance is required")
            for key in ("video", "caption", "camera_motion"):
                if not c.get(key):
                    raise ToolError(f"candidate {i}: {key} is required")
            host = {key: self._host(caller, c[key]) for key in ("video", "caption", "pose") if c.get(key)}
            for key, path in host.items():
                if not path.is_file():           # container paths only: host paths never reach the agent
                    raise ToolError(f"candidate {i}: {key} {c[key]} does not exist (data_ingest moves each "
                                    "staged file into the archive, so an already ingested file is gone)")
            built.append(Candidate(
                video=host["video"], caption=host["caption"], pose=host.get("pose"),
                camera_motion=c["camera_motion"], provenance=c["provenance"],
                license=c.get("license"), derived_from=list(c.get("derived_from") or [])))
        conn = open_db(self.run_dir)
        try:
            ingestor = Ingestor(self.cfg, self.run_dir, conn, self.recorder, self._leakage_checker())
            results = ingestor.ingest(built, node_id=caller.node)
        finally:
            conn.close()
        return [{"accepted": r.accepted, "clip_id": r.clip_id, "formats": r.formats,
                 "warnings": r.warnings, "reasons": r.reasons} for r in results]

    def query(self, caller, filter: dict) -> dict:
        conn = open_db(self.run_dir)
        try:
            clips = ClipStore(conn).all()
            usage = scores_by_clip(conn)
        finally:
            conn.close()
        fmt, motion = filter.get("format"), filter.get("camera_motion")
        wanted = set(filter.get("clip_ids") or [])
        out = []
        for clip in clips:
            if fmt and not any(f == fmt or f.startswith(fmt + ":") for f in clip["formats"]):
                continue
            if motion and clip["camera_motion"] != motion:
                continue
            if wanted and clip["clip_id"] not in wanted:
                continue
            if filter.get("ingested_by") and clip.get("ingested_by") != filter["ingested_by"]:
                continue
            out.append(clip_record(clip, usage))
        offset, limit = int(filter.get("offset") or 0), int(filter.get("limit") or QUERY_LIMIT)
        page = out[offset:offset + limit]
        return {"total": len(out), "returned": len(page), "clips": page}

    def commit(self, caller, parent: str | None, datasets: dict, message: str) -> dict:
        conn = open_db(self.run_dir)
        try:
            store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))
            try:
                commit_id = store.commit(parent, datasets, message, node_id=caller.node,
                                         attempt=caller.attempt)
            except CommitError as exc:
                raise ToolError(str(exc)) from exc
            stats = dataset_stats(store.manifest(commit_id), ClipStore(conn).all())
        finally:
            conn.close()
        return {"commit_id": commit_id, "datasets": stats}

    def recipe_check(self, caller, recipe: dict, data_commit: str) -> dict:
        scratch = caller.workspace_host.parent / "recipe_check" / uuid.uuid4().hex
        conn = open_db(self.run_dir)
        try:
            store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))
            try:
                store.manifest(data_commit)
            except Exception as exc:
                raise ToolError(f"unknown data commit {data_commit!r}") from exc
            parent_commit = _parent_commit(conn, caller.node)
            with self.gpu_lock:          # the describe step is a GPU job
                result = Gate(self.cfg, store, self.recorder).check(
                    recipe, data_commit, parent_commit, caller.node, scratch, self.run_dir, self.gpus)
        finally:
            conn.close()
            shutil.rmtree(scratch, ignore_errors=True)
        return {"ok": result.ok, "failures": result.failures}


def _parent_commit(conn, node_id: str) -> str | None:
    nodes = NodeStore(conn)
    try:
        parent_id = nodes.get(node_id)["parent_id"]
        return nodes.get(parent_id)["data_commit"] if parent_id else None
    except KeyError:
        return None


def clip_source(clip: dict, by_id: dict[str, dict]) -> str:
    """Where a clip's footage came from: `rollout:<generator>` or `hf:<repo>`. A derived clip takes
    the source of the clips it names in derived_from; one that names none is just `derived`."""
    provenance = clip["provenance"]
    if provenance.get("kind") == "rollout":
        return f"rollout:{provenance.get('generator')}"
    if provenance.get("kind") == "hf_dataset":
        return f"hf:{provenance.get('repo')}"
    parents = [by_id[i] for i in clip.get("derived_from") or [] if i in by_id]
    if parents:
        return "+".join(sorted({clip_source(p, by_id) for p in parents}))
    return str(provenance.get("kind"))


def dataset_stats(manifest: dict, clips: list[dict]) -> dict[str, dict]:
    """Per-dataset summary of a commit manifest: format, prompt_mode, weight, clip count and
    clip count per source (data_commit's result row and the context's lineage entries)."""
    by_id = {c["clip_id"]: c for c in clips}
    out = {}
    for name, d in manifest["datasets"].items():
        sources: dict[str, int] = {}
        for clip_id in d["clips"]:
            source = clip_source(by_id[clip_id], by_id)
            sources[source] = sources.get(source, 0) + 1
        out[name] = {"format": d["format"], "prompt_mode": d["prompt_mode"], "weight": d["weight"],
                     "clips": len(d["clips"]), "sources": sources}
    return out


def clip_record(clip: dict, usage: dict[str, list[float]]) -> dict:
    """Shape one archive clip row for data_query's result (the context
    bundle reuses this instead of re-deriving it from ClipStore rows)."""
    return {"clip_id": clip["clip_id"], "formats": clip["formats"],
            "camera_motion": clip["camera_motion"], "metadata": clip["metadata"],
            "warnings": clip["warnings"], "provenance": clip["provenance"],
            "license": clip.get("license"), "derived_from": clip.get("derived_from"),
            "ingested_by": clip.get("ingested_by"),
            "used_by_scores": usage.get(clip["clip_id"], [])}


def scores_by_clip(conn) -> dict[str, list[float]]:
    """Scores of scored nodes whose data commit contains each clip."""
    out: dict[str, list[float]] = {}
    rows = conn.execute("SELECT n.score, c.manifest FROM nodes n JOIN data_commits c "
                        "ON n.data_commit = c.commit_id WHERE n.status = 'scored'").fetchall()
    for row in rows:
        clip_ids = {cid for d in json.loads(row["manifest"])["datasets"].values() for cid in d["clips"]}
        for cid in clip_ids:
            out.setdefault(cid, []).append(float(row["score"]))
    return out


def register_data_tools(mcp, kit, tools: DataTools) -> None:
    @mcp.tool(name="video_probe", description="Frame count, fps, coded size, rotation, pixel "
              "aspect and display aspect of a video under /workspace.")
    async def video_probe(path: str, ctx: Context) -> dict:
        return await kit.call(ctx, "video_probe", {"path": path}, lambda c: tools.probe(c, path))

    @mcp.tool(name="data_ingest", description="Ingest staged candidates. Each: video, "
              "caption, optional pose (container paths under /workspace/staging), camera_motion "
              "'moving'|'static', provenance, optional license and derived_from. Ingest MOVES each "
              "staged file into the archive: copy it first if you still need it. Returns accepted + "
              "clip_id + eligible formats, or rejected + reasons, per candidate.")
    async def data_ingest(candidates: list[dict[str, Any]], ctx: Context) -> list[dict]:
        return await kit.call(ctx, "data_ingest", {"candidates": candidates},
                              lambda c: tools.ingest(c, candidates))

    @mcp.tool(name="data_query", description="Search the archive-wide clip pool. Filter keys: "
              "format, camera_motion, clip_ids, ingested_by, limit (default 100), offset (for the next page; "
              "`total` counts all matches). Each clip includes provenance, "
              "metadata, eligible formats and the scores of nodes that trained on it.")
    async def data_query(filter: dict[str, Any], ctx: Context) -> dict:
        return await kit.call(ctx, "data_query", {"filter": filter}, lambda c: tools.query(c, filter))

    @mcp.tool(name="data_commit", description="Create an immutable data commit. "
              "datasets: {name: {format, prompt_mode, weight, clips: [clip_id]}}. Set prompt_mode only for "
              "format video_timed_prompts_camera; omit it for every other format.")
    async def data_commit(parent: str | None, datasets: dict[str, Any], message: str,
                          ctx: Context) -> dict:
        return await kit.call(ctx, "data_commit",
                              {"parent": parent, "datasets": datasets, "message": message},
                              lambda c: tools.commit(c, parent, datasets, message))

    @mcp.tool(name="recipe_check", description="Run every recipe-gate check on a recipe "
              "and data commit without consuming an attempt. `recipe` is a flat {tunable key: value} map, "
              "e.g. {\"optimizer.lr\": 1e-4}, with no wrapper key. Returns ok and the failures.")
    async def recipe_check(recipe: dict[str, Any], data_commit: str, ctx: Context) -> dict:
        return await kit.call(ctx, "recipe_check", {"recipe": recipe, "data_commit": data_commit},
                              lambda c: tools.recipe_check(c, recipe, data_commit))
