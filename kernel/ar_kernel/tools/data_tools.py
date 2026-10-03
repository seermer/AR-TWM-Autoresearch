"""Data tools: video_probe, data_ingest, data_query, data_commit, recipe_check."""
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from pydantic import Field

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
from .gpu_jobs import items_schema
from ..process_digest import _shape
from .server import FROM_FILE, SHOWN, ToolError, listed, publish, refuse

QUERY_LIMIT = 100           # clips per page: a clip record is about 1,200 characters


class DataTools:
    def __init__(self, cfg, run_dir: Path, recorder, gpus: list[int], gpu_lock: threading.Lock,
                 require_free=lambda: None) -> None:
        self.cfg, self.run_dir, self.recorder = cfg, Path(run_dir), recorder
        self.gpus, self.gpu_lock, self.require_free = list(gpus), gpu_lock, require_free
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

    def ingest(self, caller, candidates) -> dict:
        candidates = listed(caller, candidates, "candidates")
        built, bad = [], {}
        for i, c in enumerate(candidates):
            if not isinstance(c, dict):
                bad[i] = "not an object"
                continue
            missing = [key for key in ("video", "caption", "camera_motion", "provenance") if not c.get(key)]
            if missing:
                bad[i] = f"{', '.join(missing)} is required"
                continue
            host = {}
            for key in ("video", "caption", "pose"):
                if c.get(key):
                    if not isinstance(c[key], str):
                        bad.setdefault(i, f"{key} must be the path of a file")      # the first problem of a candidate
                        continue
                    try:
                        host[key] = to_host(caller, c[key])
                    except PathError as exc:
                        bad.setdefault(i, f"{key}: {exc}")
                        continue
                    if not host[key].is_relative_to(caller.staging_host):
                        bad.setdefault(i, f"{key} {c[key]} is not under /workspace/staging: copy or move it there")
                    elif not host[key].is_file():      # container paths only: host paths never reach the agent
                        bad.setdefault(i, f"{key} {c[key]} does not exist (data_ingest moves each staged file "
                                         "into the archive, so an already ingested file is gone)")
            if bad:
                continue
            built.append(Candidate(
                video=host["video"], caption=host["caption"], pose=host.get("pose"),
                camera_motion=c["camera_motion"], provenance=c["provenance"],
                license=c.get("license"), derived_from=list(c.get("derived_from") or [])))
        refuse(caller, "data_ingest", candidates, bad)
        conn = open_db(self.run_dir)
        try:
            ingestor = Ingestor(self.cfg, self.run_dir, conn, self.recorder, self._leakage_checker())
            results = ingestor.ingest(built, node_id=caller.node)
        finally:
            conn.close()
        rows = [{"index": i, "video": c["video"], "accepted": r.accepted, "clip_id": r.clip_id, "formats": r.formats,
                 "warnings": r.warnings, "reasons": r.reasons} for i, (c, r) in enumerate(zip(candidates, results))]
        # the panel's ingest table: one row per candidate, with what the agent staged
        self.recorder.event("ingest.result", node=caller.node, phase=caller.phase, attempt=caller.attempt,
                            component="tools", payload={"rows": [
                                {**row, **{key: c.get(key) for key in ("caption", "pose", "camera_motion")}}
                                for row, c in zip(rows, candidates)]})
        return {"accepted": sum(r["accepted"] for r in rows), "rejected": sum(not r["accepted"] for r in rows),
                "rejected_for": _counted(r for row in rows if not row["accepted"] for r in row["reasons"]),
                "warnings": _counted(w for row in rows for w in row["warnings"]),
                "result_file": publish(caller, f"data_ingest-{uuid.uuid4().hex[:8]}", rows)}

    def query(self, caller, filter: dict) -> dict:
        conn = open_db(self.run_dir)
        try:
            clips = pool_clips(conn)
            usage = scores_by_clip(conn)
        finally:
            conn.close()
        fmt, motion = filter.get("format"), filter.get("camera_motion")
        wanted = set(listed(caller, filter.get("clip_ids"), "clip_ids"))
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
        datasets = {name: {**d, "clips": listed(caller, d.get("clips"), f"{name}.clips")} if isinstance(d, dict) else d
                    for name, d in datasets.items()}
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
                try:
                    self.require_free()
                except RuntimeError as exc:
                    raise ToolError(str(exc)) from exc
                result = Gate(self.cfg, store, self.recorder).check(
                    recipe, data_commit, parent_commit, caller.node, scratch, self.run_dir, self.gpus)
        finally:
            conn.close()
            shutil.rmtree(scratch, ignore_errors=True)
        return {"ok": result.ok, "failures": result.failures}


_VARIES = re.compile(r"'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?")      # quoted text and numbers of one message


def _counted(texts) -> list[dict]:
    """The distinct messages (paths, quoted text and numbers taken out), most frequent first, with one
    example each."""
    shapes: dict[str, list] = {}
    for text in texts:
        shapes.setdefault(_VARIES.sub("#", _shape(text)), []).append(text)
    return [{"count": len(found), "example": found[0]}
            for found in sorted(shapes.values(), key=len, reverse=True)[:SHOWN]]


def _parent_commit(conn, node_id: str) -> str | None:
    nodes = NodeStore(conn)
    try:
        parent_id = nodes.get(node_id)["parent_id"]
        return nodes.get(parent_id)["data_commit"] if parent_id else None
    except KeyError:
        return None


def pool_clips(conn) -> list[dict]:
    """The clips agents may use: all of them except those a quarantined node ingested."""
    hidden = {n["node_id"] for n in NodeStore(conn).all() if n["status"] == "quarantined"}
    return [c for c in ClipStore(conn).all() if c.get("ingested_by") not in hidden]


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


_FORMATS = "video_caption_camera, video_timed_prompts_camera (prompt_mode per_chunk or segment), video_caption_static"
_STAGED = "absolute container path under /workspace/staging/"
Candidates = Annotated[list[dict[str, Any]] | str, items_schema("the clips to ingest, one candidate each", {
    "video": {"type": "string", "description": f"the mp4: {_STAGED}"},
    "caption": {"type": "string", "description": f"the caption JSON file (not the text): {_STAGED}. It holds "
                                                 "{\"caption\": str} and, for timed prompts, \"segments\""},
    "pose": {"type": "string", "description": f"the pose npz with cam_c2w: {_STAGED}. Required for 'moving', "
                                              "forbidden for 'static'"},
    "camera_motion": {"type": "string", "enum": ["moving", "static"],
                      "description": "'static' only when a measurement shows the camera does not move"},
    "provenance": {"type": "object", "description": "where the clip came from: the record hf_download returned, "
                   "the one in a rollout's `candidate`, or for a clip you derived {\"kind\": \"derived\", "
                   "\"from\": [...], \"transform\": \"what you did\"}"},
    "license": {"type": "string", "description": "the source's license, when known"},
    "derived_from": {"type": "array", "items": {"type": "string"},
                     "description": "ids of archive clips this clip was made from"}},
    ["video", "caption", "camera_motion", "provenance"])]


def register_data_tools(mcp, kit, tools: DataTools) -> None:
    @mcp.tool(name="video_probe", description="Frame count, fps, coded size, rotation, pixel "
              "aspect and display aspect of a video under /workspace.")
    async def video_probe(path: Annotated[str, Field(description="a video file under /workspace")],
                          ctx: Context) -> dict:
        return await kit.call(ctx, "video_probe", {"path": path}, lambda c: tools.probe(c, path))

    @mcp.tool(name="data_ingest", description="Ingest staged candidates into the run's clip pool. The formats "
              "a clip must meet are in skill data_formats. Ingest MOVES each staged file into the archive: copy "
              "it first if you still need it. Send every candidate in one call. Returns how many were accepted "
              "and rejected, the rejection reasons and warnings with counts, and `result_file`: a JSON array "
              "under /workspace/staging/results/ with, per candidate, index, video, accepted, clip_id, formats, "
              "warnings and reasons. Take the clip ids from that file with a script. Read the reasons and change "
              "the candidate; a clip the kernel excludes cannot be made acceptable.")
    async def data_ingest(candidates: Candidates, ctx: Context) -> dict:
        return await kit.call(ctx, "data_ingest", {"candidates": candidates},
                              lambda c: tools.ingest(c, candidates))

    @mcp.tool(name="data_query", description="Search the clip pool of the whole run. Every argument narrows the "
              "result; with none, all clips are listed. Each clip has its provenance, metadata, eligible formats "
              "and the scores of the nodes that trained on it. `total` counts all matches; page with `offset`.")
    async def data_query(
            ctx: Context,
            format: Annotated[str | None, Field(description="keep clips eligible for this format, e.g. "
                              "'video_caption_camera' or 'video_timed_prompts_camera:per_chunk'")] = None,
            camera_motion: Annotated[Literal["moving", "static"] | None, Field(description="keep clips with this camera motion")] = None,
            clip_ids: Annotated[list[str] | str | None, Field(description=f"keep only these clip ids, {FROM_FILE}")] = None,
            ingested_by: Annotated[str | None, Field(description="keep clips this node ingested, e.g. the id of the node being built")] = None,
            limit: Annotated[int, Field(description="clips per page")] = QUERY_LIMIT,
            offset: Annotated[int, Field(description="skip this many matches")] = 0) -> dict:
        given = dict(format=format, camera_motion=camera_motion, clip_ids=clip_ids, ingested_by=ingested_by,
                     limit=limit, offset=offset)
        filter = {key: value for key, value in given.items() if value is not None}
        return await kit.call(ctx, "data_query", filter, lambda c: tools.query(c, filter))

    @mcp.tool(name="data_commit", description="Create an immutable data commit: the training set of this node.")
    async def data_commit(
            parent: Annotated[str | None, Field(description="the commit this one follows (the parent node's data commit), or null")],
            datasets: Annotated[dict[str, Any], Field(description="{name: {format, prompt_mode, weight, clips}}. "
                                f"clips is a list of clip ids, {FROM_FILE}. format is one of: {_FORMATS}. Set prompt_mode only for "
                                "video_timed_prompts_camera; omit it otherwise. weight is the dataset's sampling "
                                "weight. Each dataset needs at least as many clips as training GPUs")],
            message: Annotated[str, Field(description="what this commit contains")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "data_commit",
                              {"parent": parent, "datasets": datasets, "message": message},
                              lambda c: tools.commit(c, parent, datasets, message))

    @mcp.tool(name="recipe_check", description="Run every pre-training check on a recipe and a data commit "
              "without using up an attempt. Returns ok and the failures. Among the checks: steps_per_epoch = "
              "floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps) must be at least 1, and "
              "optimizer.epochs * steps_per_epoch at least optimizer.max_steps, where epoch_windows is the "
              "largest, over the commit's datasets, of ceil(clips / (weight / total_weight)).")
    async def recipe_check(
            recipe: Annotated[dict[str, Any], Field(description="a flat {tunable key: value} map, e.g. "
                              "{\"optimizer.lr\": 1e-4}, with no wrapper key")],
            data_commit: Annotated[str, Field(description="the commit id data_commit returned")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "recipe_check", {"recipe": recipe, "data_commit": data_commit},
                              lambda c: tools.recipe_check(c, recipe, data_commit))
