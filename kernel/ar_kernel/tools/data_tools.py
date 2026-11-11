"""Data tools: video_probe, data_ingest, data_query, data_fetch, data_commit, recipe_check."""
from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, TypedDict

from mcp.server.mcpserver import Context
from pydantic import Field

from .. import fs, telemetry
from ..archive.activity import add_activity
from ..archive.blobs import KINDS, BlobStore
from ..archive.clips import ClipStore, hidden
from ..archive.commits import BadClips, CommitError, CommitStore, limit_text
from ..archive.db import open_db
from ..archive.nodes import NodeStore
from ..data.annotation import DERIVED_KINDS
from ..data.ingest import Candidate, Ingestor
from ..data.leakage import LeakageChecker
from ..data.probe import readable_video
from ..faults import disk_fault
from ..train.gate import Gate
from .context import STAGING, WORKSPACE, PathError, open_file, to_host
from .gpu_jobs import items_schema
from ..process_digest import _shape
from .server import FROM_FILE, SHOWN, ToolError, listed, publish, refuse


def clip_id_list(caller, value) -> list[str]:
    ids = listed(caller, value, "clip_ids")
    if not all(isinstance(i, str) for i in ids):
        raise ToolError("clip_ids: every item must be a clip id, a string. This list holds other things, e.g. "
                        "whole clip records: pass each record's `clip_id` instead")
    return ids


class DataTools:
    def __init__(self, cfg, run_dir: Path, gpus: list[int], gpu_lock: threading.Lock,
                 require_free=lambda caller: None) -> None:
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.gpus, self.gpu_lock, self.require_free = list(gpus), gpu_lock, require_free
        self._leakage: LeakageChecker | None = None
        self._leakage_lock = threading.Lock()
        self.new_clips_per_commit = int(cfg.get("improve_recipe.new_clips_per_commit"))
        self._passed: dict[str, set] = {}          # token -> (recipe, data commit) recipe_check passed

    def _leakage_checker(self) -> LeakageChecker:
        """Lazily build one shared, read-only held-out image index under the initialization lock."""
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
        named = path if path.startswith("/") else str(WORKSPACE / path)
        host = self._host(caller, named)
        try:
            fd = open_file(caller, host)
        except PathError as exc:
            raise ToolError(str(exc)) from exc
        except OSError as exc:
            if disk_fault(exc):
                raise
            raise ToolError(f"{path} cannot be read: {exc.strerror}") from exc
        with open(fd, "rb") as handle:
            telemetry.keep(named, handle, "read")
        try:
            info = readable_video(host)
        except ValueError as exc:
            raise ToolError(f"{path} is not a readable video: {exc}") from exc
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
            missing = [key for key in ("video", "annotation", "pose", "derived_from") if not c.get(key)]
            if missing:
                bad[i] = f"{', '.join(missing)} is required"
                continue
            host = {}
            for key in ("video", "annotation", "pose"):
                if not isinstance(c[key], str):
                    bad.setdefault(i, f"{key} must be the path of a file")      # the first problem of a candidate
                    continue
                try:
                    host[key] = to_host(caller, c[key])
                except PathError as exc:
                    telemetry.error(exc, "a bad item")
                    bad.setdefault(i, f"{key}: {exc}")
                    continue
                if not host[key].is_relative_to(caller.staging_host):
                    bad.setdefault(i, f"{key} {c[key]} is not under /workspace/staging: copy or move it there")
                elif not host[key].is_file():      # container paths only: host paths never reach the agent
                    bad.setdefault(i, f"{key} {c[key]} does not exist")
            if bad:
                continue
            built.append(Candidate(video=host["video"], annotation=host["annotation"], pose=host["pose"],
                                   derived_from=c["derived_from"], named={k: c[k] for k in host}))
        refuse(caller, "data_ingest", candidates, bad)
        with closing(open_db(self.run_dir)) as conn:
            ingestor = Ingestor(self.cfg, self.run_dir, conn, self._leakage_checker())
            results = ingestor.ingest(built, node_id=caller.node, attempt=caller.attempt)
        if results and all(r.checker_failed for r in results):     # a counted tool error, not the agent's
            raise RuntimeError(f"the clip checker failed on every candidate: {results[0].reasons[0]}")
        rows = [{"index": i, "video": c["video"], "accepted": r.accepted, "clip_id": r.clip_id,
                 "warnings": r.warnings, "reasons": r.reasons} for i, (c, r) in enumerate(zip(candidates, results))]
        return {"accepted": sum(r["accepted"] for r in rows), "rejected": sum(not r["accepted"] for r in rows),
                "rejected_for": _counted(r for row in rows if not row["accepted"] for r in row["reasons"]),
                "warnings": _counted(w for row in rows for w in row["warnings"]),
                "result_file": publish(caller, f"data_ingest-{uuid.uuid4().hex[:8]}", rows)}

    def query(self, caller, filter: dict) -> dict:
        with closing(open_db(self.run_dir)) as conn:
            clips = pool_clips(conn)
            usage = scores_by_clip(conn)
        wanted = set(clip_id_list(caller, filter.get("clip_ids")))
        in_commit = None
        if filter.get("data_commit"):
            in_commit = {}                                  # clip id -> the commit's datasets that hold it
            for name, d in json.loads(self.usable(filter["data_commit"])["manifest"])["datasets"].items():
                for clip_id in d["clips"]:
                    in_commit.setdefault(clip_id, []).append(name)
        out = []
        for clip in clips:
            if in_commit is not None and clip["clip_id"] not in in_commit:
                continue
            if wanted and clip["clip_id"] not in wanted:
                continue
            if filter.get("ingested_by") and clip.get("ingested_by") != filter["ingested_by"]:
                continue
            out.append({**clip_record(clip, usage), **({"datasets": in_commit[clip["clip_id"]]} if in_commit else {})})
        return {"total": len(out), "clips": out[:SHOWN],
                "result_file": publish(caller, f"data_query-{uuid.uuid4().hex[:8]}", out)}

    def fetch(self, caller, clip_ids) -> dict:
        wanted = list(dict.fromkeys(clip_id_list(caller, clip_ids)))
        with closing(open_db(self.run_dir)) as conn:
            pool = {clip["clip_id"]: clip for clip in pool_clips(conn)}
            blobs = BlobStore(self.run_dir, conn)
            for clip in (pool[i] for i in wanted if i in pool):
                copies = [(blobs.path(clip[f"{kind}_digest"], kind), f"{kind}s/{clip['clip_id']}{suffix}")
                          for kind, suffix in KINDS.items()]
                for blob, rel in copies:            # a copy: a hard link would let the agent change the archive's file
                    data = Path(blob).read_bytes()
                    fs.place_bytes(data, caller.staging_host, f"pool_clips/{rel}")
                    telemetry.keep(str(STAGING / "pool_clips" / rel), data, "handed")
        return {"folder": str(STAGING / "pool_clips"), "copied": sum(i in pool for i in wanted),
                "not_in_pool": [i for i in wanted if i not in pool]}

    def commit(self, caller, datasets: dict, message: str, include: list[str] = ()) -> dict:
        datasets = {name: {**d, "clips": listed(caller, d.get("clips"), f"{name}.clips")} if isinstance(d, dict) else d
                    for name, d in datasets.items()}
        inherited: dict[str, dict] = {}
        for commit_id in include:
            for name, d in json.loads(self.usable(commit_id)["manifest"])["datasets"].items():
                if name in inherited and inherited[name] != d and name not in datasets:
                    raise ToolError(f"dataset {name} is in more than one included commit, with different content: "
                                    "pass it in `datasets` as it should be")
                inherited[name] = d
        datasets = {**inherited, **datasets}
        with closing(open_db(self.run_dir)) as conn:
            store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))
            try:
                commit_id, new = store.commit(datasets, message, node_id=caller.node, attempt=caller.attempt,
                                              cap=self.new_clips_per_commit)
            except BadClips as exc:
                telemetry.error(exc, "a refusal naming each bad clip")
                refuse(caller, f"data_commit-{exc.dataset}", exc.clips, exc.bad)
            except CommitError as exc:
                raise ToolError(str(exc)) from exc
            stats = dataset_stats(store.manifest(commit_id), ClipStore(conn).all())
        return {"commit_id": commit_id, "datasets": stats, "new_clips": new}

    def usable(self, commit_id: str) -> dict:
        """The data commit's row; ToolError when it is unknown, obsolete or failed."""
        with closing(open_db(self.run_dir)) as conn:
            try:
                return CommitStore(conn, None, None).usable(commit_id)
            except KeyError:
                raise ToolError(f"unknown data commit {commit_id!r}") from None
            except CommitError as exc:
                raise ToolError(str(exc)) from None

    def forget(self, token: str) -> None:
        self._passed.pop(token, None)

    def recipe_check(self, caller, recipe: dict, data_commit: str) -> dict:
        self.usable(data_commit)
        failures = self._gate(caller, recipe, data_commit)
        if not failures:
            self._passed.setdefault(caller.token, set()).add((json.dumps(recipe, sort_keys=True), data_commit))
        return {"ok": not failures, "failures": failures}

    def snapshot_gate(self, caller, recipe: dict, data_commit: str) -> list[str]:
        """The gate's findings for a snapshot; none when recipe_check passed the same recipe and data commit
        in this phase attempt."""
        if (json.dumps(recipe, sort_keys=True), data_commit) in self._passed.get(caller.token, ()):
            return []
        return self._gate(caller, recipe, data_commit)

    def _gate(self, caller, recipe: dict, data_commit: str) -> list[str]:
        scratch = caller.workspace_host.parent / "recipe_check" / uuid.uuid4().hex
        try:
            with closing(open_db(self.run_dir)) as conn:
                store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))     # callers checked usable
                parent_commit = _parent_commit(conn, caller.node)
                if not self.gpu_lock.acquire(blocking=False):       # the describe step is a GPU job
                    raise ToolError("a GPU job is running and the check needs the GPUs: "
                                    "call again once your jobs have finished (job_wait).")
                try:
                    self.require_free(caller)
                    failures = Gate(self.cfg, store).check(
                        recipe, data_commit, parent_commit, caller.node, scratch, self.run_dir, self.gpus)
                finally:
                    self.gpu_lock.release()
        finally:
            fs.rmtree(scratch, ignore_errors="a check's scratch folder")
        telemetry.decision("recipe.gate", {"recipe": recipe, "data_commit": data_commit, "parent_commit": parent_commit},
                           ["pass", "refuse"], "refuse" if failures else "pass", "\n".join(failures) or "no findings")
        add_activity(self.run_dir, node=caller.node, phase=caller.phase, attempt=caller.attempt, kind="gate",
                     ok=not failures)
        return failures


class Dataset(TypedDict, total=False):
    """Typed so that a number reaches the kernel as a number: values in an untyped map arrive as strings."""
    weight: float
    clips: list[str] | str


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
    except KeyError as exc:
        telemetry.error(exc, "no parent data commit")
        return None


def pool_clips(conn) -> list[dict]:
    """The clips agents may use: every clip not marked hidden."""
    return [c for c in ClipStore(conn).all() if not hidden(c["ingested_by"])]


def clip_source(clip: dict) -> str:
    """What a clip was made from, as its `derived_from` says: `<kind>:<name>`."""
    return f"{clip['derived_from']['kind']}:{clip['derived_from']['name']}"


def dataset_stats(manifest: dict, clips: list[dict]) -> dict[str, dict]:
    """Per-dataset summary of a commit manifest: weight, clip count and clip count per source
    (data_commit's result row and the context's lineage entries)."""
    by_id = {c["clip_id"]: c for c in clips}
    out = {}
    for name, d in manifest["datasets"].items():
        sources: dict[str, int] = {}
        for clip_id in d["clips"]:
            source = clip_source(by_id[clip_id])
            sources[source] = sources.get(source, 0) + 1
        out[name] = {"weight": d["weight"], "clips": len(d["clips"]), "sources": sources}
    return out


def clip_record(clip: dict, usage: dict[str, list[float]]) -> dict:
    """Shape one archive clip row for data_query's result (the context
    bundle reuses this instead of re-deriving it from ClipStore rows)."""
    return {"clip_id": clip["clip_id"], "metadata": clip["metadata"], "warnings": clip["warnings"],
            "derived_from": clip["derived_from"], "ingested_by": clip.get("ingested_by"),
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


_STAGED = "absolute container path under /workspace/staging/"
Candidates = Annotated[list[dict[str, Any]] | str, items_schema("the clips to ingest, one candidate each", {
    "video": {"type": "string", "description": f"the mp4, 960x544: {_STAGED}"},
    "annotation": {"type": "string", "description": f"the annotation JSON file (not the text): {_STAGED}. Its "
                                                    "shape is in knowledge file data_formats"},
    "pose": {"type": "string", "description": f"the pose npz with cam_c2w, as annotate_camera writes it: {_STAGED}"},
    "derived_from": {"type": "object", "description": "what the clip was made from, for a person to read",
                     "properties": {"kind": {"type": "string", "enum": list(DERIVED_KINDS)},
                                    "name": {"type": "string", "description": "the model, the dataset or the "
                                                                              "method, by name"}},
                     "required": ["kind", "name"]}},
    ["video", "annotation", "pose", "derived_from"])]


def register_data_tools(mcp, kit, tools: DataTools) -> None:
    cap = getattr(tools, "new_clips_per_commit", None)
    limit = f" {limit_text(cap)}" if cap is not None else ""
    new = ", and `new_clips`: how many of its clips are new" if cap is not None else ""

    @mcp.tool(name="video_probe", description="Frame count, fps, coded size, rotation, pixel "
              "aspect and display aspect of a video under /workspace.")
    async def video_probe(path: Annotated[str, Field(description="a video file under /workspace (a relative path "
                                                             "resolves against /workspace)")],
                          ctx: Context) -> dict:
        return await kit.call(ctx, "video_probe", {"path": path}, lambda c: tools.probe(c, path))

    @mcp.tool(name="data_ingest", description="Ingest staged candidates into the run's clip pool. What "
              "a clip must meet is in knowledge file data_formats. Ingest copies each staged file into the archive and "
              "leaves yours where it is. Send every candidate in one call. Returns how many were accepted "
              "and rejected, the rejection reasons and warnings with counts, and `result_file`: a JSON array "
              "under /workspace/staging/results/ with, per candidate, index, video, accepted, clip_id, "
              "warnings and reasons. Take the clip ids from that file with a script. Read the reasons and change "
              "the candidate; a clip the kernel excludes cannot be made acceptable.")
    async def data_ingest(candidates: Candidates, ctx: Context) -> dict:
        return await kit.call(ctx, "data_ingest", {"candidates": candidates},
                              lambda c: tools.ingest(c, candidates))

    @mcp.tool(name="data_query", description="Search the clip pool of the whole run. Every argument narrows the "
              "result; with none, all clips are listed. Each clip has what it was derived from, its metadata "
              "and the scores of the nodes that trained on it. The reply holds `total`, the first "
              f"{SHOWN} clips, and `result_file`: a JSON file with every match.")
    async def data_query(
            ctx: Context,
            clip_ids: Annotated[list[str] | str | None, Field(description=f"keep only these clip ids, {FROM_FILE}")] = None,
            ingested_by: Annotated[str | None, Field(description="keep clips this node ingested, e.g. the id of the node being built")] = None,
            data_commit: Annotated[str | None, Field(description="keep the clips of this data commit; each clip then has `datasets`, the commit's datasets that hold it")] = None) -> dict:
        given = dict(clip_ids=clip_ids, ingested_by=ingested_by, data_commit=data_commit)
        filter = {key: value for key, value in given.items() if value is not None}
        return await kit.call(ctx, "data_query", filter, lambda c: tools.query(c, filter))

    @mcp.tool(name="data_fetch", description="Copy the files of clips in the pool to /workspace/staging/pool_clips/, "
              "to look at them: any clip data_query lists, also one that no data commit holds. Each clip gives "
              "videos/<clip_id>.mp4, annotations/<clip_id>.json (the annotation it was ingested with), "
              "captions/<clip_id>.json (the training text made from it, one prompt per turn) and "
              "poses/<clip_id>.npz. Returns the folder, the number of clips copied and "
              "the ids that are not in the pool.")
    async def data_fetch(
            clip_ids: Annotated[list[str] | str, Field(description=f"the clips to copy: a list of clip ids, {FROM_FILE}")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "data_fetch", {"clip_ids": clip_ids}, lambda c: tools.fetch(c, clip_ids))

    @mcp.tool(name="data_commit", description="Create an immutable data commit: the tentative whole training set of "
              "this node. It holds the datasets of the commits in `include` plus the ones in `datasets`. Every node's "
              "improve_recipe phase runs in subphases. When a subphase ends, the data commit of the snapshot its "
              f"result names is kept, and the subphase's other data commits can no longer be used.{limit} Returns the "
              f"commit id, each dataset's weight and clip counts{new}.")
    async def data_commit(
            datasets: Annotated[dict[str, Dataset], Field(description="{name: {weight, clips}}. "
                                f"clips is a list of clip ids, {FROM_FILE}. weight is the dataset's sampling "
                                "weight. Each dataset needs at least as many clips as training GPUs")],
            message: Annotated[str, Field(description="what this commit contains")],
            ctx: Context,
            include: Annotated[list[str] | None, Field(description="kept data commits of earlier nodes (the parent's, or "
                               "any other finished node's) or data commits of this node that can still be used, whose "
                               "datasets this commit "
                               "takes over as they are: name, weight and clips. A dataset in `datasets` with the same "
                               "name replaces the inherited one whole; "
                               "to stop training on one, pass it with weight 0")] = None) -> dict:
        return await kit.call(ctx, "data_commit",
                              {"datasets": datasets, "message": message, "include": include or []},
                              lambda c: tools.commit(c, datasets, message, include or []))

    @mcp.tool(name="recipe_check", description="Run every pre-training check on a recipe and a data commit. "
              "It uses the GPUs, so it is refused while a GPU job "
              "runs. Returns ok and the failures. Among the checks: the commit must have "
              "enough clips for the step count. One pass over the data draws, for the dataset that needs the "
              "most, ceil(clips / (weight / total_weight)) clips; that many, divided by the GPU count and then "
              "by optimizer.grad_accum_steps (each rounded down), is the optimizer steps of one pass. It must be "
              "at least 1, and optimizer.epochs passes must give at least optimizer.max_steps steps.")
    async def recipe_check(
            recipe: Annotated[dict[str, int | float], Field(description="a flat {tunable key: value} map, e.g. "
                              "{\"optimizer.lr\": 1e-4}, with no wrapper key")],
            data_commit: Annotated[str, Field(description="the commit id data_commit returned")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "recipe_check", {"recipe": recipe, "data_commit": data_commit},
                              lambda c: tools.recipe_check(c, recipe, data_commit))
