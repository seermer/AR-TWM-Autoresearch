"""caption_videos: caption clips with a local video model, as a GPU job.

Each job runs a bridge (bridges/caption_vllm.py) that starts `vllm serve` (the `captioner` config) on the node's
GPUs, sends every clip file to it through the OpenAI-compatible endpoint, and always stops it again. No video bytes
pass through the agent or the paid agent model.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import Context
from pydantic import Field

from .context import WORKSPACE, PathError, open_file, to_host
from .. import fs, telemetry
from ..faults import disk_fault
from ..subproc import free_port
from .jobs import job_dir
from .server import FROM_FILE, ToolError, listed, refuse
from .vllm_server import gpu_memory_mib, serve_command

TOOL = "caption_videos"
BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "caption_vllm.py"
SYSTEM = (
    "You are the video reader of a research pipeline. Each request gives one video clip and says what to "
    "say about it. Answer that request about this clip, in plain text: no preamble, no markdown, no mention "
    "of these instructions. Your text is used as it stands, so write exactly what the request asks for.")


def container_path(path: str) -> str:
    """Relative paths resolve against /workspace, like the agent's own working directory."""
    return path if path.startswith("/") else str(WORKSPACE / path)


def clip_host_path(caller, path: str) -> Path:
    host = to_host(caller, container_path(path))
    if not host.is_file():
        raise PathError(f"{path} is not a file")
    return host


def input_error(exc: Exception) -> str:
    """An item's staging error for its result; a disk fault raises."""
    if disk_fault(exc):
        raise exc
    return f"input: {exc}"


def stage_clip(caller, path: str, dst: Path) -> None:
    """Hard-link (or copy) the caller's clip to `dst` without trusting the path between check and use.

    The agent keeps running while the job runs, so any directory on the path can be swapped for
    a link to a host directory after `clip_host_path` checked it. The file is opened with
    O_NOFOLLOW, the open file's real path must still be inside the caller's mounts, and what
    lands at `dst` must be that same inode (or a copy read from that open file). The bytes are recorded
    from that open file once staged."""
    fd = open_file(caller, clip_host_path(caller, path))
    try:
        info = os.fstat(fd)
        real = os.readlink(f"/proc/self/fd/{fd}")
        try:
            fs.link(real, dst, follow_symlinks=False)
        except OSError as exc:                         # e.g. another filesystem: copy the open file
            telemetry.error(exc, "a copy instead")
            with open(os.dup(fd), "rb") as fin:
                fs.write_bytes(dst, fin.read())
        else:
            linked = os.stat(dst, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                fs.remove(dst)
                raise PathError(f"{path} changed while it was being staged")
        with open(os.dup(fd), "rb") as fin:
            fin.seek(0)
            telemetry.keep(path, fin, "read")
    finally:
        os.close(fd)


class CaptionBackend:
    """JobQueue backend for caption_videos. Args: {"paths": [container paths], "prompt": str}.
    Result: {"clips": {path: {"caption"} | {"error"}}, "load_s", "gpu_memory_mib", "gpu_memory_released"}."""
    name = tool = TOOL

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.gpu_memory = registry, gpu_memory
        self.block = cfg.get("captioner") or {}
        self.max_items = self.block.get("max_items")

    def server_command(self, port: int, media_dir: Path) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "captioner", media_dir)

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = job_dir(self.run_dir, job)
        media = work / "media"                    # the only directory the server may read
        fs.mkdir(media)
        clips: dict[str, Path] = {}
        results: dict[str, dict] = {}
        for i, path in enumerate(job.args["paths"]):
            dst = media / f"{i}{Path(path).suffix}"   # a hard link: the server resolves symlinks
            try:
                stage_clip(caller, path, dst)
                clips[path] = dst
            except (PathError, OSError) as exc:
                telemetry.error(exc, "the item's input error")
                results[path] = {"error": input_error(exc)}

        if not clips:
            return {"clips": results, "load_s": None, "gpu_memory_mib": None, "gpu_memory_released": None}
        from .gpu_jobs import run_bridge           # imports this module
        port, peak = free_port(), {}

        def sample() -> None:
            for g, used in (self.gpu_memory(self.gpus) or {}).items():
                peak[g] = max(peak.get(g, 0), used)

        paths = list(clips)
        spec = {"bridge": BRIDGE, "env": c["env"], "job": {
            "serve": {"argv": self.server_command(port, media), "env": {"HF_HUB_OFFLINE": "1"}, "port": port},
            "system": SYSTEM, "prompt": job.args["prompt"], **{k: c[k] for k in (
                "concurrent_requests", "clip_timeout_s", "reasoning_effort")}}}

        def row(m: dict) -> dict:           # the agent gets the caption; the reasoning stays in the messages
            return {"caption": m["info"]["caption"]} if m["ok"] else {"error": m["error"][:2000]}   # whole in telemetry

        try:
            ran = run_bridge(self, job, work, spec, [{"index": i, "path": str(clips[p])} for i, p in enumerate(paths)],
                             cancel, report, row, done=len(results), poll=sample)
        finally:
            fs.rmtree(media, ignore_errors="the job's own links")
        for i, r in ran["rows"].items():
            results[paths[i]] = {"error": r["error"]} if "index" in r else r
        results = {p: results[p] for p in job.args["paths"] if p in results}     # the submitted order
        ready = ran["ready"]
        return {"clips": results, "load_s": ready["info"]["load_s"] if ready else None,
                "gpu_memory_mib": {**ran["gpu_memory_mib"], "peak": peak or None},
                "gpu_memory_released": ran["gpu_memory_released"]}


def submit(q, caller, paths, prompt: str) -> dict:
    """Validate a caption_videos call and queue its job; returns {"job_id"}."""
    paths = listed(caller, paths, "paths")
    if not paths:
        raise ToolError("paths is empty")
    cap = getattr(q.backends.get(TOOL), "max_items", None)
    if cap and len(paths) > cap:
        raise ToolError(f"at most {cap} items per job: got {len(paths)}; send the rest in another job")
    if not prompt.strip():
        raise ToolError("prompt is empty")
    bad = {}
    for n, path in enumerate(paths):
        try:
            clip_host_path(caller, str(path))
        except PathError as exc:
            telemetry.error(exc, "a bad item")
            bad[n] = str(exc)
    refuse(caller, TOOL, paths, bad)
    return {"job_id": q.submit(caller, TOOL, {"paths": [container_path(str(p)) for p in paths], "prompt": prompt})}


def register_caption_tool(mcp, kit, q) -> None:
    """caption_videos queues a job for the queue's `caption_videos` backend (the real
    CaptionBackend, or a fake in smoke runs)."""
    cap = getattr(q.backends.get(TOOL), "max_items", None)

    @mcp.tool(name=TOOL, description="Caption video clips with the kernel's local video model, which sees frames "
              "sampled across the whole clip. A GPU job: returns {job_id} at once; collect it with job_wait. "
              "Every job loads the model again, so send every clip of one prompt in one call. The finished "
              "job's result file (job_wait gives its path) holds `clips`: each path mapped to {caption} or "
              "{error}. What it can be used for: knowledge file caption_videos."
              + (f" At most {cap} clips per job." if cap else ""))
    async def caption_videos(
            paths: Annotated[list[str] | str, Field(description="video files under /workspace (relative paths "
                                                    f"resolve against /workspace), {FROM_FILE}")],
            prompt: Annotated[str, Field(description="the instruction sent with every clip, e.g. 'Describe the scene "
                              "in one sentence.'")],
            ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, TOOL, {"paths": paths, "prompt": prompt},
                              lambda c: submit(q, c, paths, prompt))
