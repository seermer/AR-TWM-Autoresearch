"""caption_videos: caption clips with a local video model, as a GPU job.

Each job starts `vllm serve` (the `captioner` config) on the node's GPUs, sends every clip
file to it through the OpenAI-compatible endpoint, and always stops it again. The kernel
sends the video; no video bytes pass through the agent or the paid agent model.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import openai
from mcp.server.mcpserver import Context
from pydantic import Field

from .context import WORKSPACE, PathError, open_file, to_host
from ..faults import disk_fault
from ..telemetry import files
from ..subproc import free_port
from .jobs import job_dir
from .server import FROM_FILE, ToolError, listed, refuse
from .vllm_server import VllmServer, gpu_memory_mib, local_client, serve_command, wait_gpu_release

TOOL = "caption_videos"
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
            os.link(real, dst, follow_symlinks=False)
        except OSError:                                # e.g. another filesystem: copy the open file
            with open(os.dup(fd), "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            linked = os.stat(dst, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                dst.unlink()
                raise PathError(f"{path} changed while it was being staged")
        os.lseek(fd, 0, os.SEEK_SET)                   # a copy's dup moved the shared offset
        with open(os.dup(fd), "rb") as fin:
            files.record(fin, direction="in", path=container_path(path))
    finally:
        os.close(fd)


class CaptionBackend:
    """JobQueue backend for caption_videos. Args: {"paths": [container paths], "prompt": str}.
    Result: {"clips": {path: {"caption"} | {"error"}}, "load_s", "gpu_memory_mib", "gpu_memory_released"}."""
    name = tool = TOOL

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib, poll_s: float = 2.0) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory, self.poll_s = registry, recorder, gpu_memory, poll_s
        self.max_items = (cfg.get("captioner") or {}).get("max_items")

    def server_command(self, port: int, media_dir: Path) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "captioner", media_dir)

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = job_dir(self.run_dir, job)
        media = work / "media"                    # the only directory the server may read
        media.mkdir(parents=True)
        clips: dict[str, Path] = {}
        results: dict[str, dict] = {}
        for i, path in enumerate(job.args["paths"]):
            dst = media / f"{i}{Path(path).suffix}"   # a hard link: the server resolves symlinks
            try:
                stage_clip(caller, path, dst)
                clips[path] = dst
            except (PathError, OSError) as exc:
                results[path] = {"error": input_error(exc)}

        if not clips:
            return {"clips": results, "load_s": None, "gpu_memory_mib": None, "gpu_memory_released": None}
        port = free_port()
        before = self.gpu_memory(self.gpus)
        peak = dict(before or {})

        def sample() -> None:
            now = self.gpu_memory(self.gpus) or {}
            for g, used in now.items():
                peak[g] = max(peak.get(g, 0), used)

        server = VllmServer(c["env"], self.server_command(port, media), self.gpus, port, cwd=work,
                            log_path=work / "vllm.log", recorder=self.recorder, node=job.node,
                            phase=TOOL, poll_s=self.poll_s, label="caption server")
        base, load_s, outcome = server.base_url, None, "failed"
        try:
            load_s = server.start(cancel, on_poll=sample)
            with local_client(base, float(c["clip_timeout_s"]), server.api_key) as client:
                if load_s is not None:
                    self.recorder.event("caption.server_ready", node=job.node, component="tools",
                                        job_id=job.id, load_s=load_s,
                                        payload={"gpus": self.gpus, "model": c["model"]})
                # Up to concurrent_requests requests at once, so the server batches them; each clip
                # still gets its own event, and results keep the submitted order.
                done_lock, captioned = threading.Lock(), [0]

                def one(path: str, file: Path) -> None:
                    if cancel.is_set():
                        return
                    t0 = time.monotonic()
                    recorded = self._caption(client, file, job.args["prompt"], c)
                    result = {k: v for k, v in recorded.items() if k != "reasoning"}   # the agent gets the caption
                    latency = time.monotonic() - t0
                    with done_lock:
                        results[path] = result
                        sample()
                        captioned[0] += 1
                        self.recorder.event("caption.clip", node=job.node, component="tools", job_id=job.id,
                                            latency_s=latency, ok="caption" in result,
                                            payload={"path": path, **recorded})
                        report({"captioned": captioned[0], "total": len(clips)})

                with ThreadPoolExecutor(max_workers=max(1, min(int(c["concurrent_requests"]), len(clips)))) as pool:
                    for f in [pool.submit(one, path, file) for path, file in clips.items()]:
                        f.result()
            outcome = "done"
        finally:
            exit_code = server.stop()
            shutil.rmtree(media, ignore_errors=True)
            # run_cancellable reports every stop we make as a cancel (-15, subproc.cancelled);
            # this event says why the server stopped.
            self.recorder.event("caption.server_stopped", node=job.node, component="tools", job_id=job.id,
                                reason="cancelled" if cancel.is_set() else outcome,
                                exit_code=exit_code)
        # A cancelled job may be part of JobQueue.shutdown, whose join deadline a full wait would
        # overrun.
        timeout = float(c["memory_release_timeout_s"])
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, before,
                                           min(timeout, 5.0) if cancel.is_set() else timeout)
        if released is False:
            self.recorder.event("caption.gpu_not_released", node=job.node, component="tools",
                                job_id=job.id, payload={"before": before, "after": after})
        results = {p: results[p] for p in job.args["paths"] if p in results}     # the submitted order
        return {"clips": results, "load_s": load_s,
                "gpu_memory_mib": {"before": before, "peak": peak or None, "after": after},
                "gpu_memory_released": released}

    def _caption(self, client: openai.OpenAI, file: Path, prompt: str, c: dict) -> dict:
        # No max_tokens: the model reasons first, and a cap could cut the caption off.
        try:
            done = client.chat.completions.create(
                model="captioner", reasoning_effort=c["reasoning_effort"],
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": [{"type": "video_url", "video_url": {"url": file.as_uri()}},
                                                       {"type": "text", "text": prompt}]}])
        except openai.OpenAIError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        message = done.choices[0].message
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        text = (message.content or "").strip()
        out = {"caption": text} if text else {"error": "empty caption"}
        return {**out, "reasoning": reasoning} if reasoning else out


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
