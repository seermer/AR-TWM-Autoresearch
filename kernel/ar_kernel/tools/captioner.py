"""caption_videos: caption clips with a local video model, as a GPU job.

A job starts `vllm serve` (the `captioner` config) on the node's GPUs unless the last caption job left
it loaded, and sends every clip file to it through the OpenAI-compatible endpoint. The kernel sends the
video; no video bytes pass through the agent or the paid agent model.
"""
from __future__ import annotations

import os
import shutil
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import httpx
from mcp.server.mcpserver import Context
from pydantic import Field

from .context import WORKSPACE, PathError, to_container, to_host
from ..subproc import free_port
from .server import FROM_FILE, ToolError, listed, refuse
from .vllm_server import VllmServer, gpu_memory_mib, serve_command, wait_gpu_release

TOOL = "caption_videos"


def container_path(path: str) -> str:
    """Relative paths resolve against /workspace, like the agent's own working directory."""
    return path if path.startswith("/") else str(WORKSPACE / path)


def clip_host_path(caller, path: str) -> Path:
    host = to_host(caller, container_path(path))
    if not host.is_file():
        raise PathError(f"{path} is not a file")
    return host


def stage_clip(caller, path: str, dst: Path) -> None:
    """Hard-link (or copy) the caller's clip to `dst` without trusting the path between check and use.

    The agent keeps running while the job runs, so any directory on the path can be swapped for
    a link to a host directory after `clip_host_path` checked it. The file is opened with
    O_NOFOLLOW, the open file's real path must still be inside the caller's mounts, and what
    lands at `dst` must be that same inode (or a copy read from that open file)."""
    src = clip_host_path(caller, path)
    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)   # NONBLOCK: a swapped-in FIFO
    try:
        info = os.fstat(fd)
        real = os.readlink(f"/proc/self/fd/{fd}")
        to_container(caller, Path(real))               # PathError if it now lies outside the mounts
        if not stat.S_ISREG(info.st_mode):
            raise PathError(f"{path} is not a file")
        try:
            os.link(real, dst, follow_symlinks=False)
        except OSError:                                # e.g. another filesystem: copy the open file
            with open(os.dup(fd), "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            return
        linked = os.stat(dst, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
            dst.unlink()
            raise PathError(f"{path} changed while it was being staged")
    finally:
        os.close(fd)


class CaptionBackend:
    """JobQueue backend for caption_videos. Args: {"paths": [container paths], "prompt": str}.
    Result: {"clips": {path: {"caption"} | {"error"}}, "load_s"} (load_s is None when the server was
    already loaded). The server stays loaded after a job (`warm`), so the next caption job starts at once;
    the queue calls `release()` before another kind of job, after `keep_warm_s` idle and at phase end."""
    name = tool = TOOL

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib, poll_s: float = 2.0) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory, self.poll_s = registry, recorder, gpu_memory, poll_s
        self.keep_warm_s = float(cfg.get("captioner.keep_warm_s"))
        self.media = self.run_dir / "jobs" / "caption_media"     # the only directory the server may read
        self._server, self._before, self._node = None, None, "run"

    def server_command(self, port: int, media_dir: Path) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "captioner", media_dir)

    @property
    def warm(self) -> bool:
        return self._server is not None

    def _start(self, job, cancel) -> float | None:
        c = self.cfg.get("captioner")
        work = self.run_dir / "jobs" / job.id
        self._before, self._node = self.gpu_memory(self.gpus), job.node
        port = free_port()
        self._server = VllmServer(c["env"], self.server_command(port, self.media), self.gpus, port, cwd=work, log_path=work / "vllm.log", recorder=self.recorder, node=job.node,
                                  phase=TOOL, poll_s=self.poll_s, label="caption server")
        load_s = self._server.start(float(c["startup_timeout_s"]), cancel)
        if load_s is not None:
            self.recorder.event("caption.server_ready", node=job.node, component="tools", job_id=job.id,
                                load_s=load_s, payload={"gpus": self.gpus, "model": c["model"]})
        return load_s

    def release(self, reason: str = "released") -> None:
        """Stop the server and wait for its GPU memory to come back."""
        server, self._server = self._server, None
        if server is None:
            return
        exit_code = server.stop()
        # run_cancellable reports every stop we make as a cancel (-15, subproc.cancelled);
        # this event says why the server stopped.
        self.recorder.event("caption.server_stopped", node=self._node, component="tools", reason=reason,
                            exit_code=exit_code)
        # A cancel may be part of JobQueue.shutdown, whose join deadline a full wait would overrun.
        timeout = float(self.cfg.get("captioner.memory_release_timeout_s"))
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, self._before,
                                           min(timeout, 5.0) if reason == "cancelled" else timeout)
        if released is False:
            self.recorder.event("caption.gpu_not_released", node=self._node, component="tools",
                                payload={"before": self._before, "after": after})

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        media = self.media / job.id
        media.mkdir(parents=True)
        (self.run_dir / "jobs" / job.id).mkdir(parents=True, exist_ok=True)
        clips: dict[str, Path] = {}
        results: dict[str, dict] = {}
        for i, path in enumerate(job.args["paths"]):
            dst = media / f"{i}{Path(path).suffix}"   # a hard link: the server resolves symlinks
            try:
                stage_clip(caller, path, dst)
                clips[path] = dst
            except (PathError, OSError) as exc:
                results[path] = {"error": str(exc)}
        load_s, outcome = None, "failed"
        try:
            if clips and (self._server is None or not self._server.alive):
                self.release("died")
                load_s = self._start(job, cancel)
            if clips and not cancel.is_set():
                base = self._server.base_url
                with httpx.Client(timeout=float(c["clip_timeout_s"]), trust_env=False) as http:
                    # Up to max_num_seqs requests at once, so the server batches them; each clip
                    # still gets its own event, and results keep the submitted order.
                    done_lock, captioned = threading.Lock(), [0]

                    def one(path: str, file: Path) -> None:
                        if cancel.is_set():
                            return
                        t0 = time.monotonic()
                        recorded = self._caption(http, base, file, job.args["prompt"], c)
                        result = {k: v for k, v in recorded.items() if k != "reasoning"}   # the agent gets the caption
                        latency = time.monotonic() - t0
                        with done_lock:
                            results[path] = result
                            captioned[0] += 1
                            self.recorder.event("caption.clip", node=job.node, component="tools", job_id=job.id,
                                                latency_s=latency, ok="caption" in result,
                                                payload={"path": path, **recorded})
                            report({"captioned": captioned[0], "total": len(clips)})

                    with ThreadPoolExecutor(max_workers=max(1, min(int(c["max_num_seqs"]), len(clips)))) as pool:
                        for f in [pool.submit(one, path, file) for path, file in clips.items()]:
                            f.result()
            outcome = "done"
        finally:
            shutil.rmtree(media, ignore_errors=True)
            if cancel.is_set() or outcome != "done":
                self.release("cancelled" if cancel.is_set() else outcome)
        results = {p: results[p] for p in job.args["paths"] if p in results}     # the submitted order
        return {"clips": results, "load_s": load_s}

    def _caption(self, http: httpx.Client, base: str, file: Path, prompt: str, c: dict) -> dict:
        # No max_tokens: the model reasons first, and a cap could cut the caption off.
        body = {"model": "captioner", "temperature": 0.0, "reasoning_effort": c["reasoning_effort"],
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": file.as_uri()}},
                    {"type": "text", "text": prompt}]}]}
        try:
            r = http.post(f"{base}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[-2000:]}"}
        try:
            message = r.json()["choices"][0]["message"]
            text, reasoning = message["content"], message.get("reasoning") or message.get("reasoning_content")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return {"error": f"unexpected response ({type(exc).__name__}): {r.text[-2000:]}"}
        out = {"caption": text.strip()} if text and text.strip() else {"error": "empty caption"}
        return {**out, "reasoning": reasoning} if reasoning else out


def submit(q, caller, paths, prompt: str) -> dict:
    """Validate a caption_videos call and queue its job; returns {"job_id"}."""
    paths = listed(caller, paths, "paths")
    if not paths:
        raise ToolError("paths is empty")
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
    @mcp.tool(name=TOOL, description="Caption video clips with the kernel's local video model, which sees the "
              "whole clip. A GPU job: returns {job_id} at once; collect it with job_wait. Loading the model takes "
              "minutes, then seconds per clip; the model stays loaded for a few minutes after a job, so a caption "
              "job that follows another starts at once. Send every clip of one prompt in one call. The finished "
              "job's result file (job_wait gives its path) holds `clips`: each path mapped to {caption} or "
              "{error}. Captions are text only: for data_ingest, write {\"caption\": \"<text>\"} to a JSON file "
              "under /workspace/staging/ with a script that reads the result file.")
    async def caption_videos(
            paths: Annotated[list[str] | str, Field(description="video files under /workspace (relative paths "
                                                    f"resolve against /workspace), {FROM_FILE}")],
            prompt: Annotated[str, Field(description="the instruction sent with every clip, e.g. 'Write one factual "
                              "caption (1-3 sentences) describing the scene and how the camera moves.'")],
            ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, TOOL, {"paths": paths, "prompt": prompt},
                              lambda c: submit(q, c, paths, prompt))
