"""caption_videos (spec 10): caption clips with a local video model, as a GPU job.

A job starts `vllm serve` (the `captioner` config) on the node's GPUs unless one is already
running, and sends every clip file to it through the OpenAI-compatible endpoint. The server then
stays up for the next caption call, and is stopped as soon as anything else wants the GPUs: the
run's GpuLock calls `CaptionBackend.release()` whenever another holder takes the lock, and a
failed or cancelled job stops it at once. The kernel sends the video; no video bytes pass through
the agent or the paid agent model.
"""
from __future__ import annotations

import os
import shutil
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import Context

from .context import WORKSPACE, PathError, to_container, to_host
from ..subproc import free_port
from .server import ToolError
from .vllm_server import (VllmServer, gpu_memory_mib, serve_command,  # noqa: F401 -- re-exported
                          wait_gpu_release, _tail)

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
    Result: {"clips": {path: {"caption"} | {"error"}}, "load_s", "gpu_memory_mib", "gpu_memory_released"}."""
    name = tool = TOOL

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib, poll_s: float = 2.0) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory, self.poll_s = registry, recorder, gpu_memory, poll_s
        self.keeps_gpu_warm = True            # JobQueue takes the GPU lock without releasing this server
        self._server: VllmServer | None = None
        self._before: dict | None = None      # GPU memory before the server started
        self._node = "run"                    # the node whose job started the server

    def server_command(self, port: int, media_dir: Path) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "captioner", media_dir)

    def release(self, reason: str = "released", node: str | None = None, job_id: str | None = None):
        """Stop the warm server, if any, and wait for its GPU memory. Called with the GPU lock held
        (by the lock itself, or by a job that is ending). Returns (memory after, released)."""
        server, self._server = self._server, None
        if server is None:
            return None, None
        node = node or self._node
        exit_code = server.stop()
        # run_cancellable reports every stop we make as a cancel (-15, subproc.cancelled);
        # this event says why the server stopped.
        self.recorder.event("caption.server_stopped", node=node, component="tools", job_id=job_id,
                            reason=reason, exit_code=exit_code)
        # A cancelled job may be part of JobQueue.shutdown, whose join deadline a full wait would
        # overrun; the check that GPU memory is free before each phase is the real guard.
        timeout = float(self.cfg.get("captioner.memory_release_timeout_s"))
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, self._before,
                                           min(timeout, 5.0) if reason == "cancelled" else timeout)
        if released is False:
            self.recorder.event("caption.gpu_not_released", node=node, component="tools", job_id=job_id,
                                payload={"before": self._before, "after": after})
        return after, released

    def _start_server(self, job, cancel, sample) -> float | None:
        """Load time in seconds (0.0 when the warm server is reused), None when cancelled while loading."""
        c = self.cfg.get("captioner")
        if self._server is not None and self._server.alive():
            self.recorder.event("caption.server_reused", node=job.node, component="tools", job_id=job.id)
            return 0.0
        self.release("dead", job.node, job.id)              # a server that died between jobs
        work = self.run_dir / "caption_server"
        work.mkdir(parents=True, exist_ok=True)
        port = free_port()
        self._before, self._node = self.gpu_memory(self.gpus), job.node
        self._server = VllmServer(c["env"], self.server_command(port, self.run_dir / "caption_media"),
                                  self.gpus, port, cwd=work, log_path=work / "vllm.log",
                                  recorder=self.recorder, node=job.node, phase=TOOL, poll_s=self.poll_s,
                                  label="caption server")
        load_s = self._server.start(float(c["startup_timeout_s"]), cancel, on_poll=sample)
        if load_s is not None:
            self.recorder.event("caption.server_ready", node=job.node, component="tools",
                                job_id=job.id, load_s=load_s, payload={"gpus": self.gpus, "model": c["model"]})
        return load_s

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        media = self.run_dir / "caption_media" / job.id     # under the only directory the server may read
        media.mkdir(parents=True)
        clips: dict[str, Path] = {}
        results: dict[str, dict] = {}
        for i, path in enumerate(job.args["paths"]):
            dst = media / f"{i}{Path(path).suffix}"   # a hard link: the server resolves symlinks
            try:
                stage_clip(caller, path, dst)
                clips[path] = dst
            except (PathError, OSError) as exc:
                results[path] = {"error": str(exc)}

        if not clips:
            shutil.rmtree(media, ignore_errors=True)
            return {"clips": results, "load_s": None, "gpu_memory_mib": None, "gpu_memory_released": None}
        peak = dict(self.gpu_memory(self.gpus) or {})

        def sample() -> None:
            now = self.gpu_memory(self.gpus) or {}
            for g, used in now.items():
                peak[g] = max(peak.get(g, 0), used)

        load_s, outcome, after, released = None, "failed", None, None
        try:
            load_s = self._start_server(job, cancel, sample)
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
                        sample()
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
            if outcome != "done" or cancel.is_set():        # only a clean job leaves the server warm
                after, released = self.release("cancelled" if cancel.is_set() else outcome, job.node, job.id)
        if outcome == "done" and not cancel.is_set():
            sample()
            after = self.gpu_memory(self.gpus)
        results = {p: results[p] for p in job.args["paths"] if p in results}     # the submitted order
        return {"clips": results, "load_s": load_s,
                "gpu_memory_mib": {"before": self._before, "peak": peak or None, "after": after},
                "gpu_memory_released": released}

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


def submit(q, caller, paths: list[str], prompt: str) -> dict:
    """Validate a caption_videos call and queue its job; returns {"job_id"}."""
    if not paths:
        raise ToolError("paths is empty")
    if not prompt.strip():
        raise ToolError("prompt is empty")
    for path in paths:
        try:
            clip_host_path(caller, path)
        except PathError as exc:
            raise ToolError(str(exc)) from exc
    return {"job_id": q.submit(caller, TOOL, {"paths": [container_path(p) for p in paths], "prompt": prompt})}


def register_caption_tool(mcp, kit, q) -> None:
    """caption_videos queues a job for the queue's `caption_videos` backend (the real
    CaptionBackend, or a fake in smoke runs)."""
    @mcp.tool(name=TOOL, description="Caption video clips with the kernel's local video model. A GPU job: "
              "returns {job_id} at once; collect the result with job_wait. Loading the model takes minutes, "
              "so caption many clips per call. `paths`: video files under /workspace (relative paths "
              "resolve against /workspace); `prompt`: the instruction sent with every clip. The finished "
              "job's result.clips maps each path to {caption} or {error}.")
    async def caption_videos(paths: list[str], prompt: str, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, TOOL, {"paths": paths, "prompt": prompt},
                              lambda c: submit(q, c, paths, prompt))
