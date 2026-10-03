"""Asynchronous GPU jobs. One job at a time, under the run's GPU lock. A finished job's result goes to a
file in the caller's staging directory; the caller is told the path and a summary."""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Protocol

from pydantic import Field

from mcp.server.mcpserver import Context

from ..subproc import KILL_FORCE_GRACE_SECONDS, KILL_GRACE_SECONDS, run_in_env
from .server import SHOWN, ToolError, publish

_DEFAULT_POLL_S = 1.0
# Worst case a worker thread can be stuck inside a real backend's run_cancellable:
# one poll interval before a cancel is even noticed, then run_in_env's own
# SIGTERM grace and SIGKILL grace. shutdown()'s join must outlast this, or a
# still-running kill sequence gets cut off when the process (its daemon thread
# included) exits.
_SHUTDOWN_JOIN_S = _DEFAULT_POLL_S + KILL_GRACE_SECONDS + KILL_FORCE_GRACE_SECONDS + 5
_RELEASE = ""               # queued to wake the worker so that it frees a backend it keeps warm


class JobBackend(Protocol):
    """A backend may stay `warm` after a job (a loaded model server): the worker then keeps the GPU lock,
    runs the next job at once if it is for the same backend, and otherwise calls `release()` first, as it
    does after `keep_warm_s` without a job and when a phase ends."""
    name: str
    tool: str

    def run(self, job: "Job", cancel: threading.Event, report: Callable[[dict], None]) -> dict: ...


@dataclass
class Job:
    id: str
    backend: str
    args: dict
    node: str
    token: str
    caller: Any = None                # who submitted it: where its result file goes
    state: str = "queued"
    progress: dict = field(default_factory=dict)
    result_file: str | None = None    # /workspace/staging/results/<backend>-<id>.json, once finished
    summary: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None


_TERMINAL = {"done", "failed", "cancelled"}


def summarise(result: dict) -> dict:
    """What a caller is told of a finished job: how many items, how many failed and the first errors."""
    rows = result["items"] if isinstance(result.get("items"), list) else \
        [{"path": path, **row} for path, row in (result.get("clips") or {}).items()]
    failed = [row for row in rows if "error" in row]
    return {"items": len(rows), "failed": len(failed),
            "errors": [{"item": row.get("index", row.get("path")), "error": str(row["error"])[:300]}
                       for row in failed[:SHOWN]]}


class JobQueue:
    """Runs at most one job at a time on the run's GPUs, under `gpu_lock`.

    A caller only ever sees or cancels jobs its own token submitted (`_own`);
    `cancel_for_token` is how a phase's container exit sweeps its orphaned jobs
    so an agent that disappears never keeps holding the GPUs.
    """

    def __init__(self, recorder, gpu_lock: threading.Lock, wait_cap_s: float) -> None:
        self.recorder, self.gpu_lock, self.wait_cap_s = recorder, gpu_lock, float(wait_cap_s)
        self._backends: dict[str, JobBackend] = {}
        self._jobs: dict[str, Job] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._cond = threading.Condition()
        self._todo: "queue.Queue[str | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="ar-gpu-jobs", daemon=True)
        self._worker.start()

    def register(self, backend: JobBackend) -> None:
        self._backends[backend.name] = backend

    @property
    def backends(self) -> dict[str, JobBackend]:
        return dict(self._backends)

    def _own(self, caller, job_id: str) -> Job:
        """The caller's job with this id, or with this unique id prefix (8+ characters)."""
        mine = {i: j for i, j in self._jobs.items() if j.token == caller.token}
        if job_id in mine:
            return mine[job_id]
        matches = [i for i in mine if len(job_id) >= 8 and i.startswith(job_id)]
        if len(matches) == 1:
            return mine[matches[0]]
        if matches:
            raise ToolError(f"job id prefix {job_id!r} matches {len(matches)} jobs: {', '.join(matches)}")
        raise ToolError(f"no job {job_id!r} for this caller; yours: {', '.join(mine) or 'none'}")

    def _view(self, job: Job) -> dict:
        # args are left out: the caller sent them, and echoing every item on each poll only fills its context
        return {key: getattr(job, key) for key in ("id", "backend", "node", "state", "progress", "result_file",
                                                   "summary", "error", "created", "started", "finished")}

    def submit(self, caller, backend_name: str, args: dict) -> str:
        if backend_name not in self._backends:
            raise ToolError(f"generator {backend_name!r} is not enabled on this run")
        job = Job(id=uuid.uuid4().hex, backend=backend_name, args=dict(args),
                  node=caller.node, token=caller.token, caller=caller)
        with self._cond:
            self._jobs[job.id] = job
            self._cancel[job.id] = threading.Event()
        self.recorder.event("job.submitted", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="tools", job_id=job.id,
                            payload={"backend": backend_name, "args": args})
        self._todo.put(job.id)
        return job.id

    def status(self, caller, job_id: str) -> dict:
        with self._cond:
            return self._view(self._own(caller, job_id))

    def wait(self, caller, job_id: str, timeout_s: float) -> dict:
        # A client can send nan/inf; min() and max() would pass nan through and never time out.
        timeout_s = float(timeout_s)
        capped = self.wait_cap_s if not math.isfinite(timeout_s) else max(0.0, min(timeout_s, self.wait_cap_s))
        deadline = time.monotonic() + capped
        with self._cond:
            job = self._own(caller, job_id)
            while job.state not in _TERMINAL:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._view(job)

    def cancel(self, caller, job_id: str) -> dict:
        with self._cond:
            job = self._own(caller, job_id)
            self._cancel_locked(job)
            return self._view(job)

    def active_for_token(self, token: str) -> int:
        """Jobs queued or running for `token` (a liveness signal: a phase whose
        own GPU job is still working is not stalled)."""
        with self._cond:
            return sum(1 for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL)

    def cancel_for_token(self, token: str) -> int:
        with self._cond:
            live = [j for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL]
            for job in live:
                self._cancel_locked(job)
        self._todo.put(_RELEASE)          # the phase is over: nothing may stay loaded on the GPUs
        return len(live)

    def _cancel_locked(self, job: Job) -> None:
        self._cancel[job.id].set()
        if job.state == "queued":
            job.state, job.finished = "cancelled", time.time()
            self._cond.notify_all()

    def shutdown(self) -> None:
        with self._cond:
            for job in self._jobs.values():
                if job.state not in _TERMINAL:
                    self._cancel_locked(job)
        self._todo.put(None)
        self._worker.join(timeout=_SHUTDOWN_JOIN_S)
        if self._worker.is_alive():
            raise RuntimeError(
                f"GPU job worker did not stop within {_SHUTDOWN_JOIN_S}s of shutdown; "
                "a job may still be holding a GPU"
            )

    def _loop(self) -> None:
        warm = None                       # a backend still loaded on the GPUs: the GPU lock stays held

        def cool() -> None:
            nonlocal warm
            if warm is not None:
                try:
                    warm.release()
                finally:
                    warm = None
                    self.gpu_lock.release()

        while True:
            try:
                job_id = self._todo.get(timeout=warm.keep_warm_s if warm else None)
            except queue.Empty:
                job_id = _RELEASE
            if not job_id:
                cool()
                if job_id is None:
                    return
                continue
            with self._cond:
                job = self._jobs[job_id]
            backend, cancel = self._backends[job.backend], self._cancel[job_id]
            if warm is not backend:
                cool()

            def report(progress: dict, _job=job) -> None:
                with self._cond:
                    _job.progress = dict(progress)
                    self._cond.notify_all()

            if warm is None:
                self.gpu_lock.acquire()
            result, final, error, tb = None, "cancelled", None, None
            try:
                # Running starts once the GPUs are held, so gpu_seconds excludes the lock
                # wait; a job cancelled before or during that wait is already final.
                with self._cond:
                    if job.state != "queued":
                        continue
                    job.state, job.started = "running", time.time()
                    self._cond.notify_all()
                try:
                    result = backend.run(job, cancel, report)
                    final = "cancelled" if cancel.is_set() else "done"
                except Exception as exc:                  # noqa: BLE001 -- a job failure, not a crash
                    final, error, tb = "failed", f"{type(exc).__name__}: {exc}", traceback.format_exc()
            finally:
                if getattr(backend, "warm", False):
                    warm = backend
                else:
                    warm = None
                    self.gpu_lock.release()
            result_file = summary = None
            if result is not None:
                summary = summarise(result)
                try:
                    result_file = publish(job.caller, f"{job.backend}-{job.id}", result)
                except OSError as exc:                    # the phase ended and took its staging with it
                    error = f"the result could not be written: {type(exc).__name__}: {exc}"
            with self._cond:
                job.state, job.error, job.finished = final, error, time.time()
                job.result_file, job.summary = result_file, summary
                self._cond.notify_all()
            self.recorder.event("job.finished", node=job.node, component="tools", job_id=job.id,
                                state=final, gpu_seconds=job.finished - job.started,
                                payload={"result": result, "error": error,
                                         "traceback": tb if final == "failed" else None})


def run_cancellable(env: str, args: list[str], *, cwd: Path, cancel: threading.Event,
                    extra_env: dict | None = None, log_path: Path, poll_s: float = _DEFAULT_POLL_S,
                    recorder=None, node: str = "run", phase: str = "-") -> int:
    """Run a GPU job's command via `run_in_env`, killing its whole process group on cancel.
    Backends launch every command through this. Returns the exit code, or -15 whenever `cancel`
    caused the kill (also when the job ignored SIGTERM and needed SIGKILL)."""
    result = run_in_env(env, args, cwd=cwd, extra_env=extra_env, log_path=log_path,
                        cancel=cancel, poll_s=poll_s, recorder=recorder, node=node, phase=phase)
    return -15 if result.cancelled else result.returncode


def register_job_tools(mcp, kit, q: JobQueue) -> None:
    @mcp.tool(name="job_status", description="State and progress of a GPU job and, once it finished, "
              "`result_file` (the full result as JSON, under /workspace/staging/results/) and `summary` (items, "
              "how many failed, the first errors).")
    async def job_status(job_id: Annotated[str, Field(description="the job_id a GPU tool returned (its first 8 characters are enough)")], ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_status", {"job_id": job_id}, lambda c: q.status(c, job_id))

    @mcp.tool(name="job_wait", description="Wait for a GPU job, at most 300 s per call; returns "
              "state 'running' if it has not finished: call again to keep waiting. A finished job gives "
              "`result_file` and `summary` as job_status does.")
    async def job_wait(job_id: Annotated[str, Field(description="the job_id a GPU tool returned (its first 8 characters are enough)")], ctx: Context,
                       timeout_s: Annotated[float, Field(description="seconds to wait, at most 300; call again if the job is still running")] = 300) -> dict[str, Any]:
        return await kit.call(ctx, "job_wait", {"job_id": job_id, "timeout_s": timeout_s},
                              lambda c: q.wait(c, job_id, timeout_s))

    @mcp.tool(name="job_cancel", description="Cancel a queued or running GPU job.")
    async def job_cancel(job_id: Annotated[str, Field(description="the job_id a GPU tool returned (its first 8 characters are enough)")], ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_cancel", {"job_id": job_id}, lambda c: q.cancel(c, job_id))
