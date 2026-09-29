"""Asynchronous GPU jobs (spec 10). One job at a time, under the run's GPU lock."""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from mcp.server.mcpserver import Context

from ..subproc import KILL_FORCE_GRACE_SECONDS, KILL_GRACE_SECONDS, run_in_env
from .server import ToolError

_DEFAULT_POLL_S = 1.0
# Worst case a worker thread can be stuck inside a real backend's run_cancellable:
# one poll interval before a cancel is even noticed, then run_in_env's own
# SIGTERM grace and SIGKILL grace. shutdown()'s join must outlast this, or a
# still-running kill sequence gets cut off when the process (its daemon thread
# included) exits.
_SHUTDOWN_JOIN_S = _DEFAULT_POLL_S + KILL_GRACE_SECONDS + KILL_FORCE_GRACE_SECONDS + 5


class JobBackend(Protocol):
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
    state: str = "queued"
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None


_TERMINAL = {"done", "failed", "cancelled"}


class JobQueue:
    """Runs at most one job at a time on the run's GPUs, under `gpu_lock`.

    A caller only ever sees or cancels jobs its own token submitted (`_own`);
    `cancel_for_token` is how a phase's container exit sweeps its orphaned jobs
    (Task 15) so an agent that disappears never keeps holding the GPUs.
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
        view = asdict(job)
        view.pop("token")
        return view

    def submit(self, caller, backend_name: str, args: dict) -> str:
        if backend_name not in self._backends:
            raise ToolError(f"generator {backend_name!r} is not enabled on this run")
        job = Job(id=uuid.uuid4().hex, backend=backend_name, args=dict(args),
                  node=caller.node, token=caller.token)
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
        own GPU job is still working is not stalled, spec 14.5)."""
        with self._cond:
            return sum(1 for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL)

    def cancel_for_token(self, token: str) -> int:
        with self._cond:
            live = [j for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL]
            for job in live:
                self._cancel_locked(job)
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
        while True:
            job_id = self._todo.get()
            if job_id is None:
                return
            with self._cond:
                job = self._jobs[job_id]
            cancel = self._cancel[job_id]

            def report(progress: dict, _job=job) -> None:
                with self._cond:
                    _job.progress = dict(progress)
                    self._cond.notify_all()

            tb = None
            with self.gpu_lock:
                # Running starts once the GPUs are held, so gpu_seconds excludes the lock
                # wait; a job cancelled before or during that wait is already final.
                with self._cond:
                    if job.state != "queued":
                        continue
                    job.state, job.started = "running", time.time()
                    self._cond.notify_all()
                try:
                    result = self._backends[job.backend].run(job, cancel, report)
                    final, error = ("cancelled" if cancel.is_set() else "done"), None
                except Exception as exc:                  # noqa: BLE001 -- a job failure, not a crash
                    result, final = None, "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    tb = traceback.format_exc()
            with self._cond:
                job.state, job.result, job.error, job.finished = final, result, error, time.time()
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
    @mcp.tool(name="job_status", description="State, progress and (when finished) the result of a GPU job.")
    async def job_status(job_id: str, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_status", {"job_id": job_id}, lambda c: q.status(c, job_id))

    @mcp.tool(name="job_wait", description="Wait for a GPU job, at most 300 s per call; returns "
              "state 'running' if it has not finished. Call again to keep waiting.")
    async def job_wait(job_id: str, ctx: Context, timeout_s: float = 300) -> dict[str, Any]:
        return await kit.call(ctx, "job_wait", {"job_id": job_id, "timeout_s": timeout_s},
                              lambda c: q.wait(c, job_id, timeout_s))

    @mcp.tool(name="job_cancel", description="Cancel a queued or running GPU job.")
    async def job_cancel(job_id: str, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_cancel", {"job_id": job_id}, lambda c: q.cancel(c, job_id))
