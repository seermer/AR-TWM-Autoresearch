import os
import threading
import time

import pytest

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import jobs as jobs_module
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue, run_cancellable
from ar_kernel.tools.server import ToolError


def _cancel_once_child_appears(pidfile, cancel: threading.Event, deadline_s: float = 60) -> None:
    """Trigger `cancel` once `pidfile` shows up, bounded by `deadline_s`.

    Waiting for the actual readiness signal (rather than a fixed timer) avoids
    racing `conda run`'s own startup latency; the bounded deadline still fails
    the test promptly if the launch never happens at all.
    """
    def watch() -> None:
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline and not pidfile.exists():
            time.sleep(0.1)
        cancel.set()

    threading.Thread(target=watch, daemon=True).start()


class SleepyBackend:
    name, tool = "sleepy", "rollout_fake"

    def run(self, job, cancel, report):
        for i in range(int(job.args.get("steps", 3))):
            if cancel.is_set():
                return {"cancelled_at": i}
            report({"step": i + 1})
            time.sleep(job.args.get("dt", 0.05))
        return {"clip": "/workspace/staging/fake.mp4"}


class BrokenBackend:
    name, tool = "broken", "rollout_broken"

    def run(self, job, cancel, report):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=1.0)
    q.register(SleepyBackend())
    q.register(BrokenBackend())
    reg = TokenRegistry(rec)
    a = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    b = reg.issue(node="n2", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    yield q, a, b, rec
    q.shutdown()


def test_submit_returns_immediately_and_wait_collects_the_result(env):
    q, a, _, _ = env
    started = time.monotonic()
    job_id = q.submit(a, "sleepy", {"steps": 3})
    assert time.monotonic() - started < 0.5
    out = q.wait(a, job_id, 30)
    assert out["state"] == "done" and out["result"] == {"clip": "/workspace/staging/fake.mp4"}
    assert out["progress"] == {"step": 3}


def test_wait_is_capped_and_returns_running(env):
    """Spec 10: job.wait never blocks past tools.job_wait_max_s (1 s here)."""
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 200, "dt": 0.05})
    started = time.monotonic()
    out = q.wait(a, job_id, 10_000)
    assert out["state"] == "running" and time.monotonic() - started < 2.5


def test_backend_failure_is_a_failed_job_not_a_crash(env):
    q, a, _, _ = env
    out = q.wait(a, q.submit(a, "broken", {}), 30)
    assert out["state"] == "failed" and "out of memory" in out["error"]


def test_jobs_run_one_at_a_time(env):
    q, a, _, _ = env
    first = q.submit(a, "sleepy", {"steps": 20, "dt": 0.05})
    second = q.submit(a, "sleepy", {"steps": 1})
    time.sleep(0.2)
    assert q.status(a, first)["state"] == "running"
    assert q.status(a, second)["state"] == "queued"


def test_cancel_stops_a_running_job(env):
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 500, "dt": 0.05})
    time.sleep(0.2)
    q.cancel(a, job_id)
    assert q.wait(a, job_id, 30)["state"] == "cancelled"


def test_callers_cannot_see_each_others_jobs(env):
    q, a, b, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 1})
    with pytest.raises(ToolError, match="no job"):
        q.status(b, job_id)
    with pytest.raises(ToolError, match="no job"):
        q.cancel(b, job_id)


def test_cancel_for_token_clears_an_ended_phase(env):
    q, a, _, _ = env
    running = q.submit(a, "sleepy", {"steps": 500, "dt": 0.05})
    queued = q.submit(a, "sleepy", {"steps": 500})
    time.sleep(0.2)
    assert q.cancel_for_token(a.token) == 2
    assert q.wait(a, running, 30)["state"] == "cancelled"
    assert q.status(a, queued)["state"] == "cancelled"


def test_unknown_backend_is_a_tool_error(env):
    q, a, _, _ = env
    with pytest.raises(ToolError, match="not enabled"):
        q.submit(a, "wan22", {})


def test_wait_caps_a_non_finite_timeout(env):
    """A lax-parsing client (e.g. JSON-RPC "nan") can hand job_wait a non-finite
    timeout_s. min(nan, cap) is nan whenever nan sorts first, which never
    satisfies `remaining <= 0`, so this must not be allowed to busy-spin past
    the cap -- both NaN and +inf must still return at the cap."""
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 200, "dt": 0.05})
    for timeout_s in (float("nan"), float("inf")):
        started = time.monotonic()
        out = q.wait(a, job_id, timeout_s)
        assert out["state"] == "running"
        assert time.monotonic() - started < 2.5


def test_shutdown_raises_if_the_worker_outlives_the_join_timeout(tmp_path, monkeypatch):
    """A worker wedged past its join timeout (its backend hung inside a kill
    path) must not exit shutdown() silently -- the caller needs to know a job
    might still be holding a GPU."""
    class StuckBackend:
        name, tool = "stuck", "rollout_stuck"

        def run(self, job, cancel, report):
            time.sleep(1.0)
            return {}

    rec = Recorder(tmp_path)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=1.0)
    q.register(StuckBackend())
    reg = TokenRegistry(rec)
    a = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                 staging_host=tmp_path)
    q.submit(a, "stuck", {})
    time.sleep(0.1)  # let the worker pick the job up before we shrink the join window
    monkeypatch.setattr(jobs_module, "_SHUTDOWN_JOIN_S", 0.05)
    with pytest.raises(RuntimeError, match="did not stop"):
        q.shutdown()
    q._worker.join(timeout=5)  # let the real work finish so no thread leaks past the test


def test_run_cancellable_kills_the_whole_group(tmp_path):
    """Plan 3 backends launch torch jobs through this; a cancel must not orphan ranks."""
    pidfile = tmp_path / "child.pid"
    cancel = threading.Event()
    _cancel_once_child_appears(pidfile, cancel)
    code = run_cancellable("autoresearcher", ["bash", "-c", f"sleep 600 & echo $! > {pidfile}; wait"],
                           cwd=tmp_path, cancel=cancel, log_path=tmp_path / "job.log")
    assert code == -15
    child = int(pidfile.read_text())
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/stat") as fh:
                if fh.read().split()[2] == "Z":
                    break
        except (ProcessLookupError, FileNotFoundError):
            break
        time.sleep(0.2)
    else:
        pytest.fail("grandchild survived the cancel")


def test_run_cancellable_records_a_distinct_cancel_event_and_reports_minus_15(tmp_path):
    """A cancelled GPU job must be distinguishable in telemetry from a normal
    exit, and must always report -15 to the caller regardless of which signal
    actually finished the process group off."""
    pidfile = tmp_path / "child.pid"
    cancel = threading.Event()
    _cancel_once_child_appears(pidfile, cancel)
    rec = Recorder(tmp_path)
    code = run_cancellable("autoresearcher", ["bash", "-c", f"sleep 600 & echo $! > {pidfile}; wait"],
                           cwd=tmp_path, cancel=cancel, log_path=tmp_path / "job2.log",
                           recorder=rec, node="n1", phase="rollout")
    assert code == -15
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert "subproc.cancelled" in kinds, kinds
    assert "subproc.end" not in kinds, kinds


def test_job_waiting_for_the_gpu_lock_stays_queued_and_its_gpu_seconds_exclude_the_wait(env):
    """The GPUs are shared with recipe_check: a job is not running until it holds them."""
    q, a, _, _ = env
    with q.gpu_lock:
        job_id = q.submit(a, "sleepy", {"steps": 1, "dt": 0.01})
        time.sleep(0.3)
        view = q.status(a, job_id)
        assert view["state"] == "queued" and view["started"] is None
        released = time.time()
    done = q.wait(a, job_id, 30)
    assert done["state"] == "done" and done["started"] >= released
    assert done["finished"] - done["started"] < 0.3      # what job.finished records as gpu_seconds


def test_active_for_token_counts_queued_and_running_jobs(env):
    q, a, b, _ = env
    gate = threading.Event()

    class BlockingBackend:
        name, tool = "blocking", "rollout_blocking"

        def run(self, job, cancel, report):
            gate.wait(30)
            return {}

    q.register(BlockingBackend())
    assert q.active_for_token(a.token) == 0
    job_id = q.submit(a, "blocking", {})
    deadline = time.monotonic() + 5
    while q.status(a, job_id)["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert q.active_for_token(a.token) == 1
    assert q.active_for_token(b.token) == 0
    gate.set()
    assert q.wait(a, job_id, 30)["state"] == "done"
    assert q.active_for_token(a.token) == 0


def test_job_cancelled_while_waiting_for_the_gpu_lock_never_reaches_the_backend(env):
    q, a, _, _ = env
    calls = []

    class RecordingBackend:
        name, tool = "recording", "rollout_recording"

        def run(self, job, cancel, report):
            calls.append(job.id)
            return {}

    q.register(RecordingBackend())
    with q.gpu_lock:
        job_id = q.submit(a, "recording", {})
        time.sleep(0.3)                      # the worker has dequeued it and waits for the lock
        assert q.cancel(a, job_id)["state"] == "cancelled"
    follow_up = q.submit(a, "recording", {})
    assert q.wait(a, follow_up, 30)["state"] == "done"
    assert q.status(a, job_id)["state"] == "cancelled" and calls == [follow_up]
