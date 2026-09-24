import os
import threading
import time

import pytest

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue, run_cancellable
from ar_kernel.tools.server import ToolError


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


def test_run_cancellable_kills_the_whole_group(tmp_path):
    """Plan 3 backends launch torch jobs through this; a cancel must not orphan ranks.

    The cancel fires once the child's pid file appears (readiness signal) rather
    than after a fixed timer, so this does not race conda run's startup latency;
    the wait itself is bounded so a launch failure still fails the test promptly.
    """
    pidfile = tmp_path / "child.pid"
    cancel = threading.Event()

    def cancel_when_ready() -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not pidfile.exists():
            time.sleep(0.1)
        cancel.set()

    threading.Thread(target=cancel_when_ready, daemon=True).start()
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
