from conftest import job_result
import os
import threading
import time

import pytest

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
    q = JobQueue(tmp_path, threading.Lock(), wait_cap_s=1.0)
    q.register(SleepyBackend())
    q.register(BrokenBackend())
    reg = TokenRegistry()
    a = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    b = reg.issue(node="n2", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    yield q, a, b, tmp_path
    q.shutdown()


def test_submit_returns_immediately_and_wait_collects_the_result(env):
    q, a, _, _ = env
    started = time.monotonic()
    job_id = q.submit(a, "sleepy", {"steps": 3})
    assert time.monotonic() - started < 0.5
    out = q.wait(a, job_id, 30)
    assert out["state"] == "done" and job_result(out) == {"clip": "/workspace/staging/fake.mp4"}
    assert out["progress"] == {"step": 3}


def test_wait_is_capped_and_returns_running(env):
    """Job.wait never blocks past tools.job_wait_max_s (1 s here)."""
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
    assert q.wait(a, job_id, 30)["state"] == "canceled"


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
    assert q.wait(a, running, 30)["state"] == "canceled"
    assert q.status(a, queued)["state"] == "canceled"


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

    q = JobQueue(tmp_path, threading.Lock(), wait_cap_s=1.0)
    q.register(StuckBackend())
    reg = TokenRegistry()
    a = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                 staging_host=tmp_path)
    q.submit(a, "stuck", {})
    time.sleep(0.1)  # let the worker pick the job up before we shrink the join window
    monkeypatch.setattr(jobs_module, "_SHUTDOWN_JOIN_S", 0.05)
    with pytest.raises(RuntimeError, match="did not stop"):
        q.shutdown()
    q._worker.join(timeout=5)  # let the real work finish so no thread leaks past the test


def test_run_cancellable_kills_the_whole_group(tmp_path):
    """The GPU backends launch torch jobs through this; a cancel must not orphan ranks."""
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


def test_run_cancellable_reports_minus_15_for_a_cancel(tmp_path):
    """A cancelled GPU job always reports -15 to the caller, whichever signal finished the process group off."""
    pidfile = tmp_path / "child.pid"
    cancel = threading.Event()
    _cancel_once_child_appears(pidfile, cancel)
    code = run_cancellable("autoresearcher", ["bash", "-c", f"sleep 600 & echo $! > {pidfile}; wait"],
                           cwd=tmp_path, cancel=cancel, log_path=tmp_path / "job2.log")
    assert code == -15


def test_job_waiting_for_the_gpu_lock_stays_queued_and_its_gpu_seconds_exclude_the_wait(env):
    """The GPUs are shared with recipe_check: a job is not running until it holds them."""
    q, a, _, _ = env
    with q.gpu_lock:
        job_id = q.submit(a, "sleepy", {"steps": 1, "dt": 0.01})
        time.sleep(1.5)                                 # longer than the limit below: a counted wait fails it
        view = q.status(a, job_id)
        assert view["state"] == "queued" and view["started"] is None
        released = time.time()
    done = q.wait(a, job_id, 30)
    assert done["state"] == "done" and done["started"] >= released
    assert done["finished"] - done["started"] < 1.0      # what job.finished records as gpu_seconds


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
        assert q.cancel(a, job_id)["state"] == "canceled"
    follow_up = q.submit(a, "recording", {})
    assert q.wait(a, follow_up, 30)["state"] == "done"
    assert q.status(a, job_id)["state"] == "canceled" and calls == [follow_up]


def test_a_unique_id_prefix_finds_the_job(env):
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 1})
    assert q.wait(a, job_id[:8], 30)["id"] == job_id


def test_a_mistyped_id_lists_the_callers_own_jobs_and_never_anothers(env):
    q, a, b, _ = env
    mine, theirs = q.submit(a, "sleepy", {"steps": 1}), q.submit(b, "sleepy", {"steps": 1})
    with pytest.raises(ToolError, match=mine) as exc:
        q.status(a, "0" * 32)
    assert theirs not in str(exc.value)
    with pytest.raises(ToolError, match="no job"):
        q.status(a, theirs[:8])                       # another caller's prefix does not resolve


def test_a_short_or_ambiguous_prefix_is_refused(env):
    q, a, _, _ = env
    ids = [q.submit(a, "sleepy", {"steps": 1}) for _ in range(30)]
    with pytest.raises(ToolError, match="no job"):
        q.status(a, ids[0][:7])                       # under 8 characters never resolves
    twin = next((p for p in {i[:8] for i in ids} if sum(j.startswith(p) for j in ids) > 1), None)
    if twin:                                          # 8 hex characters: a clash is very unlikely
        with pytest.raises(ToolError, match="matches 2 jobs"):
            q.status(a, twin)


def test_a_job_fails_without_running_while_the_gpus_are_in_use(tmp_path):
    """The agent's container sees the run's GPUs: a job that starts beside its processes would run out of memory."""
    busy = [True]

    def require_free(caller):
        if busy[0]:
            raise RuntimeError("the GPUs are not free: GPU 0 has 3200 MiB in use")
    q = JobQueue(tmp_path, threading.Lock(), wait_cap_s=5.0, require_free=require_free)
    q.register(SleepyBackend())
    a = TokenRegistry().issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                                 staging_host=tmp_path)
    try:
        out = q.wait(a, q.submit(a, "sleepy", {"steps": 1}), 30)
        assert out["state"] == "failed" and "GPU 0 has 3200 MiB in use" in out["error"]
        assert out["progress"] == {} and not q.gpu_lock.locked()
        busy[0] = False
        assert q.wait(a, q.submit(a, "sleepy", {"steps": 1}), 30)["state"] == "done"
    finally:
        q.shutdown()


def test_drain_waits_for_a_cancelled_running_job(env):
    q, a, _, _ = env
    job = q.submit(a, "sleepy", {"steps": 40, "dt": 0.05})
    deadline = time.monotonic() + 10
    while q.status(a, job)["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    q.cancel_for_token(a.token)
    q.drain(a.token)
    assert q.status(a, job)["state"] == "canceled"


def test_a_job_folder_is_named_by_its_phase_attempt(env, tmp_path):
    q, a, _, _ = env
    job = q._jobs[q.submit(a, "sleepy", {"steps": 1})]
    assert jobs_module.job_dir(tmp_path, job) == tmp_path / "jobs" / "n1" / "improve_recipe-1" / job.id


class ScriptedBackend:
    """Each run raises or returns the next of `outcomes`."""
    name, tool = "scripted", "rollout_scripted"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def run(self, job, cancel, report):
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _queue(tmp_path, outcomes):
    q = JobQueue(tmp_path, threading.Lock(), wait_cap_s=30)
    q.register(ScriptedBackend(outcomes))
    caller = TokenRegistry().issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                                      staging_host=tmp_path)
    run = lambda: q.wait(caller, q.submit(caller, "scripted", {}), 30)
    return q, caller, run


def test_job_failures_count_per_tool_and_refusals_do_not(tmp_path):
    q, caller, run = _queue(tmp_path, [ToolError("bad item"), ToolError("bad item"),
                                       RuntimeError("vLLM exited during startup"), {"items": [{"error": "OOM"}]}])
    try:
        assert [run()["state"] for _ in range(2)] == ["failed", "failed"] and q.health._streak == {}
        assert run()["state"] == "failed" and run()["state"] == "done"
        assert q.health._streak == {"rollout_scripted job": 2}
        assert not caller.fault.event.is_set()
    finally:
        q.shutdown()


def test_input_errors_are_left_out_and_one_good_item_ends_the_streak(tmp_path):
    bad_input = {"items": [{"error": "input: no such file"}, {"error": "input: not 960x544"}]}
    mixed = {"items": [{"error": "worker crashed"}, {"clip": "c.mp4"}]}
    failed = {"items": [{"error": "input: no such file"}, {"error": "worker crashed"}]}
    q, caller, run = _queue(tmp_path, [failed, mixed, failed, bad_input, bad_input, failed])
    try:
        for _ in range(5):
            assert run()["state"] == "done"
        assert q.health._streak == {"rollout_scripted job": 1}
        assert run()["state"] == "done" and q.health._streak == {"rollout_scripted job": 2}
    finally:
        q.shutdown()


def test_a_disk_error_or_kernel_fault_in_a_job_stops_the_phase(tmp_path):
    from ar_kernel.faults import KernelFault
    for n, exc in enumerate((OSError(28, "No space left on device"), KernelFault("GPUs held by pid 9"))):
        q, caller, run = _queue(tmp_path / str(n), [exc])
        try:
            assert run()["state"] == "failed" and caller.fault.event.is_set() and q.health._streak == {}
        finally:
            q.shutdown()


def test_a_cancelled_job_ends_canceled_whatever_it_raised(tmp_path):
    q = JobQueue(tmp_path, threading.Lock(), wait_cap_s=30)
    started = threading.Event()

    class Dying:
        name, tool = "dying", "rollout_dying"

        def run(self, job, cancel, report):
            started.set()
            cancel.wait(30)
            raise RuntimeError("killed")
    q.register(Dying())
    caller = TokenRegistry().issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                                      staging_host=tmp_path)
    try:
        job = q.submit(caller, "dying", {})
        assert started.wait(10)
        q.cancel_for_token(caller.token)
        q.drain(caller.token)
        assert q.status(caller, job)["state"] == "canceled" and not caller.fault.event.is_set()
    finally:
        q.shutdown()


def _raise_once(exc, real):
    """`real`, except that its first call raises `exc`."""
    calls = []

    def wrapped(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return real(*a, **k)
    return wrapped


def test_a_publish_that_cannot_serialize_fails_the_job_and_the_next_job_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_module, "publish", _raise_once(TypeError("not JSON serializable"), jobs_module.publish))
    q, caller, run = _queue(tmp_path, [{"items": [{"clip": "a.mp4"}]}, {"items": [{"clip": "b.mp4"}]}])
    try:
        first = run()
        assert first["state"] == "failed" and first["error"] == "TypeError: not JSON serializable"
        assert run()["state"] == "done"
    finally:
        q.shutdown()


def test_a_raising_judge_fails_the_job_and_the_worker_lives(tmp_path):
    q, caller, run = _queue(tmp_path, [{"items": [{"clip": "a.mp4"}]}, {"items": [{"clip": "b.mp4"}]}])
    q._judge = _raise_once(RuntimeError("judge broke"), q._judge)
    try:
        assert run()["state"] == "failed" and run()["state"] == "done"
    finally:
        q.shutdown()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")     # the worker dies, as intended
def test_a_base_exception_in_the_worker_still_ends_the_job_so_drain_returns(tmp_path):
    q, caller, run = _queue(tmp_path, [SystemExit(1)])
    try:
        job = q.submit(caller, "scripted", {})
        done = threading.Event()
        threading.Thread(target=lambda: (q.drain(caller.token), done.set()), daemon=True).start()
        assert done.wait(10) and q.status(caller, job)["state"] == "failed"
        q._worker.join(10)
        assert not q._worker.is_alive()
    finally:
        q.shutdown()


def _finished(q, caller, backend) -> str:
    job_id = q.submit(caller, backend, {"steps": 1})
    while (state := q.wait(caller, job_id, 30)["state"]) not in ("done", "failed"):     # waits are capped at 1 s
        pass
    return state


def test_each_finished_job_is_an_activity_row(env):
    from ar_kernel.archive.activity import activity
    from ar_kernel.archive.db import open_db
    q, a, _, run = env
    for backend in ("sleepy", "broken"):
        _finished(q, a, backend)
    q.shutdown()                                         # the worker writes the row once the job is final
    rows = [(r["phase"], r["attempt"], r["kind"], r["name"], r["ok"], r["detail"])
            for r in activity(open_db(run), "n1")]
    assert rows == [("improve_recipe", 1, "job", "rollout_fake", 1, "done"),
                    ("improve_recipe", 1, "job", "rollout_broken", 0, "failed")]


def test_a_job_row_that_cannot_be_written_stops_the_callers_phase(env, monkeypatch):
    from ar_kernel.faults import KernelFault

    def broken(*a, **k):
        raise KernelFault("the archive could not record activity: disk I/O error")
    monkeypatch.setattr(jobs_module, "add_activity", broken)
    q, a, _, _ = env
    assert _finished(q, a, "sleepy") == "done"
    assert a.fault.event.wait(5) and a.fault.reason.startswith("rollout_fake job: the archive could not record")
    assert _finished(q, a, "sleepy") == "done"                                           # the worker lives on


def _traced(run):
    from ar_kernel import telemetry
    telemetry.open(run)
    telemetry.start("ar.run")
    return telemetry


def _cancels_and_end(run, job_id):
    from conftest import telemetry_payload, telemetry_records
    records = telemetry_records(run)
    [start] = [r for r in records if r["kind"] == "start" and r["name"] == "job.rollout_fake"
               and telemetry_payload(run, r["inputs"])["job"] == job_id]
    cancels = [r for r in records if r["kind"] == "start" and r["name"] == "job.cancel"
               and telemetry_payload(run, r["inputs"])["job"] == job_id]
    ends = [r for r in records if r["kind"] == "end" and r["span"] == start["span"]]
    return start, cancels, ends


def test_a_job_is_one_span_from_submit_to_its_end_and_each_cancel_is_a_span_under_the_canceller(env):
    from ar_kernel.telemetry import tree
    q, a, _, run = env
    telemetry = _traced(run)
    with telemetry.span("tool.rollout_fake") as submitter:
        running = q.submit(a, "sleepy", {"steps": 500, "dt": 0.05})
        queued = q.submit(a, "sleepy", {"steps": 500})
        submitter.output = {}
    time.sleep(0.2)
    with telemetry.span("phase.close") as closer:
        assert q.cancel_for_token(a.token) == 2
        closer.output = {}
    done = q.submit(a, "sleepy", {"steps": 1})
    for job_id, state in ((running, "canceled"), (done, "done")):
        deadline = time.monotonic() + 30                    # wait() returns after wait_cap_s (1 s) here
        while q.status(a, job_id)["state"] != state and time.monotonic() < deadline:
            q.wait(a, job_id, 1)
        assert q.status(a, job_id)["state"] == state
    with telemetry.span("tool.job_cancel") as late:
        q.cancel(a, done)                                   # already final: no cancel recorded
        late.output = {}
    for job_id in (running, queued):
        start, cancels, ends = _cancels_and_end(run, job_id)
        assert start["parent"] == submitter.id
        assert [c["parent"] for c in cancels] == [closer.id] and [e["status"] for e in ends] == ["cancelled"]
    start, cancels, ends = _cancels_and_end(run, done)
    assert cancels == [] and [e["status"] for e in ends] == ["ok"]
    assert tree.check(tree.load(run)) == []


def test_a_failed_job_ends_its_span_with_the_error(env):
    from conftest import telemetry_records
    q, a, _, run = env
    _traced(run)
    job_id = q.submit(a, "broken", {})
    assert q.wait(a, job_id, 30)["state"] == "failed"
    [end] = [r for r in telemetry_records(run) if r["kind"] == "end" and r["name"] == "job.rollout_broken"]
    assert end["status"] == "error" and end["error"]
