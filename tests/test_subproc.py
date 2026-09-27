"""run_in_env must be safe to use from an unattended multi-day loop.

Review I6: subprocess.run(timeout=) killed only the `conda run` process, so
torchrun ranks and WBench multiprocessing workers were orphaned and kept holding
GPUs -- the exact failure the Task 14 verification hit, where two orphaned spawn
workers held ~19 GB for 35 minutes and made the next job OOM. A timeout also
wrote no event and dropped the output captured so far.
"""
import os
import time

import pytest

from ar_kernel.config import REPO_ROOT
from ar_kernel.liveness import Liveness, tree_mark
from ar_kernel.subproc import SubprocTimeout, conda_command, run_in_env
from ar_kernel.telemetry.recorder import Recorder

ENV = "autoresearcher"


def test_conda_command_by_name_uses_dash_n():
    assert conda_command("autoresearcher", ["python", "-c", "1"]) == \
        ["conda", "run", "--no-capture-output", "-n", "autoresearcher", "python", "-c", "1"]


def test_conda_command_with_a_slash_is_a_repo_relative_prefix():
    assert conda_command(".envs/gen-zimage", ["x"]) == \
        ["conda", "run", "--no-capture-output", "-p", str(REPO_ROOT / ".envs" / "gen-zimage"), "x"]


def test_conda_command_with_an_absolute_path_is_used_as_is():
    assert conda_command("/tmp/some/env", ["x"]) == \
        ["conda", "run", "--no-capture-output", "-p", "/tmp/some/env", "x"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_success_returns_output_and_code(tmp_path):
    proc = run_in_env(ENV, ["python", "-c", "print('hello'); import sys; sys.exit(3)"], cwd=tmp_path)
    assert proc.returncode == 3
    assert "hello" in proc.stdout


def test_timeout_kills_the_whole_process_group(tmp_path):
    """A grandchild (standing in for a torchrun rank / spawn worker) must die too."""
    pidfile = tmp_path / "child.pid"
    script = f"sleep 600 & echo $! > {pidfile}; echo started; wait"
    with pytest.raises(SubprocTimeout):
        run_in_env(ENV, ["bash", "-c", script], cwd=tmp_path, timeout=8)
    child = int(pidfile.read_text())
    deadline = time.time() + 15
    while _alive(child) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(child), "grandchild survived the timeout -- it would keep holding the GPU"


def test_timeout_records_an_event_with_the_partial_output(tmp_path):
    rec = Recorder(tmp_path)
    with pytest.raises(SubprocTimeout) as info:
        run_in_env(ENV, ["bash", "-c", "echo progress-before-hang; sleep 600"],
                   cwd=tmp_path, timeout=8, recorder=rec, node="n1", phase="train")
    assert "progress-before-hang" in (info.value.output or "")
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert "subproc.error" in kinds, kinds
    err = next(e for e in rec.read_events("n1") if e["type"] == "subproc.error")
    assert "progress-before-hang" in str(rec.load_payload(err["payload"]))


def test_log_path_streams_output_to_disk(tmp_path):
    """Long jobs (48 h of training) must not buffer everything in memory until exit."""
    log = tmp_path / "logs" / "job.log"
    proc = run_in_env(ENV, ["bash", "-c", "echo to-out; echo to-err 1>&2"], cwd=tmp_path, log_path=log)
    assert proc.returncode == 0
    text = log.read_text()
    assert "to-out" in text and "to-err" in text
    assert "to-out" in proc.stdout


def test_a_stalled_job_is_killed_by_liveness(tmp_path):
    rec = Recorder(tmp_path)
    lv = Liveness(1, probe_window_s=1, extension_frac=0.25, signals=[lambda: 0])
    started = time.monotonic()
    with pytest.raises(SubprocTimeout):
        run_in_env(ENV, ["python", "-c", "import time; time.sleep(120)"], cwd=tmp_path,
                   recorder=rec, liveness=lv, poll_s=0.2)
    assert time.monotonic() - started < 30
    err = [e for e in rec.read_events() if e["type"] == "subproc.error"][-1]
    assert "no sign of progress" in rec.load_payload(err["payload"])["message"]


def test_a_job_that_keeps_writing_is_extended(tmp_path):
    log = tmp_path / "log.txt"
    lv = Liveness(1, probe_window_s=1.5, extension_frac=0.5, signals=[lambda: tree_mark(log)])
    code = "import time\nfor i in range(8):\n    print(i, flush=True); time.sleep(0.5)\n"
    proc = run_in_env(ENV, ["python", "-c", code], cwd=tmp_path, liveness=lv, poll_s=0.2, log_path=log)
    assert proc.returncode == 0 and lv.extensions >= 1
