"""run_in_env must be safe to use from an unattended multi-day loop.

Killing only the `conda run` process orphans torchrun ranks and WBench multiprocessing workers,
which keep holding GPUs. A cancel therefore stops the whole process group.
"""
import os
import threading
import time

import pytest

from ar_kernel.config import REPO_ROOT
from ar_kernel.subproc import conda_command, run_in_env
from ar_kernel.telemetry.recorder import Recorder

ENV = "autoresearcher"


def test_conda_command_refuses_an_env_that_is_not_in_the_project(tmp_path, monkeypatch):
    (tmp_path / ".envs" / "half-made").mkdir(parents=True)          # no conda-meta: not an env
    monkeypatch.setattr("ar_kernel.subproc.REPO_ROOT", tmp_path)
    for name in ("autoresearcher", "half-made"):
        with pytest.raises(FileNotFoundError, match="no conda env"):
            conda_command(name, ["python", "-c", "1"])


def test_conda_command_by_name_prefers_the_project_env(tmp_path, monkeypatch):
    (tmp_path / ".envs" / "alayaworld" / "conda-meta").mkdir(parents=True)
    monkeypatch.setattr("ar_kernel.subproc.REPO_ROOT", tmp_path)
    assert conda_command("alayaworld", ["x"]) == \
        ["conda", "run", "--no-capture-output", "-p", str(tmp_path / ".envs" / "alayaworld"), "x"]


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


def test_cancel_kills_the_whole_process_group(tmp_path):
    """A grandchild (standing in for a torchrun rank / spawn worker) must die too."""
    pidfile = tmp_path / "child.pid"
    script = f"sleep 600 & echo $! > {pidfile}; echo started; wait"
    cancel = threading.Event()
    threading.Thread(target=lambda: (_wait_for(pidfile), cancel.set()), daemon=True).start()
    proc = run_in_env(ENV, ["bash", "-c", script], cwd=tmp_path, cancel=cancel, poll_s=0.2)
    assert proc.cancelled and "started" in proc.stdout
    child = int(pidfile.read_text())
    deadline = time.time() + 15
    while _alive(child) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(child), "grandchild survived the cancel -- it would keep holding the GPU"


def _wait_for(path, seconds=60):
    deadline = time.time() + seconds
    while not (path.exists() and path.read_text().strip()) and time.time() < deadline:
        time.sleep(0.1)


def test_log_path_streams_output_to_disk(tmp_path):
    """Long jobs (48 h of training) must not buffer everything in memory until exit."""
    log = tmp_path / "logs" / "job.log"
    proc = run_in_env(ENV, ["bash", "-c", "echo to-out; echo to-err 1>&2"], cwd=tmp_path, log_path=log)
    assert proc.returncode == 0
    text = log.read_text()
    assert "to-out" in text and "to-err" in text
    assert "to-out" in proc.stdout


def test_process_groups_are_recorded_while_running(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_PGID_DIR", str(tmp_path / "pgids"))
    seen = []

    def check():
        seen.extend(p.name for p in (tmp_path / "pgids").iterdir())
    t = threading.Timer(3.0, check)
    t.start()
    run_in_env(ENV, ["python", "-c", "import time; time.sleep(6)"], cwd=tmp_path)
    t.join()
    assert len(seen) == 1 and not any((tmp_path / "pgids").iterdir())


def test_an_interrupted_kill_keeps_the_group_record(tmp_path, monkeypatch):
    """A second Ctrl-C during the kill sequence: the group may still be alive, so the record
    must stay for kill_recorded_groups on resume."""
    import signal
    from ar_kernel import subproc
    monkeypatch.setenv("AR_PGID_DIR", str(tmp_path / "pgids"))

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(subproc, "_wait", interrupted)
    monkeypatch.setattr(subproc, "_kill_group", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_in_env(ENV, ["python", "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    (marker,) = (tmp_path / "pgids").iterdir()
    os.killpg(int(marker.name), signal.SIGKILL)
