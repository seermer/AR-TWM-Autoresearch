import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ar_kernel import cli
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import GpuPolicyError
from ar_kernel.control import Control

REPO = Path(__file__).resolve().parents[1]


def test_run_refuses_an_existing_run_without_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["run", "--run-id", "r1", "--max-nodes", "3"]) == 2
    assert "--resume" in capsys.readouterr().err


def test_stop_writes_the_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["stop", "--run-id", "r1"]) == 0
    assert (tmp_path / "r1" / "control" / "stop").exists()


def test_a_new_run_without_max_nodes_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    assert cli.main(["run", "--run-id", "r2"]) == 2
    assert "--max-nodes" in capsys.readouterr().err and not (tmp_path / "r2").exists()


def _existing_run(tmp_path, monkeypatch, gpus_default="0,1,2,3", root="scored"):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run = tmp_path / "r1"
    (run / "config").mkdir(parents=True)
    raw = yaml.safe_load((REPO / "configs" / "kernel.yaml").read_text())
    judge = {"kind": "local", "model": raw["captioner"]["model"], "url": None}
    (run / "config" / "run.json").write_text(json.dumps(
        {"metric_set": ["m"], "case_ids": ["1"], "versions": {}, "judge": judge}))
    raw["gpus"]["default"] = gpus_default
    (run / "config" / "kernel.yaml").write_text(yaml.safe_dump(raw))
    Control(run).save_args(max_nodes=1)
    if root is not None:
        nodes = NodeStore(open_db(run))
        nodes.create("root", None, 0)
        if root == "scored":
            nodes.record_score("root", 0.78, ["m"], {"m": 0.78})
        elif root != "running":
            nodes.set_status("root", root)
    return run


class Reached(Exception):
    pass


def test_resume_claims_the_loop_before_its_cleanup(tmp_path, monkeypatch):
    run = _existing_run(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(cli, "check_visible", lambda gpus: None)

    def cleanup(control):
        seen.append(Control(run).alive_pid())
        raise Reached
    monkeypatch.setattr(cli, "kill_recorded_groups", cleanup)
    with pytest.raises(Reached):
        cli.main(["run", "--run-id", "r1", "--resume"])
    assert seen == [os.getpid()]


def test_resume_resolves_gpus_from_the_runs_frozen_config(tmp_path, monkeypatch):
    _existing_run(tmp_path, monkeypatch, gpus_default="4,5,6,7")     # the live config says 0,1,2,3
    seen = []

    def visible(gpus):
        seen.append(gpus)
        raise Reached
    monkeypatch.setattr(cli, "check_visible", visible)
    with pytest.raises(Reached):
        cli.main(["run", "--run-id", "r1", "--resume"])
    assert seen == [[4, 5, 6, 7]]


def test_a_new_run_without_the_llm_settings_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(cli, "load_dotenv", lambda path, env: [])
    monkeypatch.setattr(cli, "check_visible", lambda gpus: None)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    assert cli.main(["run", "--run-id", "r2", "--max-nodes", "1"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err and not (tmp_path / "r2").exists()


def test_a_new_run_on_invisible_gpus_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")

    def invisible(gpus):
        raise GpuPolicyError("GPU(s) [3] are not visible")
    monkeypatch.setattr(cli, "check_visible", invisible)
    with pytest.raises(GpuPolicyError):
        cli.main(["run", "--run-id", "r2", "--max-nodes", "1"])
    assert not (tmp_path / "r2").exists()


def test_stop_without_a_running_loop_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["stop", "--run-id", "r1"]) == 0
    assert "no loop is running" in capsys.readouterr().out


@pytest.mark.parametrize("root", ["running", "eval_failed", None])
def test_resume_refuses_a_run_whose_root_was_never_scored(tmp_path, monkeypatch, capsys, root):
    run = _existing_run(tmp_path, monkeypatch, root=root)
    (run / "nodes" / "root").mkdir(parents=True)
    (run / "nodes" / "root" / "debug.txt").write_text("kept")
    assert cli.main(["run", "--run-id", "r1", "--resume"]) == 2
    assert "cannot be resumed" in capsys.readouterr().err
    assert Control(run).alive_pid() is None                        # refused before claiming the run
    assert (run / "nodes" / "root" / "debug.txt").read_text() == "kept"   # nothing cleaned up


def test_a_new_run_with_an_unreachable_git_remote_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(cli, "load_dotenv", lambda path, env: [])
    monkeypatch.setattr(cli, "check_visible", lambda gpus: None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    missing = tmp_path / "no_such_repo.git"
    assert cli.main(["run", "--run-id", "r2", "--max-nodes", "1", "--git-remote", str(missing)]) == 2
    assert "cannot reach --git-remote" in capsys.readouterr().err and not (tmp_path / "r2").exists()


def test_resume_pushes_to_the_git_remote_the_run_was_started_with(tmp_path, monkeypatch):
    run = _existing_run(tmp_path, monkeypatch)
    Control(run).save_args(git_remote="git@example.com:me/runs.git")
    monkeypatch.setattr(cli, "check_visible", lambda gpus: None)
    made = []

    class FakeRepo:
        check_remote = staticmethod(lambda url: None)

        def __init__(self, path, **kw):
            made.append(kw)

        def push(self):
            made.append("pushed")
    monkeypatch.setattr(cli, "AgentsRepo", FakeRepo)

    def cleanup(control):
        raise Reached
    monkeypatch.setattr(cli, "kill_recorded_groups", cleanup)
    with pytest.raises(Reached):
        cli.main(["run", "--run-id", "r1", "--resume"])
    assert made[0]["remote"] == "git@example.com:me/runs.git" and made[0]["namespace"] == "r1"
    assert made[1] == "pushed"


def test_resume_refuses_when_the_runs_git_remote_is_unreachable(tmp_path, monkeypatch, capsys):
    run = _existing_run(tmp_path, monkeypatch)
    Control(run).save_args(git_remote=str(tmp_path / "gone.git"))
    assert cli.main(["run", "--run-id", "r1", "--resume"]) == 2
    assert "cannot reach the run's git remote" in capsys.readouterr().err
    assert Control(run).alive_pid() is None


# ---- stop and resume around a kernel that will not exit (acceptance_20260928) ----

def _patch_until_drive(monkeypatch, drive):
    monkeypatch.setattr(cli, "check_visible", lambda gpus: None)
    monkeypatch.setattr(cli, "build_run_kit", lambda *a, **k: SimpleNamespace(budget=None))
    monkeypatch.setattr(cli, "Loop", lambda *a, **k: None)
    monkeypatch.setattr(cli, "Monitor", lambda *a, **k: None)
    monkeypatch.setattr(cli, "drive", drive)
    exits = []

    def exit_process(code, recorder):
        exits.append(code)
        raise Reached
    monkeypatch.setattr(cli, "exit_process", exit_process)
    return exits


@pytest.mark.parametrize("outcome,code", [("return", 0), ("raise", 1)])
def test_run_always_ends_the_process_after_drive(tmp_path, monkeypatch, outcome, code):
    _existing_run(tmp_path, monkeypatch)

    def drive(*a, **k):
        if outcome == "raise":
            raise RuntimeError("service thread(s) did not stop within the join timeout: ar-tools")
        return "force stop (SIGTERM)"
    exits = _patch_until_drive(monkeypatch, drive)
    with pytest.raises(Reached):
        cli.main(["run", "--run-id", "r1", "--resume"])
    assert exits == [code]


def test_resume_clears_partial_downloads_left_in_hf_tmp(tmp_path, monkeypatch):
    from ar_kernel.telemetry.recorder import Recorder
    run = _existing_run(tmp_path, monkeypatch)
    (run / "hf_tmp" / "6963b829").mkdir(parents=True)
    (run / "hf_tmp" / "6963b829" / "part.mp4").write_bytes(b"x")
    exits = _patch_until_drive(monkeypatch, lambda *a, **k: "max_nodes reached")
    with pytest.raises(Reached):
        cli.main(["run", "--run-id", "r1", "--resume"])
    assert exits == [0] and list((run / "hf_tmp").iterdir()) == []
    cleanup = [Recorder(run).load_payload(e["payload"]) for e in Recorder(run).read_events()
               if e["type"] == "run.resume_cleanup"]
    assert cleanup and cleanup[-1]["hf_tmp"] == ["6963b829"]


def _fake_loop(run, on_term):
    """A process holding the run's pid file; on SIGTERM it `exit`s after 1 s, or `ignore`s it."""
    import subprocess
    import sys
    from ar_kernel.subproc import proc_start_time
    handler = {"exit": "lambda *a: (time.sleep(1), sys.exit(0))", "ignore": "signal.SIG_IGN"}[on_term]
    proc = subprocess.Popen([sys.executable, "-c", "import signal, sys, time\n"
                             f"signal.signal(signal.SIGTERM, {handler})\nprint('up', flush=True)\n"
                             "time.sleep(60)\n"], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "up"
    (run / "control").mkdir(parents=True, exist_ok=True)
    (run / "control" / "loop.pid").write_text(f"{proc.pid} {proc_start_time(proc.pid)}")
    return proc


def test_force_stop_waits_until_the_process_has_exited(tmp_path, monkeypatch, capsys):
    run = _existing_run(tmp_path, monkeypatch)
    proc = _fake_loop(run, "exit")
    try:
        assert cli.main(["stop", "--run-id", "r1", "--force"]) == 0
        assert proc.poll() is not None                     # gone before stop returned
        assert "stopped" in capsys.readouterr().out
    finally:
        proc.kill()
        proc.wait()


def test_force_stop_says_when_the_process_outlives_the_wait(tmp_path, monkeypatch, capsys):
    run = _existing_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "FORCE_STOP_WAIT_S", 1.0)
    proc = _fake_loop(run, "ignore")
    try:
        assert cli.main(["stop", "--run-id", "r1", "--force"]) == 1
        err = capsys.readouterr().err
        assert f"still running (pid {proc.pid})" in err and "kill -9" in err
    finally:
        proc.kill()
        proc.wait()
