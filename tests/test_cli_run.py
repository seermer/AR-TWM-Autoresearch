import os
from pathlib import Path

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
    (run / "config" / "run.json").write_text('{"metric_set": ["m"], "case_ids": ["1"], "versions": {}}')
    raw = yaml.safe_load((REPO / "configs" / "kernel.yaml").read_text())
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
