"""Portability checks must FAIL on a non-portable tree.

A check that only ever passes proves nothing, so every case here builds the
broken condition and asserts it is caught. The motivating incident: WBench's
MegaSAM weight symlinks still pointed at a previous checkout after the tree
moved, every navigation case failed, the tool exited 0 anyway, and the run
produced a report silently missing five metrics.
"""
import os

import pytest
import yaml

from ar_kernel.config import KernelConfig
from ar_kernel.doctor import report, run_checks


def _tree(tmp_path, *, worldmodel="../WorldModel", wbench="../WBench"):
    """A minimal sibling layout: <root>/{AutoResearcher,WorldModel,WBench}."""
    root = tmp_path / "AutoResearcher"
    (root / "configs").mkdir(parents=True)
    (tmp_path / "WorldModel").mkdir()
    (tmp_path / "WBench").mkdir()
    (root / "configs" / "kernel.yaml").write_text(yaml.safe_dump({
        "paths": {"worldmodel": worldmodel, "wbench": wbench, "runs_dir": "runs"},
        "gpus": {"default": "0,1,2,3", "min_count": 4},
    }))
    return root


def _levels(findings, check_prefix):
    return {f.level for f in findings if f.check.startswith(check_prefix)}


def test_relative_paths_pass(tmp_path):
    cfg = KernelConfig.load(_tree(tmp_path) / "configs" / "kernel.yaml")
    assert _levels(run_checks(cfg), "paths.") == {"ok"}


def test_absolute_path_in_config_is_a_failure(tmp_path):
    """An absolute path pins the tree to one machine."""
    root = _tree(tmp_path, worldmodel=str(tmp_path / "WorldModel"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    findings = run_checks(cfg)
    assert "fail" in _levels(findings, "paths.worldmodel")
    assert any("absolute" in f.detail for f in findings if f.check == "paths.worldmodel")


def test_broken_symlink_is_a_failure(tmp_path):
    """Exactly the MegaSAM case: a link left pointing at the old location."""
    root = _tree(tmp_path)
    os.symlink(str(tmp_path / "gone" / "weights"), str(tmp_path / "WBench" / "torchhub"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    findings = run_checks(cfg)
    assert "fail" in _levels(findings, "symlink")
    assert any("torchhub" in f.detail for f in findings if f.check == "symlink")


def test_intact_symlink_passes(tmp_path):
    root = _tree(tmp_path)
    (tmp_path / "WBench" / "real").mkdir()
    os.symlink(str(tmp_path / "WBench" / "real"), str(tmp_path / "WBench" / "link"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert _levels(run_checks(cfg), "symlink") == {"ok"}


def test_missing_sibling_repo_is_a_failure(tmp_path):
    root = _tree(tmp_path)
    (tmp_path / "WBench").rmdir()
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert "fail" in _levels(run_checks(cfg), "layout.wbench")


def test_absolute_dotenv_symlink_is_a_failure(tmp_path):
    """A relative .env link survives a move; an absolute one does not."""
    root = _tree(tmp_path)
    (root / ".env").write_text("OPENAI_API_KEY=x\n")
    os.symlink(str(root / ".env"), str(tmp_path / "WBench" / ".env"))   # absolute
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    findings = run_checks(cfg)
    assert "fail" in _levels(findings, "dotenv.WBench")


def test_relative_dotenv_symlink_passes(tmp_path):
    root = _tree(tmp_path)
    (root / ".env").write_text("OPENAI_API_KEY=x\n")
    os.symlink("../AutoResearcher/.env", str(tmp_path / "WBench" / ".env"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert _levels(run_checks(cfg), "dotenv.WBench") == {"ok"}


def test_report_exit_codes(tmp_path, capsys):
    root = _tree(tmp_path, worldmodel=str(tmp_path / "WorldModel"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert report(run_checks(cfg)) == 1          # a failure is non-zero
    clean = KernelConfig.load(_tree(tmp_path / "clean") / "configs" / "kernel.yaml")
    findings = [f for f in run_checks(clean) if f.level != "fail"]
    assert report(findings) == 0
    assert report(findings, strict=True) == (1 if any(f.level == "warn" for f in findings) else 0)
