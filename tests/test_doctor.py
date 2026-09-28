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


def test_kernel_weight_table_matches_wbench_verify_install():
    """WBENCH_WEIGHTS mirrors WBench's own table; fail loudly if they drift."""
    import ast
    from ar_kernel.doctor import WBENCH_WEIGHTS
    src = (KernelConfig.load().wbench / "tools" / "verify_install.py").read_text()
    tree = ast.parse(src)
    upstream = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "weight_checks" for t in node.targets))
    assert WBENCH_WEIGHTS == upstream


def test_missing_gpu_metric_weight_is_reported(tmp_path):
    from ar_kernel.doctor import WBENCH_WEIGHTS, wbench_weight_problems
    root = _tree(tmp_path)
    weights = tmp_path / "WBench" / "weights"
    for rel in WBENCH_WEIGHTS.values():
        (weights / rel).parent.mkdir(parents=True, exist_ok=True)
        (weights / rel).write_bytes(b"w")
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert wbench_weight_problems(cfg) == []
    (weights / WBENCH_WEIGHTS["MegaSAM"]).unlink()
    assert any("MegaSAM" in p for p in wbench_weight_problems(cfg))


def test_broken_link_under_wbench_weights_is_reported(tmp_path):
    """Exactly the MegaSAM incident: hub/torchhub pointing at the old checkout."""
    from ar_kernel.doctor import WBENCH_WEIGHTS, wbench_weight_problems
    root = _tree(tmp_path)
    weights = tmp_path / "WBench" / "weights"
    for rel in WBENCH_WEIGHTS.values():
        (weights / rel).parent.mkdir(parents=True, exist_ok=True)
        (weights / rel).write_bytes(b"w")
    (weights / "hub").mkdir()
    os.symlink("/home/old-checkout/WBench/weights/torch_hub", str(weights / "hub" / "torchhub"))
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert any("torchhub" in p for p in wbench_weight_problems(cfg))


def test_missing_prefix_env_of_an_enabled_tool_is_a_failure(tmp_path):
    from ar_kernel.doctor import _prefix_envs
    cfg = KernelConfig(raw={"images": {"enabled": True, "env": ".envs/gen-zimage"},
                            "generators": {"wan22": {"env": ".envs/gen-wan22",
                                                     "variants": {"ti2v-5b": {"enabled": False}}}}},
                       repo_root=tmp_path)
    found = {f.check: f.level for f in _prefix_envs(cfg)}
    assert found == {"env.images": "fail"}                # disabled wan22 is not checked
    (tmp_path / ".envs" / "gen-zimage" / "conda-meta").mkdir(parents=True)
    assert {f.check: f.level for f in _prefix_envs(cfg)} == {"env.images": "ok"}


def test_env_in_the_project_is_ok_a_named_one_warns_and_a_missing_one_fails(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from ar_kernel.doctor import _envs
    cfg = KernelConfig(raw={"captioner": {"env": "vllm-env"}}, repo_root=tmp_path)
    (tmp_path / ".envs" / "alayaworld" / "conda-meta").mkdir(parents=True)
    listing = "# conda environments:\nbase  /x\nwbench-main  /y\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=listing))
    found = {f.check: f.level for f in _envs(cfg)}
    assert found["env.alayaworld"] == "ok"                 # lives in <repo>/.envs
    assert found["env.wbench-main"] == "warn"              # only a named env of the conda install
    assert found["env.wbench-vp"] == "fail" and found["env.vllm-env"] == "fail"


def test_dangling_links_in_the_package_cache_are_not_reported(tmp_path):
    root = _tree(tmp_path)
    pkgs = root / ".cache" / "conda" / "pkgs" / "python-3.12" / "compiler_compat"
    pkgs.mkdir(parents=True)
    os.symlink("../bin/ld", str(pkgs / "ld"))                       # dangling by design
    cfg = KernelConfig.load(root / "configs" / "kernel.yaml")
    assert _levels(run_checks(cfg), "symlink") == {"ok"}
