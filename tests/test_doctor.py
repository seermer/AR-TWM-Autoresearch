"""Portability checks must FAIL on a non-portable tree.

A check that only ever passes proves nothing, so every case here builds the
broken condition and asserts it is caught.
"""
import os
import subprocess

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
        "gpus": {"num_gpus": 4, "min_count": 4, "free_below_mib": 512},
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
    """A link left pointing at the old location is reported."""
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
    """hub/torchhub pointing at the old checkout is reported."""
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
                            "generators": {"wan22": {"env": ".envs/gen-wan22", "enabled": False}}},
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


def test_links_an_agent_left_in_a_run_are_not_tree_problems(tmp_path):
    """Links made by agents in their workspaces and pointing at container paths are not reported as broken."""
    from ar_kernel.doctor import _broken_links
    for folder in ("runs/r1/nodes/n1/workspace", ".obsolete_runs/r0", "weights"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "link.mp4").symlink_to("/workspace/work/clips/a.mp4")
    found = _broken_links(tmp_path, (tmp_path / "runs", tmp_path / ".obsolete_runs"))
    assert [link for link, _ in found] == [tmp_path / "weights" / "link.mp4"]


def test_sandbox_checks_name_what_is_missing(monkeypatch):
    from ar_kernel import doctor
    cfg = KernelConfig.load()
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert [(f.level, f.check) for f in doctor._sandbox(cfg)] == [("fail", "bwrap")]
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(doctor, "project_env", lambda name, root: None)
    assert [(f.level, f.check) for f in doctor._sandbox(cfg)] == [("fail", "env.agent")]


def test_sandbox_checks_run_the_contract_import_and_count_the_chosen_gpus(monkeypatch):
    from ar_kernel import doctor
    from ar_kernel.config import GpuPolicyError
    cfg = KernelConfig.load()
    seen = []

    def sandbox(cfg, command, gpus=()):
        seen.append((command, list(gpus)))
        out = "".join(f"GPU {g}: card\n" for g in gpus) if command[0] == "nvidia-smi" else ""
        return subprocess.CompletedProcess(command, 0, out, "")
    monkeypatch.setattr(doctor, "_in_sandbox", sandbox)
    assert {f.check: f.level for f in doctor._sandbox(cfg)} == {
        "bwrap": "ok", "env.agent": "ok", "gpus.choice": "ok", "gpus.sandbox": "ok"}
    assert seen[1][0][:2] == ["python", "-c"] and seen[2] == (["nvidia-smi", "-L"], [0, 1, 2, 3])

    def busy(cfg, env):
        raise GpuPolicyError("need 4 free GPUs (at most 512 MiB in use), found 2: GPU 0 9000 MiB")
    monkeypatch.setattr(doctor, "choose_gpus", busy)
    found = {f.check: f for f in doctor._sandbox(cfg)}
    assert found["gpus.choice"].level == "fail" and "found 2" in found["gpus.choice"].detail
    assert "gpus.sandbox" not in found


@pytest.mark.sandbox
def test_a_real_sandbox_starts_and_imports_the_contract():
    from ar_kernel import doctor
    r = doctor._in_sandbox(KernelConfig.load(), ["python", "-c", "import ar_contract.run, ar_contract.client"])
    assert r.returncode == 0, r.stderr


def test_a_tool_that_does_not_fit_the_gpus_is_a_failure(monkeypatch):
    from ar_kernel import doctor
    from ar_kernel.config import GpuPolicyError
    cfg = KernelConfig.load()
    monkeypatch.setattr(doctor, "check_tools_fit", lambda cfg, gpus: None)
    assert [f.level for f in doctor._fit(cfg)] == ["ok"]
    monkeypatch.setattr(doctor, "check_tools_fit", lambda cfg, gpus: (_ for _ in ()).throw(
        GpuPolicyError("enabled tools do not fit GPUs 0,1: generators.h3 needs 70 GiB")))
    [found] = doctor._fit(cfg)
    assert (found.level, found.check) == ("fail", "gpus.fit") and "generators.h3 needs 70 GiB" in found.detail


def _torch_env(root, name, cuda):
    v = root / ".envs" / name / "lib" / "python3.10" / "site-packages" / "torch"
    v.mkdir(parents=True)
    (root / ".envs" / name / "conda-meta").mkdir()
    (v / "version.py").write_text(f"__version__ = '2.4.0'\ncuda: Optional[str] = '{cuda}'\n")


def test_gpu_checks_fail_on_mixed_or_unsupported_cards(monkeypatch):
    from ar_kernel import doctor
    cfg = KernelConfig.load()
    for smi_out in ("NVIDIA A100, 8.0\nNVIDIA H100, 9.0\n", "Tesla V100, 7.0\n"):
        monkeypatch.setattr(doctor, "smi", lambda args, out=smi_out: out)
        [found] = doctor.gpu_checks(cfg)
        assert (found.level, found.check) == ("fail", "gpu.host")


def test_gpu_checks_fail_an_env_built_for_a_newer_cuda_than_the_driver(tmp_path, monkeypatch):
    from ar_kernel import doctor
    cfg = KernelConfig.load(_tree(tmp_path) / "configs" / "kernel.yaml")
    _torch_env(cfg.repo_root, "vllm", "13.2")
    _torch_env(cfg.repo_root, "alayaworld", "12.8")
    _torch_env(cfg.repo_root, "agent", "12.4")
    monkeypatch.setattr(doctor, "smi", lambda args: "NVIDIA H100 80GB HBM3, 9.0\n" * 4)
    monkeypatch.setattr(doctor, "driver_cuda", lambda: (12, 8))
    monkeypatch.setattr(doctor, "choose_gpus", lambda cfg, env: [2, 3, 4, 5])
    probed = []

    def probe(env, gpu, snippets):
        probed.append((env.name, gpu))
        return subprocess.CompletedProcess([], 1 if env.name == "agent" else 0, "", "no kernel image")
    monkeypatch.setattr(doctor, "_probe", probe)
    found = {f.check: f for f in doctor.gpu_checks(cfg)}
    assert {k: f.level for k, f in found.items()} == {
        "gpu.host": "ok", "gpu.driver": "ok", "gpu.agent": "fail", "gpu.alayaworld": "ok", "gpu.vllm": "fail"}
    assert "CUDA 13.2" in found["gpu.vllm"].detail and "no kernel image" in found["gpu.agent"].detail
    assert sorted(probed) == [("agent", 2), ("alayaworld", 2)]
