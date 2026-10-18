"""`ar doctor`: everything that has made this tree non-portable in practice, checked in one
place (e.g. WBench weight links left dangling by a moved checkout silently dropped five metrics)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import GpuPolicyError, KernelConfig
from .guards import check_tools_fit, choose_gpus, device_minors
from .sandbox.runner import Mounts, _bwrap_args
from .subproc import cache_env, project_env

ENVS = ("alayaworld", "wbench-main", "wbench-vp", "autoresearcher")


@dataclass
class Finding:
    level: str          # "ok" | "warn" | "fail"
    check: str
    detail: str


# WBench's required GPU-metric weights, relative to WBench/weights. Mirrors
# `weight_checks` in WBench/tools/verify_install.py; tests/test_doctor.py fails if
# the two drift apart.
WBENCH_WEIGHTS = {
    "CLIP ViT-L/14": "clip/ViT-L-14.pt",
    "Aesthetic": "aesthetic/sa_0_4_vit_l_14_linear.pth",
    "RAFT": "raft/raft-things.pth",
    "TransNetV2": "transnetv2/transnetv2-pytorch-weights.pth",
    "DreamSim": "dreamsim/dino_vitb16_pretrain.pth",
    "HPSv3": "HPSv3/HPSv3.safetensors",
    "SAM2 (native .pt)": "sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt",
    "MegaSAM": "megasam/megasam_final.pth",
    "DA3-GIANT-1.1": "DA3-GIANT-1.1/config.json",
    "Qwen2-VL (HPSv3 base)": "Qwen2-VL-7B-Instruct/config.json",
}


def _broken_links(root: Path, skip: tuple[Path, ...] = ()) -> list[tuple[Path, str]]:
    """(link, target) for every dangling symlink under `root`, except those under a `skip` folder."""
    out = []
    for link in sorted(Path(root).rglob("*")):
        if any(folder in link.parents for folder in skip):
            continue
        try:
            if link.is_symlink() and not link.exists():
                out.append((link, os.readlink(str(link))))
        except OSError:
            continue
    return out


def wbench_weight_problems(cfg: KernelConfig) -> list[str]:
    """Missing GPU-metric weights, or broken links under WBench/weights. bootstrap_run refuses
    to start a run on any of these, as well as `ar doctor` reporting them."""
    weights = cfg.wbench / "weights"
    problems = [f"{name} weights missing: {weights / rel}"
                for name, rel in WBENCH_WEIGHTS.items() if not (weights / rel).exists()]
    problems += [f"broken symlink under WBench/weights: {link} -> {target}"
                 for link, target in _broken_links(weights)]
    return problems


def _wbench_weights(cfg: KernelConfig) -> list[Finding]:
    problems = wbench_weight_problems(cfg) if (cfg.wbench / "weights").is_dir() else \
        [f"{cfg.wbench / 'weights'} does not exist"]
    return [Finding("fail", "wbench.weights", p) for p in problems] or \
        [Finding("ok", "wbench.weights", "all required GPU-metric weights present")]


def _rel_paths(cfg: KernelConfig) -> list[Finding]:
    out = []
    for key in ("paths.worldmodel", "paths.wbench", "paths.runs_dir"):
        raw = str(cfg.get(key))
        if Path(raw).is_absolute():
            out.append(Finding("fail", key,
                               f"absolute path {raw!r}: pins the tree to this machine, "
                               f"use a path relative to the AutoResearcher repo"))
        else:
            out.append(Finding("ok", key, raw))
    return out


def _siblings(cfg: KernelConfig) -> list[Finding]:
    out = []
    for name, p in (("worldmodel", cfg.worldmodel), ("wbench", cfg.wbench)):
        if not p.is_dir():
            out.append(Finding("fail", f"layout.{name}", f"{p} is missing"))
        elif p.resolve().parent != cfg.repo_root.resolve().parent:
            out.append(Finding("warn", f"layout.{name}",
                               f"{p.resolve()} is not a sibling of {cfg.repo_root.resolve()}; "
                               f"the relative paths in kernel.yaml assume a side-by-side layout"))
        else:
            out.append(Finding("ok", f"layout.{name}", str(p.resolve())))
    return out


def _symlinks(cfg: KernelConfig) -> list[Finding]:
    """Broken links are the concrete way a move has broken this tree before. The package cache
    under `.cache/` is skipped: conda's extracted packages hold relative links whose targets exist
    only once a package is linked into an env. Runs are skipped too: agents leave links in their
    workspaces that point at container paths."""
    skip = (Path(cfg.repo_root) / ".cache", cfg.runs_dir, Path(cfg.repo_root) / ".obsolete_runs")
    out = [Finding("fail", "symlink", f"broken: {link} -> {target} "
                                      f"(typical after moving the tree; re-run the tool that creates it)")
           for root in (cfg.wbench, cfg.worldmodel, cfg.repo_root) for link, target in _broken_links(root, skip)]
    return out or [Finding("ok", "symlink", "no broken symlinks")]


def _envs(cfg: KernelConfig) -> list[Finding]:
    """Every env the project runs in must live in `<repo>/.envs/<name>`.
    A named env of the conda installation still works but is reported, so the tree is not
    quietly depending on something outside it."""
    try:
        raw = subprocess.run(["conda", "env", "list"], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raw, listing_error = "", str(exc)
    else:
        listing_error = None
    names = {line.split()[0] for line in raw.splitlines()
             if line.strip() and not line.startswith("#")}
    out = []
    for e in dict.fromkeys((*ENVS, str(cfg.get("captioner.env") or ""))):
        if not e or "/" in e:
            continue
        local = project_env(e, cfg.repo_root)
        if local:
            out.append(Finding("ok", f"env.{e}", str(local)))
        elif e in names:
            out.append(Finding("warn", f"env.{e}", "a named conda env outside the project; "
                                                   "move it to .envs/"))
        elif listing_error:
            out.append(Finding("warn", f"env.{e}", f"not in .envs/ and conda could not list environments: {listing_error}"))
        else:
            out.append(Finding("fail", f"env.{e}", "missing; create it in .envs/ before running"))
    return out


def _tools() -> list[Finding]:
    return [Finding("ok", f"tool.{t}", shutil.which(t)) if shutil.which(t)
            else Finding("fail", f"tool.{t}", "not on PATH")
            for t in ("ffprobe", "ffmpeg", "conda", "git", "curl", "wget", "unzip", "gcc")]   # + the agent's host tools


def _in_sandbox(cfg: KernelConfig, command: list[str], gpus=()) -> subprocess.CompletedProcess:
    """`command` in an agent sandbox over empty folders."""
    with tempfile.TemporaryDirectory(dir=cache_env()["TMPDIR"]) as tmp:
        dirs = {k: Path(tmp) / k for k in ("agent", "workspace", "staging", "context", "sockets")}
        for d in dirs.values():
            d.mkdir()
        mounts = Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"],
                        context=dirs["context"], contract=cfg.repo_root / "contract", sockets=dirs["sockets"],
                        agent_readonly=True)
        argv, env = _bwrap_args(mounts, command, {}, cfg.repo_root / ".envs" / "agent", 1, device_minors(list(gpus)))
        return subprocess.run(argv, env=env, capture_output=True, text=True)


def _sandbox(cfg: KernelConfig) -> list[Finding]:
    """Agents run in bwrap with .envs/agent, on the run's cards: the first gpus.num_gpus free ones."""
    if not shutil.which("bwrap"):
        return [Finding("fail", "bwrap", "not on PATH: install bubblewrap")]
    if project_env("agent", cfg.repo_root) is None:
        return [Finding("fail", "env.agent", "no .envs/agent: run scripts/setup_envs.sh agent")]
    out = []
    r = _in_sandbox(cfg, ["true"])
    if r.returncode != 0:
        return [Finding("fail", "bwrap", f"a sandbox does not start (unprivileged user namespaces allowed?): "
                                         f"{r.stderr.strip()}")]
    out.append(Finding("ok", "bwrap", "a sandbox starts"))
    r = _in_sandbox(cfg, ["python", "-c", "import ar_contract.run, ar_contract.client"])
    out.append(Finding("ok", "env.agent", "imports the contract in a sandbox") if r.returncode == 0
               else Finding("fail", "env.agent", f"cannot import the contract in a sandbox: {r.stderr.strip()[-2000:]}"))
    try:
        gpus = choose_gpus(cfg, os.environ)
    except GpuPolicyError as exc:
        return out + [Finding("fail", "gpus.choice", str(exc))]
    out.append(Finding("ok", "gpus.choice", f"GPUs {gpus}"))
    try:
        r = _in_sandbox(cfg, ["nvidia-smi", "-L"], gpus)
    except GpuPolicyError as exc:
        return out + [Finding("fail", "gpus.sandbox", str(exc))]
    seen = [line for line in r.stdout.splitlines() if line.startswith("GPU ")]
    out.append(Finding("ok", "gpus.sandbox", f"a sandbox sees {len(seen)} GPUs") if r.returncode == 0 and len(seen) == len(gpus)
               else Finding("fail", "gpus.sandbox", f"a sandbox on GPUs {gpus} sees {len(seen)}: {r.stdout}{r.stderr}".strip()))
    return out


def _fit(cfg: KernelConfig) -> list[Finding]:
    try:
        check_tools_fit(cfg, choose_gpus(cfg, os.environ))
    except GpuPolicyError as exc:
        return [Finding("fail", "gpus.fit", str(exc))]
    return [Finding("ok", "gpus.fit", "every enabled tool's estimate fits the run's GPUs")]


def _dotenv(cfg: KernelConfig) -> list[Finding]:
    out = []
    for repo in (cfg.repo_root, cfg.worldmodel, cfg.wbench):
        p = Path(repo) / ".env"
        if not p.exists() and not p.is_symlink():
            out.append(Finding("warn", f"dotenv.{Path(repo).name}", "no .env; without VLM_API_KEY in the shell the VLM metrics use the local judge"))
        elif p.is_symlink() and Path(os.readlink(str(p))).is_absolute():
            out.append(Finding("fail", f"dotenv.{Path(repo).name}",
                               f"symlink target is absolute ({os.readlink(str(p))}); use a relative target"))
        elif p.is_symlink() and not p.exists():
            out.append(Finding("fail", f"dotenv.{Path(repo).name}", "symlink is broken"))
        else:
            out.append(Finding("ok", f"dotenv.{Path(repo).name}", "present"))
    return out


def _prefix_envs(cfg: KernelConfig) -> list[Finding]:
    """Enabled tools whose `env` is a conda-prefix path inside the repo."""
    blocks = {"annotate": cfg.get("annotate") or {}, "images": cfg.get("images") or {},
              **{f"generators.{k}": v for k, v in (cfg.get("generators") or {}).items()}}
    out = []
    for name, block in blocks.items():
        env = str(block.get("env") or "")
        enabled = bool(block.get("enabled") or block.get("variants"))
        if "/" not in env or not enabled:
            continue
        path = Path(env) if Path(env).is_absolute() else cfg.repo_root / env
        out.append(Finding("ok", f"env.{name.split('.')[-1]}", str(path))
                   if (path / "conda-meta").is_dir()
                   else Finding("fail", f"env.{name.split('.')[-1]}",
                                f"prefix env missing at {path}"))
    return out


def run_checks(cfg: KernelConfig | None = None) -> list[Finding]:
    cfg = cfg or KernelConfig.load()
    findings: list[Finding] = []
    findings += _rel_paths(cfg)
    findings += _siblings(cfg)
    findings += _tools()
    findings += _sandbox(cfg)
    findings += _fit(cfg)
    findings += _envs(cfg)
    findings += _prefix_envs(cfg)
    findings += _dotenv(cfg)
    findings += _symlinks(cfg)
    findings += _wbench_weights(cfg)
    return findings


def report(findings: list[Finding], strict: bool = False) -> int:
    for f in findings:
        if f.level == "ok":
            continue
        print(f"[{f.level.upper():>4}] {f.check}: {f.detail}")
    n_fail = sum(f.level == "fail" for f in findings)
    n_warn = sum(f.level == "warn" for f in findings)
    print(f"\n{len(findings)} checks, {n_fail} failed, {n_warn} warnings"
          + ("" if n_fail or n_warn else " - tree looks portable"))
    return 1 if n_fail or (strict and n_warn) else 0
