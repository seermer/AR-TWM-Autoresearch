"""`ar doctor`: everything that has made this tree non-portable in practice, checked in one
place (e.g. WBench weight links left dangling by a moved checkout silently dropped five metrics)."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import KernelConfig
from .subproc import project_env

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


def _broken_links(root: Path, skip: Path | None = None) -> list[tuple[Path, str]]:
    """(link, target) for every dangling symlink under `root`, except those under `skip`."""
    out = []
    for link in sorted(Path(root).rglob("*")):
        if skip is not None and skip in link.parents:
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
    only once a package is linked into an env."""
    skip = Path(cfg.repo_root) / ".cache"
    out = [Finding("fail", "symlink", f"broken: {link} -> {target} "
                                      f"(typical after moving the tree; re-run the tool that creates it)")
           for root in (cfg.wbench, cfg.worldmodel, cfg.repo_root) for link, target in _broken_links(root, skip)]
    return out or [Finding("ok", "symlink", "no broken symlinks")]


def _envs(cfg: KernelConfig) -> list[Finding]:
    """Every env the project runs in must live in `<repo>/.envs/<name>` (user rule, 2026-09-28).
    A named env of the conda installation still works but is reported, so the tree is not
    quietly depending on something outside it."""
    try:
        raw = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=60).stdout
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
                                                   "move it to .envs/ (docs/PORTABILITY.md)"))
        elif listing_error:
            out.append(Finding("warn", f"env.{e}", f"not in .envs/ and conda could not list environments: {listing_error}"))
        else:
            out.append(Finding("fail", f"env.{e}", "missing; create it in .envs/ before running"))
    return out


def _tools() -> list[Finding]:
    return [Finding("ok", f"tool.{t}", shutil.which(t)) if shutil.which(t)
            else Finding("fail", f"tool.{t}", "not on PATH")
            for t in ("ffprobe", "ffmpeg", "conda", "git")]


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
    """Enabled tools whose `env` is a conda-prefix path inside the repo (Plan 3 disk ruling)."""
    blocks = {"annotate": cfg.get("annotate") or {}, "images": cfg.get("images") or {},
              **{f"generators.{k}": v for k, v in (cfg.get("generators") or {}).items()}}
    out = []
    for name, block in blocks.items():
        env = str(block.get("env") or "")
        variants = block.get("variants")
        enabled = (any((v or {}).get("enabled") for v in variants.values()) if variants
                   else bool(block.get("enabled")))
        if "/" not in env or not enabled:
            continue
        path = Path(env) if Path(env).is_absolute() else cfg.repo_root / env
        out.append(Finding("ok", f"env.{name.split('.')[-1]}", str(path))
                   if (path / "conda-meta").is_dir()
                   else Finding("fail", f"env.{name.split('.')[-1]}",
                                f"prefix env missing at {path}; see docs/PORTABILITY.md"))
    return out


def run_checks(cfg: KernelConfig | None = None) -> list[Finding]:
    cfg = cfg or KernelConfig.load()
    findings: list[Finding] = []
    findings += _rel_paths(cfg)
    findings += _siblings(cfg)
    findings += _tools()
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
