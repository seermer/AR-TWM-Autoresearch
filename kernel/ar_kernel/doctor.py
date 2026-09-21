"""Portability / environment preflight.

Everything that made this tree non-portable in practice, checked in one place.
The failure that motivated it: WBench's MegaSAM weight symlinks still pointed at
a previous checkout location after the tree moved, so every navigation case
failed, the tool still exited 0, and the run produced a report silently missing
five metrics.

Run: `ar doctor` (add --strict to exit non-zero on warnings).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import KernelConfig

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


def _broken_links(root: Path) -> list[tuple[Path, str]]:
    out = []
    for link in sorted(Path(root).rglob("*")):
        try:
            if link.is_symlink() and not link.exists():
                out.append((link, os.readlink(str(link))))
        except OSError:
            continue
    return out


def wbench_weight_problems(cfg: KernelConfig) -> list[str]:
    """Missing GPU-metric weights, or broken links under WBench/weights.

    Run by bootstrap_run as well as `ar doctor`: a moved checkout left MegaSAM's
    weight links dangling, every navigation case failed, and the report came out
    missing five metrics. Metric preflight only checked the VLM key and the VP
    weights, so nothing stopped the run starting.
    """
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
    """Broken links are the concrete way a move has broken this tree before."""
    out = []
    for root in (cfg.wbench, cfg.worldmodel, cfg.repo_root):
        for link in sorted(Path(root).rglob("*")):
            try:
                if not link.is_symlink():
                    continue
            except OSError:
                continue
            if link.exists():
                continue
            target = os.readlink(str(link))
            out.append(Finding("fail", "symlink",
                               f"broken: {link} -> {target} "
                               f"(typical after moving the tree; re-run the tool that creates it)"))
    return out or [Finding("ok", "symlink", "no broken symlinks")]


def _envs() -> list[Finding]:
    try:
        raw = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return [Finding("warn", "conda", f"could not list environments: {exc}")]
    names = {line.split()[0] for line in raw.splitlines()
             if line.strip() and not line.startswith("#")}
    return [Finding("ok", f"env.{e}", "present") if e in names
            else Finding("fail", f"env.{e}", "missing; create it before running")
            for e in ENVS]


def _tools() -> list[Finding]:
    return [Finding("ok", f"tool.{t}", shutil.which(t)) if shutil.which(t)
            else Finding("fail", f"tool.{t}", "not on PATH")
            for t in ("ffprobe", "ffmpeg", "conda", "git")]


def _dotenv(cfg: KernelConfig) -> list[Finding]:
    out = []
    for repo in (cfg.repo_root, cfg.worldmodel, cfg.wbench):
        p = Path(repo) / ".env"
        if not p.exists() and not p.is_symlink():
            out.append(Finding("warn", f"dotenv.{Path(repo).name}", "no .env; VLM metrics are excluded unless VLM_API_KEY is set in the shell"))
        elif p.is_symlink() and Path(os.readlink(str(p))).is_absolute():
            out.append(Finding("fail", f"dotenv.{Path(repo).name}",
                               f"symlink target is absolute ({os.readlink(str(p))}); use a relative target"))
        elif p.is_symlink() and not p.exists():
            out.append(Finding("fail", f"dotenv.{Path(repo).name}", "symlink is broken"))
        else:
            out.append(Finding("ok", f"dotenv.{Path(repo).name}", "present"))
    return out


def run_checks(cfg: KernelConfig | None = None) -> list[Finding]:
    cfg = cfg or KernelConfig.load()
    findings: list[Finding] = []
    findings += _rel_paths(cfg)
    findings += _siblings(cfg)
    findings += _tools()
    findings += _envs()
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
