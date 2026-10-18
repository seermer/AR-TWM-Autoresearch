"""Run alerts and the start-time GPU choice. There is deliberately no wait for idle GPUs."""
from __future__ import annotations

import re
import subprocess
from typing import Mapping

from .config import GpuPolicyError


def alert(recorder, kind: str, message: str, *, level: str = "error", **payload) -> None:
    recorder.event("alert", payload={"kind": kind, "level": level, "message": message, **payload},
                   kind=kind, level=level, message=message)


def smi(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _nvidia_smi(args: list[str]) -> str:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    except OSError as exc:
        raise GpuPolicyError(f"nvidia-smi failed: {exc}") from exc
    if r.returncode != 0:
        raise GpuPolicyError(f"nvidia-smi {' '.join(args)} exited {r.returncode}: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def gpu_memory_used() -> dict[int, int]:
    """Memory in use on every card, in MiB, by nvidia-smi index."""
    out = _nvidia_smi(["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"])
    return {int(i): int(m) for i, m in (line.split(",") for line in out.splitlines() if line.strip())}


def choose_gpus(cfg, env: Mapping[str, str], used=None) -> list[int]:
    """The first `gpus.num_gpus` cards (nvidia-smi index) holding at most `gpus.free_below_mib`."""
    if env.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise GpuPolicyError("unset CUDA_VISIBLE_DEVICES: the run picks its cards by gpus.num_gpus")
    count, group = cfg.get("gpus.num_gpus"), int(cfg.get("gpus.min_count"))
    if not isinstance(count, int) or count <= 0 or count % group:
        raise GpuPolicyError(f"gpus.num_gpus must be a positive multiple of {group}, got {count!r}")
    limit = int(cfg.get("gpus.free_below_mib"))
    memory = (used or gpu_memory_used)()
    free = [g for g in sorted(memory) if memory[g] <= limit]
    if len(free) < count:
        raise GpuPolicyError(f"need {count} free GPUs (at most {limit} MiB in use), found {len(free)}: "
                             + ", ".join(f"GPU {g} {m} MiB" for g, m in sorted(memory.items())))
    return free[:count]


def device_minors(gpus: list[int]) -> list[int]:
    """Each card's /dev/nvidia<minor> number; it can differ from the nvidia-smi index."""
    out = []
    for g in gpus:
        found = re.search(r"Minor Number\s*:\s*(\d+)", _nvidia_smi(["-q", "-i", str(g)]))
        if found is None:
            raise GpuPolicyError(f"nvidia-smi -q -i {g} shows no Minor Number")
        out.append(int(found.group(1)))
    return out


def gpu_total_gib(gpus: list[int]) -> dict[int, int] | None:
    """Each card's memory in whole GiB (a 24 GB card reports 23.99)."""
    out = smi(["--query-gpu=index,memory.total", "--format=csv,noheader,nounits"])
    if out is None:
        return None
    rows = [[x.strip() for x in line.split(",")] for line in out.splitlines()]
    total = {int(r[0]): round(int(r[1]) / 1024) for r in rows if len(r) == 2 and r[0].isdigit() and r[1].isdigit()}
    return {g: total[g] for g in gpus if g in total}


def check_tools_fit(cfg, gpus: list[int], totals=gpu_total_gib) -> None:
    """Raise when an enabled GPU tool's `min_total_gpu_gib` (its config block's estimate of the
    memory it needs across the run's cards) exceeds what the run's cards have: the run would
    otherwise start and the tool fail for the agent."""
    cards = totals(gpus)
    if not cards:
        return
    blocks = {"captioner": cfg.get("captioner") or {},
              **{k: b for k in ("annotate", "images") if (b := cfg.get(k) or {}).get("enabled")},
              **{f"generators.{k}": b for k, b in (cfg.get("generators") or {}).items()
                 if b.get("enabled") or b.get("variants")}}
    whole = sum(cards.values())
    problems = [f"{name} needs {block['min_total_gpu_gib']} GiB" for name, block in blocks.items()
                if block.get("min_total_gpu_gib", 0) > whole]
    if problems:
        raise GpuPolicyError(f"enabled tools do not fit GPUs {','.join(map(str, gpus))} ({whole} GiB in total): "
                             + "; ".join(problems) + ". Give the run larger or more GPUs.")
