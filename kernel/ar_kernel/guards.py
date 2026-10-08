"""Run alerts and the start-time GPU visibility check. There is deliberately no
wait for idle GPUs (user decision 2026-09-27)."""
from __future__ import annotations

import subprocess

from .config import GpuPolicyError


def alert(recorder, kind: str, message: str, *, level: str = "error", **payload) -> None:
    recorder.event("alert", payload={"kind": kind, "level": level, "message": message, **payload},
                   kind=kind, level=level, message=message)


def smi(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def visible_gpus() -> set[int] | None:
    out = smi(["--query-gpu=index", "--format=csv,noheader"])
    return None if out is None else {int(x) for x in out.split() if x.strip().isdigit()}


def check_visible(gpus: list[int], visible=visible_gpus) -> None:
    seen = visible()
    if seen is None:
        return
    missing = [g for g in gpus if g not in seen]
    if missing:
        raise GpuPolicyError(f"GPU(s) {missing} are not visible to nvidia-smi (visible: {sorted(seen)})")


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
