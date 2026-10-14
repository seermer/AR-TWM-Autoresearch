"""Run alerts and the start-time GPU visibility check. There is deliberately no
wait for idle GPUs."""
from __future__ import annotations

import math

import pynvml

from .config import GpuPolicyError


def alert(recorder, kind: str, message: str, *, level: str = "error", **payload) -> None:
    recorder.event("alert", payload={"kind": kind, "level": level, "message": message, **payload},
                   kind=kind, level=level, message=message)


_nvml_ready = False


def nvml(read):
    """`read()` through NVML (initialised on first use), or None when the driver cannot be reached.
    NVML indices are nvidia-smi's, i.e. PCI bus order."""
    global _nvml_ready
    try:
        if not _nvml_ready:
            pynvml.nvmlInit()
            _nvml_ready = True
        return read()
    except pynvml.NVMLError:
        return None


def memory_used_mib(handle) -> int:
    """Used memory as nvidia-smi reports it: the v2 count, rounded up to whole MiB."""
    return math.ceil(pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2).used / 2**20)


def visible_gpus() -> set[int] | None:
    return nvml(lambda: set(range(pynvml.nvmlDeviceGetCount())))


def check_visible(gpus: list[int], visible=visible_gpus) -> None:
    seen = visible()
    if seen is None:
        return
    missing = [g for g in gpus if g not in seen]
    if missing:
        raise GpuPolicyError(f"GPU(s) {missing} are not visible to nvidia-smi (visible: {sorted(seen)})")


def gpu_total_gib(gpus: list[int]) -> dict[int, int] | None:
    """Each card's memory in whole GiB (a 24 GB card reports 23.99)."""
    return nvml(lambda: {g: round(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(g)).total / 2**30)
                         for g in gpus if g < pynvml.nvmlDeviceGetCount()})


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
