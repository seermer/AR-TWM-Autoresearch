"""Run alerts and the start-time GPU choice. There is deliberately no wait for idle GPUs."""
from __future__ import annotations

import re
import subprocess
from typing import Mapping

import psutil

from .config import GpuPolicyError
from .subproc import memory_cgroup, memory_limit


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


WORKER_TOOLS = ("annotate", "images", "generators.wan22", "generators.ltx25")


def check_tools_fit(cfg, gpus: list[int], totals=gpu_total_gib,
                    limit=lambda: memory_limit(memory_cgroup()), host_total=lambda: psutil.virtual_memory().total) -> None:
    """Raise when an enabled GPU tool does not fit the run: its `min_total_gpu_gib` (an estimate of the memory
    it needs across the run's cards) exceeds the cards, its `workers` (x `gpus_per_worker`) exceed the GPU
    count, or its host RAM (`workers` x `peak_rss_gib` + `shared_gib` + `host.kernel_reserve_gib`) exceeds
    `host.ram_gib`; or when this job's memory limit (without one, the host's RAM) is below `host.ram_gib`."""
    blocks = {"captioner": cfg.get("captioner") or {},
              **{k: b for k in ("annotate", "images") if (b := cfg.get(k) or {}).get("enabled")},
              **{f"generators.{k}": b for k, b in (cfg.get("generators") or {}).items()
                 if b.get("enabled")}}
    missing = [k for k in ("host.ram_gib", "host.kernel_reserve_gib", "captioner.tensor_parallel")
               if not isinstance(cfg.get(k), int)]
    missing += [f"{k}.workers" for k in WORKER_TOOLS if k in blocks and not isinstance(blocks[k].get("workers"), int)]
    if missing:
        raise GpuPolicyError(f"the config needs an int for {', '.join(missing)} (a run started before these keys "
                             "existed cannot be resumed)")
    problems = []
    if (tp := blocks["captioner"]["tensor_parallel"]) > len(gpus):
        problems.append(f"captioner.tensor_parallel {tp} > {len(gpus)} GPUs")
    for name, block in blocks.items():
        if "workers" in block and block["workers"] * block.get("gpus_per_worker", 1) > len(gpus):
            problems.append(f"{name}.workers {block['workers']} x {block.get('gpus_per_worker', 1)} GPUs > {len(gpus)} GPUs")
    ram, reserve = cfg.get("host.ram_gib"), cfg.get("host.kernel_reserve_gib")
    for name, block in blocks.items():
        if "peak_rss_gib" in block:
            need = block.get("workers", 1) * block["peak_rss_gib"] + block.get("shared_gib", 0) + reserve
            if need > ram:
                problems.append(f"{name} needs {need:g} GiB of host RAM (workers x peak_rss_gib + shared_gib + "
                                f"host.kernel_reserve_gib) > host.ram_gib {ram:g}")
    found = limit()
    if found and found[1] / 2**30 < ram:
        problems.append(f"this job's memory limit is {found[1] / 2**30:.0f} GiB ({found[0]}) < host.ram_gib {ram:g}")
    elif not found and host_total() / 2**30 < ram:
        problems.append(f"this host has {host_total() / 2**30:.0f} GiB of RAM < host.ram_gib {ram:g}")
    cards = totals(gpus)
    whole = sum(cards.values()) if cards else 0
    problems += [f"{name} needs {block['min_total_gpu_gib']} GiB of GPU memory, the GPUs have {whole}"
                 for name, block in blocks.items() if cards and block.get("min_total_gpu_gib", 0) > whole]
    if problems:
        raise GpuPolicyError(f"enabled tools do not fit this machine (GPUs {','.join(map(str, gpus))}): " + "; ".join(problems))
