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
