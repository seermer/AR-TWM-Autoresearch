from __future__ import annotations
import shutil, subprocess, time
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env

def available_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")

def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def wait_for_ram(cfg: KernelConfig, recorder, node_id: str, threshold_gb: float | None = None,
                 poll_s: float = 5.0, alert_after_s: float | None = None) -> None:
    threshold = threshold_gb if threshold_gb is not None else float(cfg.get("eval.free_ram_before_render_gb"))
    deadline = alert_after_s if alert_after_s is not None else float(cfg.get("eval.ram_wait_alert_min")) * 60
    subprocess.run(["sync"], check=True)
    started = time.monotonic()
    while available_ram_gb() < threshold:
        if time.monotonic() - started > deadline:
            recorder.event("eval.ram_wait_alert", node=node_id, phase="eval",
                           payload={"available_gb": available_ram_gb(), "threshold_gb": threshold})
            raise TimeoutError(f"host RAM stayed below {threshold} GB for {deadline}s")
        time.sleep(poll_s)

def merge_lora(cfg: KernelConfig, checkpoint: Path, rank: int, alpha: int, run_dir: Path,
               recorder, node_id: str) -> Path:
    slot = Path(run_dir) / "merge_slot"
    if slot.exists():
        shutil.rmtree(slot)
    minimum = float(cfg.get("disk.merge_min_free_gb"))
    if free_disk_gb(Path(run_dir)) < minimum:
        raise RuntimeError(f"less than {minimum} GB free; refusing to merge")
    with recorder.span("merge", node=node_id, phase="eval", payload={"checkpoint": str(checkpoint)}):
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/merge_lora_for_rollout.py",
             "--ckpt_dir", str(checkpoint),
             "--base_transformer", str(cfg.worldmodel / "weights/alaya-world-ar/transformer.pt"),
             "--output", str(slot), "--lora_rank", str(rank), "--lora_alpha", str(alpha)],
            cwd=cfg.worldmodel, timeout=7200, recorder=recorder, node=node_id, phase="eval")
    if proc.returncode != 0:
        raise RuntimeError(f"merge failed (rc={proc.returncode}): {proc.stdout[-4000:]}")
    wait_for_ram(cfg, recorder, node_id)
    return slot
