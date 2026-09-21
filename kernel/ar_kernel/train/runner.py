from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env, _tail

RECIPE_SIGNATURES = (
    "CUDA out of memory", "torch.OutOfMemoryError", "loss=nan", "loss=inf",
    "no enabled data sources", "Loaded 0 samples",
)
# Match only tokens that a FAILING run emits. A healthy multi-GPU run prints the
# "NCCL version ..." banner and several ProcessGroupNCCL.cpp warnings, so a bare
# "NCCL" substring classifies every successful distributed run as an infra failure.
INFRA_SIGNATURES = (
    "exitcode: -9", "No space left on device", "CUDA driver error", "Killed",
    "NCCL error", "ncclInternalError", "ncclSystemError", "ncclUnhandledCudaError",
    "ncclRemoteError", "NCCL communicator was aborted",
    "Watchdog caught collective operation timeout",
    "DistBackendError",
    # loader.py::_text_encoder_disabled, raised when the precache is incomplete.
    "missed the on-disk embedding cache",
)
TRAIN_LINE = re.compile(
    r"\[Train\] step=(?P<step>\d+) epoch=(?P<epoch>\d+).*?"
    r"loss=(?P<loss>[\d.naif]+) grad=(?P<grad>[\d.naif]+) lr=(?P<lr>[\d.e+-]+) time=(?P<time>[\d.]+)s"
)

@dataclass
class TrainOutcome:
    checkpoint: Path | None
    failure: str
    log_path: Path
    metrics: list[dict] = field(default_factory=list)

def classify_failure(log: str, returncode: int) -> str:
    if any(sig in log for sig in RECIPE_SIGNATURES):
        return "recipe"
    if any(sig in log for sig in INFRA_SIGNATURES):
        return "infra"
    return "none" if returncode == 0 else "infra"

def parse_train_lines(log: str) -> list[dict]:
    rows = []
    for match in TRAIN_LINE.finditer(log):
        rows.append({"step": int(match["step"]), "epoch": int(match["epoch"]),
                     "loss": float(match["loss"]), "grad": float(match["grad"]),
                     "lr": float(match["lr"]), "time": float(match["time"])})
    return rows

def newest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in Path(output_dir).glob("checkpoint-*")
                  if (p / "lora.safetensors").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))

class TrainRunner:
    def __init__(self, cfg: KernelConfig, recorder) -> None:
        self.cfg = cfg
        self.recorder = recorder

    def precache(self, resolved: Path, gpus: list[int], node_id: str) -> None:
        two = ",".join(str(g) for g in gpus[:2])
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/precache_train_text_embeds.py", "--config", str(resolved),
             "--device-map", "auto"],
            cwd=self.cfg.worldmodel,
            extra_env={"CUDA_VISIBLE_DEVICES": two, "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"},
            timeout=7200, recorder=self.recorder, node=node_id, phase="precache")
        if proc.returncode != 0:
            raise RuntimeError(f"prompt precache failed (rc={proc.returncode}): {_tail(proc)}")

    def train(self, resolved: Path, gpus: list[int], node_id: str, node_dir: Path) -> TrainOutcome:
        import yaml
        config = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        output_dir = Path(config["run"]["output_dir"])
        log_path = Path(node_dir) / "train" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.recorder.span("train", node=node_id, phase="train",
                                payload={"config": str(resolved), "gpus": gpus}):
            proc = run_in_env(
                "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
                cwd=self.cfg.worldmodel,
                extra_env={"CONFIG_PATH": str(resolved),
                           "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                           "LOG_FILTER": "all", "ALAYA_LOG_MEMORY": "1",
                           "ALAYA_DATASET_CACHE_DIR": str(Path(node_dir) / "dataset_cache")},
                timeout=int(48 * 3600), recorder=self.recorder, node=node_id, phase="train")
        log = proc.stdout + proc.stderr
        log_path.write_text(log, encoding="utf-8")
        failure = classify_failure(log, proc.returncode)
        checkpoint = newest_checkpoint(output_dir)
        if failure == "none" and checkpoint is None:
            failure = "recipe"
        return TrainOutcome(checkpoint=checkpoint, failure=failure, log_path=log_path,
                            metrics=parse_train_lines(log))
