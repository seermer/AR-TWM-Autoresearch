from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import KernelConfig
from ..liveness import Liveness, tree_mark
from ..subproc import SubprocTimeout, output_tail, run_in_env

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

@dataclass
class TrainOutcome:
    checkpoint: Path | None
    failure: str
    log_path: Path
    detail: str = ""        # human-readable reason, surfaced to the agent on failure

def classify_failure(log: str, returncode: int) -> str:
    if any(sig in log for sig in RECIPE_SIGNATURES):
        return "recipe"
    if any(sig in log for sig in INFRA_SIGNATURES):
        return "infra"
    return "none" if returncode == 0 else "infra"

DIVERGENCE_SIGNATURES = ("loss=nan", "loss=inf")


def final_failure(log: str, returncode: int, checkpoint_exists: bool) -> str:
    """Outcome of a finished training run.

    A clean exit that wrote a checkpoint is trusted unless the loss diverged: a
    loose infra token (e.g. "Killed" from some helper process) must not discard a
    good run. Anything else goes through signature classification, which lets
    log content override an exit code.
    """
    if returncode == 0 and checkpoint_exists:
        return "recipe" if any(sig in log for sig in DIVERGENCE_SIGNATURES) else "none"
    failure = classify_failure(log, returncode)
    return "recipe" if failure == "none" and not checkpoint_exists else failure


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
            raise RuntimeError(f"prompt precache failed (rc={proc.returncode}): {output_tail(proc)}")

    def train(self, resolved: Path, gpus: list[int], node_id: str, node_dir: Path) -> TrainOutcome:
        config = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        output_dir = Path(config["run"]["output_dir"])
        log_path = Path(node_dir) / "train" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        liveness = Liveness.from_config(self.cfg, float(self.cfg.get("timeouts.train_s")),
                                        signals=[lambda: tree_mark(log_path)])
        with self.recorder.span("train", node=node_id, phase="train",
                                payload={"config": str(resolved), "gpus": gpus}):
            try:
                # Streams to train.log as it runs rather than buffering up to 48 h
                # of output in memory until exit.
                proc = run_in_env(
                    "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
                    cwd=self.cfg.worldmodel,
                    extra_env={"CONFIG_PATH": str(resolved),
                               "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                               "LOG_FILTER": "all", "ALAYA_LOG_MEMORY": "1",
                               "ALAYA_DATASET_CACHE_DIR": str(Path(node_dir) / "dataset_cache")},
                    timeout=None, liveness=liveness, recorder=self.recorder, node=node_id,
                    phase="train", log_path=log_path)
            except SubprocTimeout:
                return TrainOutcome(checkpoint=None, failure="infra", log_path=log_path,
                                    detail=f"training stalled: {liveness.reason}")
        log = proc.stdout
        checkpoint = newest_checkpoint(output_dir)
        failure = final_failure(log, proc.returncode, checkpoint is not None)
        detail = ""
        if failure == "recipe" and proc.returncode == 0 and checkpoint is None:
            detail = "training exited cleanly but wrote no checkpoint"
        if checkpoint is not None and failure == "none":
            (checkpoint / "trainer_state.pt").unlink(missing_ok=True)
        return TrainOutcome(checkpoint=checkpoint, failure=failure, log_path=log_path, detail=detail)
