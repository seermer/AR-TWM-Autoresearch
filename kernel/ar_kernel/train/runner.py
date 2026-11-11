from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import yaml

from .. import fs, telemetry
from ..config import KernelConfig
from ..faults import KernelFault, NodeFault
from ..subproc import output_tail, run_in_env

DIVERGENCE_SIGNATURES = ("loss=nan", "loss=inf")
# Match only tokens that a FAILING run emits. A healthy multi-GPU run prints the
# "NCCL version ..." banner and several ProcessGroupNCCL.cpp warnings, so a bare
# "NCCL" substring would name every successful distributed run.
MACHINE_SIGNATURES = (
    "CUDA out of memory", "torch.OutOfMemoryError",
    "exitcode: -9", "No space left on device", "CUDA driver error", "Killed",
    "NCCL error", "ncclInternalError", "ncclSystemError", "ncclUnhandledCudaError",
    "ncclRemoteError", "NCCL communicator was aborted",
    "Watchdog caught collective operation timeout",
    "DistBackendError",
    # loader.py::_text_encoder_disabled, raised when the precache is incomplete.
    "missed the on-disk embedding cache",
)
NONFINITE_BITS = {"BF16": (np.uint16, 0x7F80), "F16": (np.uint16, 0x7C00), "F32": (np.uint32, 0x7F800000)}


def _matched(log: str, signatures) -> tuple[str, int, str] | None:
    """(signature, line number, line) of the first signature found, in signature order."""
    for sig in signatures:
        if sig in log:
            n, line = next((n, line) for n, line in enumerate(log.splitlines(), 1) if sig in line)
            return sig, n, line
    return None


def outcome(log: str, returncode: int, checkpoint: Path | None, log_name: str) -> Path:
    """Classify checkpoint success, diverged loss, machine errors, and other crashes, in that order.
    Machine errors and clean exits without a checkpoint are kernel faults; divergence and other crashes are the node's."""
    diverged = _matched(log, DIVERGENCE_SIGNATURES)
    machine = None if diverged or (returncode == 0 and checkpoint is not None) else _matched(log, MACHINE_SIGNATURES)
    if returncode == 0 and checkpoint is not None and not diverged:
        chosen, reason, fault = "checkpoint", "a clean exit with a checkpoint and no diverged loss", None
    elif diverged:
        chosen, reason = "node fault", f"the training loss diverged ({diverged[0]}); log: {log_name}"
        fault = NodeFault(reason)
    elif machine:
        chosen, reason = "kernel fault", f"training hit a machine error ({machine[0]}); log: {log_name}"
        fault = KernelFault(reason)
    elif returncode == 0:
        chosen, reason = "kernel fault", f"training exited cleanly but wrote no checkpoint; log: {log_name}"
        fault = KernelFault(reason)
    else:
        chosen, reason = "node fault", f"training crashed (exit code {returncode}); log: {log_name}"
        fault = NodeFault(reason)
    telemetry.decision("train.outcome", {"returncode": returncode, "checkpoint": str(checkpoint) if checkpoint else None,
                                         "matched": [list(m) for m in (diverged, machine) if m]},
                       ["checkpoint", "node fault", "kernel fault"], chosen, reason)
    if fault is not None:
        raise fault
    return checkpoint


def nonfinite_tensors(path: Path) -> list[str]:
    """The tensors of a .safetensors file holding NaN or inf (floating point: an all-ones exponent)."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)
    bad = []
    for name, info in header.items():
        if name == "__metadata__" or info["dtype"] not in NONFINITE_BITS:
            continue
        kind, mask = NONFINITE_BITS[info["dtype"]]
        begin, end = info["data_offsets"]
        if np.any(data[begin:end].view(kind) & kind(mask) == kind(mask)):
            bad.append(name)
    return bad


def newest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in Path(output_dir).glob("checkpoint-*")
                  if (p / "lora.safetensors").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))

class TrainRunner:
    def __init__(self, cfg: KernelConfig) -> None:
        self.cfg = cfg

    def precache(self, resolved: Path, gpus: list[int]) -> None:
        two = ",".join(str(g) for g in gpus[:2])
        with telemetry.span("train.precache", {"config": Path(resolved).read_text(encoding="utf-8")}) as sp:
            proc = run_in_env(
                "alayaworld",
                ["python", "scripts/tools/precache_train_text_embeds.py", "--config", str(resolved),
                 "--device-map", "auto"],
                cwd=self.cfg.worldmodel,
                extra_env={"CUDA_VISIBLE_DEVICES": two, "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"})
            if proc.returncode != 0:            # the gate's dry run of the same pass passed at submit
                raise KernelFault(f"prompt precache failed (rc={proc.returncode}): {output_tail(proc)}")
            sp.output = {"returncode": 0}

    def train(self, resolved: Path, gpus: list[int], node_dir: Path) -> Path:
        """The checkpoint; raises NodeFault or KernelFault (`outcome`), or NodeFault for a LoRA holding NaN/inf."""
        text = Path(resolved).read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        output_dir = Path(config["run"]["output_dir"])
        log_path = Path(node_dir) / "train" / "train.log"
        fs.mkdir(log_path.parent)
        with telemetry.span("train", {"config": text, "gpus": list(gpus)}) as sp:
            # Streams to train.log as it runs rather than buffering days of output in memory until exit.
            proc = run_in_env(
                "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
                cwd=self.cfg.worldmodel,
                extra_env={"CONFIG_PATH": str(resolved),
                           "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                           "LOG_FILTER": "all", "ALAYA_LOG_MEMORY": "1",
                           "ALAYA_DATASET_CACHE_DIR": str(Path(node_dir) / "dataset_cache")},
                log_path=log_path)
            log_name = str(Path("attempts", Path(node_dir).name, "train", "train.log"))     # under /nodes/<node>/
            checkpoint = outcome(proc.stdout, proc.returncode, newest_checkpoint(output_dir), log_name)
            bad = nonfinite_tensors(checkpoint / "lora.safetensors")
            telemetry.decision("train.nonfinite", {"tensors": bad}, ["keep", "node fault"],
                               "node fault" if bad else "keep",
                               f"{len(bad)} tensors hold NaN or inf" if bad else "every tensor is finite")
            if bad:
                raise NodeFault(f"the trained LoRA holds NaN or inf in {len(bad)} tensors, e.g. {bad[0]}; log: {log_name}")
            fs.remove(checkpoint / "trainer_state.pt")
            sp.output = {"checkpoint": str(checkpoint), "files": fs.files(checkpoint)}
        return checkpoint
