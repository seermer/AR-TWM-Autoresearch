import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.faults import KernelFault, NodeFault
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train import runner as runner_mod
from ar_kernel.train.runner import TrainRunner, newest_checkpoint, nonfinite_tensors, outcome

CUDA_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
NCCL = "RuntimeError: NCCL communicator was aborted on rank 2"
HOST_OOM = "torchrun ... failed (exitcode: -9)"
HEALTHY = "[Train] step=2 loss=0.26 grad=0.05 lr=5.00e-05 time=7.9s"
CK = Path("checkpoint-2")


def test_a_clean_exit_with_a_checkpoint_succeeds():
    assert outcome(HEALTHY, 0, CK, "log") == CK


def test_a_clean_exit_is_not_undone_by_a_loose_log_token():
    """A benign 'Killed' line from a helper must not discard a good run."""
    assert outcome(f"{HEALTHY}\nsome helper process Killed during cleanup\n", 0, CK, "log") == CK


def test_the_real_successful_run_log_succeeds():
    """A healthy 4-GPU run prints the NCCL version banner and ProcessGroupNCCL.cpp warnings: a bare
    "NCCL" signature would have failed every run. Fixture: the verbatim log of a real 2-step run."""
    log = (Path(__file__).parent / "fixtures" / "real_successful_train.log").read_text()
    assert "NCCL version" in log and "ProcessGroupNCCL.cpp" in log
    assert outcome(log, 0, CK, "log") == CK


@pytest.mark.parametrize("returncode, checkpoint", [(0, CK), (1, None)])
def test_a_diverged_loss_is_the_nodes_even_beside_a_machine_error(returncode, checkpoint):
    with pytest.raises(NodeFault, match="loss=nan"):
        outcome(f"[Train] step=3 loss=nan grad=inf\n{CUDA_OOM}\n", returncode, checkpoint, "log")


@pytest.mark.parametrize("log", [CUDA_OOM, NCCL, HOST_OOM, "NCCL error: unhandled system error",
                                 "RuntimeError: a prompt missed the on-disk embedding cache. Re-run ..."])
def test_oom_and_machine_errors_are_the_kernels(log):
    with pytest.raises(KernelFault, match="machine error"):
        outcome(log, 1, None, "attempts/improve_recipe-1/train/train.log")


def test_a_clean_exit_without_a_checkpoint_is_the_kernels():
    with pytest.raises(KernelFault, match="no checkpoint"):
        outcome(HEALTHY, 0, None, "log")


def test_any_other_crash_is_the_nodes():
    with pytest.raises(NodeFault, match="exit code 1.*log: attempts/x/train/train.log"):
        outcome("Traceback ...\nValueError: Loaded 0 samples", 1, None, "attempts/x/train/train.log")


def test_newest_checkpoint_picks_the_highest_step(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "lora.safetensors").touch()
    assert newest_checkpoint(tmp_path).name == "checkpoint-300"


def test_newest_checkpoint_ignores_dirs_without_lora(tmp_path):
    (tmp_path / "checkpoint-400").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-100" / "lora.safetensors").touch()
    assert newest_checkpoint(tmp_path).name == "checkpoint-100"


def write_safetensors(path: Path, tensors: dict[str, tuple[str, bytes]]) -> None:
    header, offset, blob = {"__metadata__": {"format": "pt"}}, 0, b""
    for name, (dtype, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [len(data) // {"F32": 4}.get(dtype, 2)],
                        "data_offsets": [offset, offset + len(data)]}
        offset, blob = offset + len(data), blob + data
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + blob)


def test_nonfinite_tensors_finds_nan_and_inf_in_every_float_type(tmp_path):
    bf16 = lambda values: (np.array(values, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    path = tmp_path / "lora.safetensors"
    write_safetensors(path, {"ok_bf16": ("BF16", bf16([0.5, -2.0])), "nan_bf16": ("BF16", bf16([1.0, np.nan])),
                             "inf_f16": ("F16", np.array([np.inf], np.float16).tobytes()),
                             "ok_f32": ("F32", np.array([3.0, 1e30], np.float32).tobytes()),
                             "inf_f32": ("F32", np.array([-np.inf], np.float32).tobytes())})
    assert nonfinite_tensors(path) == ["nan_bf16", "inf_f16", "inf_f32"]


def _resolved(tmp_path):
    out = tmp_path / "node" / "train" / "outputs"
    cfg = tmp_path / "train_config.yaml"
    cfg.write_text(f"run: {{output_dir: {out}}}\n")
    return cfg, out


def _training(outputs, lora: bytes):
    def fake_run(env, args, **kw):
        ck = outputs / "checkpoint-2"
        ck.mkdir(parents=True)
        for name in ("history_encoder.pt", "trainer_state.pt"):
            (ck / name).write_text("x")
        write_safetensors(ck / "lora.safetensors", {"a": ("F32", lora)})
        kw["log_path"].write_text(HEALTHY + "\n")
        return subprocess.CompletedProcess(args, 0, kw["log_path"].read_text(), "")
    return fake_run


def test_training_and_its_precache_run_without_a_time_limit(tmp_path, monkeypatch):
    resolved, outputs = _resolved(tmp_path)
    seen = []
    train = _training(outputs, np.zeros(2, np.float32).tobytes())

    def fake_run(env, args, **kw):
        seen.append(kw)
        return train(env, args, **kw) if "log_path" in kw else subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(runner_mod, "run_in_env", fake_run)
    runner = TrainRunner(KernelConfig.load(), Recorder(tmp_path))
    runner.precache(resolved, [0, 1, 2, 3], "n1")
    runner.train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert len(seen) == 2 and not any("timeout" in kw or "liveness" in kw for kw in seen)


def test_a_failed_precache_is_the_kernels(tmp_path, monkeypatch):
    resolved, _ = _resolved(tmp_path)
    monkeypatch.setattr(runner_mod, "run_in_env", lambda env, args, **kw: subprocess.CompletedProcess(args, 1, "", "OOM"))
    with pytest.raises(KernelFault, match="precache"):
        TrainRunner(KernelConfig.load(), Recorder(tmp_path)).precache(resolved, [0, 1], "n1")


def test_trainer_state_is_deleted_after_success(tmp_path, monkeypatch):
    resolved, outputs = _resolved(tmp_path)
    monkeypatch.setattr(runner_mod, "run_in_env", _training(outputs, np.ones(2, np.float32).tobytes()))
    checkpoint = TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert not (checkpoint / "trainer_state.pt").exists() and (checkpoint / "lora.safetensors").exists()


def test_a_lora_holding_nan_is_the_nodes(tmp_path, monkeypatch):
    resolved, outputs = _resolved(tmp_path)
    monkeypatch.setattr(runner_mod, "run_in_env", _training(outputs, np.array([1, np.nan], np.float32).tobytes()))
    with pytest.raises(NodeFault, match="NaN or inf in 1 tensors"):
        TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
