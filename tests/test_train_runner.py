import subprocess
from pathlib import Path
from ar_kernel.train.runner import classify_failure, newest_checkpoint
from ar_kernel.config import KernelConfig
from ar_kernel.subproc import SubprocTimeout
from ar_kernel.train import runner as runner_mod
from ar_kernel.train.runner import TrainRunner
from ar_kernel.telemetry.recorder import Recorder

CUDA_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
NCCL = "RuntimeError: NCCL communicator was aborted on rank 2"
HOST_OOM = "torchrun ... failed (exitcode: -9)"

def test_cuda_oom_is_a_recipe_failure():
    assert classify_failure(CUDA_OOM, 1) == "recipe"

def test_nan_loss_is_a_recipe_failure():
    assert classify_failure("[Train] step=3 loss=nan grad=inf lr=5.00e-05 time=8s", 1) == "recipe"

def test_host_oom_and_nccl_are_infra_failures():
    assert classify_failure(HOST_OOM, 1) == "infra"
    assert classify_failure(NCCL, 1) == "infra"

def test_clean_run_has_no_failure():
    assert classify_failure("[Train] step=2 loss=0.26 grad=0.05 lr=5.00e-05 time=7.9s", 0) == "none"

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


def test_real_successful_run_log_is_not_classified_as_failure():
    """Regression: a healthy 4-GPU run prints the NCCL version banner and several
    ProcessGroupNCCL.cpp warnings. A bare "NCCL" infra signature classified every
    such run as an infra failure, which would have made the loop discard every
    node it ever trained. Fixture is the verbatim log of the real 2-step run from
    Task 14 (checkpoint-2 written, loss 0.431 -> 0.252)."""
    log = (Path(__file__).parent / "fixtures" / "real_successful_train.log").read_text()
    assert "NCCL version" in log, "fixture must retain the benign NCCL banner"
    assert "ProcessGroupNCCL.cpp" in log, "fixture must retain the benign NCCL warnings"
    assert classify_failure(log, 0) == "none"


def test_genuine_nccl_error_is_still_infra():
    log = "some output\nNCCL error: unhandled system error\n"
    assert classify_failure(log, 1) == "infra"


def test_incomplete_prompt_precache_is_infra():
    log = ("RuntimeError: text encoder is disabled (ALAYA_SKIP_TEXT_ENCODER=1) but a "
           "prompt missed the on-disk embedding cache. Re-run ...")
    assert classify_failure(log, 1) == "infra"


def test_clean_exit_with_checkpoint_is_not_overridden_by_a_loose_log_token():
    """Review minor: signatures were checked before the return code even when rc=0
    and a checkpoint existed, so e.g. a benign 'Killed' line in a healthy log
    discarded a good run. Only a divergence signature may override a clean exit."""
    from ar_kernel.train.runner import final_failure
    healthy = "[Train] step=2 loss=0.25\nsome helper process Killed during cleanup\n"
    assert final_failure(healthy, 0, checkpoint_exists=True) == "none"


def test_clean_exit_with_nan_loss_is_still_a_recipe_failure():
    from ar_kernel.train.runner import final_failure
    assert final_failure("[Train] step=2 loss=nan grad=inf\n", 0, checkpoint_exists=True) == "recipe"


def test_nonzero_exit_still_goes_through_signature_classification():
    from ar_kernel.train.runner import final_failure
    assert final_failure("torch.OutOfMemoryError: CUDA out of memory", 1, checkpoint_exists=False) == "recipe"
    assert final_failure("NCCL error: unhandled system error", 1, checkpoint_exists=False) == "infra"


def _resolved(tmp_path):
    out = tmp_path / "node" / "train" / "outputs"
    cfg = tmp_path / "train_config.yaml"
    cfg.write_text(f"run: {{output_dir: {out}}}\n")
    return cfg, out


def test_a_stalled_training_run_is_an_infra_failure_reported_to_the_agent(tmp_path, monkeypatch):
    resolved, _ = _resolved(tmp_path)
    seen = {}

    def fake_run(env, args, **kw):
        seen.update(kw)
        kw["liveness"].reason = "no sign of progress for 600s after the 172800s soft timeout"
        raise SubprocTimeout(args, 0, output="[Setup] rank 0/4\n")
    monkeypatch.setattr(runner_mod, "run_in_env", fake_run)
    out = TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert seen["timeout"] is None and seen["liveness"].soft_s == 172800
    assert (out.checkpoint, out.failure) == (None, "infra") and "stalled" in out.detail


def test_trainer_state_is_deleted_after_success(tmp_path, monkeypatch):
    resolved, outputs = _resolved(tmp_path)

    def fake_run(env, args, **kw):
        ck = outputs / "checkpoint-2"
        ck.mkdir(parents=True)
        for name in ("lora.safetensors", "history_encoder.pt", "trainer_state.pt"):
            (ck / name).write_text("x")
        kw["log_path"].write_text("[Train] step=2 epoch=0 loss=0.25 grad=0.05 lr=5.00e-05 time=7.8s\n")
        return subprocess.CompletedProcess(args, 0, kw["log_path"].read_text(), "")
    monkeypatch.setattr(runner_mod, "run_in_env", fake_run)
    out = TrainRunner(KernelConfig.load(), Recorder(tmp_path)).train(resolved, [0, 1, 2, 3], "n1", tmp_path / "node")
    assert out.failure == "none" and not (out.checkpoint / "trainer_state.pt").exists()
    assert (out.checkpoint / "lora.safetensors").exists()
