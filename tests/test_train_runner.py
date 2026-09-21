from pathlib import Path
from ar_kernel.train.runner import classify_failure, parse_train_lines, newest_checkpoint

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

def test_parse_train_lines_extracts_metrics():
    log = ("[Train] step=1 epoch=0 source=cam video=abc fs=0 fe=120 K=4 sigma=0.99 "
           "loss=0.261719 grad=0.0579 lr=5.00e-05 time=7.92s\n"
           "[Train] step=2 epoch=0 source=cam video=def fs=8 fe=128 K=4 sigma=0.98 "
           "loss=0.251000 grad=0.0611 lr=5.00e-05 time=7.81s\n")
    rows = parse_train_lines(log)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["loss"] == 0.251
    assert rows[0]["lr"] == 5e-05

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


def test_real_successful_run_log_parses_its_train_lines():
    log = (Path(__file__).parent / "fixtures" / "real_successful_train.log").read_text()
    rows = parse_train_lines(log)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["loss"] == 0.431641 and rows[1]["loss"] == 0.251953
