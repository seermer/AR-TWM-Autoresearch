import pytest
from ar_kernel.config import REPO_ROOT, KernelConfig, GpuPolicyError, resolve_gpus

def test_loads_defaults_and_resolves_paths():
    cfg = KernelConfig.load()
    assert cfg.get("gpus.min_count") == 4
    assert cfg.worldmodel.is_dir() and cfg.wbench.is_dir()
    assert cfg.get("train.lora_allowlist") == [[16, 16], [32, 32], [64, 64]]

def test_gpu_list_defaults_when_env_unset():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {}) == [0, 1, 2, 3]

def test_gpu_list_uses_env_verbatim_including_gpu5():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,2,4,5"}) == [0, 2, 4, 5]

def test_gpu_list_keeps_whole_groups_of_four(caplog):
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}) == [0, 1, 2, 3]
    assert "[4, 5] are left out" in caplog.text
    assert resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}) == [0, 1, 2, 3, 4, 5, 6, 7]

def test_gpu_list_below_min_count_is_refused():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="at least 4"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2"})

def test_gpu_list_rejects_non_integer_entries():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="not an integer"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2,gpu3"})


def test_dotenv_fills_missing_keys_without_overriding_the_shell(tmp_path):
    """Nothing loaded .env, so filling it in did not enable VLM metrics
    even though the docs said it would. Shell variables must still win."""
    from ar_kernel.config import load_dotenv
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nVLM_API_KEY=from-file\nOPENAI_MODEL='gpt-x'\n"
                        "EMPTY_ONE=\nALREADY_SET=from-file\n\nnot a pair\n")
    env = {"ALREADY_SET": "from-shell"}
    applied = load_dotenv(env_file, env)
    assert env["VLM_API_KEY"] == "from-file"
    assert env["OPENAI_MODEL"] == "gpt-x"            # quotes stripped
    assert env["ALREADY_SET"] == "from-shell"        # shell wins
    assert "EMPTY_ONE" not in env                    # empty placeholders do not count
    assert set(applied) == {"VLM_API_KEY", "OPENAI_MODEL"}


def test_missing_dotenv_is_not_an_error(tmp_path):
    from ar_kernel.config import load_dotenv
    env = {}
    assert load_dotenv(tmp_path / "nope.env", env) == [] and env == {}


def test_duplicate_gpu_indices_are_rejected():
    """'0,0,0,0' passed the >= 4 count check while naming one physical GPU."""
    import pytest
    from ar_kernel.config import GpuPolicyError, resolve_gpus
    with pytest.raises(GpuPolicyError, match="duplicate"):
        resolve_gpus(KernelConfig.load(), {"CUDA_VISIBLE_DEVICES": "0,0,0,0"})


def test_for_run_reads_the_frozen_snapshot_with_this_checkout_as_root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text("budget: {max_usd: 5}\npaths: {runs_dir: runs}\n")
    cfg = KernelConfig.for_run(tmp_path)
    assert cfg.get("budget.max_usd") == 5
    assert cfg.repo_root == REPO_ROOT                       # not runs/<id>: sibling paths still resolve
