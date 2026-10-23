import pytest
from ar_kernel.config import REPO_ROOT, KernelConfig, GpuPolicyError

def test_loads_defaults_and_resolves_paths():
    cfg = KernelConfig.load()
    assert cfg.get("gpus.min_count") == 4
    assert cfg.worldmodel.is_dir() and cfg.wbench.is_dir()
    assert cfg.get("train.lora_allowlist") == [[16, 16], [32, 32], [64, 64]]

def _choose(num_gpus=4, env=None, used=None):
    from ar_kernel.guards import choose_gpus
    cfg = KernelConfig(raw={"gpus": {"num_gpus": num_gpus, "min_count": 4, "free_below_mib": 512}}, repo_root=None)
    return choose_gpus(cfg, env or {}, used=lambda: used or {g: 1 for g in range(8)})

def test_gpu_choice_takes_the_first_free_cards():
    assert _choose() == [0, 1, 2, 3]
    assert _choose(used={0: 1, 1: 9000, 2: 512, 3: 1, 4: 600, 5: 1, 6: 1}) == [0, 2, 3, 5]
    assert _choose(8) == list(range(8))

def test_gpu_choice_with_too_few_free_cards_names_every_card():
    with pytest.raises(GpuPolicyError, match="need 4 free GPUs.*found 3.*GPU 1 9000 MiB"):
        _choose(used={0: 1, 1: 9000, 2: 1, 3: 1})

def test_gpu_count_must_be_a_positive_multiple_of_min_count():
    for bad in (0, 6, -4, "4"):
        with pytest.raises(GpuPolicyError, match="positive multiple of 4"):
            _choose(bad)

def test_gpu_choice_refuses_cuda_visible_devices():
    with pytest.raises(GpuPolicyError, match="unset CUDA_VISIBLE_DEVICES"):
        _choose(env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert _choose(env={"CUDA_VISIBLE_DEVICES": " "}) == [0, 1, 2, 3]

def test_gpu_choice_raises_when_nvidia_smi_fails(monkeypatch):
    from ar_kernel import guards
    monkeypatch.setattr(guards.subprocess, "run", lambda *a, **k: __import__("subprocess").CompletedProcess(a, 9, "", "no driver"))
    with pytest.raises(GpuPolicyError, match="exited 9: no driver"):
        guards._nvidia_smi(["--query-gpu=index,memory.used"])
    with pytest.raises(GpuPolicyError, match="exited 9"):
        guards.device_minors([0])


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


def test_for_run_reads_the_frozen_snapshot_with_this_checkout_as_root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text("budget: {max_usd: 5}\npaths: {runs_dir: runs}\n")
    cfg = KernelConfig.for_run(tmp_path)
    assert cfg.get("budget.max_usd") == 5
    assert cfg.repo_root == REPO_ROOT                       # not runs/<id>: sibling paths still resolve


def test_overlays_apply_in_order_and_keep_keys_they_leave_out(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text("gpus: {num_gpus: 8}\ncaptioner: {model: m1, max_num_seqs: 64}\n")
    second.write_text("captioner: {model: m2}\n")
    cfg = KernelConfig.load(overlays=[first, second])
    assert cfg.get("gpus.num_gpus") == 8 and cfg.get("gpus.min_count") == 4
    assert cfg.get("captioner.model") == "m2" and cfg.get("captioner.max_num_seqs") == 64
    assert cfg.overlay == {"gpus": {"num_gpus": 8}, "captioner": {"model": "m2", "max_num_seqs": 64}}
    assert KernelConfig.load().overlay is None


def test_base_recipe_is_the_file_the_config_names(tmp_path):
    from ar_kernel.config import run_config_path
    overlay = tmp_path / "o.yaml"
    overlay.write_text("train: {base_recipe: configs/base_recipe_h100.yaml}\n")
    assert run_config_path(KernelConfig.load(), tmp_path, "base_recipe.yaml") == REPO_ROOT / "configs" / "base_recipe.yaml"
    assert run_config_path(KernelConfig.load(overlays=[overlay]), tmp_path, "base_recipe.yaml") \
        == REPO_ROOT / "configs" / "base_recipe_h100.yaml"
