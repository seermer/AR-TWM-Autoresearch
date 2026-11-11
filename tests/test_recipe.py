import pytest, yaml
from ar_kernel.config import KernelConfig
from ar_kernel.train.recipe import build_resolved_config, steps_per_epoch, lora_of
from ar_kernel.train.recipe import TUNABLE_KEYS, validate_recipe_values

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"
MANIFEST = {"datasets": {"cam": {"weight": 1.0, "clips": ["x"] * 40}}}

def test_resolved_config_injects_datasets_and_kernel_paths(tmp_path):
    resolved = build_resolved_config(
        BASE, {"optimizer.lr": 1e-4}, {"cam": tmp_path / "view" / "cam"}, MANIFEST,
        node_id="n1", node_dir=tmp_path / "n1", run_dir=tmp_path)
    assert resolved["data"]["sources"] == {}
    assert resolved["data"]["datasets"]["cam"]["root"] == str(tmp_path / "view" / "cam")
    assert resolved["data"]["datasets"]["cam"] == {
        "root": str(tmp_path / "view" / "cam"), "format": "video_timed_prompts_camera", "weight": 1.0}
    assert resolved["optimizer"]["lr"] == pytest.approx(1e-4)
    assert resolved["run"]["name"] == "node_n1"
    assert resolved["runtime"]["text_embed_cache_dir"] == str(tmp_path / "cache" / "text_embed")
    assert resolved["validation"]["enabled"] is False

def test_steps_per_epoch_matches_the_standard_epoch_definition():
    manifest = {"datasets": {
        "a": {"weight": 1.0, "clips": ["x"] * 6},
        "b": {"weight": 1.0, "clips": ["y"] * 6}}}
    windows, steps = steps_per_epoch(manifest, n_gpus=4, grad_accum=1)
    assert windows == 12 and steps == 3

def test_steps_per_epoch_is_zero_for_a_tiny_dataset():
    manifest = {"datasets": {"a": {"weight": 1.0, "clips": ["x"] * 12}}}
    assert steps_per_epoch(manifest, n_gpus=4, grad_accum=4) == (12, 0)


def test_lora_of_reads_the_resolved_config(tmp_path):
    p = tmp_path / "train_config.yaml"
    p.write_text("lora: {rank: 32, alpha: 16}\n")
    assert lora_of(p) == (32, 16)


@pytest.mark.parametrize("key", sorted(TUNABLE_KEYS))
def test_tunable_numeric_boundaries(key):
    check = lambda value: validate_recipe_values({key: value})
    assert check(1) == []
    assert check(-1) and check(True) and check("1")
    if key in {"optimizer.lr", "optimizer.weight_decay", "optimizer.max_grad_norm"}:
        assert check(0.5) == []
        assert check(float("nan")) and check(float("inf"))
    else:
        assert check(0.5)
    assert bool(check(0)) == (key not in {"optimizer.weight_decay", "optimizer.warmup_steps"})


def test_unknown_recipe_keys_are_left_to_the_gate():
    assert validate_recipe_values({"unknown": None}) == []
