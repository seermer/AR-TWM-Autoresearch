import pytest, yaml
from ar_kernel.config import KernelConfig
from ar_kernel.train.recipe import build_resolved_config, steps_per_epoch, lora_of

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"
MANIFEST = {"datasets": {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                 "weight": 1.0, "clips": ["x"] * 40}}}

def test_resolved_config_injects_datasets_and_kernel_paths(tmp_path):
    resolved = build_resolved_config(
        CFG, BASE, {"optimizer.lr": 1e-4}, {"cam": tmp_path / "view" / "cam"}, MANIFEST,
        node_id="n1", node_dir=tmp_path / "n1", run_dir=tmp_path)
    assert resolved["data"]["sources"] == {}
    assert resolved["data"]["datasets"]["cam"]["root"] == str(tmp_path / "view" / "cam")
    assert resolved["data"]["datasets"]["cam"]["format"] == "video_caption_camera"
    assert resolved["optimizer"]["lr"] == pytest.approx(1e-4)
    assert resolved["run"]["name"] == "node_n1"
    assert resolved["runtime"]["text_embed_cache_dir"] == str(tmp_path / "cache" / "text_embed")
    assert resolved["validation"]["enabled"] is False

def test_prompt_mode_is_emitted_only_for_timed_datasets(tmp_path):
    manifest = {"datasets": {"pencil": {"format": "video_timed_prompts_camera",
                                        "prompt_mode": "per_chunk", "weight": 2.0,
                                        "clips": ["a"] * 8}}}
    resolved = build_resolved_config(CFG, BASE, {}, {"pencil": tmp_path / "p"}, manifest,
                                     node_id="n2", node_dir=tmp_path / "n2", run_dir=tmp_path)
    assert resolved["data"]["datasets"]["pencil"]["prompt_mode"] == "per_chunk"

def test_steps_per_epoch_matches_the_standard_epoch_definition():
    manifest = {"datasets": {
        "a": {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": ["x"] * 6},
        "b": {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": ["y"] * 6}}}
    windows, steps = steps_per_epoch(manifest, n_gpus=4, grad_accum=1)
    assert windows == 12 and steps == 3

def test_steps_per_epoch_is_zero_for_a_tiny_dataset():
    manifest = {"datasets": {"a": {"format": "video_caption_camera", "prompt_mode": None,
                                   "weight": 1.0, "clips": ["x"] * 12}}}
    assert steps_per_epoch(manifest, n_gpus=4, grad_accum=4) == (12, 0)


def test_lora_of_reads_the_resolved_config(tmp_path):
    p = tmp_path / "train_config.yaml"
    p.write_text("lora: {rank: 32, alpha: 16}\n")
    assert lora_of(p) == (32, 16)
