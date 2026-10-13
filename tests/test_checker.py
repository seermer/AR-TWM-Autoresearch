from ar_kernel.config import KernelConfig
from ar_kernel.data.checker import check_clip
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"


def _root(tmp_path, seconds, segments):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=seconds, fps=24)
    write_caption(root / "captions" / "c1.json", segments=segments)
    write_poses(root / "poses" / "c1.npz", n_frames=int(seconds * 24))
    return root


def test_a_clip_with_trainable_prompts_passes(tmp_path):
    report = check_clip(CFG, _root(tmp_path, 10.0, [
        {"time_range_s": [0.0, 5.0], "prompt": "A bright room seen head on."},
        {"time_range_s": [5.0, 10.0], "prompt": "The same room in the rain."}]), BASE)
    assert report["ok"] is True and report["errors"] == []


def test_a_clip_without_timed_prompts_fails(clip_dir):
    report = check_clip(CFG, clip_dir, BASE)
    assert report["ok"] is False and any("segments" in e for e in report["errors"])


def test_a_prompt_too_short_to_train_is_an_error(tmp_path):
    report = check_clip(CFG, _root(tmp_path, 8.0, [
        {"time_range_s": [0.0, 7.0], "prompt": "A bright room seen head on."},
        {"time_range_s": [7.0, 8.0], "prompt": "The same room in the rain."}]), BASE)
    assert report["ok"] is False and any("to be trained on" in e for e in report["errors"])
