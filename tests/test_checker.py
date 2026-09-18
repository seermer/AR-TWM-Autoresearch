from ar_kernel.config import KernelConfig
from ar_kernel.data.checker import check_clip_formats
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
BASE = CFG.repo_root / "configs" / "base_recipe.yaml"

def test_moving_camera_clip_is_eligible_for_camera_format(clip_dir):
    result = check_clip_formats(CFG, clip_dir, "moving", BASE)
    assert result["video_caption_camera"]["ok"] is True
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is False  # no segments

def test_short_clip_is_rejected_with_the_checker_message(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=1.0)
    write_caption(root / "captions" / "c1.json")
    write_poses(root / "poses" / "c1.npz", n_frames=30)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_caption_camera"]["ok"] is False
    assert any("shorter than one training window" in e
               for e in result["video_caption_camera"]["errors"])

def test_round_aligned_segments_pass_per_chunk(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=8.0, fps=24)
    write_caption(root / "captions" / "c1.json", segments=[
        {"time_range_s": [0.0, 3.7083], "prompt": "A bright room seen head on."},
        {"time_range_s": [3.7083, 8.0], "prompt": "The same room as a pencil sketch."},
    ])
    write_poses(root / "poses" / "c1.npz", n_frames=192)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is True

def test_misaligned_segments_fail_per_chunk_but_pass_segment(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4", seconds=10.0, fps=24)
    write_caption(root / "captions" / "c1.json", segments=[
        {"time_range_s": [0.0, 5.0], "prompt": "A bright room seen head on."},
        {"time_range_s": [5.0, 10.0], "prompt": "The same room as a pencil sketch."},
    ])
    write_poses(root / "poses" / "c1.npz", n_frames=240)
    result = check_clip_formats(CFG, root, "moving", BASE)
    assert result["video_timed_prompts_camera:per_chunk"]["ok"] is False
    assert result["video_timed_prompts_camera:segment"]["ok"] is True

def test_static_clip_needs_no_poses(tmp_path):
    root = tmp_path / "ds"
    make_mp4(root / "videos" / "c1.mp4")
    write_caption(root / "captions" / "c1.json")
    result = check_clip_formats(CFG, root, "static", BASE)
    assert result["video_caption_static"]["ok"] is True
    assert "video_caption_camera" not in result
