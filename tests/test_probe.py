import pytest
from ar_kernel.data.probe import probe_video, aspect_ok
from conftest import make_mp4

def test_probe_reads_fps_frames_and_size(clip_dir):
    info = probe_video(clip_dir / "videos" / "c1.mp4")
    assert info.fps == pytest.approx(30.0, abs=0.01)
    assert info.frames == 120
    assert (info.width, info.height) == (736, 414)
    assert info.duration == pytest.approx(4.0, abs=0.05)

def test_aspect_ok_accepts_16_9_and_rejects_4_3(tmp_path):
    wide = probe_video(make_mp4(tmp_path / "w.mp4", width=1280, height=720))
    narrow = probe_video(make_mp4(tmp_path / "n.mp4", width=640, height=480))
    assert aspect_ok(wide, 0.02) is True
    assert aspect_ok(narrow, 0.02) is False


import subprocess


def _rotated(src, dst, degrees=90):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(degrees),
                    "-i", str(src), "-c", "copy", str(dst)], check=True)
    return dst


def _with_sar(dst, width, height, sar):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={width}x{height}:rate=24", "-t", "3",
                    "-vf", f"setsar={sar}", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def test_rotation_is_reported(tmp_path):
    info = probe_video(_rotated(make_mp4(tmp_path / "a.mp4"), tmp_path / "r.mp4"))
    assert info.rotation == 90


def test_rotated_landscape_is_not_sixteen_by_nine(tmp_path):
    """Coded 736x414 but displayed portrait: must fail the aspect check."""
    info = probe_video(_rotated(make_mp4(tmp_path / "a.mp4"), tmp_path / "r.mp4"))
    assert not aspect_ok(info, 0.02)


def test_anamorphic_clip_that_displays_sixteen_by_nine_passes(tmp_path):
    info = probe_video(_with_sar(tmp_path / "s.mp4", 552, 414, "4/3"))
    assert abs(info.sar - 4 / 3) < 1e-6
    assert aspect_ok(info, 0.02)


def test_square_pixel_four_by_three_still_fails(tmp_path):
    assert not aspect_ok(probe_video(_with_sar(tmp_path / "f.mp4", 552, 414, "1")), 0.02)


def test_unrotated_clip_reports_zero_rotation(tmp_path):
    info = probe_video(make_mp4(tmp_path / "plain.mp4"))
    assert info.rotation == 0 and info.sar == 1.0
