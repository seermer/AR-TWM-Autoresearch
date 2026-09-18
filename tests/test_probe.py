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
