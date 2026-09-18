import shutil, subprocess
from ar_kernel.config import KernelConfig
from ar_kernel.data.leakage import LeakageChecker
from conftest import make_mp4

CFG = KernelConfig.load()

def _video_from_image(image, out, seconds=3.0, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
                    "-t", str(seconds), "-r", str(fps), "-vf", "scale=736:414",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out

def test_wbench_first_frame_is_rejected(tmp_path):
    case_image = CFG.wbench / "data" / "images" / "case_2.jpg"
    video = _video_from_image(case_image, tmp_path / "leak.mp4")
    verdict = LeakageChecker(CFG).check(video)
    assert verdict.rejected is True
    assert verdict.matches and verdict.matches[0]["case_id"] == "2"

def test_unrelated_synthetic_clip_is_accepted(tmp_path):
    video = make_mp4(tmp_path / "ok.mp4")
    assert LeakageChecker(CFG).check(video).rejected is False

def test_flat_frame_cannot_trigger_rejection_alone(tmp_path):
    flat = tmp_path / "flat.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=gray:size=736x414:rate=24:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(flat)], check=True)
    verdict = LeakageChecker(CFG).check(flat)
    assert verdict.rejected is False
