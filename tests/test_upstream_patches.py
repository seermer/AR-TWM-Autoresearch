import os, subprocess
from pathlib import Path
from ar_kernel.config import KernelConfig

CFG = KernelConfig.load()

def test_vp_accepts_absolute_work_dir_from_any_cwd(tmp_path):
    videos = tmp_path / "wd" / "mymodel" / "videos"
    videos.mkdir(parents=True)
    out = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "wbench-main", "python",
         str(CFG.wbench / "tools" / "run_visual_plausibility.py"),
         "--work_dir", str(tmp_path / "wd"), "--model", "mymodel"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert str(videos) in out.stdout, out.stdout + out.stderr

def test_launcher_refuses_fewer_gpus_than_minimum():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2", "ALAYA_LAUNCH_DRY_RUN": "1"}
    proc = subprocess.run(["bash", str(CFG.worldmodel / "scripts/finetune/lowcompute_4x4090.sh")],
                          cwd=CFG.worldmodel, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "at least 4" in (proc.stdout + proc.stderr)

def test_launcher_accepts_any_indices_including_gpu5():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,5", "ALAYA_LAUNCH_DRY_RUN": "1"}
    proc = subprocess.run(["bash", str(CFG.worldmodel / "scripts/finetune/lowcompute_4x4090.sh")],
                          cwd=CFG.worldmodel, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,5" in proc.stdout
