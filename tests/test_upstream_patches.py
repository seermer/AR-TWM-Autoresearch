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

def test_a_failed_judge_call_leaves_the_case_unscored_instead_of_counting_as_wrong():
    """WBench counted a judge call that failed after its retries as a wrong answer. It raises now,
    so main.py stores `score: None` for the case and a later vlm pass asks again."""
    code = (
        "from src.metrics.interaction.vlm_interaction import _execute_binary_tasks\n"
        "class Down:\n"
        "    def ask(self, *a, **k): raise ConnectionError('judge down')\n"
        "tasks = [(1, 'Q1', 'q?', 'yes', [])]\n"
        "try:\n"
        "    _execute_binary_tasks(Down(), tasks, [(1, 'jump')], 1, 'event_edit_adherence')\n"
        "except ConnectionError as exc:\n"
        "    print('raised', exc)\n"
    )
    out = subprocess.run(["conda", "run", "--no-capture-output", "-n", "wbench-main", "python", "-c", code],
                         cwd=CFG.wbench, capture_output=True, text=True, timeout=600)
    assert "raised judge down" in out.stdout, out.stdout + out.stderr
