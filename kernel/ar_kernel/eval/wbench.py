from __future__ import annotations
import json
from pathlib import Path

from ..config import KernelConfig
from ..liveness import Liveness, tree_mark
from ..subproc import run_in_env, _tail
from .score import VLM_METRICS

def run_wbench_phases(cfg: KernelConfig, work_dir: Path, model: str, gpus: list[int],
                      metric_set: list[str], recorder, node_id: str) -> dict:
    gpu_arg = ",".join(str(g) for g in gpus)
    phases = ["precompute", "gpu"]
    if any(m in VLM_METRICS for m in metric_set):
        phases.append("vlm")
    phases.append("report")
    for phase in phases:
        with recorder.span(f"wbench.{phase}", node=node_id, phase="eval"):
            liveness = Liveness.from_config(cfg, float(cfg.get("timeouts.eval_s")),
                                            signals=[lambda: tree_mark(Path(work_dir) / model)])
            proc = run_in_env(
                "wbench-main",
                ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                 "--phase", phase, "--gpus", gpu_arg],
                cwd=cfg.wbench, timeout=None, liveness=liveness, recorder=recorder, node=node_id,
                phase="eval")
        if proc.returncode != 0:
            raise RuntimeError(f"wbench {phase} failed (rc={proc.returncode}): {_tail(proc)}")
    if "visual_plausibility" in metric_set:
        with recorder.span("wbench.visual_plausibility", node=node_id, phase="eval"):
            liveness = Liveness.from_config(cfg, float(cfg.get("timeouts.eval_s")),
                                            signals=[lambda: tree_mark(Path(work_dir) / model)])
            proc = run_in_env(
                "wbench-vp",
                ["python", "tools/run_visual_plausibility.py", "--model", model,
                 "--work_dir", str(work_dir),
                 "--model_path", str(cfg.wbench / cfg.get("eval.vp_weights"))],
                cwd=cfg.wbench, extra_env={"CUDA_VISIBLE_DEVICES": gpu_arg},
                timeout=None, liveness=liveness, recorder=recorder, node=node_id, phase="eval")
        if proc.returncode != 0:
            raise RuntimeError(f"visual_plausibility failed (rc={proc.returncode}): {_tail(proc)}")
        with recorder.span("wbench.report2", node=node_id, phase="eval"):
            proc = run_in_env("wbench-main",
                              ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                               "--phase", "report", "--gpus", gpu_arg],
                              cwd=cfg.wbench, timeout=3600, recorder=recorder, node=node_id,
                              phase="eval")
        if proc.returncode != 0:
            # Otherwise the failure surfaces later as a misleading "metrics missing".
            raise RuntimeError(f"wbench report (after visual_plausibility) failed "
                               f"(rc={proc.returncode}): {_tail(proc)}")
    report_path = Path(work_dir) / model / "evaluation" / "report.json"
    return json.loads(report_path.read_text())
