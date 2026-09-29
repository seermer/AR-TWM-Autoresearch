from __future__ import annotations
import json
from pathlib import Path

from ..config import KernelConfig
from ..liveness import Liveness, tree_mark
from ..subproc import run_in_env, _tail
from ..tools.vllm_server import gpu_memory_mib, wait_gpu_release
from .judge import Judge, judge_env, judge_server


def run_wbench_phases(cfg: KernelConfig, work_dir: Path, model: str, gpus: list[int],
                      metric_set: list[str], recorder, node_id: str, judge: Judge | None = None) -> dict:
    """precompute, gpu (twice), vlm (behind a local judge server when `judge` is local),
    visual plausibility, report. The metric set is always all 22, so no phase is optional."""
    gpu_arg = ",".join(str(g) for g in gpus)

    def wbench(env: str, args: list[str], name: str, extra_env: dict | None = None, timeout=None) -> None:
        with recorder.span(f"wbench.{name}", node=node_id, phase="eval"):
            liveness = Liveness.from_config(cfg, float(cfg.get("timeouts.eval_s")),
                                            signals=[lambda: tree_mark(Path(work_dir) / model)]) \
                if timeout is None else None
            proc = run_in_env(env, args, cwd=cfg.wbench, extra_env=extra_env, timeout=timeout,
                              liveness=liveness, recorder=recorder, node=node_id, phase="eval")
        if proc.returncode != 0:
            raise RuntimeError(f"wbench {name} failed (rc={proc.returncode}): {_tail(proc)}")

    def main_phase(phase: str, extra_env: dict | None = None, timeout=None) -> None:
        wbench("wbench-main", ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                               "--phase", phase, "--gpus", gpu_arg], phase, extra_env, timeout)

    main_phase("precompute")
    main_phase("gpu")
    # The second pass recomputes only cases whose first attempt failed (e.g. a transient CUDA OOM),
    # and costs seconds when nothing failed.
    main_phase("gpu")
    if judge is not None and judge.kind == "local":
        _vlm_with_local_judge(cfg, judge, gpus, Path(work_dir), recorder, node_id, main_phase)
    else:
        main_phase("vlm")
    wbench("wbench-vp", ["python", "tools/run_visual_plausibility.py", "--model", model,
                         "--work_dir", str(work_dir),
                         "--model_path", str(cfg.wbench / cfg.get("eval.vp_weights"))],
           "visual_plausibility", {"CUDA_VISIBLE_DEVICES": gpu_arg})
    main_phase("report", timeout=3600)
    report_path = Path(work_dir) / model / "evaluation" / "report.json"
    return json.loads(report_path.read_text())


def _vlm_with_local_judge(cfg, judge, gpus, work_dir, recorder, node_id, main_phase) -> None:
    """Serve the judge on all the GPUs for the vlm phase only, then free them for what follows."""
    server = judge_server(cfg, judge, gpus, work_dir.parent / "judge", recorder, node_id)
    before = gpu_memory_mib(gpus)
    try:
        with recorder.span("wbench.judge_server", node=node_id, phase="eval"):
            load_s = server.start(float(cfg.get("captioner.startup_timeout_s")))
        recorder.event("judge.server_ready", node=node_id, component="eval", load_s=load_s,
                       payload={"gpus": gpus, "model": judge.model})
        main_phase("vlm", extra_env=judge_env(cfg, judge, server.base_url))
    finally:
        server.stop()
        _, released = wait_gpu_release(gpu_memory_mib, gpus, before, float(cfg.get("captioner.memory_release_timeout_s")))
        if released is False:
            recorder.event("judge.gpu_not_released", node=node_id, component="eval", payload={"before": before})
