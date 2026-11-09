from __future__ import annotations
import json
from pathlib import Path

from .. import telemetry
from ..config import KernelConfig
from ..subproc import output_tail, run_in_env
from ..tools.vllm_server import gpu_memory_mib, wait_gpu_release
from .judge import Judge, judge_env, judge_server


def run_wbench_phases(cfg: KernelConfig, work_dir: Path, model: str, gpus: list[int],
                      metric_set: list[str], judge: Judge | None = None) -> dict:
    """precompute, gpu (twice), vlm (twice, behind a local judge server when `judge` is local),
    visual plausibility, report. Every phase runs whatever the metric set."""
    with telemetry.span("eval.wbench", {"model": model, "work_dir": str(work_dir), "gpus": list(gpus),
                                        "metric_set": list(metric_set),
                                        "judge": None if judge is None else [judge.kind, judge.model]}) as sp:
        sp.output = _phases(cfg, work_dir, model, gpus, judge)
        return sp.output


def _phases(cfg: KernelConfig, work_dir: Path, model: str, gpus: list[int], judge: Judge | None) -> dict:
    gpu_arg = ",".join(str(g) for g in gpus)

    def wbench(env: str, args: list[str], name: str, extra_env: dict | None = None) -> None:
        proc = run_in_env(env, args, cwd=cfg.wbench, extra_env=extra_env)
        if proc.returncode != 0:
            raise RuntimeError(f"wbench {name} failed (rc={proc.returncode}): {output_tail(proc)}")

    def main_phase(phase: str, extra_env: dict | None = None, args: list[str] = ()) -> None:
        wbench("wbench-main", ["python", "main.py", "--model", model, "--work_dir", str(work_dir),
                               "--data", str(cfg.eval_data), "--phase", phase, "--gpus", gpu_arg, *args],
               phase, extra_env)

    def vlm(extra_env: dict | None = None) -> None:
        args = ["--vlm_workers", str(cfg.get("eval.local_judge_workers"))] if extra_env else []
        main_phase("vlm", extra_env, args)
        # A failed judge call leaves its case unscored; the second pass asks again for those only.
        # A case still unscored after it fails the eval on its case count (score_from_report).
        main_phase("vlm", extra_env, args)

    main_phase("precompute")
    main_phase("gpu")
    # The second pass recomputes only cases whose first attempt failed (e.g. a transient CUDA OOM),
    # and costs seconds when nothing failed.
    main_phase("gpu")
    if judge is not None and judge.kind == "local":
        _vlm_with_local_judge(cfg, judge, gpus, Path(work_dir), vlm)
    else:
        vlm()
    wbench("wbench-vp", ["python", "tools/run_visual_plausibility.py", "--model", model,
                         "--work_dir", str(work_dir),
                         "--model_path", str(cfg.wbench / cfg.get("eval.vp_weights"))],
           "visual_plausibility", {"CUDA_VISIBLE_DEVICES": gpu_arg})
    main_phase("report")
    report_path = Path(work_dir) / model / "evaluation" / "report.json"
    return json.loads(report_path.read_text())


def _vlm_with_local_judge(cfg, judge, gpus, work_dir, vlm) -> None:
    """Serve the judge on all the GPUs for the vlm phase only, then free them for what follows."""
    server = judge_server(cfg, judge, gpus, work_dir.parent / "judge")
    before = gpu_memory_mib(gpus)
    with telemetry.span("eval.judge", {"model": judge.model, "gpus": list(gpus)}) as sp:
        w = telemetry.writer()
        if w is not None:
            w.add_redaction(server.api_key)
        try:
            sp.output = {"load_s": server.start()}
            vlm(judge_env(cfg, judge, server.base_url, server.api_key))
        finally:
            sp.output = {**(sp.output or {}), "exit": server.stop()}
            after, released = wait_gpu_release(gpu_memory_mib, gpus, before,
                                               float(cfg.get("captioner.memory_release_timeout_s")))
            sp.output.update(gpu_memory_mib={"before": before, "after": after}, released=released)
