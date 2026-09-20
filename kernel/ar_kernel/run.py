from __future__ import annotations
import datetime as dt
import json, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .archive.db import open_db
from .config import KernelConfig, resolve_gpus
from .eval.merge import merge_lora
from .eval.render import build_render_config, render_proxy
from .eval.score import (aggregates, cleanup_eval, resolve_metric_set, score_from_report)
from .eval.wbench import run_wbench_phases
from .telemetry.recorder import Recorder

@dataclass
class RunContext:
    run_dir: Path
    conn: object
    recorder: Recorder
    gpus: list[int]
    metric_set: list[str]
    case_ids: list[str]
    versions: dict

def preflight_metrics(cfg: KernelConfig, env: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """The run's metric set plus one human-readable reason per exclusion."""
    metrics = resolve_metric_set(cfg, env)   # single source of truth for the rule
    excluded = []
    if not env.get("VLM_API_KEY", "").strip():
        excluded.append("VLM metrics excluded: VLM_API_KEY is empty")
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        excluded.append(f"visual_plausibility excluded: weights missing at {vp_weights}")
    return metrics, excluded

def _git_state(repo: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty

def bootstrap_run(cfg: KernelConfig, run_id: str | None, env: Mapping[str, str]) -> RunContext:
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_dir / run_id
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache" / "text_embed").mkdir(parents=True, exist_ok=True)
    gpus = resolve_gpus(cfg, env)
    recorder = Recorder(run_dir, redact=[v for k, v in env.items()
                                         if k in {"OPENAI_API_KEY", "VLM_API_KEY", "HF_TOKEN"} and v])
    shutil.copy2(cfg.repo_root / "configs" / "kernel.yaml", run_dir / "config" / "kernel.yaml")
    shutil.copy2(cfg.repo_root / "configs" / "base_recipe.yaml", run_dir / "config" / "base_recipe.yaml")
    wm_sha, wm_dirty = _git_state(cfg.worldmodel)
    wb_sha, wb_dirty = _git_state(cfg.wbench)
    kernel_sha, kernel_dirty = _git_state(cfg.repo_root)
    versions = {"worldmodel_sha": wm_sha, "worldmodel_dirty": wm_dirty,
                "wbench_sha": wb_sha, "wbench_dirty": wb_dirty,
                "kernel_sha": kernel_sha, "kernel_dirty": kernel_dirty}
    (run_dir / "config" / "versions.json").write_text(json.dumps(versions, indent=2))
    metric_set, excluded = preflight_metrics(cfg, env)
    case_ids = (cfg.repo_root / "configs" / "proxy_cases.txt").read_text().strip().split(",")
    recorder.event("run.start", payload={"gpus": gpus, "metric_set": metric_set,
                                         "excluded_metrics": excluded, "versions": versions,
                                         "case_ids": case_ids})
    if wm_dirty or wb_dirty:
        recorder.event("run.warning", payload={"message": "sibling repo has uncommitted changes",
                                               "worldmodel_dirty": wm_dirty, "wbench_dirty": wb_dirty})
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=recorder, gpus=gpus,
                      metric_set=metric_set, case_ids=case_ids, versions=versions)

def score_node(cfg: KernelConfig, ctx: RunContext, node_id: str, checkpoint: Path | None,
               rank: int, alpha: int) -> tuple[float, dict]:
    node_dir = ctx.run_dir / "nodes" / node_id
    work_dir = node_dir / "eval" / "work_dirs"
    model = f"ar_{ctx.run_dir.name}_n{node_id}"
    videos_dir = work_dir / model / "videos"
    if checkpoint is None:
        merged, history = None, cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
    else:
        merged = merge_lora(cfg, checkpoint, rank, alpha, ctx.run_dir, ctx.recorder, node_id)
        history = Path(checkpoint) / "history_encoder.pt"
    render_config = build_render_config(cfg, merged, history, videos_dir, ctx.case_ids, node_dir)
    render_proxy(cfg, render_config, ctx.gpus, node_id, ctx.recorder, ctx.case_ids)
    report = run_wbench_phases(cfg, work_dir, model, ctx.gpus, ctx.metric_set, ctx.recorder, node_id)
    score, per_metric = score_from_report(report, ctx.metric_set)
    strata = aggregates(cfg, work_dir / model / "evaluation", ctx.case_ids)
    ctx.recorder.event("eval.scored", node=node_id, phase="eval",
                       payload={"score": score, "metrics": per_metric, "strata": strata})
    removed = cleanup_eval(work_dir, model)
    if merged is not None and merged.exists():
        shutil.rmtree(merged)
        removed.append("merge_slot")
    ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})
    return score, {"metrics": per_metric, "strata": strata, "report": report}
