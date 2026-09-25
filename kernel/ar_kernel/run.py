from __future__ import annotations
import datetime as dt
import json, shutil, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .archive.db import open_db
from .doctor import wbench_weight_problems
from .config import SNAPSHOT_FILES, KernelConfig, resolve_gpus, run_config_path
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
    expected_n: dict = field(default_factory=dict)   # metric -> case count on the proxy

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

class RunNotFound(FileNotFoundError):
    """No run with this id exists."""


class PreflightError(RuntimeError):
    """The environment cannot produce a complete, comparable score."""


_SECRET_KEYS = {"OPENAI_API_KEY", "VLM_API_KEY", "HF_TOKEN"}


def _recorder(run_dir: Path, env: Mapping[str, str]) -> Recorder:
    return Recorder(run_dir, redact=[v for k, v in env.items() if k in _SECRET_KEYS and v])


def bootstrap_run(cfg: KernelConfig, run_id: str | None, env: Mapping[str, str]) -> RunContext:
    """Create a run, freezing its configs and recording versions. Idempotent: for
    an existing run it attaches instead, and never overwrites the snapshot."""
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_dir / run_id
    if (run_dir / "config" / "run.json").exists():
        return attach_run(cfg, run_id, env)
    gpus = resolve_gpus(cfg, env)          # before creating anything
    problems = wbench_weight_problems(cfg)
    if problems:
        raise PreflightError("WBench is not runnable; fix before starting a run "
                             "(`ar doctor` for details):\n  " + "\n  ".join(problems))
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache" / "text_embed").mkdir(parents=True, exist_ok=True)
    recorder = _recorder(run_dir, env)
    for name in SNAPSHOT_FILES:
        shutil.copy2(cfg.repo_root / "configs" / name, run_dir / "config" / name)
    wm_sha, wm_dirty = _git_state(cfg.worldmodel)
    wb_sha, wb_dirty = _git_state(cfg.wbench)
    kernel_sha, kernel_dirty = _git_state(cfg.repo_root)
    versions = {"worldmodel_sha": wm_sha, "worldmodel_dirty": wm_dirty,
                "wbench_sha": wb_sha, "wbench_dirty": wb_dirty,
                "kernel_sha": kernel_sha, "kernel_dirty": kernel_dirty}
    (run_dir / "config" / "versions.json").write_text(json.dumps(versions, indent=2))
    metric_set, excluded = preflight_metrics(cfg, env)
    case_ids = (run_dir / "config" / "proxy_cases.txt").read_text().strip().split(",")
    expected_n = _expected_case_counts(cfg, metric_set, recorder)
    # run.json is written last: its presence is what marks the run as created.
    (run_dir / "config" / "run.json").write_text(json.dumps(
        {"run_id": run_id, "metric_set": metric_set, "excluded_metrics": excluded,
         "case_ids": case_ids, "versions": versions, "expected_n": expected_n}, indent=2))
    recorder.event("run.start", payload={"gpus": gpus, "metric_set": metric_set,
                                         "excluded_metrics": excluded, "versions": versions,
                                         "case_ids": case_ids})
    if wm_dirty or wb_dirty:
        recorder.event("run.warning", payload={"message": "sibling repo has uncommitted changes",
                                               "worldmodel_dirty": wm_dirty, "wbench_dirty": wb_dirty})
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=recorder, gpus=gpus,
                      metric_set=metric_set, case_ids=case_ids, versions=versions,
                      expected_n=expected_n)


REFERENCE_REPORT = Path("reference") / "wbench_alayaworld_proxy" / "report.json"


def _expected_case_counts(cfg: KernelConfig, metric_set: list[str], recorder) -> dict:
    """Per-metric case counts on the proxy subset, from the reference report.

    Counts are fixed by case metadata, so every node must match them; a smaller
    count means a precompute step dropped cases. Metrics the reference lacks (the
    VLM ones) get no expectation.
    """
    path = cfg.repo_root / REFERENCE_REPORT
    if not path.exists():
        recorder.event("run.warning", payload={
            "message": f"no reference report at {path}; per-metric case counts will not be checked"})
        return {}
    full = json.loads(path.read_text())["full"]
    return {m: int(full[m]["n"]) for m in metric_set if m in full and "n" in full[m]}


def attach_run(cfg: KernelConfig, run_id: str, env: Mapping[str, str]) -> RunContext:
    """Open an existing run without modifying its snapshot or appending run.start.

    For `ar status`, `ar score-node`, and resuming. A missing run is an error --
    it used to be created silently, so a typo'd run id made a fresh run.
    """
    run_dir = cfg.runs_dir / run_id
    meta_path = run_dir / "config" / "run.json"
    if not meta_path.exists():
        raise RunNotFound(f"no run {run_id!r} under {cfg.runs_dir}")
    meta = json.loads(meta_path.read_text())
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=_recorder(run_dir, env),
                      gpus=resolve_gpus(cfg, env), metric_set=meta["metric_set"],
                      case_ids=meta["case_ids"], versions=meta["versions"],
                      expected_n=meta.get("expected_n", {}))


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
    score, per_metric = score_from_report(report, ctx.metric_set, ctx.expected_n)
    if "per_case" not in report:
        ctx.recorder.event("eval.warning", node=node_id, phase="eval", payload={
            "message": "report.json has no per_case scores (older WBench); aggregates are empty"})
    agg = aggregates(cfg, report, ctx.case_ids, ctx.metric_set)
    ctx.recorder.event("eval.scored", node=node_id, phase="eval",
                       payload={"score": score, "metrics": per_metric, "aggregates": agg})
    removed = cleanup_eval(work_dir, model)
    if merged is not None and merged.exists():
        shutil.rmtree(merged)
        removed.append("merge_slot")
    ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})
    return score, {"metrics": per_metric, "aggregates": agg, "report": report}
