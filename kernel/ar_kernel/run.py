from __future__ import annotations
import datetime as dt
import json, os, shutil, subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .archive.db import open_db
from .doctor import wbench_weight_problems
from .config import SNAPSHOT_FILES, KernelConfig, resolve_gpus
from .eval.judge import Judge, resolve_judge
from .eval.merge import merge_lora
from .eval.render import build_render_config, render_proxy
from .eval.score import DIMENSION_METRICS, UNIVERSAL_METRICS, aggregates, cleanup_eval, score_from_report
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
    judge: Judge | None = None                       # who answers the VLM metrics

def preflight_metrics(cfg: KernelConfig) -> list[str]:
    """The run's metric set (always all 22); PreflightError when it cannot be computed."""
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        raise PreflightError("the full metric set cannot be computed:\n  "
                             f"visual_plausibility needs the VP weights at {vp_weights}")
    return list(DIMENSION_METRICS)

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
    metric_set = preflight_metrics(cfg)
    judge = resolve_judge(cfg, env)
    case_ids = (run_dir / "config" / "proxy_cases.txt").read_text().strip().split(",")
    expected_n = initial_expected_n(case_ids)
    # run.json is written last: its presence is what marks the run as created.
    (run_dir / "config" / "run.json").write_text(json.dumps(
        {"run_id": run_id, "metric_set": metric_set, "judge": asdict(judge),
         "case_ids": case_ids, "versions": versions, "expected_n": expected_n}, indent=2))
    recorder.event("run.start", payload={"gpus": gpus, "metric_set": metric_set, "judge": asdict(judge), "versions": versions,
                                         "case_ids": case_ids})
    if wm_dirty or wb_dirty:
        recorder.event("run.warning", payload={"message": "sibling repo has uncommitted changes",
                                               "worldmodel_dirty": wm_dirty, "wbench_dirty": wb_dirty})
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=recorder, gpus=gpus,
                      metric_set=metric_set, case_ids=case_ids, versions=versions,
                      expected_n=expected_n, judge=judge)


def initial_expected_n(case_ids: list[str]) -> dict:
    """Case counts known up front: the universal metrics cover every proxy case. The root's
    report adds the rest (score_node), and every later node must match them."""
    return {m: len(case_ids) for m in UNIVERSAL_METRICS}


def _record_root_counts(ctx: RunContext, report: dict) -> None:
    ctx.expected_n = {m: int(report["full"][m]["n"]) for m in ctx.metric_set}
    path = ctx.run_dir / "config" / "run.json"
    meta = json.loads(path.read_text())
    meta["expected_n"] = ctx.expected_n
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)


def attach_run(cfg: KernelConfig, run_id: str, env: Mapping[str, str],
               check_judge: bool = True) -> RunContext:
    """Open an existing run without modifying its snapshot or appending run.start.

    For `ar status`, `ar score-node`, and resuming. A missing run is an error --
    it used to be created silently, so a typo'd run id made a fresh run.
    """
    run_dir = cfg.runs_dir / run_id
    meta_path = run_dir / "config" / "run.json"
    if not meta_path.exists():
        raise RunNotFound(f"no run {run_id!r} under {cfg.runs_dir}")
    meta = json.loads(meta_path.read_text())
    judge = _stored_judge(cfg, env, meta) if check_judge else None
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=_recorder(run_dir, env),
                      gpus=resolve_gpus(cfg, env), metric_set=meta["metric_set"],
                      case_ids=meta["case_ids"], versions=meta["versions"],
                      expected_n=meta.get("expected_n", {}), judge=judge)


def _stored_judge(cfg: KernelConfig, env: Mapping[str, str], meta: dict) -> Judge:
    """The judge the run was created with. Scores from a different judge are not comparable, so
    an environment that would pick another one is refused."""
    stored, now = Judge(**meta["judge"]), resolve_judge(cfg, env)
    if (stored.kind, stored.model) != (now.kind, now.model):
        raise PreflightError(
            f"this run was scored by the {stored.kind} judge {stored.model!r}, but the environment now "
            f"selects the {now.kind} judge {now.model!r}; set VLM_API_KEY and VLM_MODEL_NAME "
            "(or unset VLM_API_KEY) to match, or start a new run")
    return stored


def score_node(cfg: KernelConfig, ctx: RunContext, node_id: str, checkpoint: Path | None,
               rank: int, alpha: int) -> tuple[float, dict]:
    node_dir = ctx.run_dir / "nodes" / node_id
    work_dir = node_dir / "eval" / "work_dirs"
    model = f"ar_{ctx.run_dir.name}_n{node_id}"
    videos_dir = work_dir / model / "videos"
    merged = None

    def cleanup() -> None:
        removed = cleanup_eval(work_dir, model)
        slot = ctx.run_dir / "merge_slot"
        if slot.exists():
            shutil.rmtree(slot)
            removed.append("merge_slot")
        ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})

    try:
        if checkpoint is None:
            history = cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
        else:
            merged = merge_lora(cfg, checkpoint, rank, alpha, ctx.run_dir, ctx.recorder, node_id)
            history = Path(checkpoint) / "history_encoder.pt"
        render_config = build_render_config(cfg, merged, history, videos_dir, ctx.case_ids, node_dir)
        render_proxy(cfg, render_config, ctx.gpus, node_id, ctx.recorder, ctx.case_ids)
        report = run_wbench_phases(cfg, work_dir, model, ctx.gpus, ctx.metric_set, ctx.recorder, node_id,
                                   ctx.judge)
        score, per_metric = score_from_report(report, ctx.metric_set, ctx.expected_n)
        if node_id == "root":
            _record_root_counts(ctx, report)
        agg = aggregates(cfg, report, ctx.case_ids, ctx.metric_set)
        ctx.recorder.event("eval.scored", node=node_id, phase="eval",
                           payload={"score": score, "metrics": per_metric, "aggregates": agg})
    except Exception:
        # A BaseException that is not an Exception (KeyboardInterrupt, SystemExit, the
        # loop's force-stop signal) skips this: an interrupted node's files are never
        # deleted, only a real eval/scoring failure's transient dirs are.
        cleanup()
        raise
    cleanup()
    return score, {"metrics": per_metric, "aggregates": agg, "report": report}
