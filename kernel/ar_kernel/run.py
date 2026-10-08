from __future__ import annotations
import datetime as dt
import hashlib, json, os, shutil, subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .archive.db import open_db, open_db_readonly
from .archive.nodes import NodeStore
from .control import Control
from .doctor import wbench_weight_problems
from .config import OVERLAY_SNAPSHOT, SNAPSHOT_FILES, KernelConfig, resolve_gpus
from .eval.judge import Judge, resolve_judge
from .eval.lora import concat_eval_lora
from .eval.render import build_render_config, render_proxy
from .eval.score import (DIMENSION_METRICS, UNIVERSAL_METRICS, aggregates, case_counts, cleanup_eval,
                         score_from_report)
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

def _cases(cfg: KernelConfig, case_ids: list[str]) -> list[dict]:
    return [json.loads((cfg.eval_data / "cases" / f"case_{i}.json").read_text()) for i in case_ids]


def preflight_metrics(cfg: KernelConfig, case_ids: list[str]) -> list[str]:
    """The run's metric set: every metric its cases list; PreflightError when it cannot be computed."""
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        raise PreflightError("the metric set cannot be computed:\n  "
                             f"visual_plausibility needs the VP weights at {vp_weights}")
    listed = {metric for case in _cases(cfg, case_ids) for metric in case["metric_list"]}
    if unknown := sorted(listed - set(DIMENSION_METRICS)):
        raise PreflightError(f"the cases list metrics the kernel does not know: {unknown}")
    return [metric for metric in DIMENSION_METRICS if metric in listed]

def _head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _uncommitted(repo: Path) -> bool:
    """Modified or untracked (not ignored) files in the repo."""
    return bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip())

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
    # A run is identified by the three commits it ran on (its scores, and a reused root, are only
    # comparable for the same code), so nothing may be uncommitted.
    repos = {"worldmodel": cfg.worldmodel, "wbench": cfg.wbench, "kernel": cfg.repo_root}
    dirty = [str(path) for path in repos.values() if _uncommitted(path)]
    if dirty:
        raise PreflightError("uncommitted changes in " + ", ".join(dirty) + "; commit them before starting a run")
    problems = wbench_weight_problems(cfg)
    if problems:
        raise PreflightError("WBench is not runnable; fix before starting a run "
                             "(`ar doctor` for details):\n  " + "\n  ".join(problems))
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache" / "text_embed").mkdir(parents=True, exist_ok=True)
    recorder = _recorder(run_dir, env)
    for name in SNAPSHOT_FILES:
        shutil.copy2(cfg.repo_root / "configs" / name, run_dir / "config" / name)
    shutil.copy2(cfg.eval_cases, run_dir / "config" / "proxy_cases.txt")
    if cfg.overlay is not None:
        shutil.copy2(cfg.overlay, run_dir / "config" / OVERLAY_SNAPSHOT)
    versions = {f"{name}_sha": _head(path) for name, path in repos.items()}
    (run_dir / "config" / "versions.json").write_text(json.dumps(versions, indent=2))
    judge = resolve_judge(cfg, env)
    ordered = (run_dir / "config" / "proxy_cases.txt").read_text().strip().split(",")
    size = int(cfg.get("eval.proxy_size"))
    if not 0 < size <= len(ordered):
        raise PreflightError(f"eval.proxy_size must be between 1 and {len(ordered)}, got {size}")
    case_ids = ordered[:size]
    metric_set = preflight_metrics(cfg, case_ids)
    expected_n = initial_expected_n(cfg, case_ids)
    # run.json is written last: its presence is what marks the run as created.
    (run_dir / "config" / "run.json").write_text(json.dumps(
        {"run_id": run_id, "metric_set": metric_set, "judge": asdict(judge),
         "case_ids": case_ids, "versions": versions, "expected_n": expected_n}, indent=2))
    recorder.event("run.start", payload={"gpus": gpus, "metric_set": metric_set, "judge": asdict(judge), "versions": versions,
                                         "case_ids": case_ids})
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), recorder=recorder, gpus=gpus,
                      metric_set=metric_set, case_ids=case_ids, versions=versions,
                      expected_n=expected_n, judge=judge)


def initial_expected_n(cfg: KernelConfig, case_ids: list[str]) -> dict:
    """Case counts known up front: the universal metrics cover every proxy case, and the case
    files say which cases each judged metric applies to (so a judge call that never succeeded
    fails the root too). The root's report adds the rest (score_node), and every later node
    must match them."""
    cases = _cases(cfg, case_ids)
    expected = {m: len(case_ids) for m in UNIVERSAL_METRICS}
    for metric in ("scene_adherence", "subject_adherence", "causal_fidelity"):
        expected[metric] = sum(bool(c.get(metric)) for c in cases)
    for kind in ("event_edit", "subject_action", "perspective_switch"):
        expected[f"{kind}_adherence"] = sum(any(i.get("type") == kind for i in c["interactions"]) for c in cases)
    return expected


def record_root_counts(ctx: RunContext, expected_n: dict) -> None:
    ctx.expected_n = dict(expected_n)
    path = ctx.run_dir / "config" / "run.json"
    meta = json.loads(path.read_text())
    meta["expected_n"] = ctx.expected_n
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)


def root_key(ctx: RunContext, cfg: KernelConfig) -> str:
    """What the root's score depends on: the same key means the same unedited model, cases, metrics,
    judge, judge settings and kernel eval code, so a root scored by any earlier run can be reused.
    The score weights are left out: a reused root's metric means are weighed by the reusing run."""
    local = ctx.judge is not None and ctx.judge.kind == "local"
    eval_code = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted((Path(__file__).parent / "eval").glob("*.py")))).hexdigest()
    # the scored cases' own files (text, first frame, mask): the benchmark folder may not be in a git repo
    data = hashlib.sha256(b"".join(
        p.read_bytes() for i in ctx.case_ids for part in ("cases", "images", "masks")
        for p in sorted((cfg.eval_data / part).glob(f"case_{i}[._]*")))).hexdigest()
    fields = {"metric_set": ctx.metric_set, "data": data, "case_ids": ctx.case_ids,
              "judge": [ctx.judge.kind, ctx.judge.model] if ctx.judge else None,
              "judge_settings": cfg.get("eval.judge"), "vp_weights": cfg.get("eval.vp_weights"),
              # a local judge is the captioner's server: how it is served decides what it sees
              "local_judge": {k: v for k, v in cfg.get("captioner").items()
                              if not k.endswith("timeout_s")} if local else None,
              "eval_code": eval_code,
              "versions": {k: ctx.versions.get(k) for k in ("worldmodel_sha", "wbench_sha")}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def attach_run(cfg: KernelConfig, run_id: str, env: Mapping[str, str]) -> RunContext:
    """Open an existing run without modifying its snapshot or appending run.start.

    For resuming. A missing run is an error -- it used to be created silently, so a
    typo'd run id made a fresh run.
    """
    run_dir = cfg.runs_dir / run_id
    meta_path = run_dir / "config" / "run.json"
    if not meta_path.exists():
        raise RunNotFound(f"no run {run_id!r} under {cfg.runs_dir}")
    meta = json.loads(meta_path.read_text())
    judge = _stored_judge(cfg, env, meta)
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
               rank: int) -> tuple[float, dict]:
    node_dir = ctx.run_dir / "nodes" / node_id
    work_dir = node_dir / "eval" / "work_dirs"
    model = f"ar_{ctx.run_dir.name}_{node_id}"
    videos_dir = work_dir / model / "videos"
    eval_lora = None

    def cleanup() -> None:
        removed = cleanup_eval(work_dir, model)
        if eval_lora is not None and eval_lora.exists():
            shutil.rmtree(eval_lora)
            removed.append("lora")
        ctx.recorder.event("eval.cleanup", node=node_id, phase="eval", payload={"removed": removed})

    try:
        if checkpoint is None:
            history = cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
        else:
            eval_lora = concat_eval_lora(cfg, checkpoint, node_dir, ctx.recorder, node_id)
            history = Path(checkpoint) / "history_encoder.pt"
        render_config = build_render_config(cfg, eval_lora, rank, history, videos_dir, ctx.case_ids, node_dir)
        render_proxy(cfg, render_config, ctx.gpus, node_id, ctx.recorder, ctx.case_ids)
        report = run_wbench_phases(cfg, work_dir, model, ctx.gpus, ctx.metric_set, ctx.recorder, node_id,
                                   ctx.judge)
        score, per_metric = score_from_report(report, ctx.metric_set, ctx.expected_n,
                                              cfg.get("eval.score_weights"))
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
    return score, {"metrics": per_metric, "aggregates": agg, "report": report,
                   "counts": case_counts(report, ctx.metric_set)}


def rescore_node(cfg: KernelConfig, run_id: str, node_id: str, env: Mapping[str, str]) -> tuple[float, Path]:
    """Score a node of an existing run again, with the run's cases, metrics, judge and weights.
    Reads the run and writes only under scores/<run>-<node>-<time>/; the run is not modified."""
    run_dir = cfg.runs_dir / run_id
    meta_path = run_dir / "config" / "run.json"
    if not meta_path.exists():
        raise RunNotFound(f"no run {run_id!r} under {cfg.runs_dir}")
    meta = json.loads(meta_path.read_text())
    run_cfg = KernelConfig.for_run(run_dir)
    conn = open_db_readonly(run_dir, writer_alive=Control(run_dir).alive_pid() is not None)
    try:
        node = NodeStore(conn).get(node_id)
    finally:
        conn.close()
    if node["checkpoint_path"] is None and node["parent_id"] is not None:
        raise ValueError(f"node {node_id!r} has no checkpoint to score")
    checkpoint = run_dir / node["checkpoint_path"] if node["checkpoint_path"] else None
    out = cfg.scores_dir / f"{run_id}-{node_id}-{dt.datetime.now():%Y%m%d_%H%M%S}"
    ctx = RunContext(run_dir=out, conn=None, recorder=_recorder(out, env), gpus=resolve_gpus(run_cfg, env),
                     metric_set=meta["metric_set"], case_ids=meta["case_ids"], versions=meta["versions"],
                     expected_n=meta["expected_n"], judge=_stored_judge(run_cfg, env, meta))
    score, detail = score_node(run_cfg, ctx, node_id, checkpoint, node["lora_rank"] or 0)
    (out / "score.json").write_text(json.dumps(
        {"run_id": run_id, "node": node_id, "score": score, "metrics": detail["metrics"],
         "aggregates": detail["aggregates"]}, indent=1))
    return score, out
