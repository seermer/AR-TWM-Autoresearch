from __future__ import annotations
import datetime as dt
import hashlib, json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping
import yaml

from . import fs, subproc, telemetry
from .archive.db import open_db, open_db_readonly
from .archive.nodes import NodeStore
from .control import Control
from .doctor import ENVS, wbench_weight_problems
from .config import OVERLAY_SNAPSHOT, SNAPSHOT_FILES, KernelConfig, live_config_path
from .eval.judge import Judge, resolve_judge
from .eval.lora import concat_eval_lora
from .eval.render import build_render_config, render_proxy
from .eval.score import (DIMENSION_METRICS, aggregates, case_counts, cleanup_eval,
                         score_from_report)
from .eval.wbench import run_wbench_phases
from .guards import choose_gpus

@dataclass
class RunContext:
    run_dir: Path
    conn: object
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
    return subproc.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True).stdout.strip()


def _uncommitted(repo: Path) -> bool:
    """Modified or untracked (not ignored) files in the repo."""
    return bool(subproc.run(["git", "-C", str(repo), "status", "--porcelain"], check=True).stdout.strip())

class RunNotFound(FileNotFoundError):
    """No run with this id exists."""


class PreflightError(RuntimeError):
    """The environment cannot produce a complete, comparable score."""


@dataclass
class NewRun:
    """What a new run is created with, from its start checks."""
    run_id: str
    judge: Judge
    case_ids: list[str]
    metric_set: list[str]
    expected_n: dict
    versions: dict


def _repos(cfg: KernelConfig) -> dict[str, Path]:
    return {"worldmodel": cfg.worldmodel, "wbench": cfg.wbench, "kernel": cfg.repo_root}


def check_new_run(cfg: KernelConfig, run_id: str | None, env: Mapping[str, str]) -> NewRun:
    """A new run's start checks: uncommitted changes, WBench weights, the judge, the proxy cases. Writes nothing."""
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # A run is identified by the three commits it ran on (its scores, and a reused root, are only
    # comparable for the same code), so nothing may be uncommitted.
    dirty = [str(path) for path in _repos(cfg).values() if _uncommitted(path)]
    if dirty:
        raise PreflightError("uncommitted changes in " + ", ".join(dirty) + "; commit them before starting a run")
    problems = wbench_weight_problems(cfg)
    if problems:
        raise PreflightError("WBench is not runnable; fix before starting a run "
                             "(`ar doctor` for details):\n  " + "\n  ".join(problems))
    judge = resolve_judge(cfg, env)
    ordered = cfg.eval_cases.read_text().strip().split(",")
    size = int(cfg.get("eval.proxy_size"))
    if not 0 < size <= len(ordered):
        raise PreflightError(f"eval.proxy_size must be between 1 and {len(ordered)}, got {size}")
    case_ids = ordered[:size]
    return NewRun(run_id, judge, case_ids, preflight_metrics(cfg, case_ids), initial_expected_n(cfg, case_ids),
                  {f"{name}_sha": _head(path) for name, path in _repos(cfg).items()})


def bootstrap_run(cfg: KernelConfig, new: NewRun, gpus: list[int]) -> RunContext:
    """Create the checked run, freezing its configs and recording versions."""
    run_dir = cfg.runs_dir / new.run_id
    fs.mkdir(run_dir / "config")
    fs.mkdir(run_dir / "cache" / "text_embed")
    for name in SNAPSHOT_FILES:
        fs.copy(live_config_path(cfg, name), run_dir / "config" / name)
    fs.copy(cfg.eval_cases, run_dir / "config" / "proxy_cases.txt")
    if cfg.overlay is not None:
        fs.write_text(run_dir / "config" / OVERLAY_SNAPSHOT, yaml.safe_dump(cfg.overlay, sort_keys=False))
    fs.write_text(run_dir / "config" / "versions.json", json.dumps(new.versions, indent=2))
    # run.json is written last: its presence is what marks the run as created.
    fs.write_text(run_dir / "config" / "run.json", json.dumps(
        {"run_id": new.run_id, "metric_set": new.metric_set, "judge": asdict(new.judge),
         "case_ids": new.case_ids, "versions": new.versions, "expected_n": new.expected_n}, indent=2))
    return RunContext(run_dir=run_dir, conn=open_db(run_dir), gpus=gpus,
                      metric_set=new.metric_set, case_ids=new.case_ids, versions=new.versions,
                      expected_n=new.expected_n, judge=new.judge)


def record_envs(cfg: KernelConfig) -> None:
    """`conda list` of every env the run uses, each a process span with the list as its output."""
    tools = [cfg.get(k) or {} for k in ("annotate", "images")] + list((cfg.get("generators") or {}).values())
    envs = dict.fromkeys(("autoresearcher", "agent", *ENVS, cfg.get("captioner.env"),
                          *(b["env"] for b in tools if b.get("enabled") and b.get("env"))))
    for env in envs:
        subproc.run(["conda", "list", "-p", str(subproc.env_path(env)), "--json"])


def initial_expected_n(cfg: KernelConfig, case_ids: list[str]) -> dict:
    """Case counts known up front: each case file lists the metrics that apply to it, so a root with a
    missing or failed case fails. The root's report then sets the counts every later node must match."""
    return dict(Counter(metric for case in _cases(cfg, case_ids) for metric in case["metric_list"]))


def record_root_counts(ctx: RunContext, expected_n: dict) -> None:
    ctx.expected_n = dict(expected_n)
    path = ctx.run_dir / "config" / "run.json"
    meta = json.loads(path.read_text())
    meta["expected_n"] = ctx.expected_n
    fs.write_text(path, json.dumps(meta, indent=2), atomic=True)


def root_key(ctx: RunContext, cfg: KernelConfig) -> str:
    return key_of(root_key_fields(ctx, cfg))


def key_of(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def root_key_fields(ctx: RunContext, cfg: KernelConfig) -> dict:
    """What the root's score depends on, hashed by root_key: the same key means the same unedited model, cases, metrics,
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
                              if not k.endswith("timeout_s") and k not in ("concurrent_requests", "min_gpu_gib")}
                             if local else None,
              "eval_code": eval_code,
              "render_overrides": cfg.get("eval.render_overrides"),
              "versions": {k: ctx.versions.get(k) for k in ("worldmodel_sha", "wbench_sha")}}
    return fields


def _meta(cfg: KernelConfig, run_id: str) -> dict:
    """A missing run is an error, so a mistyped run id cannot make a fresh run."""
    meta_path = cfg.runs_dir / run_id / "config" / "run.json"
    if not meta_path.exists():
        raise RunNotFound(f"no run {run_id!r} under {cfg.runs_dir}")
    return json.loads(meta_path.read_text())


def check_resume(cfg: KernelConfig, run_id: str, env: Mapping[str, str]) -> Judge:
    """A resume's start check: the environment selects the judge the run was created with. Writes nothing."""
    return _stored_judge(cfg, env, _meta(cfg, run_id))


def attach_run(cfg: KernelConfig, run_id: str, judge: Judge, gpus: list[int]) -> RunContext:
    """Open an existing, checked run without modifying its snapshot."""
    meta = _meta(cfg, run_id)
    return RunContext(run_dir=cfg.runs_dir / run_id, conn=open_db(cfg.runs_dir / run_id), gpus=gpus,
                      metric_set=meta["metric_set"], case_ids=meta["case_ids"], versions=meta["versions"],
                      expected_n=meta.get("expected_n", {}), judge=judge)


def code_changes(cfg: KernelConfig, versions: dict) -> list[str]:
    """Each repo whose commit differs from the run's recorded one, or that has uncommitted changes."""
    out = []
    for name, path in _repos(cfg).items():
        was, now = versions.get(f"{name}_sha"), _head(path)
        if was != now:
            out.append(f"{name} was at {str(was)[:8]}, is at {now[:8]}")
        if _uncommitted(path):
            out.append(f"{name} has uncommitted changes")
    return out


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
    with telemetry.span("eval", {"node": node_id, "checkpoint": str(checkpoint) if checkpoint else None,
                                 "rank": rank, "cases": list(ctx.case_ids), "metric_set": list(ctx.metric_set)}) as sp:
        score, scored = _score(cfg, ctx, node_dir, work_dir, model, videos_dir, checkpoint, rank)
        sp.output = {"score": score, **scored}
    return score, scored


def _score(cfg, ctx, node_dir, work_dir, model, videos_dir, checkpoint, rank) -> tuple[float, dict]:
    eval_lora = None

    def cleanup() -> None:
        cleanup_eval(work_dir, model)
        if eval_lora is not None:
            fs.rmtree(eval_lora)

    try:
        if checkpoint is None:
            history = cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
        else:
            eval_lora = concat_eval_lora(cfg, checkpoint, node_dir)
            history = Path(checkpoint) / "history_encoder.pt"
        render_config = build_render_config(cfg, eval_lora, rank, history, videos_dir, ctx.case_ids, node_dir)
        render_proxy(cfg, render_config, ctx.gpus, ctx.case_ids)
        report = run_wbench_phases(cfg, work_dir, model, ctx.gpus, ctx.metric_set, ctx.judge)
        score, per_metric = score_from_report(report, ctx.metric_set, ctx.expected_n,
                                              cfg.get("eval.score_weights"))
        agg = aggregates(cfg, report, ctx.case_ids, ctx.metric_set)
    except Exception:
        # A BaseException that is not an Exception (KeyboardInterrupt, SystemExit, the
        # loop's force-stop signal) skips this: a stopped run's eval files stay until
        # resume clears them.
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
    running = [d.name for d in sorted(cfg.runs_dir.iterdir()) if Control(d).alive_pid() is not None]
    if running:
        raise PreflightError(f"a loop is running ({', '.join(running)}) and holds the GPUs; score a node when it has stopped")
    run_cfg = KernelConfig.for_run(run_dir)
    conn = open_db_readonly(run_dir, writer_alive=False)
    try:
        node = NodeStore(conn).get(node_id)
    finally:
        conn.close()
    if node["checkpoint_path"] is None and node["parent_id"] is not None:
        raise ValueError(f"node {node_id!r} has no checkpoint to score")
    checkpoint = run_dir / node["checkpoint_path"] if node["checkpoint_path"] else None
    out = cfg.scores_dir / f"{run_id}-{node_id}-{dt.datetime.now():%Y%m%d_%H%M%S}"
    fs.mkdir(out)
    ctx = RunContext(run_dir=out, conn=None, gpus=choose_gpus(run_cfg, env),
                     metric_set=meta["metric_set"], case_ids=meta["case_ids"], versions=meta["versions"],
                     expected_n=meta["expected_n"], judge=_stored_judge(run_cfg, env, meta))
    score, detail = score_node(run_cfg, ctx, node_id, checkpoint, node["lora_rank"] or 0)
    fs.write_text(out / "score.json", json.dumps(
        {"run_id": run_id, "node": node_id, "score": score, "metrics": detail["metrics"],
         "aggregates": detail["aggregates"]}, indent=1))
    return score, out
