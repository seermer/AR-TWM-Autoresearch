from __future__ import annotations
import json, math, shutil
from pathlib import Path
from typing import Mapping

from ..config import KernelConfig

DIMENSION_METRICS = [
    "aesthetic_quality", "imaging_quality", "temporal_flickering", "dynamic_degree",
    "motion_smoothness", "hpsv3_quality",
    "background_consistency", "segment_continuity", "perspective_consistency",
    "subject_consistency", "geometric_consistency", "photometric_consistency",
    "spatial_consistency", "gated_spatial_consistency",
    "navigation_trajectory", "event_edit_adherence", "subject_action_adherence",
    "perspective_switch_adherence",
    "scene_adherence", "subject_adherence",
    "visual_plausibility", "causal_fidelity",
]
VLM_METRICS = {"scene_adherence", "subject_adherence", "causal_fidelity",
               "event_edit_adherence", "subject_action_adherence", "perspective_switch_adherence"}
# _megasam_tmp: MegaSAM scratch, which WBench now writes beside its output rather than
# inside WBench; a killed run leaves ~1 GB per in-flight case there.
REGENERABLE = ("da3_cache", "megasam", "masks", "_navi_videos_tmp", "_megasam_tmp")

def resolve_metric_set(cfg: KernelConfig, env: Mapping[str, str]) -> list[str]:
    metrics = list(DIMENSION_METRICS)
    if not env.get("VLM_API_KEY", "").strip():
        metrics = [m for m in metrics if m not in VLM_METRICS]
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        metrics = [m for m in metrics if m != "visual_plausibility"]
    return metrics

class ScoreError(ValueError):
    """The report cannot yield a score comparable with the rest of the run."""


def score_from_report(report: dict, metric_set: list[str],
                      expected_n: dict[str, int] | None = None) -> tuple[float, dict]:
    """Mean of the run's fixed metric set. Refuses, rather than scores, a report
    that is incomplete, non-finite, or computed over the wrong cases.

    `expected_n` maps metric -> case count. Per-metric counts are fixed by case
    metadata on the proxy subset, so a smaller n means a precompute step silently
    dropped cases (DA3 or SAM2 failing on some) and the mean is over a different
    case set than every other node's.
    """
    full = report["full"]
    missing = [m for m in metric_set if m not in full]
    if missing:
        raise KeyError(f"metrics missing from the report: {missing}")
    per_metric = {m: float(full[m]["mean"]) for m in metric_set}
    bad = [m for m, v in per_metric.items() if not math.isfinite(v)]
    if bad:
        raise ScoreError(f"metric means not finite: {bad}")
    if expected_n:
        wrong = {m: (full[m].get("n"), expected_n[m]) for m in metric_set
                 if m in expected_n and full[m].get("n") != expected_n[m]}
        if wrong:
            raise ScoreError("metrics computed over the wrong number of cases (got vs expected): "
                             + ", ".join(f"{m} {got} vs {want}" for m, (got, want) in wrong.items()))
    return sum(per_metric.values()) / len(per_metric), per_metric

NAV_PARTS = ("navigation_accuracy", "navigation_consistency")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregates(cfg: KernelConfig, report: dict, case_ids: list[str], metric_set: list[str]) -> dict:
    """Agent-facing aggregates of one node's WBench report, over the run's metric set:
    per-metric means, per-dimension means (WBench's grouping) and, per coarse stratum
    of the case metadata, the mean of each case's metric mean.

    Reads report["per_case"], which WBench's generate_report fills after parsing every
    metric's file layout. navigation_trajectory is the mean of its two components, as
    in WBench. No case ids leave this function.

    An older WBench report with no "per_case" degrades to empty aggregates rather than
    raising: score_node still has a usable score (from report["full"]) and just logs
    the aggregates as unavailable.
    """
    if "per_case" not in report:
        return {"metrics": {}, "dimensions": {},
                "strata": {"interaction_type": {}, "category": {}, "perspective": {}}}
    wanted = set(metric_set)
    per_metric: dict[str, list[float]] = {}
    per_case_mean: dict[str, float] = {}
    for case_id, scores in report["per_case"].items():
        for metric, value in scores.items():
            per_metric.setdefault(metric, []).append(float(value))
        case_scores = [float(v) for m, v in scores.items() if m in wanted]
        nav = [float(scores[p]) for p in NAV_PARTS if p in scores]
        if "navigation_trajectory" in wanted and nav:
            case_scores.append(_mean(nav))
        if case_scores:
            per_case_mean[case_id] = _mean(case_scores)
    means = {m: _mean(v) for m, v in per_metric.items()}
    if all(p in means for p in NAV_PARTS):
        means["navigation_trajectory"] = _mean([means[p] for p in NAV_PARTS])
    metrics = {m: means[m] for m in metric_set if m in means}
    dimensions = {dim: _mean([metrics[m] for m in names if m in metrics])
                  for dim, names in report["dimensions"].items() if any(m in metrics for m in names)}

    strata: dict[str, dict[str, list[float]]] = {"interaction_type": {}, "category": {}, "perspective": {}}
    for case_id in case_ids:
        if case_id not in per_case_mean:
            continue
        case = json.loads((cfg.wbench / "data" / "cases" / f"case_{case_id}.json").read_text())
        settings = case.get("settings") or {}
        mean = per_case_mean[case_id]
        for t in sorted({i.get("type") for i in case.get("interactions") or []}, key=str):
            strata["interaction_type"].setdefault(str(t), []).append(mean)
        strata["category"].setdefault(str((settings.get("scene") or {}).get("category")), []).append(mean)
        strata["perspective"].setdefault(str(settings.get("perspective")), []).append(mean)
    return {"metrics": metrics, "dimensions": dimensions,
            "strata": {axis: {key: _mean(v) for key, v in buckets.items()}
                       for axis, buckets in strata.items()}}

def cleanup_eval(work_dir: Path, model: str) -> list[str]:
    removed = []
    for name in REGENERABLE:
        target = Path(work_dir) / model / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return sorted(removed)
