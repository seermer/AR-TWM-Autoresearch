from __future__ import annotations
import json, math, shutil
from pathlib import Path

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
# _megasam_tmp: MegaSAM scratch, which WBench now writes beside its output rather than
# inside WBench; a killed run leaves ~1 GB per in-flight case there.
REGENERABLE = ("da3_cache", "megasam", "masks", "_navi_videos_tmp", "_megasam_tmp")

# Computed for every proxy case, so their n is known before any node is scored. The other
# metrics apply to a subset of cases; their n is taken from the root's report.
UNIVERSAL_METRICS = (
    "aesthetic_quality", "imaging_quality", "temporal_flickering", "dynamic_degree",
    "motion_smoothness", "hpsv3_quality", "background_consistency", "segment_continuity",
    "geometric_consistency", "photometric_consistency",
)

class ScoreError(ValueError):
    """The report cannot yield a score comparable with the rest of the run."""


def weighted_score(per_metric: dict[str, float], weights: dict[str, float] | None) -> float:
    """Weighted mean of the metric means; a metric without a weight weighs 1."""
    weights = weights or {}
    total = sum(weights.get(m, 1.0) for m in per_metric)
    return sum(v * weights.get(m, 1.0) for m, v in per_metric.items()) / total


def score_from_report(report: dict, metric_set: list[str], expected_n: dict[str, int] | None = None,
                      weights: dict[str, float] | None = None) -> tuple[float, dict]:
    """Weighted mean of the run's fixed metric set. Refuses, rather than scores, a report
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
    return weighted_score(per_metric, weights), per_metric

NAV_PARTS = ("navigation_accuracy", "navigation_consistency")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregates(cfg: KernelConfig, report: dict, case_ids: list[str], metric_set: list[str]) -> dict:
    """Agent-facing aggregates of one node's WBench report, over the run's metric set:
    per-metric means, per-dimension means (WBench's grouping) and, for each group of cases
    sharing an interaction type, scene category or perspective, the same per-dimension means
    over that group's cases alone.

    navigation_trajectory is the mean of its two components, as in WBench. No case ids
    leave this function.
    """
    wanted = set(metric_set)
    per_case: dict[str, dict[str, float]] = {}
    for case_id, scores in report["per_case"].items():
        case = {m: float(v) for m, v in scores.items() if m in wanted}
        nav = [float(scores[p]) for p in NAV_PARTS if p in scores]
        if "navigation_trajectory" in wanted and nav:
            case["navigation_trajectory"] = _mean(nav)
        per_case[case_id] = case

    def metric_means(ids) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for case_id in ids:
            for metric, value in per_case.get(case_id, {}).items():
                values.setdefault(metric, []).append(value)
        return {m: _mean(values[m]) for m in metric_set if m in values}

    def dimension_means(metrics: dict[str, float]) -> dict[str, float]:
        return {dim: _mean([metrics[m] for m in names if m in metrics])
                for dim, names in report["dimensions"].items() if any(m in metrics for m in names)}

    groups: dict[str, dict[str, list[str]]] = {"interaction_type": {}, "category": {}, "perspective": {}}
    for case_id in case_ids:
        if not per_case.get(case_id):
            continue
        case = json.loads((cfg.wbench / "data" / "cases" / f"case_{case_id}.json").read_text())
        settings = case.get("settings") or {}
        for t in sorted({i.get("type") for i in case.get("interactions") or []}, key=str):
            groups["interaction_type"].setdefault(str(t), []).append(case_id)
        groups["category"].setdefault(str((settings.get("scene") or {}).get("category")), []).append(case_id)
        groups["perspective"].setdefault(str(settings.get("perspective")), []).append(case_id)
    metrics = metric_means(per_case)
    return {"metrics": metrics, "dimensions": dimension_means(metrics),
            "strata": {axis: {name: dimension_means(metric_means(ids)) for name, ids in buckets.items()}
                       for axis, buckets in groups.items()}}

def cleanup_eval(work_dir: Path, model: str) -> list[str]:
    removed = []
    for name in REGENERABLE:
        target = Path(work_dir) / model / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return sorted(removed)
