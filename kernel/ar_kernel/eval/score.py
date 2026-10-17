from __future__ import annotations
import json, math, shutil
from pathlib import Path

from ar_contract.metrics import METRICS

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
# What agents are shown instead of the benchmark's own names. Real name -> (alias, dimension alias,
# what it measures); the alias's dimension and meaning are in ar_contract. Applied in memory when a
# context is built; nothing on disk uses an alias.
_ALIASES = {
    "aesthetic_quality": "frame_aesthetics",
    "imaging_quality": "frame_clarity",
    "temporal_flickering": "flicker_free",
    "dynamic_degree": "motion_amount",
    "motion_smoothness": "smooth_motion",
    "hpsv3_quality": "human_preference",
    "background_consistency": "background_stability",
    "segment_continuity": "no_hard_cuts",
    "perspective_consistency": "subject_framing_stability",
    "subject_consistency": "subject_stability",
    "geometric_consistency": "geometry_stability",
    "photometric_consistency": "appearance_stability",
    "spatial_consistency": "revisit_match",
    "gated_spatial_consistency": "revisit_match_strict",
    "navigation_trajectory": "camera_path_accuracy",
    "event_edit_adherence": "follows_event_instruction",
    "subject_action_adherence": "follows_subject_action",
    "perspective_switch_adherence": "follows_viewpoint_change",
    "scene_adherence": "scene_matches_description",
    "subject_adherence": "subject_matches_description",
    "visual_plausibility": "looks_plausible",
    "causal_fidelity": "cause_and_effect",
}
AGENT_METRICS = {name: (alias, *METRICS[alias]) for name, alias in _ALIASES.items()}
AGENT_DIMENSIONS = {"interaction": "control", "setting": "description", "physical": "physics"}
AGENT_AXES = {"perspective": "viewpoint"}


def agent_metrics(metrics: dict) -> dict:
    return {AGENT_METRICS[name][0]: value for name, value in metrics.items()}


def _agent_dimensions(dimensions: dict) -> dict:
    return {AGENT_DIMENSIONS.get(name, name): value for name, value in dimensions.items()}


def agent_aggregates(aggregates: dict | None) -> dict | None:
    """`aggregates()` as an agent sees it: every metric, dimension and group axis renamed."""
    if not aggregates:
        return None
    return {"metrics": agent_metrics(aggregates.get("metrics") or {}),
            "dimensions": _agent_dimensions(aggregates.get("dimensions") or {}),
            "groups": {AGENT_AXES.get(axis, axis): {name: _agent_dimensions(dims) for name, dims in groups.items()}
                       for axis, groups in (aggregates.get("strata") or {}).items()}}


def metric_guide(metric_set: list[str], weights: dict | None) -> dict:
    """What each scored metric measures and how much it weighs in the score. Nothing about how it is judged."""
    return {AGENT_METRICS[name][0]: {"dimension": AGENT_METRICS[name][1], "weight": float((weights or {}).get(name, 1.0)),
                                     "measures": AGENT_METRICS[name][2]}
            for name in metric_set}


# _megasam_tmp: MegaSAM scratch, which WBench now writes beside its output rather than
# inside WBench; a killed run leaves ~1 GB per in-flight case there.
REGENERABLE = ("da3_cache", "megasam", "masks", "_navi_videos_tmp", "_megasam_tmp")

NAV_PARTS = ("navigation_accuracy", "navigation_consistency")


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
        wrong = {m: (full[m].get("n"), expected_n[m]) for m in (*metric_set, *NAV_PARTS)
                 if m in expected_n and m in full and full[m].get("n") != expected_n[m]}
        if wrong:
            raise ScoreError("metrics computed over the wrong number of cases (got vs expected): "
                             + ", ".join(f"{m} {got} vs {want}" for m, (got, want) in wrong.items()))
    return weighted_score(per_metric, weights), per_metric


def case_counts(report: dict, metric_set: list[str]) -> dict[str, int]:
    """The case count behind each scored mean, and behind each half of navigation_trajectory."""
    return {m: int(report["full"][m]["n"]) for m in (*metric_set, *NAV_PARTS) if m in report["full"]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregates(cfg: KernelConfig, report: dict, case_ids: list[str], metric_set: list[str]) -> dict:
    """Agent-facing aggregates of one node's WBench report, over the run's metric set:
    per-metric means, per-dimension means (WBench's grouping) and, for each group of cases
    sharing a scene category or perspective, the same per-dimension means
    over that group's cases alone.

    navigation_trajectory is the mean of its two components' means, as WBench scores it (a case
    may lack one component, so this is not the mean of per-case means). No case ids leave this
    function.
    """
    wanted = {*metric_set, *(NAV_PARTS if "navigation_trajectory" in metric_set else ())}
    per_case = {case_id: {m: float(v) for m, v in scores.items() if m in wanted}
                for case_id, scores in report["per_case"].items()}

    def metric_means(ids) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for case_id in ids:
            for metric, value in per_case.get(case_id, {}).items():
                values.setdefault(metric, []).append(value)
        nav = [_mean(values[p]) for p in NAV_PARTS if p in values]
        if "navigation_trajectory" in wanted and nav:
            values["navigation_trajectory"] = nav
        return {m: _mean(values[m]) for m in metric_set if m in values}

    def dimension_means(metrics: dict[str, float]) -> dict[str, float]:
        return {dim: _mean([metrics[m] for m in names if m in metrics])
                for dim, names in report["dimensions"].items() if any(m in metrics for m in names)}

    groups: dict[str, dict[str, list[str]]] = {"category": {}, "perspective": {}}
    for case_id in case_ids:
        if not per_case.get(case_id):
            continue
        case = json.loads((cfg.eval_data / "cases" / f"case_{case_id}.json").read_text())
        settings = case.get("settings") or {}
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
