from __future__ import annotations
import json, shutil
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
REGENERABLE = ("da3_cache", "megasam", "masks", "_navi_videos_tmp")

def resolve_metric_set(cfg: KernelConfig, env: Mapping[str, str]) -> list[str]:
    metrics = list(DIMENSION_METRICS)
    if not env.get("VLM_API_KEY", "").strip():
        metrics = [m for m in metrics if m not in VLM_METRICS]
    vp_weights = cfg.wbench / cfg.get("eval.vp_weights")
    if not vp_weights.exists():
        metrics = [m for m in metrics if m != "visual_plausibility"]
    return metrics

def score_from_report(report: dict, metric_set: list[str]) -> tuple[float, dict]:
    full = report["full"]
    missing = [m for m in metric_set if m not in full]
    if missing:
        raise KeyError(f"metrics missing from the report: {missing}")
    per_metric = {m: float(full[m]["mean"]) for m in metric_set}
    return sum(per_metric.values()) / len(per_metric), per_metric

def aggregates(cfg: KernelConfig, eval_dir: Path, case_ids: list[str]) -> dict:
    """Per-metric means plus means by coarse stratum; no case ids leave this function."""
    cases = {}
    for case_id in case_ids:
        path = cfg.wbench / "data" / "cases" / f"case_{case_id}.json"
        case = json.loads(path.read_text())
        settings = case.get("settings") or {}
        cases[case_id] = {
            "types": sorted({i.get("type") for i in case.get("interactions") or []}),
            "category": ((settings.get("scene") or {}).get("category")),
            "perspective": settings.get("perspective"),
        }
    per_case: dict[str, dict] = {}
    for metric_dir in Path(eval_dir).iterdir():
        if not metric_dir.is_dir():
            continue
        for result in metric_dir.glob("case_*.json"):
            case_id = result.stem.replace("case_", "")
            data = json.loads(result.read_text())
            if isinstance(data.get("score"), (int, float)):
                per_case.setdefault(case_id, {})[metric_dir.name] = float(data["score"])
    strata: dict[str, dict[str, list[float]]] = {"interaction_type": {}, "category": {}, "perspective": {}}
    for case_id, metrics in per_case.items():
        mean = sum(metrics.values()) / len(metrics) if metrics else None
        if mean is None or case_id not in cases:
            continue
        facets = cases[case_id]
        for t in facets["types"]:
            strata["interaction_type"].setdefault(str(t), []).append(mean)
        strata["category"].setdefault(str(facets["category"]), []).append(mean)
        strata["perspective"].setdefault(str(facets["perspective"]), []).append(mean)
    return {axis: {key: sum(v) / len(v) for key, v in buckets.items()}
            for axis, buckets in strata.items()}

def cleanup_eval(work_dir: Path, model: str) -> list[str]:
    removed = []
    for name in REGENERABLE:
        target = Path(work_dir) / model / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return sorted(removed)
