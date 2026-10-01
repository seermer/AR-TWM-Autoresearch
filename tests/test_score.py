import json, pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.score import DIMENSION_METRICS, score_from_report, cleanup_eval

CFG = KernelConfig.load()
REPORT = json.loads((CFG.repo_root / "reference" / "wbench_alayaworld_proxy" / "report.json").read_text())

def test_dimension_metrics_are_the_22_wbench_metrics():
    assert len(DIMENSION_METRICS) == 22
    assert "navigation_trajectory" in DIMENSION_METRICS
    assert "navigation_accuracy" not in DIMENSION_METRICS

def test_score_is_the_mean_over_the_metric_set():
    metric_set = [m for m in DIMENSION_METRICS if m in REPORT["full"]]
    score, per_metric = score_from_report(REPORT, metric_set)
    expected = sum(REPORT["full"][m]["mean"] for m in metric_set) / len(metric_set)
    assert score == pytest.approx(expected)
    assert per_metric["aesthetic_quality"] == pytest.approx(REPORT["full"]["aesthetic_quality"]["mean"])

def test_weights_shift_the_score_toward_the_weighted_metrics():
    report = _report(event_edit_adherence=(0.2, 12), aesthetic_quality=(0.8, 50), imaging_quality=(0.8, 50))
    metric_set = ["event_edit_adherence", "aesthetic_quality", "imaging_quality"]
    score, per_metric = score_from_report(report, metric_set, weights={"event_edit_adherence": 2})
    assert score == pytest.approx((2 * 0.2 + 0.8 + 0.8) / 4)
    assert per_metric["event_edit_adherence"] == 0.2


def test_the_four_weighted_grades_are_half_the_score():
    weights = CFG.get("eval.score_weights")
    assert sorted(weights) == ["causal_fidelity", "event_edit_adherence", "perspective_switch_adherence",
                               "subject_action_adherence"]
    assert sum(weights.values()) == len(DIMENSION_METRICS) - len(weights)
    report = _report(**{m: ((1.0 if m in weights else 0.0), 1) for m in DIMENSION_METRICS})
    assert score_from_report(report, DIMENSION_METRICS, weights=weights)[0] == pytest.approx(0.5)


def test_missing_metric_raises(tmp_path):
    with pytest.raises(KeyError, match="visual_plausibility"):
        score_from_report(REPORT, ["aesthetic_quality", "visual_plausibility"])

def test_cleanup_removes_regenerable_dirs_only(tmp_path):
    model_dir = tmp_path / "work_dirs" / "m"
    for name in ("videos", "evaluation", "da3_cache", "megasam", "masks", "_navi_videos_tmp"):
        (model_dir / name).mkdir(parents=True)
    removed = cleanup_eval(tmp_path / "work_dirs", "m")
    assert sorted(removed) == ["_navi_videos_tmp", "da3_cache", "masks", "megasam"]
    assert (model_dir / "videos").exists() and (model_dir / "evaluation").exists()


def _report(**metrics):
    return {"full": {m: {"mean": mean, "n": n} for m, (mean, n) in metrics.items()}}


def test_metric_computed_on_fewer_cases_than_expected_is_refused():
    """Review I5: if DA3 failed on 10 of 40 cases, geometric consistency was
    averaged over 30 and the node scored normally on a different case set --
    a wrong result that looks clean."""
    import pytest
    from ar_kernel.eval.score import ScoreError
    report = _report(geometric_consistency=(0.88, 30), aesthetic_quality=(0.57, 40))
    with pytest.raises(ScoreError, match="geometric_consistency.*30.*40"):
        score_from_report(report, ["geometric_consistency", "aesthetic_quality"],
                          expected_n={"geometric_consistency": 40, "aesthetic_quality": 40})


def test_matching_case_counts_score_normally():
    report = _report(geometric_consistency=(0.8, 40), spatial_consistency=(0.6, 8))
    score, per = score_from_report(report, ["geometric_consistency", "spatial_consistency"],
                                   expected_n={"geometric_consistency": 40, "spatial_consistency": 8})
    assert abs(score - 0.7) < 1e-9


def test_metric_without_an_expectation_is_not_count_checked():
    """VLM metrics are absent from the GPU-only reference; no n to compare against."""
    score, _ = score_from_report(_report(scene_adherence=(0.5, 40)), ["scene_adherence"],
                                 expected_n={})
    assert score == 0.5


def test_nan_metric_mean_is_refused():
    """A NaN mean passed through, and sqlite stored the NaN score as NULL."""
    import pytest
    from ar_kernel.eval.score import ScoreError
    with pytest.raises(ScoreError, match="not finite"):
        score_from_report(_report(aesthetic_quality=(float("nan"), 40)), ["aesthetic_quality"])


def test_dimension_metrics_match_wbench_dimension_map():
    """DIMENSION_METRICS copies WBench's DIMENSION_MAP (main.py). If WBench adds,
    drops or renames a metric, the kernel's fixed metric set silently diverges
    from what the report produces -- fail here instead."""
    import ast
    from ar_kernel.config import KernelConfig
    src = (KernelConfig.load().wbench / "main.py").read_text()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "DIMENSION_MAP" for t in n.targets))
    upstream = [m for metrics in ast.literal_eval(node.value).values() for m in metrics]
    assert DIMENSION_METRICS == upstream


def test_cleanup_reclaims_megasam_scratch(tmp_path):
    (tmp_path / "m" / "_megasam_tmp" / "megasam_case_1_x").mkdir(parents=True)
    assert "_megasam_tmp" in cleanup_eval(tmp_path, "m")
    assert not (tmp_path / "m" / "_megasam_tmp").exists()


# A 3-case report.json written by WBench's generate_report from real per-case metric
# files of runs/manual_root (cases 136, 84, 133), so every metric file layout is covered.
FIXTURE_REPORT = json.loads((CFG.repo_root / "tests" / "fixtures" / "wbench_report" / "report.json").read_text())
FIXTURE_CASES = ["136", "84", "133"]
FIXTURE_METRICS = [m for m in DIMENSION_METRICS if m in FIXTURE_REPORT["full"]]


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)


def test_aggregates_cover_every_metric_in_the_run_metric_set():
    """Review: reading only a top-level "score" per case file missed 9 of 15 metrics
    (VQ summaries, reconstruction, navigation, ungated spatial)."""
    from ar_kernel.eval.score import aggregates
    assert len(FIXTURE_METRICS) == 15
    agg = aggregates(CFG, FIXTURE_REPORT, FIXTURE_CASES, FIXTURE_METRICS)
    assert sorted(agg["metrics"]) == sorted(FIXTURE_METRICS)
    # WBench rounds its means to 4 places, navigation_trajectory twice (components, then theirs).
    for m in FIXTURE_METRICS:
        assert agg["metrics"][m] == pytest.approx(FIXTURE_REPORT["full"][m]["mean"], abs=1e-4)


def test_aggregates_group_metric_means_by_wbench_dimension():
    from ar_kernel.eval.score import aggregates
    agg = aggregates(CFG, FIXTURE_REPORT, FIXTURE_CASES, FIXTURE_METRICS)
    dims = FIXTURE_REPORT["dimensions"]
    assert sorted(agg["dimensions"]) == ["consistency", "interaction", "quality"]
    for dim in ("quality", "consistency"):
        expected = [agg["metrics"][m] for m in dims[dim]]
        assert agg["dimensions"][dim] == pytest.approx(sum(expected) / len(expected))
    assert agg["dimensions"]["interaction"] == agg["metrics"]["navigation_trajectory"]


def test_aggregates_strata_give_each_dimension_per_group_and_leak_no_case_ids():
    from ar_kernel.eval.score import aggregates
    per_case = FIXTURE_REPORT["per_case"]
    agg = aggregates(CFG, FIXTURE_REPORT, FIXTURE_CASES, ["spatial_consistency"])
    # Only cases 136 (Indoor) and 84 (Urban) have the ungated spatial score (ret_sim).
    assert agg["strata"]["category"] == {"Indoor": {"consistency": per_case["136"]["spatial_consistency"]},
                                         "Urban": {"consistency": per_case["84"]["spatial_consistency"]}}
    full = aggregates(CFG, FIXTURE_REPORT, FIXTURE_CASES, FIXTURE_METRICS)
    # A group's dimension is the mean of that dimension's metric means over the group's cases alone.
    quality = FIXTURE_REPORT["dimensions"]["quality"]
    indoor = full["strata"]["category"]["Indoor"]
    assert sorted(indoor) == ["consistency", "interaction", "quality"]
    assert indoor["quality"] == pytest.approx(sum(per_case["136"][m] for m in quality) / len(quality))
    assert sorted(full["strata"]["interaction_type"]) == ["navigation", "subject_action"]
    assert sorted(full["strata"]["perspective"]) == ["first_person", "third_person"]
    assert not set(_keys(full)) & set(per_case)


