import json, pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.score import DIMENSION_METRICS, resolve_metric_set, score_from_report, cleanup_eval

CFG = KernelConfig.load()
REPORT = json.loads((CFG.repo_root / "reference" / "wbench_alayaworld_proxy" / "report.json").read_text())

def test_dimension_metrics_are_the_22_wbench_metrics():
    assert len(DIMENSION_METRICS) == 22
    assert "navigation_trajectory" in DIMENSION_METRICS
    assert "navigation_accuracy" not in DIMENSION_METRICS

def test_metric_set_excludes_vlm_metrics_without_a_key():
    metrics = resolve_metric_set(CFG, {"VLM_API_KEY": ""})
    assert "scene_adherence" not in metrics
    assert "aesthetic_quality" in metrics

def test_metric_set_includes_vlm_metrics_with_a_key():
    metrics = resolve_metric_set(CFG, {"VLM_API_KEY": "abc"})
    assert "causal_fidelity" in metrics

def test_score_is_the_mean_over_the_metric_set():
    metric_set = [m for m in DIMENSION_METRICS if m in REPORT["full"]]
    score, per_metric = score_from_report(REPORT, metric_set)
    expected = sum(REPORT["full"][m]["mean"] for m in metric_set) / len(metric_set)
    assert score == pytest.approx(expected)
    assert per_metric["aesthetic_quality"] == pytest.approx(REPORT["full"]["aesthetic_quality"]["mean"])

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
