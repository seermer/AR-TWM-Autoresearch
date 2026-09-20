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
