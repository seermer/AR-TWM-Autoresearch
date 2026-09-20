import json
from ar_kernel.config import KernelConfig
from ar_kernel.run import bootstrap_run, preflight_metrics

CFG = KernelConfig.load()

def test_preflight_reports_exclusions_with_reasons():
    metrics, excluded = preflight_metrics(CFG, {"VLM_API_KEY": ""})
    assert "scene_adherence" not in metrics
    assert any("VLM_API_KEY" in reason for reason in excluded)

def test_bootstrap_creates_run_layout_and_records_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))
    ctx = bootstrap_run(CFG, run_id="testrun", env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert (ctx.run_dir / "archive.db").exists()
    assert (ctx.run_dir / "config" / "kernel.yaml").exists()
    versions = json.loads((ctx.run_dir / "config" / "versions.json").read_text())
    assert "worldmodel_sha" in versions and "wbench_sha" in versions
    assert versions["worldmodel_dirty"] in (True, False)
    assert ctx.gpus == [0, 1, 2, 3]
    assert len(ctx.case_ids) == 40
    assert ctx.metric_set

def test_bootstrap_refuses_too_few_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))
    try:
        bootstrap_run(CFG, run_id="bad", env={"CUDA_VISIBLE_DEVICES": "0,1"})
    except Exception as exc:
        assert "at least 4" in str(exc)
    else:
        raise AssertionError("expected a GpuPolicyError")
