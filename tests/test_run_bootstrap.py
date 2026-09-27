import json
from pathlib import Path
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


def _runs(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))


ENV4 = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}


def test_attaching_to_a_missing_run_is_an_error_not_a_new_run(tmp_path, monkeypatch):
    """Review I7: `ar status --run-id <typo>` used to silently create a run."""
    import pytest
    from ar_kernel.run import RunNotFound, attach_run
    _runs(tmp_path, monkeypatch)
    with pytest.raises(RunNotFound):
        attach_run(CFG, "no_such_run", ENV4)
    assert not (tmp_path / "no_such_run").exists()


def test_attach_does_not_rewrite_the_run_snapshot(tmp_path, monkeypatch):
    """Review I7: every `ar status` re-copied configs over the snapshot and
    rewrote versions.json, erasing the record of what the run started on."""
    from ar_kernel.run import attach_run
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r1", env=ENV4)
    versions = ctx.run_dir / "config" / "versions.json"
    versions.write_text('{"worldmodel_sha": "the-sha-the-run-started-on"}')
    attached = attach_run(CFG, "r1", ENV4)
    assert versions.read_text() == '{"worldmodel_sha": "the-sha-the-run-started-on"}'
    assert attached.case_ids == ctx.case_ids and attached.metric_set == ctx.metric_set
    starts = [e for e in attached.recorder.read_events() if e["type"] == "run.start"]
    assert len(starts) == 1


def test_bootstrapping_an_existing_run_id_does_not_overwrite_it(tmp_path, monkeypatch):
    _runs(tmp_path, monkeypatch)
    bootstrap_run(CFG, run_id="r2", env=ENV4)
    snap = tmp_path / "r2" / "config" / "base_recipe.yaml"
    snap.write_text(snap.read_text() + "\n# edited after the run started\n")
    bootstrap_run(CFG, run_id="r2", env=ENV4)
    assert snap.read_text().endswith("# edited after the run started\n")


def test_run_uses_its_own_config_snapshot(tmp_path, monkeypatch):
    """Gate and bootstrap must read the run's snapshot, not the live repo config,
    so a mid-run edit or pull cannot change what an in-flight run does."""
    from ar_kernel.run import run_config_path
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r3", env=ENV4)
    for name in ("kernel.yaml", "base_recipe.yaml", "proxy_cases.txt"):
        assert run_config_path(CFG, ctx.run_dir, name) == ctx.run_dir / "config" / name
        assert (ctx.run_dir / "config" / name).exists()


def test_bootstrap_refuses_to_start_when_wbench_weights_are_broken(tmp_path, monkeypatch):
    """Review I8: a run must not start when it cannot produce a complete score."""
    import pytest
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    monkeypatch.setattr(run_mod, "wbench_weight_problems",
                        lambda cfg: ["MegaSAM weights missing: /x/megasam_final.pth"])
    with pytest.raises(run_mod.PreflightError, match="MegaSAM"):
        bootstrap_run(CFG, run_id="broken", env=ENV4)
    assert not (tmp_path / "broken" / "config" / "run.json").exists()


def test_score_node_degrades_and_still_succeeds_without_per_case_scores(tmp_path, monkeypatch):
    """An older WBench report.json (no "per_case") must not crash score_node after the
    score is already computed (Follow-up B item 5): aggregates come back empty and a
    warning event is recorded, but score_node still returns a score."""
    import pytest
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-score", env=ENV4)
    ctx.metric_set = ["aesthetic_quality"]
    ctx.expected_n = {}
    fake_report = {"full": {"aesthetic_quality": {"mean": 0.5, "n": 1}}}   # no "per_case"
    monkeypatch.setattr(run_mod, "build_render_config", lambda *a, **k: object())
    monkeypatch.setattr(run_mod, "render_proxy", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "run_wbench_phases", lambda *a, **k: fake_report)

    score, detail = run_mod.score_node(CFG, ctx, "n1", checkpoint=None, rank=8, alpha=16)

    assert score == pytest.approx(0.5)
    assert detail["aggregates"] == {"metrics": {}, "dimensions": {},
                                    "strata": {"interaction_type": {}, "category": {}, "perspective": {}}}
    warnings = [e for e in ctx.recorder.read_events("n1") if e["type"] == "eval.warning"]
    assert len(warnings) == 1
    assert "per_case" in ctx.recorder.load_payload(warnings[0]["payload"])["message"]


def test_score_node_cleans_up_the_merge_slot_even_when_wbench_fails(tmp_path, monkeypatch):
    """score_node must not leak the merge_slot (large merged weights) or the
    regenerable eval dirs when a later phase (WBench) blows up on GPU."""
    import pytest
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-cleanup", env=ENV4)
    ctx.metric_set = ["aesthetic_quality"]
    ctx.expected_n = {}
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()

    def fake_merge_lora(cfg, checkpoint, rank, alpha, run_dir, recorder, node_id):
        slot = Path(run_dir) / "merge_slot"
        slot.mkdir(parents=True)
        return slot
    monkeypatch.setattr(run_mod, "merge_lora", fake_merge_lora)
    monkeypatch.setattr(run_mod, "build_render_config", lambda *a, **k: object())
    monkeypatch.setattr(run_mod, "render_proxy", lambda *a, **k: None)

    def fake_run_wbench_phases(*a, **k):
        raise RuntimeError("wbench gpu failed")
    monkeypatch.setattr(run_mod, "run_wbench_phases", fake_run_wbench_phases)

    with pytest.raises(RuntimeError, match="wbench gpu failed"):
        run_mod.score_node(CFG, ctx, "n1", checkpoint=checkpoint, rank=8, alpha=16)

    assert not (ctx.run_dir / "merge_slot").exists()
    cleanups = [e for e in ctx.recorder.read_events("n1") if e["type"] == "eval.cleanup"]
    assert len(cleanups) == 1


def test_bootstrap_records_expected_case_counts_from_the_reference(tmp_path, monkeypatch):
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r_n", env=ENV4)
    assert ctx.expected_n.get("geometric_consistency") == 40
    assert ctx.expected_n.get("spatial_consistency") == 8
    from ar_kernel.run import attach_run
    assert attach_run(CFG, "r_n", ENV4).expected_n == ctx.expected_n
