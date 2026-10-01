import json
from pathlib import Path
import pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.score import DIMENSION_METRICS
from ar_kernel.run import PreflightError, bootstrap_run, initial_expected_n, preflight_metrics

CFG = KernelConfig.load()

@pytest.mark.real_preflight
def test_preflight_fails_when_vp_weights_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "wbench", property(lambda self: tmp_path))
    with pytest.raises(PreflightError, match="visual_plausibility"):
        preflight_metrics(CFG)

@pytest.mark.real_preflight
def test_preflight_passes_with_vp_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "wbench", property(lambda self: tmp_path))
    (tmp_path / CFG.get("eval.vp_weights")).mkdir(parents=True)
    assert preflight_metrics(CFG) == DIMENSION_METRICS

def test_bootstrap_creates_run_layout_and_records_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(KernelConfig, "runs_dir", property(lambda self: tmp_path))
    ctx = bootstrap_run(CFG, run_id="testrun", env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert (ctx.run_dir / "archive.db").exists()
    assert (ctx.run_dir / "config" / "kernel.yaml").exists()
    versions = json.loads((ctx.run_dir / "config" / "versions.json").read_text())
    assert "worldmodel_sha" in versions and "wbench_sha" in versions
    assert versions["worldmodel_dirty"] in (True, False)
    assert ctx.gpus == [0, 1, 2, 3]
    assert len(ctx.case_ids) == 50
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
    from ar_kernel.config import run_config_path
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


def test_score_node_cleans_up_the_eval_lora_even_when_wbench_fails(tmp_path, monkeypatch):
    """score_node must not leak the concatenated eval LoRA (several GB) or the
    regenerable eval dirs when a later phase (WBench) blows up on GPU."""
    import pytest
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-cleanup", env=ENV4)
    ctx.metric_set = ["aesthetic_quality"]
    ctx.expected_n = {}
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()

    def fake_concat(cfg, checkpoint, node_dir, recorder, node_id):
        out = Path(node_dir) / "eval" / "lora"
        out.mkdir(parents=True)
        return out
    monkeypatch.setattr(run_mod, "concat_eval_lora", fake_concat)
    monkeypatch.setattr(run_mod, "build_render_config", lambda *a, **k: object())
    monkeypatch.setattr(run_mod, "render_proxy", lambda *a, **k: None)

    def fake_run_wbench_phases(*a, **k):
        raise RuntimeError("wbench gpu failed")
    monkeypatch.setattr(run_mod, "run_wbench_phases", fake_run_wbench_phases)

    with pytest.raises(RuntimeError, match="wbench gpu failed"):
        run_mod.score_node(CFG, ctx, "n1", checkpoint=checkpoint, rank=8)

    assert not (ctx.run_dir / "nodes" / "n1" / "eval" / "lora").exists()
    cleanups = [e for e in ctx.recorder.read_events("n1") if e["type"] == "eval.cleanup"]
    assert len(cleanups) == 1


def test_score_node_skips_cleanup_on_a_base_exception(tmp_path, monkeypatch):
    """An interrupted node's files are never deleted (memory: no-pause / interrupted
    nodes). A KeyboardInterrupt (or the loop's force-stop signal) during scoring is not
    an Exception, so it must leave the eval dirs in place and propagate,
    unlike an ordinary Exception failure (the test above), which does clean up."""
    import pytest
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-interrupt", env=ENV4)
    ctx.metric_set = ["aesthetic_quality"]
    ctx.expected_n = {}
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()

    def fake_concat(cfg, checkpoint, node_dir, recorder, node_id):
        out = Path(node_dir) / "eval" / "lora"
        out.mkdir(parents=True)
        return out
    monkeypatch.setattr(run_mod, "concat_eval_lora", fake_concat)
    monkeypatch.setattr(run_mod, "build_render_config", lambda *a, **k: object())
    monkeypatch.setattr(run_mod, "render_proxy", lambda *a, **k: None)

    def fake_run_wbench_phases(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(run_mod, "run_wbench_phases", fake_run_wbench_phases)

    with pytest.raises(KeyboardInterrupt):
        run_mod.score_node(CFG, ctx, "n1", checkpoint=checkpoint, rank=8)

    assert (ctx.run_dir / "nodes" / "n1" / "eval" / "lora").exists()
    cleanups = [e for e in ctx.recorder.read_events("n1") if e["type"] == "eval.cleanup"]
    assert len(cleanups) == 0


def test_expected_n_known_before_the_root_is_scored():
    """Universal metrics cover every case; the judged ones follow the case files (case 2 has
    scene and subject adherence and one subject_action turn, cases 1 and 3 are navigation only)."""
    n = initial_expected_n(CFG, ["1", "2", "3"])
    assert n["aesthetic_quality"] == 3 and "perspective_consistency" not in n
    cases = [json.loads((CFG.wbench / "data" / "cases" / f"case_{i}.json").read_text()) for i in "123"]
    assert n["scene_adherence"] == sum("scene_adherence" in c for c in cases)
    assert n["subject_action_adherence"] == sum(
        any(i["type"] == "subject_action" for i in c["interactions"]) for c in cases)


def test_proxy_is_the_first_n_cases_of_the_ordered_list(tmp_path, monkeypatch):
    _runs(tmp_path, monkeypatch)
    ordered = (CFG.repo_root / "configs" / "proxy_cases.txt").read_text().strip().split(",")
    assert bootstrap_run(CFG, run_id="r50", env=ENV4).case_ids == ordered[:50]
    monkeypatch.setitem(CFG.raw["eval"], "proxy_size", 8)
    assert bootstrap_run(CFG, run_id="r8", env=ENV4).case_ids == ordered[:8]
    monkeypatch.setitem(CFG.raw["eval"], "proxy_size", len(ordered) + 1)
    with pytest.raises(PreflightError, match="proxy_size"):
        bootstrap_run(CFG, run_id="rbig", env=ENV4)


def test_bootstrap_records_universal_counts_and_attach_reads_them(tmp_path, monkeypatch):
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r_n", env=ENV4)
    assert ctx.expected_n["geometric_consistency"] == 50 and "spatial_consistency" not in ctx.expected_n
    from ar_kernel.run import attach_run
    assert attach_run(CFG, "r_n", ENV4).expected_n == ctx.expected_n


def test_score_reports_every_metrics_case_count_and_the_root_run_keeps_them(tmp_path, monkeypatch):
    import ar_kernel.run as run_mod
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-root", env=ENV4)
    ctx.metric_set = ["aesthetic_quality", "spatial_consistency"]
    ctx.expected_n = {"aesthetic_quality": 50}
    report = {"full": {"aesthetic_quality": {"mean": 0.5, "n": 50},
                       "spatial_consistency": {"mean": 0.4, "n": 8}}, "per_case": {}, "dimensions": {}}
    monkeypatch.setattr(run_mod, "build_render_config", lambda *a, **k: object())
    monkeypatch.setattr(run_mod, "render_proxy", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "run_wbench_phases", lambda *a, **k: report)
    _, detail = run_mod.score_node(CFG, ctx, "root", checkpoint=None, rank=0)
    run_mod.record_root_counts(ctx, detail["counts"])
    saved = json.loads((ctx.run_dir / "config" / "run.json").read_text())["expected_n"]
    assert saved == {"aesthetic_quality": 50, "spatial_consistency": 8}
    assert run_mod.attach_run(CFG, "r-root", ENV4).expected_n == saved


def test_bootstrap_records_the_judge_and_attach_refuses_a_different_one(tmp_path, monkeypatch):
    _runs(tmp_path, monkeypatch)
    ctx = bootstrap_run(CFG, run_id="r-judge", env=ENV4)
    assert ctx.judge.kind == "local"
    assert json.loads((ctx.run_dir / "config" / "run.json").read_text())["judge"]["kind"] == "local"
    from ar_kernel.run import attach_run
    assert attach_run(CFG, "r-judge", ENV4).judge == ctx.judge
    with pytest.raises(PreflightError, match="local judge.*api judge"):
        attach_run(CFG, "r-judge", {**ENV4, "VLM_API_KEY": "k"})


def _tree(root):
    return {str(f.relative_to(root)): (f.stat().st_size, f.stat().st_mtime_ns)
            for f in sorted(Path(root).rglob("*")) if f.is_file()}


def test_rescoring_a_node_reads_the_run_and_writes_only_under_scores(tmp_path, monkeypatch):
    import ar_kernel.run as run_mod
    from ar_kernel.archive.nodes import NodeStore
    _runs(tmp_path / "runs", monkeypatch)
    monkeypatch.setattr(KernelConfig, "scores_dir", property(lambda self: tmp_path / "scores"))
    ctx = bootstrap_run(CFG, run_id="r", env=ENV4)
    nodes = NodeStore(ctx.conn)
    nodes.create("root", None, 0)
    nodes.create("n1", "root", 1)
    checkpoint = ctx.run_dir / "nodes" / "n1" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    nodes.set_fields("n1", checkpoint_path="nodes/n1/checkpoint-2", lora_rank=32)
    nodes.create("n2", "root", 1)
    ctx.conn.close()
    seen = {}

    def fake_score_node(cfg, sctx, node_id, ckpt, rank):
        seen.update(run_dir=sctx.run_dir, checkpoint=ckpt, rank=rank, cases=sctx.case_ids, n=sctx.expected_n,
                    weights=cfg.get("eval.score_weights"))
        sctx.recorder.event("eval.scored", node=node_id)
        return 0.5, {"metrics": {"m": 0.5}, "aggregates": {}}
    monkeypatch.setattr(run_mod, "score_node", fake_score_node)
    before = _tree(ctx.run_dir)
    score, out = run_mod.rescore_node(CFG, "r", "n1", ENV4)
    assert _tree(ctx.run_dir) == before
    assert score == 0.5 and out.parent == tmp_path / "scores" and out.name.startswith("r-n1-")
    assert seen["run_dir"] == out and seen["checkpoint"] == checkpoint and seen["rank"] == 32
    assert seen["cases"] == ctx.case_ids and seen["n"] == ctx.expected_n and seen["weights"]
    assert json.loads((out / "score.json").read_text())["score"] == 0.5
    assert (out / "telemetry").exists()
    with pytest.raises(KeyError):
        run_mod.rescore_node(CFG, "r", "no-such-node", ENV4)
    with pytest.raises(ValueError, match="no checkpoint"):
        run_mod.rescore_node(CFG, "r", "n2", ENV4)
    with pytest.raises(run_mod.RunNotFound):
        run_mod.rescore_node(CFG, "missing", "n1", ENV4)
    assert _tree(ctx.run_dir) == before
