import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from ar_kernel.config import KernelConfig
from ar_kernel.eval.wbench import run_wbench_phases


def _make_test_env(tmp_path, monkeypatch):
    """Helper to set up a test environment."""
    import ar_kernel.eval.wbench as wbench_mod

    cfg = KernelConfig.load()
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    model = "test_model"
    model_dir = work_dir / model
    model_dir.mkdir()
    eval_dir = model_dir / "evaluation"
    eval_dir.mkdir(parents=True)

    # Create a fake report.json
    fake_report = {"full": {"test_metric": {"mean": 0.5, "n": 1}}}
    report_path = eval_dir / "report.json"
    report_path.write_text(json.dumps(fake_report))

    gpus = [0, 1]
    node_id = "n1"

    # Track which phases are called in order
    phases_called = []

    def fake_run_in_env(*args, **kwargs):
        # Extract phase from the command arguments
        cmd_args = args[1]  # Second positional arg is the command list
        for i, arg in enumerate(cmd_args):
            if arg == "--phase" and i + 1 < len(cmd_args):
                phase = cmd_args[i + 1]
                phases_called.append(phase)
                break
        # Return a mock process with returncode 0 (success)
        mock_proc = Mock()
        mock_proc.returncode = 0
        return mock_proc

    # Mock the recorder
    mock_recorder = MagicMock()
    mock_recorder.span = MagicMock(return_value=MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=None)))

    # Patch run_in_env
    monkeypatch.setattr(wbench_mod, "run_in_env", fake_run_in_env)

    return cfg, work_dir, model, gpus, fake_report, mock_recorder, node_id, phases_called


def _fake_judge_server(monkeypatch, wbench_mod, events, fail_vlm=False):
    class Server:
        base_url = "http://127.0.0.1:1"

        def start(self, timeout_s, **kw):
            events.append("server_start")
            return 1.0

        def stop(self):
            events.append("server_stop")

    monkeypatch.setattr(wbench_mod, "judge_server", lambda *a, **k: Server())
    monkeypatch.setattr(wbench_mod, "gpu_memory_mib", lambda gpus: None)


def test_local_judge_phase_order_is_server_around_vlm_then_vp_then_report(tmp_path, monkeypatch):
    import ar_kernel.eval.wbench as wbench_mod
    from ar_kernel.eval.judge import Judge
    cfg, work_dir, model, gpus, fake_report, rec, node_id, phases = _make_test_env(tmp_path, monkeypatch)
    real = wbench_mod.run_in_env
    def logged(env, args, **kw):
        if env == "wbench-vp":
            phases.append("vp")
            return Mock(returncode=0)
        if "vlm" in args:
            assert kw["extra_env"]["VLM_API_KEY"] == "local"
        return real(env, args, **kw)
    monkeypatch.setattr(wbench_mod, "run_in_env", logged)
    _fake_judge_server(monkeypatch, wbench_mod, phases)
    out = run_wbench_phases(cfg, work_dir, model, gpus, ["test_metric"], rec, node_id,
                            Judge("local", "Qwen/x", None))
    assert out == fake_report
    assert phases == ["precompute", "gpu", "gpu", "server_start", "vlm", "server_stop", "vp", "report"]


def test_the_judge_server_is_stopped_when_the_vlm_phase_fails(tmp_path, monkeypatch):
    import pytest
    import ar_kernel.eval.wbench as wbench_mod
    from ar_kernel.eval.judge import Judge
    cfg, work_dir, model, gpus, _, rec, node_id, phases = _make_test_env(tmp_path, monkeypatch)
    def failing(env, args, **kw):
        phases.append(args[args.index("--phase") + 1] if "--phase" in args else "vp")
        return Mock(returncode=1 if "vlm" in args else 0, stdout="", stderr="boom")
    monkeypatch.setattr(wbench_mod, "run_in_env", failing)
    _fake_judge_server(monkeypatch, wbench_mod, phases)
    with pytest.raises(RuntimeError, match="vlm failed"):
        run_wbench_phases(cfg, work_dir, model, gpus, ["test_metric"], rec, node_id, Judge("local", "Qwen/x", None))
    assert phases[-1] == "server_stop" and "vp" not in phases


def test_an_api_judge_starts_no_server(tmp_path, monkeypatch):
    import ar_kernel.eval.wbench as wbench_mod
    from ar_kernel.eval.judge import Judge
    cfg, work_dir, model, gpus, _, rec, node_id, phases = _make_test_env(tmp_path, monkeypatch)
    _fake_judge_server(monkeypatch, wbench_mod, phases)
    real = wbench_mod.run_in_env
    monkeypatch.setattr(wbench_mod, "run_in_env",
                        lambda env, args, **kw: Mock(returncode=0) if env == "wbench-vp" else real(env, args, **kw))
    run_wbench_phases(cfg, work_dir, model, gpus, ["test_metric"], rec, node_id, Judge("api", "d", "u"))
    assert phases == ["precompute", "gpu", "gpu", "vlm", "report"]
