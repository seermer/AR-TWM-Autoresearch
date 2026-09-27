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


def test_run_wbench_phases_retries_gpu_phase(tmp_path, monkeypatch):
    """Verify that the gpu phase is run twice to handle transient CUDA OOM errors."""
    cfg, work_dir, model, gpus, fake_report, mock_recorder, node_id, phases_called = \
        _make_test_env(tmp_path, monkeypatch)

    metric_set = ["test_metric"]

    # Run the function
    result = run_wbench_phases(cfg, work_dir, model, gpus, metric_set, mock_recorder, node_id)

    # Verify the result
    assert result == fake_report

    # Verify the phase order includes gpu twice
    assert phases_called == ["precompute", "gpu", "gpu", "report"], \
        f"Expected phase order [precompute, gpu, gpu, report], but got {phases_called}"


def test_run_wbench_phases_retries_gpu_phase_with_vlm(tmp_path, monkeypatch):
    """Verify that the gpu phase is run twice before vlm phase."""
    cfg, work_dir, model, gpus, fake_report, mock_recorder, node_id, phases_called = \
        _make_test_env(tmp_path, monkeypatch)

    # scene_adherence is a VLM metric
    metric_set = ["test_metric", "scene_adherence"]

    # Run the function
    result = run_wbench_phases(cfg, work_dir, model, gpus, metric_set, mock_recorder, node_id)

    # Verify the result
    assert result == fake_report

    # Verify the phase order includes gpu twice, then vlm, then report
    assert phases_called == ["precompute", "gpu", "gpu", "vlm", "report"], \
        f"Expected phase order [precompute, gpu, gpu, vlm, report], but got {phases_called}"
