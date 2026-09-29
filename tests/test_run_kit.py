import threading

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.run_kit import build_run_kit
from ar_kernel.telemetry.recorder import Recorder

ENV = {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "model-x", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}


def test_kit_uses_the_run_config_and_registers_enabled_tools(tmp_path, monkeypatch):
    cfg = KernelConfig(raw={**KernelConfig.load().raw, "budget": {"max_usd": 7, "usd_per_mtok": {"input": 1, "cached_input": 1, "output": 1}}}, repo_root=KernelConfig.load().repo_root)
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    kit = build_run_kit(cfg, tmp_path, [0, 1, 2, 3], Recorder(tmp_path), ENV)
    try:
        assert kit.default_model == "model-x" and kit.budget.max_usd == 7
        assert kit.queue.gpu_lock is kit.gpu_lock and kit.queue.wait_cap_s == float(cfg.get("tools.job_wait_max_s"))
    finally:
        kit.queue.shutdown()


def test_stop_survives_a_stuck_job_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    rec = Recorder(tmp_path)
    kit = build_run_kit(KernelConfig.load(), tmp_path, [0, 1, 2, 3], rec, ENV)
    stopped = []
    monkeypatch.setattr(kit.queue, "shutdown", lambda: (_ for _ in ()).throw(RuntimeError("stuck")))
    def services_stop():
        stopped.append("services")
        raise RuntimeError("service thread did not join")
    monkeypatch.setattr(kit.services, "stop", services_stop)
    monkeypatch.setattr(kit.harness, "stop", lambda: stopped.append("harness"))
    with pytest.raises(RuntimeError, match="join"):
        kit.stop()
    assert stopped == ["services", "harness"]                     # the harness still stopped
    assert [e["kind"] for e in rec.read_events() if e["type"] == "alert"] == ["shutdown"]
