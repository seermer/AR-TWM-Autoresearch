import threading

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.run_kit import build_run_kit
from ar_kernel.vcs.agents_repo import AgentsRepo

ENV = {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "model-x", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}


def test_kit_uses_the_run_config_and_registers_enabled_tools(tmp_path, monkeypatch):
    cfg = KernelConfig(raw={**KernelConfig.load().raw, "budget": {"max_usd": 7, "usd_per_mtok": {"input": 1, "cached_input": 1, "output": 1}}}, repo_root=KernelConfig.load().repo_root)
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    kit = build_run_kit(cfg, tmp_path, [0, 1, 2, 3], ENV, AgentsRepo(tmp_path / "agents.git"))
    try:
        assert kit.env.default_model == "model-x" and kit.budget.max_usd == 7
        assert kit.env.queue.gpu_lock is kit.gpu_lock and kit.env.queue.wait_cap_s == float(cfg.get("tools.job_wait_max_s"))
    finally:
        kit.env.queue.shutdown()


def test_stop_survives_a_stuck_job_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    kit = build_run_kit(KernelConfig.load(), tmp_path, [0, 1, 2, 3], ENV, AgentsRepo(tmp_path / "agents.git"))
    stopped = []
    monkeypatch.setattr(kit.env.queue, "shutdown", lambda: (_ for _ in ()).throw(RuntimeError("stuck")))
    def services_stop():
        stopped.append("services")
        raise RuntimeError("service thread did not join")
    monkeypatch.setattr(kit.services, "stop", services_stop)
    monkeypatch.setattr(kit.harness, "stop", lambda: stopped.append("harness"))
    from ar_kernel import telemetry
    from conftest import telemetry_records
    telemetry.open(tmp_path / "trace")
    telemetry.start("ar.run")
    with pytest.raises(RuntimeError, match="join"):
        kit.stop()
    assert stopped == ["services", "harness"]                     # the harness still stopped
    assert "warning: stuck" in capsys.readouterr().err
    assert [(r["name"], r["message"]) for r in telemetry_records(tmp_path / "trace") if r["kind"] == "alert"] == [
        ("run.shutdown", "stuck")]


def test_gateway_calls_tool_calls_and_job_progress_move_the_activity_clock(tmp_path, monkeypatch):
    import asyncio
    import time
    from fastapi.testclient import TestClient
    from ar_kernel.tools.jobs import JobQueue
    from ar_kernel.tools.server import ToolKit
    monkeypatch.setattr("ar_kernel.run_kit.build_gpu_backends", lambda *a, **k: [])
    kit = build_run_kit(KernelConfig.load(), tmp_path, [0, 1, 2, 3], ENV, AgentsRepo(tmp_path / "agents.git"))
    try:
        clock = kit.last_activity
        assert kit.env.queue.touch == clock.touch
        caller = kit.env.registry.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                                        staging_host=tmp_path, mock_script="smoke")
        time.sleep(0.05)
        assert clock.idle_s() >= 0.05
        reply = TestClient(kit.gateway_app).post("/v1/responses", json={"model": "model-x", "input": "hi"},
                                                 headers={"Authorization": f"Bearer {caller.token}"})
        assert reply.status_code == 200 and clock.idle_s() < 0.05
    finally:
        kit.env.queue.shutdown()

    touched = []
    tools = ToolKit(kit.env.registry, tmp_path, touch=lambda: touched.append("tool"))
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()
    assert asyncio.run(tools.call(ctx, "t", {}, lambda c: "ok")) == "ok" and touched == ["tool", "tool"]

    class Steps:
        name, tool = "steps", "rollout_fake"

        def run(self, job, cancel, report):
            report({"done": 1, "total": 2}), report({"done": 2, "total": 2})
            return {"items": []}
    queue = JobQueue(tmp_path, threading.Lock(), wait_cap_s=30, touch=lambda: touched.append("job"))
    queue.register(Steps())
    try:
        assert queue.wait(caller, queue.submit(caller, "steps", {}), 30)["state"] == "done"
    finally:
        queue.shutdown()
    assert touched == ["tool", "tool", "job", "job"]
