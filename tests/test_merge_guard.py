import pytest
from ar_kernel.config import KernelConfig
from ar_kernel.eval.merge import wait_for_ram
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()

def test_wait_for_ram_returns_when_threshold_is_met(tmp_path):
    wait_for_ram(CFG, Recorder(tmp_path), "n1", threshold_gb=0.5, poll_s=0.01, alert_after_s=1)

def test_wait_for_ram_alerts_when_never_satisfied(tmp_path):
    rec = Recorder(tmp_path)
    with pytest.raises(TimeoutError):
        wait_for_ram(CFG, rec, "n1", threshold_gb=1e9, poll_s=0.01, alert_after_s=0.05)
    assert any(e["type"] == "eval.ram_wait_alert" for e in rec.read_events("n1"))
