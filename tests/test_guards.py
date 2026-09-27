import pytest

from ar_kernel.config import GpuPolicyError
from ar_kernel.guards import alert, check_visible
from ar_kernel.telemetry.recorder import Recorder


def test_alert_is_a_run_event(tmp_path):
    rec = Recorder(tmp_path)
    alert(rec, "node_failed", "n1 ended crashed", node_id="n1")
    (event,) = [e for e in rec.read_events() if e["type"] == "alert"]
    assert (event["kind"], event["level"], event["message"]) == ("node_failed", "error", "n1 ended crashed")


def test_invisible_gpu_is_refused():
    with pytest.raises(GpuPolicyError, match="7"):
        check_visible([0, 7], visible=lambda: {0, 1, 2, 3, 4, 5})
    check_visible([0, 5], visible=lambda: {0, 1, 2, 3, 4, 5})
    check_visible([0, 7], visible=lambda: None)          # no nvidia-smi: cannot check, no refusal
