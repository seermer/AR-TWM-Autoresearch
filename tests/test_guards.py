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


def _fit_cfg(**generators):
    from ar_kernel.config import KernelConfig
    return KernelConfig(raw={"captioner": {"min_total_gpu_gib": 96}, "annotate": {"enabled": True, "min_total_gpu_gib": 48},
                             "images": {"enabled": False, "min_total_gpu_gib": 999},
                             "generators": generators}, repo_root=None)


def test_enabled_tools_must_fit_the_runs_cards():
    from ar_kernel.guards import check_tools_fit
    h3 = {"enabled": True, "min_total_gpu_gib": 96}
    ltx = {"variants": ["distilled"], "min_total_gpu_gib": 48}
    check_tools_fit(_fit_cfg(h3=h3, ltx25=ltx), [0, 1, 2, 3], totals=lambda g: {i: 24 for i in g})
    with pytest.raises(GpuPolicyError) as exc:                       # two 24 GiB cards
        check_tools_fit(_fit_cfg(h3=h3, ltx25=ltx, wan22={"enabled": False, "min_total_gpu_gib": 999}),
                        [0, 1], totals=lambda g: {i: 24 for i in g})
    text = str(exc.value)
    assert "GPUs 0,1 (48 GiB in total)" in text and text.endswith("Give the run larger or more GPUs.")
    assert "generators.h3 needs 96 GiB" in text and "captioner needs 96 GiB" in text
    assert "annotate" not in text and "ltx25" not in text            # they fit
    assert "images" not in text and "wan22" not in text              # not enabled
    check_tools_fit(_fit_cfg(h3=h3), [0, 1], totals=lambda g: None)          # no nvidia-smi: cannot check
    from ar_kernel.config import KernelConfig
    bare = KernelConfig(raw={"generators": {"h3": {"enabled": True}}}, repo_root=None)
    check_tools_fit(bare, [0], totals=lambda g: {0: 1})                    # no estimate: not checked


def test_card_sizes_are_whole_gib_and_unreadable_output_means_no_check(monkeypatch):
    from ar_kernel import guards
    monkeypatch.setattr(guards, "smi", lambda args: "0, 24564\n1, 24564\n2, 11264\n")
    assert guards.gpu_total_gib([0, 2, 7]) == {0: 24, 2: 11}
    for text in ("0, [N/A]\n", "No devices were found\n", ""):
        monkeypatch.setattr(guards, "smi", lambda args, text=text: text)
        assert not guards.gpu_total_gib([0])
    monkeypatch.setattr(guards, "smi", lambda args: None)
    assert guards.gpu_total_gib([0]) is None
