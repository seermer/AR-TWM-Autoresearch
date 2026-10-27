import pytest

from ar_kernel.config import GpuPolicyError
from ar_kernel.guards import alert
from ar_kernel.telemetry.recorder import Recorder


def test_alert_is_a_run_event(tmp_path):
    rec = Recorder(tmp_path)
    alert(rec, "node_failed", "n1 ended crashed", node_id="n1")
    (event,) = [e for e in rec.read_events() if e["type"] == "alert"]
    assert (event["kind"], event["level"], event["message"]) == ("node_failed", "error", "n1 ended crashed")


@pytest.mark.gpu
def test_device_minors_on_this_machine():
    """Index 0..5 are /dev/nvidia1, 2, 0, 3, 5, 4 here."""
    from ar_kernel.guards import device_minors
    assert device_minors([0, 1, 2, 3, 4, 5]) == [1, 2, 0, 3, 5, 4]


def _fit_cfg(host=None, captioner=None, **generators):
    from ar_kernel.config import KernelConfig
    return KernelConfig(raw={"captioner": {"min_total_gpu_gib": 96, "tensor_parallel": 2, **(captioner or {})},
                             "annotate": {"enabled": True, "min_total_gpu_gib": 48, "workers": 2},
                             "images": {"enabled": False, "min_total_gpu_gib": 999, "workers": 99},
                             "host": {"ram_gib": 240, "kernel_reserve_gib": 72, **(host or {})},
                             "generators": generators}, repo_root=None)


def _fit(cfg, gpus, cards=24, limit=None, host_gib=251):
    from ar_kernel.guards import check_tools_fit
    check_tools_fit(cfg, gpus, totals=lambda g: cards and {i: cards for i in g}, limit=lambda: limit,
                    host_total=lambda: host_gib * 2**30)


def test_enabled_tools_must_fit_the_runs_cards():
    h3 = {"enabled": True, "min_total_gpu_gib": 96}
    ltx = {"enabled": True, "min_total_gpu_gib": 48, "workers": 2}
    _fit(_fit_cfg(h3=h3, ltx25=ltx), [0, 1, 2, 3])
    with pytest.raises(GpuPolicyError) as exc:                       # two 24 GiB cards
        _fit(_fit_cfg(h3=h3, ltx25=ltx, wan22={"enabled": False, "min_total_gpu_gib": 999}), [0, 1])
    text = str(exc.value)
    assert text.startswith("enabled tools do not fit this machine (GPUs 0,1): ")
    assert "generators.h3 needs 96 GiB of GPU memory, the GPUs have 48" in text and "captioner needs 96 GiB" in text
    assert "annotate" not in text and "ltx25" not in text            # they fit
    assert "images" not in text and "wan22" not in text              # not enabled
    _fit(_fit_cfg(h3=h3), [0, 1], cards=None)                        # no nvidia-smi: cannot check
    _fit(_fit_cfg(h3={"enabled": True}), [0, 1], cards=48)           # h3 without an estimate: not checked


def test_workers_and_tensor_parallel_must_fit_the_gpu_count():
    _fit(_fit_cfg(wan22={"enabled": True, "workers": 2, "gpus_per_worker": 2}), [0, 1, 2, 3])
    with pytest.raises(GpuPolicyError) as exc:
        _fit(_fit_cfg(captioner={"tensor_parallel": 8}, ltx25={"enabled": True, "workers": 3},
                      wan22={"enabled": True, "workers": 2, "gpus_per_worker": 2}), [0, 1])
    text = str(exc.value)
    assert "captioner.tensor_parallel 8 > 2 GPUs" in text
    assert "generators.ltx25.workers 3 x 1 GPUs > 2 GPUs" in text and "generators.wan22.workers 2 x 2 GPUs" in text
    assert "annotate" not in text and "images" not in text           # fits; not enabled


def test_each_tool_must_fit_host_ram_alone():
    ltx = {"enabled": True, "workers": 3, "peak_rss_gib": 40, "shared_gib": 36}     # 120 + 36 + 72 = 228
    h3 = {"enabled": True, "peak_rss_gib": 140}                                     # one job: 140 + 72
    _fit(_fit_cfg(ltx25=ltx, h3=h3), [0, 1, 2, 3])
    with pytest.raises(GpuPolicyError) as exc:
        _fit(_fit_cfg(ltx25={**ltx, "workers": 4}, h3=h3), [0, 1, 2, 3])
    text = str(exc.value)
    assert "generators.ltx25 needs 268 GiB of host RAM" in text and "host.ram_gib 240" in text
    assert "h3" not in text


def test_a_memory_limit_below_host_ram_is_refused():
    from pathlib import Path
    _fit(_fit_cfg(), [0, 1, 2, 3], limit=(Path("/sys/fs/cgroup/job"), 240 * 2**30))
    with pytest.raises(GpuPolicyError, match=r"memory limit is 200 GiB \(/sys/fs/cgroup/job\) < host.ram_gib 240"):
        _fit(_fit_cfg(), [0, 1, 2, 3], limit=(Path("/sys/fs/cgroup/job"), 200 * 2**30))
    _fit(_fit_cfg(), [0, 1, 2, 3], limit=(Path("/sys/fs/cgroup/job"), 768 * 2**30), host_gib=100)   # the limit counts
    with pytest.raises(GpuPolicyError, match="this host has 200 GiB of RAM < host.ram_gib 240"):     # no limit
        _fit(_fit_cfg(), [0, 1, 2, 3], host_gib=200)


def test_a_config_without_the_explicit_keys_names_them():
    """e.g. the frozen config of a run started before the keys existed."""
    with pytest.raises(GpuPolicyError, match=r"needs an int for host.ram_gib, captioner.tensor_parallel, "
                                             r"generators.ltx25.workers \(a run started before"):
        _fit(_fit_cfg(host={"ram_gib": None}, captioner={"tensor_parallel": None},
                      ltx25={"enabled": True, "workers": None}, wan22={"enabled": False}), [0, 1, 2, 3])


def test_card_sizes_are_whole_gib_and_unreadable_output_means_no_check(monkeypatch):
    from ar_kernel import guards
    monkeypatch.setattr(guards, "smi", lambda args: "0, 24564\n1, 24564\n2, 11264\n")
    assert guards.gpu_total_gib([0, 2, 7]) == {0: 24, 2: 11}
    for text in ("0, [N/A]\n", "No devices were found\n", ""):
        monkeypatch.setattr(guards, "smi", lambda args, text=text: text)
        assert not guards.gpu_total_gib([0])
    monkeypatch.setattr(guards, "smi", lambda args: None)
    assert guards.gpu_total_gib([0]) is None
