import copy
import os
import subprocess
import sys
import time

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import telemetry
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.monitor import Monitor, process_memory, tree_mark
from ar_kernel.telemetry import tree

CFG = KernelConfig.load()
SILENT = lambda: 10 ** 9                       # noqa: E731 -- no agent activity for ages


@pytest.fixture
def run(tmp_path):
    telemetry.open(tmp_path)
    telemetry.start("ar.run")
    return tmp_path


def alerts(run, name=None):
    return [r for r in telemetry_records(run) if r["kind"] == "alert" and name in (None, r["name"])]


def samples(run):
    return [r for r in telemetry_records(run) if r["kind"] == "sample"]


def decisions(run, name):
    return [r for r in telemetry_records(run) if r["kind"] == "decision" and r["name"] == name]


def test_a_tick_is_one_sample_of_the_gpus_and_the_host(run):
    Monitor(CFG, run, [0], Budget(), SILENT,
            usage=lambda: {0: {"util": 97, "memory_mib": 20000, "power_w": 300, "temp_c": 70, "pids": [5]}}).tick()
    [sample] = samples(run)
    assert sample["gpus"]["0"]["util"] == 97 and sample["gpus"]["0"]["pids"] == [5]
    assert sample["disk_free_gib"] > 0 and "dev_shm" in sample["memory_gib"]
    [me] = [p for p in sample["processes"] if p["pid"] == os.getpid()]
    assert me["anon"] > 0 and "python" in me["cmd"] and set(me) == {"pid", "cmd", "anon", "file", "shmem"}


def test_host_memory_alerts_once_per_crossing(run):
    """Page cache never counts: `used` stays at the limit, anon + shmem crosses the threshold twice."""
    anon = [100.0]
    m = Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: None, processes=lambda: [],
                memory=lambda: {"limit": 768.0, "used": 768.0, "anon": anon[0], "file": 600.0, "shmem": 10.0})
    high = 0.95 * CFG.get("host.ram_gib")
    for anon[0] in (50.0, high, high, 50.0, high):
        m.tick()
    assert len(samples(run)) == 5 and samples(run)[0]["gpus"] is None
    assert len(alerts(run, "monitor.host_memory_high")) == 2


def test_an_oom_kill_at_the_job_limit_is_an_error_alert_per_increase(run):
    kills = [0]
    m = Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: None, processes=lambda: [],
                events=lambda: {"max": 3, "oom_kill": kills[0]})
    for kills[0] in (2, 2, 3, 3, 5):                       # kills before the run started do not count
        m.tick()
    found = alerts(run, "monitor.host_oom_kill")
    assert [a["message"].split()[3] for a in found] == ["1", "2"] and found[0]["level"] == "error"
    assert samples(run)[0]["memory_events"] == {"max": 3, "oom_kill": 2}


def test_process_memory_includes_children():
    child = subprocess.Popen([sys.executable, "-c", "import sys, time; x = bytearray(300 * 2**20); "
                              "x[::4096] = b'1' * len(x[::4096]); print(flush=True); time.sleep(60)"],
                             stdout=subprocess.PIPE)
    try:
        child.stdout.readline()
        (found,) = [p for p in process_memory() if p["pid"] == child.pid]
        assert found["anon"] >= 0.25
    finally:
        child.kill()
        child.wait()


def _state(run, phase="train", attempt=1):
    (run / "control").mkdir(exist_ok=True)
    (run / "control" / "state.json").write_text(
        f'{{"node": "n1", "phase": "{phase}", "attempt": {attempt}, "since": 0}}')


def test_a_silent_phase_alerts_once_and_each_check_is_a_decision(run):
    _state(run)
    m = Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: {}, clock=lambda: 10_000.0)
    m.check()
    m.check()
    assert len(alerts(run, "monitor.stall")) == 1
    first, second = decisions(run, "monitor.stall_check")
    assert (first["chosen"], second["chosen"]) == ("alert", "already alerted")
    marks = telemetry_payload(run, first["inputs"])["marks"]
    assert marks["phase_start"] == 0 and marks["train_log_or_eval"] == 0


def test_agent_activity_is_progress_not_a_stall(run):
    _state(run, phase="improve_recipe")
    Monitor(CFG, run, [0], Budget(), lambda: 60.0, usage=lambda: {}, clock=lambda: 10_000.0).check()
    assert not alerts(run, "monitor.stall") and decisions(run, "monitor.stall_check")[0]["chosen"] == "active"


def test_a_growing_train_log_is_progress_not_a_stall(run):
    _state(run)
    log = run / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "train" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text("step 100\n")
    now = time.time() + 3600                           # no agent activity for ages, but the log grew a minute ago
    os.utime(log, (now - 60, now - 60))
    Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: {}, clock=lambda: now).check()
    assert not alerts(run, "monitor.stall")


def test_eval_work_is_progress_not_a_stall(run):
    _state(run, phase="eval", attempt=0)
    out = run / "nodes" / "n1" / "eval" / "renders" / "case1.mp4"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"x")
    now = time.time() + 3600
    os.utime(out, (now - 60, now - 60))
    Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: {}, clock=lambda: now).check()
    assert not alerts(run, "monitor.stall")


def test_no_running_phase_is_no_stall_check(run):
    Monitor(CFG, run, [0], Budget(), SILENT, usage=lambda: {}).check()
    assert not decisions(run, "monitor.stall_check")


def test_the_gateway_error_rate_check_is_a_decision_and_alerts(run):
    b = Budget()
    m = Monitor(CFG, run, [0], b, SILENT, usage=lambda: {})
    m.check()
    for _ in range(6):
        b.record(503, None)
    m.check()
    assert [d["chosen"] for d in decisions(run, "monitor.error_rate_check")] == ["quiet", "alert"]
    assert len(alerts(run, "monitor.gateway_errors")) == 1


def test_gpu_outside_the_list_alert(run):
    Monitor(CFG, run, [0], Budget(), SILENT,
            usage=lambda: {3: {"util": 5, "memory_mib": 900, "power_w": 50, "temp_c": 40,
                               "pids": [os.getpid()]}}).check()
    assert alerts(run, "monitor.gpu_outside_list")


def test_a_raising_step_does_not_stop_the_checks_and_alerts_once(run):
    raw = copy.deepcopy(CFG.raw)
    raw["disk"]["alert_below_gb"] = 10 ** 9
    cfg = type(CFG)(raw=raw, repo_root=CFG.repo_root)

    def broken():
        raise RuntimeError("nvidia-smi hung")
    m = Monitor(cfg, run, [0], Budget(), SILENT, usage=broken)
    m.poll()
    m.poll()
    assert alerts(run, "monitor.disk_low")
    [failed] = alerts(run, "monitor.failed")
    assert "nvidia-smi hung" in failed["message"]
    [error] = [r for r in telemetry_records(run) if r["kind"] == "error"]           # once, with the exception
    assert telemetry_payload(run, error["error"])["message"] == "nvidia-smi hung"
    assert tree.check(tree.load(run)) == []


def test_the_thread_samples_under_the_current_span(run, monkeypatch):
    raw = copy.deepcopy(CFG.raw)
    raw["telemetry"]["gpu_sample_sec"] = 0.01
    cfg = type(CFG)(raw=raw, repo_root=CFG.repo_root)
    m = Monitor(cfg, run, [0], Budget(), SILENT, usage=lambda: None, processes=lambda: [])
    with telemetry.span("services") as sp:
        m.start()
    deadline = time.monotonic() + 30
    while not samples(run) and time.monotonic() < deadline:
        time.sleep(0.05)
    m.stop()
    assert samples(run)[0]["span"] == sp.id and not m._thread.is_alive()


def test_tree_mark_sees_new_and_grown_files(tmp_path):
    a = tree_mark(tmp_path)
    (tmp_path / "x").write_text("1")
    b = tree_mark(tmp_path)
    (tmp_path / "x").write_text("12")
    assert a != b != tree_mark(tmp_path)
    assert tree_mark(tmp_path / "missing") == (0, 0, 0)
