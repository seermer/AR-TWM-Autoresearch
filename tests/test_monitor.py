import copy
import os
import time

from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.monitor import Monitor
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()


def kinds(rec):
    return [e["kind"] for e in rec.read_events() if e["type"] == "alert"]


def test_gpu_sample_goes_to_its_own_file(tmp_path):
    rec = Recorder(tmp_path)
    m = Monitor(CFG, tmp_path, rec, [0], Budget(),
                usage=lambda gpus: {0: {"util": 97, "memory_mib": 20000, "power_w": 300, "temp_c": 70, "pids": []}})
    m.tick()
    (event,) = rec.read_events("gpu")
    assert event["gpus"]["0"]["util"] == 97 and event["payload"] is None


def test_stall_alert_once(tmp_path):
    rec = Recorder(tmp_path)
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "state.json").write_text('{"node": "n1", "phase": "train", "attempt": 1, "since": 0}')
    now = [10_000.0]
    m = Monitor(CFG, tmp_path, rec, [0], Budget(), usage=lambda gpus: {}, clock=lambda: now[0])
    m.check()
    m.check()
    assert kinds(rec).count("stall") == 1


def test_gateway_error_alert(tmp_path):
    rec, b = Recorder(tmp_path), Budget()
    for _ in range(6):
        b.record(503, None)
    Monitor(CFG, tmp_path, rec, [0], b, usage=lambda gpus: {}).check()
    assert "gateway_errors" in kinds(rec)


def test_gpu_outside_the_list_alert(tmp_path):
    rec = Recorder(tmp_path)
    m = Monitor(CFG, tmp_path, rec, [0], Budget(),
                usage=lambda gpus: {3: {"util": 5, "memory_mib": 900, "power_w": 50, "temp_c": 40,
                                        "pids": [os.getpid()]}})
    m.check()
    assert "gpu_outside_list" in kinds(rec)


def _state(run, phase="train", attempt=1):
    (run / "control").mkdir(exist_ok=True)
    (run / "control" / "state.json").write_text(
        f'{{"node": "n1", "phase": "{phase}", "attempt": {attempt}, "since": 0}}')


def test_a_growing_train_log_is_progress_not_a_stall(tmp_path):
    rec = Recorder(tmp_path)
    _state(tmp_path)
    log = tmp_path / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "train" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text("step 100\n")
    now = time.time() + 3600                           # no event for an hour, but the log grew a minute ago
    os.utime(log, (now - 60, now - 60))
    Monitor(CFG, tmp_path, rec, [0], Budget(), usage=lambda gpus: {}, clock=lambda: now).check()
    assert "stall" not in kinds(rec)


def test_eval_work_is_progress_not_a_stall(tmp_path):
    rec = Recorder(tmp_path)
    _state(tmp_path, phase="eval", attempt=0)
    out = tmp_path / "nodes" / "n1" / "eval" / "renders" / "case1.mp4"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"x")
    now = time.time() + 3600
    os.utime(out, (now - 60, now - 60))
    Monitor(CFG, tmp_path, rec, [0], Budget(), usage=lambda gpus: {}, clock=lambda: now).check()
    assert "stall" not in kinds(rec)


def test_a_failing_gpu_sample_does_not_stop_the_checks_and_is_logged_once(tmp_path):
    raw = copy.deepcopy(CFG.raw)
    raw["disk"]["alert_below_gb"] = 10 ** 9
    cfg = type(CFG)(raw=raw, repo_root=CFG.repo_root)
    rec = Recorder(tmp_path)

    def broken(gpus):
        raise RuntimeError("nvidia-smi hung")
    m = Monitor(cfg, tmp_path, rec, [0], Budget(), usage=broken)
    m.poll()
    m.poll()
    assert "disk_low" in kinds(rec)
    assert len([e for e in rec.read_events() if e["type"] == "monitor.error"]) == 1
