"""GpuJob plumbing with a fake worker (no GPU)."""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.gpu_jobs import GpuJob, canonical_hash, register_gpu_tools, split_gpus
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError, ToolKit, new_mcp
from tests.conftest import make_mp4

FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


class FakeJob(GpuJob):
    name = tool = "rollout_fake"
    kind, generator, license = "rollout", "fake-gen", "test-license"
    file_keys = ("src",)

    def check_args(self, args):
        if not args.get("items"):
            raise ToolError("items is empty")

    def produce(self, job, items, work, out, cancel, report):
        groups = split_gpus(self.gpus, 1, None)
        return self.run_workers("autoresearcher", lambda r, w: ["python", str(FAKE), "--items",
                                str(work / "items.json"), "--out", str(out), "--rank", str(r), "--world", str(w)],
                                groups, job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        return {"video": out / f"{item['index']}.mp4",
                "caption": self.write_caption(out, item, {"caption": item.get("prompt", "x")})}


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2.5, fps=24, width=736, height=414)
    q.register(FakeJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                       gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, rec, ws, staging, tmp_path / "run"
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_fake"].submit(q, caller, {"items": items, **params})


def test_split_gpus():
    assert split_gpus([0, 1, 4, 5], 1, None) == [[0], [1], [4], [5]]
    assert split_gpus([0, 1, 4, 5], 2, None) == [[0, 1], [4, 5]]
    assert split_gpus([0, 1, 4, 5], 1, 2) == [[0], [1]]
    assert split_gpus([0, 1, 2], 2, None) == [[0, 1]]


def test_run_workers_refuses_empty_gpu_groups(env):
    q, caller, rec, ws, staging, run = env
    backend = q.backends["rollout_fake"]
    with pytest.raises(ValueError):
        backend.run_workers("autoresearcher", lambda r, w: [], [], job=None, work=run, out=run,
                            total=0, cancel=threading.Event(), report=lambda p: None)


def test_max_items_and_timeout_come_from_the_backends_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)

    class ConfiguredJob(FakeJob):
        config_key = "annotate"        # configs/kernel.yaml: max_items: 64, timeout_s: 21600

    backend = ConfiguredJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec)
    assert backend.max_items == 64
    assert backend.timeout_s == 21600
    assert FakeJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec).timeout_s is None


def test_items_fan_out_one_worker_per_gpu_and_publish_candidates(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4", "prompt": f"p{i}", "seed": i} for i in range(6)]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    got = out["result"]["items"]
    assert [g["index"] for g in got] == list(range(6))
    c = got[0]["candidate"]
    job = out["id"]
    assert c["video"] == f"/workspace/staging/rollouts/{job}/0.mp4"
    assert (staging / "rollouts" / job / "0.mp4").is_file()
    assert json.loads((staging / "rollouts" / job / "0.json").read_text()) == {"caption": "p0"}
    assert c["provenance"] == {"kind": "rollout", "generator": "fake-gen", "job_id": job,
                               "inputs_hash": c["provenance"]["inputs_hash"], "seed": 0}
    assert c["license"] == "test-license" and "pose" not in c
    assert got[0]["worker"]["gpus"] == "0" and got[1]["worker"]["gpus"] == "1" and got[4]["worker"]["gpus"] == "0"
    assert not (run / "jobs" / job / "in").exists() and not (run / "jobs" / job / "out").exists()
    assert out["result"]["gpu_memory_released"] is True


def test_inputs_hash_depends_on_file_content_not_path(env):
    q, caller, rec, ws, staging, run = env
    (ws / "b.mp4").write_bytes((ws / "a.mp4").read_bytes())
    out = q.wait(caller, submit(q, caller, [{"src": "a.mp4", "seed": 1}, {"src": "b.mp4", "seed": 1},
                                           {"src": "a.mp4", "seed": 2}])["job_id"], 120)
    h = [i["candidate"]["provenance"]["inputs_hash"] for i in out["result"]["items"]]
    assert h[0] == h[1] != h[2]


def test_per_item_failure_and_worker_crash_are_item_errors(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4"}, {"src": "a.mp4", "fail": True}, {"src": "a.mp4", "crash": True}, {"src": "a.mp4"}]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done"
    by = {i["index"]: i for i in out["result"]["items"]}
    assert "candidate" in by[0] and "candidate" in by[3]
    assert by[1]["error"] == "boom"
    assert "exit code 3" in by[2]["error"] and "log tail" in by[2]["error"]


def test_a_truncated_status_file_is_an_item_error_not_a_job_failure(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4"}, {"src": "a.mp4"}, {"src": "a.mp4", "truncated": True}, {"src": "a.mp4"}]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    by = {i["index"]: i for i in out["result"]["items"]}
    assert "candidate" in by[0] and "candidate" in by[1] and "candidate" in by[3]
    assert "error" in by[2] and "JSONDecodeError" in by[2]["error"]


def test_run_workers_kills_workers_past_its_own_timeout(env):
    q, caller, rec, ws, staging, run = env
    q.backends["rollout_fake"].timeout_s = 1
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 60}])["job_id"]
    out = q.wait(caller, job_id, 60)
    assert out["state"] == "done", out
    assert "timeout" in out["result"]["items"][0]["error"]


def test_missing_or_escaping_input_is_refused_at_submit(env):
    q, caller, rec, ws, staging, run = env
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "nope.mp4"}])
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "/etc/passwd"}])
    with pytest.raises(ToolError):
        submit(q, caller, [])


def test_a_planted_link_in_staging_fails_the_item_not_the_host(env, tmp_path):
    q, caller, rec, ws, staging, run = env
    outside = tmp_path / "outside"; outside.mkdir()
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 2}])["job_id"]
    (staging / "rollouts").mkdir()
    (staging / "rollouts" / job_id).symlink_to(outside)        # swapped in while the job runs
    out = q.wait(caller, job_id, 120)
    assert "error" in out["result"]["items"][0]
    assert list(outside.iterdir()) == []


def test_cancel_kills_every_worker(env):
    q, caller, rec, ws, staging, run = env
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 300} for _ in range(4)])["job_id"]
    out_dir = run / "jobs" / job_id / "out"
    deadline = time.monotonic() + 30
    while len(list(out_dir.glob("worker*.pid"))) < 4 and time.monotonic() < deadline:
        time.sleep(0.2)
    pids = [int(p.read_text()) for p in out_dir.glob("worker*.pid")]
    assert len(pids) == 4
    q.cancel(caller, job_id)
    assert q.wait(caller, job_id, 120)["state"] == "cancelled"
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_registration_exposes_only_tools_for_backends_on_the_queue(env):
    q, caller, rec, ws, staging, run = env

    class Wan22Fake(FakeJob):
        name = tool = "rollout_wan22"

    q.register(Wan22Fake(KernelConfig.load(), run, [0, 1, 4, 5], TokenRegistry(rec), rec,
                         gpu_memory=lambda g: {i: 100 for i in g}))
    kit, mcp = ToolKit(TokenRegistry(rec), rec), new_mcp()
    register_gpu_tools(mcp, kit, q)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "rollout_wan22" in names                     # registered backend: the tool appears
    assert "annotate_camera" not in names and "generate_images" not in names   # not registered: absent
