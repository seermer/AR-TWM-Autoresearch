"""caption_videos: the vLLM caption job (Follow-up C). Unit tests run a fake vLLM server
(tests/fixtures/fake_vllm.py, no GPU); the `gpu` test runs the real model."""
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig, resolve_gpus
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.captioner import CaptionBackend, submit
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError

FAKE = Path(__file__).parent / "fixtures" / "fake_vllm.py"
REAL = KernelConfig.load()


def fake_cfg(**over) -> KernelConfig:
    c = {**REAL.get("captioner"), "env": "autoresearcher", "startup_timeout_s": 60, "clip_timeout_s": 10,
         "memory_release_timeout_s": 2, **over}
    return KernelConfig(raw={"captioner": c}, repo_root=REAL.repo_root)


class FakeServer(CaptionBackend):
    """The real command line, with `vllm serve` swapped for the fake server."""
    def server_command(self, port, media_dir):
        return ["python", str(FAKE), *super().server_command(port, media_dir)[2:]]


@pytest.fixture
def make(tmp_path):
    made = []

    def build(cfg=None):
        rec = Recorder(tmp_path / "run")
        reg = TokenRegistry(rec)
        q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
        q.register(FakeServer(cfg or fake_cfg(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                              gpu_memory=lambda gpus: {g: 100 for g in gpus}, poll_s=0.2))
        ws, staging = tmp_path / "ws", tmp_path / "staging"
        (ws / "clips").mkdir(parents=True)
        staging.mkdir()
        caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
        made.append(q)
        return q, caller, rec, ws, staging
    yield build
    for q in made:
        q.shutdown()


def _pid_gone(run_dir: Path, job_id: str) -> bool:
    pid = int((run_dir / "jobs" / job_id / "fake_vllm.pid").read_text())
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def test_batch_success_captions_every_clip_then_stops_the_server(make, tmp_path):
    q, caller, rec, ws, staging = make()
    (ws / "clips" / "a.mp4").write_bytes(b"x" * 10)
    (staging / "b.mp4").write_bytes(b"y" * 20)
    job = submit(q, caller, ["clips/a.mp4", "/workspace/staging/b.mp4"], "Describe the camera motion.")["job_id"]
    out = q.wait(caller, job, 120)
    assert out["state"] == "done", out
    assert out["args"]["paths"] == ["/workspace/clips/a.mp4", "/workspace/staging/b.mp4"]
    clips = out["result"]["clips"]
    assert clips == {"/workspace/clips/a.mp4": {"caption": "Describe the camera motion. [10 bytes, tp=4]"},
                     "/workspace/staging/b.mp4": {"caption": "Describe the camera motion. [20 bytes, tp=4]"}}
    assert out["result"]["load_s"] > 0 and out["result"]["gpu_memory_released"] is True
    assert out["progress"] == {"captioned": 2, "total": 2}
    run = tmp_path / "run"
    assert _pid_gone(run, job) and not (run / "jobs" / job / "media").exists()
    events = rec.read_events("n1")
    assert [e["ok"] for e in events if e["type"] == "caption.clip"] == [True, True]
    assert all(e["latency_s"] >= 0 for e in events if e["type"] == "caption.clip")
    assert any(e["type"] == "caption.server_ready" for e in events)


def test_a_clip_the_server_rejects_is_a_per_clip_error_not_a_failed_job(make):
    q, caller, _, ws, _ = make()
    (ws / "good.mp4").write_bytes(b"ok")
    (ws / "bad.mp4").write_bytes(b"BAD video")
    job = submit(q, caller, ["good.mp4", "bad.mp4"], "Caption.")["job_id"]
    out = q.wait(caller, job, 120)
    assert out["state"] == "done", out
    clips = out["result"]["clips"]
    assert "caption" in clips["/workspace/good.mp4"]
    assert "HTTP 400" in clips["/workspace/bad.mp4"]["error"] and "decode" in clips["/workspace/bad.mp4"]["error"]


def test_a_clip_swapped_for_a_link_after_submit_is_a_per_clip_error(make, tmp_path):
    q, caller, _, ws, _ = make()
    (ws / "a.mp4").write_bytes(b"ok")
    (ws / "b.mp4").write_bytes(b"ok")
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"secret")
    with q.gpu_lock:                                  # hold the job queued while the agent swaps the file
        job = submit(q, caller, ["a.mp4", "b.mp4"], "Caption.")["job_id"]
        (ws / "b.mp4").unlink()
        (ws / "b.mp4").symlink_to(outside)
    out = q.wait(caller, job, 120)
    assert out["state"] == "done", out
    assert "caption" in out["result"]["clips"]["/workspace/a.mp4"]
    assert "escapes its mount" in out["result"]["clips"]["/workspace/b.mp4"]["error"]


def test_cancel_stops_the_server(make, tmp_path):
    q, caller, rec, ws, _ = make(fake_cfg(extra_args=["--fake-mode", "slow"], clip_timeout_s=120))
    (ws / "a.mp4").write_bytes(b"ok")
    job = submit(q, caller, ["a.mp4"], "Caption.")["job_id"]
    deadline = time.monotonic() + 60
    while not any(e["type"] == "caption.server_ready" for e in rec.read_events("n1")):
        assert time.monotonic() < deadline, "the fake server never became ready"
        time.sleep(0.2)
    time.sleep(0.5)                                   # the caption request is in flight
    started = time.monotonic()
    q.cancel(caller, job)
    out = q.wait(caller, job, 120)
    assert out["state"] == "cancelled" and time.monotonic() - started < 30
    assert _pid_gone(tmp_path / "run", job)
    assert any(e["type"] == "subproc.cancelled" for e in rec.read_events("n1"))


def test_server_that_dies_during_startup_fails_the_job_with_its_log(make, tmp_path):
    q, caller, _, ws, _ = make(fake_cfg(extra_args=["--fake-mode", "exit"]))
    (ws / "a.mp4").write_bytes(b"ok")
    out = q.wait(caller, submit(q, caller, ["a.mp4"], "Caption.")["job_id"], 120)
    assert out["state"] == "failed"
    assert "exited during startup" in out["error"] and "CUDA out of memory" in out["error"]


def test_server_that_never_gets_ready_times_out_and_is_killed(make, tmp_path):
    q, caller, _, ws, _ = make(fake_cfg(extra_args=["--fake-mode", "hang"], startup_timeout_s=8))
    (ws / "a.mp4").write_bytes(b"ok")
    job = submit(q, caller, ["a.mp4"], "Caption.")["job_id"]
    out = q.wait(caller, job, 120)
    assert out["state"] == "failed" and "not ready within 8 s" in out["error"]
    assert _pid_gone(tmp_path / "run", job)


@pytest.mark.parametrize("paths, message", [
    ([], "paths is empty"),
    (["/etc/passwd"], "outside /workspace"),
    (["../../etc/passwd"], "escapes its mount"),
    (["missing.mp4"], "is not a file"),
    (["clips"], "is not a file"),
])
def test_paths_are_validated_before_a_job_is_queued(make, paths, message):
    q, caller, _, _, _ = make()
    with pytest.raises(ToolError, match=message):
        submit(q, caller, paths, "Caption.")


def test_empty_prompt_is_refused(make):
    q, caller, _, ws, _ = make()
    (ws / "a.mp4").write_bytes(b"ok")
    with pytest.raises(ToolError, match="prompt is empty"):
        submit(q, caller, ["a.mp4"], "  ")


def test_caption_videos_over_mcp_queues_a_job_and_reports_path_errors(tmp_path):
    """The tool as the agent sees it, with the smoke-run fake backend (never starts vLLM)."""
    import asyncio
    from ar_contract.client import mcp_session
    from fastapi import FastAPI
    from ar_kernel.contract.verify import _MockCaption
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.tools.jobs import register_job_tools
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
    rec = Recorder(tmp_path / "run")
    reg, q = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5)
    q.register(_MockCaption())
    kit, mcp = ToolKit(reg, rec), new_mcp()
    register_job_tools(mcp, kit, q)
    register_caption_tool(mcp, kit, q)
    services = RunServices(socket_dir_for(tmp_path / "run"))
    services.start(FastAPI(), build_tool_app(mcp))
    (tmp_path / "a.mp4").write_bytes(b"ok")
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path / "staging")

    async def calls():
        async with mcp_session(services.socket_dir, caller.token) as session:
            job = json.loads((await session.call_tool("caption_videos", {"paths": ["a.mp4"], "prompt": "p"}))
                             .content[0].text)["job_id"]
            done = json.loads((await session.call_tool("job_wait", {"job_id": job})).content[0].text)
            bad = await session.call_tool("caption_videos", {"paths": ["/etc/passwd"], "prompt": "p"})
            return done, bad
    try:
        done, bad = asyncio.run(calls())
    finally:
        services.stop(), q.shutdown()
    assert done["state"] == "done" and done["result"]["clips"] == {"/workspace/a.mp4": {"caption": "mock caption"}}
    assert bad.is_error and "outside /workspace" in bad.content[0].text


@pytest.mark.gpu
def test_real_model_captions_two_example_clips(tmp_path):
    """Run with an explicit device list, e.g. CUDA_VISIBLE_DEVICES=0,1,4,5 pytest -m gpu -k real_model."""
    gpus = resolve_gpus(REAL, os.environ)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=300)
    q.register(CaptionBackend(REAL, tmp_path / "run", gpus, reg, rec))
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(), staging.mkdir()
    examples = REAL.worldmodel / "data" / "examples"
    shutil.copy(examples / "video_caption_camera" / "videos" / "clip_0001.mp4", staging / "cam.mp4")
    shutil.copy(examples / "video_caption_static" / "videos" / "clip_0001.mp4", staging / "static.mp4")
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    try:
        job = submit(q, caller, ["/workspace/staging/cam.mp4", "/workspace/staging/static.mp4"],
                     "Write one factual caption (1-3 sentences) describing the scene and how the camera "
                     "moves.")["job_id"]
        out = q.wait(caller, job, 300)
        while out["state"] not in ("done", "failed", "cancelled"):
            out = q.wait(caller, job, 300)
    finally:
        q.shutdown()
    latencies = [e["latency_s"] for e in rec.read_events("gpu") if e["type"] == "caption.clip"]
    print(json.dumps({"gpus": gpus, "state": out["state"], "error": out["error"], "result": out["result"],
                      "latencies_s": latencies}, indent=2))
    assert out["state"] == "done", out["error"]
    clips = out["result"]["clips"]
    assert all(len(v.get("caption", "")) > 20 for v in clips.values()), clips
    assert out["result"]["gpu_memory_released"] is True
