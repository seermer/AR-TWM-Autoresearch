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
from ar_kernel.tools import captioner, vllm_server
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
    assert clips == {
        "/workspace/clips/a.mp4": {"caption": "Describe the camera motion. [10 bytes, tp=4, gpus=0,1,4,5]"},
        "/workspace/staging/b.mp4": {"caption": "Describe the camera motion. [20 bytes, tp=4, gpus=0,1,4,5]"}}
    assert out["result"]["load_s"] > 0 and out["result"]["gpu_memory_released"] is True
    assert out["progress"] == {"captioned": 2, "total": 2}
    run = tmp_path / "run"
    assert _pid_gone(run, job) and not (run / "jobs" / job / "media").exists()
    argv = json.loads((run / "jobs" / job / "fake_vllm.argv.json").read_text())
    assert argv[argv.index("--allowed-local-media-path") + 1] == str(run / "jobs" / job / "media")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    events = rec.read_events("n1")
    assert [(e["reason"], e["exit_code"]) for e in events if e["type"] == "caption.server_stopped"] == \
        [("done", -15)]
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


def test_a_parent_directory_swapped_for_a_link_after_the_check_is_refused(make, tmp_path, monkeypatch):
    """The agent keeps running during the job: a directory on the path swapped for a link to a
    host directory between the check and the open must not get host videos captioned."""
    q, caller, _, ws, _ = make()
    (ws / "clips" / "a.mp4").write_bytes(b"agent clip")
    host = tmp_path / "host_videos"
    host.mkdir()
    (host / "a.mp4").write_bytes(b"HOST VIDEO")
    checked = captioner.clip_host_path

    def check_then_swap(c, path):
        src = checked(c, path)                      # the job's check passes on the real directory
        (ws / "clips").rename(ws / "clips_moved")
        (ws / "clips").symlink_to(host)             # then the parent becomes a link to the host
        return src

    with q.gpu_lock:                                # submit (and its check) before the swap hook
        job = submit(q, caller, ["clips/a.mp4"], "Caption.")["job_id"]
        monkeypatch.setattr(captioner, "clip_host_path", check_then_swap)
    out = q.wait(caller, job, 120)
    assert out["state"] == "done", out
    assert "not inside this caller's workspace" in out["result"]["clips"]["/workspace/clips/a.mp4"]["error"]
    assert out["result"]["load_s"] is None           # nothing staged, so no server was started
    assert (host / "a.mp4").stat().st_nlink == 1     # never linked into the job's media dir


def test_default_tensor_parallel_is_the_largest_power_of_two_within_the_gpus(tmp_path):
    for gpus, tp in (([0, 1, 2, 3], "4"), ([0, 1, 2, 3, 4, 5], "4"), ([0, 1, 2, 3, 4, 5, 6, 7], "8")):
        cmd = CaptionBackend(REAL, tmp_path, gpus, None, None).server_command(8000, tmp_path)
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == tp


def test_the_encoder_cache_fits_every_clip_the_video_budget_allows(tmp_path):
    """vLLM's encoder cache is max(--max-num-batched-tokens, its own estimate of one video's tokens),
    and its estimate (12288 for Qwen3.8's 25165824-pixel video budget) is exceeded by clips whose
    sampled frame count is odd: a 1080p clip of 13 frames (~6.5 s at 2 fps) takes 14280 tokens.
    Such a clip was rejected with HTTP 400 in loopcheck_20260927 n1 (verification log, 2026-09-28)."""
    cmd = CaptionBackend(REAL, tmp_path, [0, 1, 2, 3], None, None).server_command(8000, tmp_path)
    assert int(cmd[cmd.index("--max-num-batched-tokens") + 1]) >= 14280


def test_the_server_takes_the_configured_batch_limits(tmp_path):
    cmd = CaptionBackend(fake_cfg(max_num_seqs=7, max_num_batched_tokens=65536), tmp_path, [0, 1, 2, 3],
                         None, None).server_command(8000, tmp_path)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "7"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "65536"


def test_the_server_runs_with_reasoning_and_a_data_parallel_encoder(tmp_path):
    cmd = CaptionBackend(REAL, tmp_path, [0, 1, 2, 3], None, None).server_command(8000, tmp_path)
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"
    assert cmd[cmd.index("--mm-encoder-tp-mode") + 1] == "data"
    assert "--speculative-config" not in cmd                      # off in the owner's verified launch
    assert "max_tokens" not in REAL.get("captioner")
    mtp = {"method": "mtp", "num_speculative_tokens": 3}
    cmd = CaptionBackend(fake_cfg(speculative_config=mtp), tmp_path, [0, 1, 2, 3], None, None).server_command(8000, tmp_path)
    assert json.loads(cmd[cmd.index("--speculative-config") + 1]) == mtp


def test_each_clips_reasoning_is_recorded_but_not_returned_to_the_agent(make):
    q, caller, rec, ws, _ = make()
    (ws / "a.mp4").write_bytes(b"x" * 10)
    out = q.wait(caller, submit(q, caller, ["a.mp4"], "Caption.")["job_id"], 120)
    assert out["state"] == "done" and set(out["result"]["clips"]["/workspace/a.mp4"]) == {"caption"}
    [clip] = [e for e in rec.read_events("n1") if e["type"] == "caption.clip"]
    assert rec.load_payload(clip["payload"])["reasoning"] == "thinking about 10 bytes"


def test_clips_are_captioned_concurrently_up_to_max_num_seqs(make):
    """Sequential requests left the server's batching unused (n1: 53 clips x 3.5 s)."""
    q, caller, rec, ws, _ = make(fake_cfg(max_num_seqs=6, extra_args=["--fake-mode", "second"]))
    names = [f"c{i}.mp4" for i in range(6)]
    for i, n in enumerate(names):
        (ws / n).write_bytes(b"v" * (i + 1))
    job = submit(q, caller, names, "Caption.")["job_id"]
    out = q.wait(caller, job, 120)
    assert out["state"] == "done", out
    assert list(out["result"]["clips"]) == [f"/workspace/{n}" for n in names]          # submitted order
    assert all("caption" in v for v in out["result"]["clips"].values())
    events = rec.read_events("n1")
    ready = next(e["ts_wall"] for e in events if e["type"] == "caption.server_ready")
    last = max(e["ts_wall"] for e in events if e["type"] == "caption.clip")
    assert last - ready < 3.5                                   # 6 one-second captions, not 6 s in a row
    assert out["progress"] == {"captioned": 6, "total": 6}


def test_unreadable_gpu_memory_is_none_not_a_crash(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, "0, [N/A]\n1, [Not Supported]\n", ""))
    assert vllm_server.gpu_memory_mib([0, 1]) is None


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
    assert [e["reason"] for e in rec.read_events("n1") if e["type"] == "caption.server_stopped"] == ["cancelled"]


def test_server_that_dies_during_startup_fails_the_job_with_its_log(make, tmp_path):
    q, caller, rec, ws, _ = make(fake_cfg(extra_args=["--fake-mode", "exit"]))
    (ws / "a.mp4").write_bytes(b"ok")
    out = q.wait(caller, submit(q, caller, ["a.mp4"], "Caption.")["job_id"], 120)
    assert out["state"] == "failed"
    assert "exited during startup" in out["error"] and "CUDA out of memory" in out["error"]
    assert [e["reason"] for e in rec.read_events("n1") if e["type"] == "caption.server_stopped"] == ["failed"]


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
