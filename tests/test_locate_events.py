"""locate_events: the frames, the span rule, and the job against a fake model server
(tests/fixtures/fake_vllm.py, no GPU)."""
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import locate_vllm

from conftest import job_result, make_mp4, job_folder
from ar_kernel.config import KernelConfig
from ar_kernel.tools import locate_events as le
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError

FAKE = Path(__file__).parent / "fixtures" / "fake_vllm.py"
REAL = KernelConfig.load()


def test_a_clip_is_decoded_at_24_frames_a_second(tmp_path):
    make_mp4(tmp_path / "a.mp4", seconds=2.5, fps=24, width=320, height=180)
    make_mp4(tmp_path / "b.mp4", seconds=1.0, fps=30, width=320, height=180)
    frames = locate_vllm.decode(str(tmp_path / "a.mp4"), 320, 180, 160, 24)
    assert len(frames) == 60 and frames[0].size == (160, 90)
    assert len(locate_vllm.decode(str(tmp_path / "b.mp4"), 320, 180, 160, 24)) == 24


def test_ask_parses_the_strict_answer_and_refuses_any_other_shape():
    def stub(parsed, content="..."):
        message = SimpleNamespace(parsed=parsed, content=content, reasoning_content="why")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **kw: calls.append(kw) or SimpleNamespace(choices=[SimpleNamespace(message=message)]))))
    calls = []
    answer = locate_vllm.Answer(happening_now=True, already_happened=False)
    assert locate_vllm.ask(stub(answer), "data:x", "SYS", "Rain.", "low") == (
        {"happening_now": True, "already_happened": False}, "why")
    assert calls[0]["response_format"] is locate_vllm.Answer and calls[0]["model"] == "locator"
    assert calls[0]["messages"][1]["content"][1] == {"type": "text", "text": "Event: Rain."}
    with pytest.raises(ValueError, match="no answer in the asked shape"):
        locate_vllm.ask(stub(None, "I cannot tell."), "data:x", "SYS", "Rain.", "low")


@pytest.mark.parametrize("there, span", [
    ([0, 0, 0], None),
    ([0, 1, 1, 1], (1, 3, 0)),
    ([0, 1, 0, 1, 1, 1], (1, 5, 1)),             # one frame without it inside the run
    ([1, 0, 0, 0], (0, 0, 0)),
    ([1, 0, 1], (0, 2, 1)),                      # a tie: the earliest start, then the longest run
    ([1, 0, 0, 1], (0, 0, 1)),
])
def test_the_span_is_the_run_of_frames_that_fits_best(there, span):
    assert le.best_span([bool(x) for x in there]) == span


def test_a_span_starts_six_frames_before_its_first_frame_and_ends_after_its_last():
    def a(now, done=False):
        return {"happening_now": now, "already_happened": done}
    late = [a(False)] * 10 + [a(True)] * 5 + [a(False, True)] * 5 + [a(False)] + [a(True)]
    early = [a(False)] * 2 + [a(True)] * 4
    got = le.spans(["A dog walks in.", "A cat sits.", "Rain begins to fall."], [late, early, [a(False)] * 6])
    assert got == [{"event": "A dog walks in.", "start_s": round(4 / 24, 4), "end_s": round(22 / 24, 4),
                    "disagreeing_s": round(1 / 24, 4)},
                   {"event": "A cat sits.", "start_s": 0.0, "end_s": 0.25, "disagreeing_s": 0.0},
                   {"event": "Rain begins to fall.", "start_s": None, "end_s": None, "disagreeing_s": 0}]


def fake_cfg(captioner=None, **over) -> KernelConfig:
    c = {**REAL.get("captioner"), "env": "autoresearcher", "memory_release_timeout_s": 2,
         **(captioner or {})}
    return KernelConfig(raw={"captioner": c, "locate_events": {**REAL.get("locate_events"), **over}},
                        repo_root=REAL.repo_root)


class FakeServer(le.LocateEventsBackend):
    """The real command line, with `vllm serve` swapped for the fake server."""
    def server_command(self, port):
        return ["python", str(FAKE), *super().server_command(port)[2:]]


@pytest.fixture
def env(tmp_path):
    reg = TokenRegistry()
    q = JobQueue(tmp_path / "run", threading.Lock(), wait_cap_s=120)
    q.register(FakeServer(fake_cfg(), tmp_path / "run", [0, 1, 2, 3], reg,
                          gpu_memory=lambda gpus: {g: 100 for g in gpus}))
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2.5, fps=24, width=320, height=180)
    make_mp4(ws / "b.mp4", seconds=1.0, fps=24, width=320, height=180)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, tmp_path / "run"
    q.shutdown()


def test_a_job_asks_about_every_event_in_every_frame_and_returns_spans_only(env):
    q, caller, run = env
    items = [{"video": "a.mp4", "events": ["A dog  walks in.", "NEVER a cat."]}, {"video": "b.mp4", "events": ["Rain."]}]
    out = q.wait(caller, le.submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    clips = job_result(out)["clips"]
    assert clips == {
        "/workspace/a.mp4": {"events": [
            {"event": "A dog walks in.", "start_s": 0.0, "end_s": 2.5, "disagreeing_s": 0.0},
            {"event": "NEVER a cat.", "start_s": None, "end_s": None, "disagreeing_s": 0}]},
        "/workspace/b.mp4": {"events": [{"event": "Rain.", "start_s": 0.0, "end_s": 1.0, "disagreeing_s": 0.0}]}}
    asked = {m["index"]: m for m in messages(run, out["id"]) if m["type"] == "item"}
    assert [[len(e) for e in asked[i]["info"]["answers"]] for i in (0, 1)] == [[60, 60], [24]]  # every event, every frame
    assert all(asked[i]["ok"] for i in (0, 1))
    assert all(r == "looked at one frame" for m in asked.values() for e in m["info"]["reasonings"] for r in e)
    assert out["progress"] == {"done": 2, "total": 2}
    argv = json.loads((job_folder(run, out["id"]) / "fake_vllm.argv.json").read_text())
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 0}
    assert not (job_folder(run, out["id"]) / "media").exists()


def test_an_answer_off_the_asked_shape_is_that_clips_error_and_kept(env):
    q, caller, run = env
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["BROKEN."]},
                                               {"video": "b.mp4", "events": ["Rain."]}])["job_id"], 120)
    clips = job_result(out)["clips"]
    assert clips["/workspace/a.mp4"]["error"].startswith("ValidationError") and "events" in clips["/workspace/b.mp4"]
    asked = {m["index"]: m for m in messages(run, out["id"]) if m["type"] == "item"}
    assert not asked[0]["ok"] and asked[0]["info"]["answers"] == [[None] * 60]
    assert all(r.startswith("ValidationError") for r in asked[0]["info"]["reasonings"][0])     # each frame's error
    assert asked[1]["ok"]


@pytest.mark.parametrize("items, message", [
    ([], "items is empty"),
    ([{"video": "a.mp4"}], "an item is"),
    ([{"video": "a.mp4", "events": []}], "non-empty list of sentences"),
    ([{"video": "a.mp4", "events": ["x", " "]}], "non-empty list of sentences"),
    ([{"video": "missing.mp4", "events": ["x"]}], "is not a file"),
    ([{"video": "a.mp4", "events": ["x"]}, {"video": "/workspace/a.mp4", "events": ["y"]}], "listed twice"),
])
def test_bad_items_are_refused_before_a_job_is_queued(env, items, message):
    q, caller, _ = env
    with pytest.raises(ToolError, match=message):
        le.submit(q, caller, items)


def messages(run, job_id) -> list[dict]:
    path = job_folder(run, job_id) / "messages.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _server_gone(folder: Path) -> bool:
    pid_file = folder / "fake_vllm.pid"
    if not pid_file.exists():                   # stopped before it wrote its pid (under load), or never started
        return True
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(int(pid_file.read_text()), 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def bridge_here(tmp_path, monkeypatch, videos: list[Path]) -> list[dict]:
    """bridges/locate_vllm.py run in this process (so its decode can be swapped) against the fake server; its
    messages."""
    from ar_kernel.subproc import free_port
    work, port = tmp_path / "work", free_port()
    work.mkdir()
    b = REAL.get("locate_events")
    command = FakeServer(fake_cfg(), work, [0, 1, 2, 3], None, None).server_command(port)
    job = {"work": str(work), "serve": {"argv": [sys.executable, *command[1:]], "env": {"HF_HUB_OFFLINE": "1"},
                                        "port": port},
           "system": le.SYSTEM, "fps": 24, "frame_width": 160, "concurrent_requests": 4, "request_timeout_s": 10,
           "reasoning_effort": b["reasoning_effort"],
           "items": [{"index": i, "video": str(v), "width": 320, "height": 180, "events": ["Rain."]}
                     for i, v in enumerate(videos)]}
    (work / "job.json").write_text(json.dumps(job))
    monkeypatch.setattr(sys, "argv", ["locate_vllm.py", "--job", str(work / "job.json")])
    monkeypatch.chdir(work)
    assert locate_vllm.main() == 0
    assert _server_gone(work)
    return [json.loads(line) for line in (work / "messages.jsonl").read_text().splitlines()]


def test_the_server_loads_while_the_frames_are_decoded(tmp_path, monkeypatch):
    make_mp4(tmp_path / "a.mp4", seconds=1.0, fps=24, width=320, height=180)
    real_decode, seen = locate_vllm.decode, []

    def decode(*args):                                   # the server process starts before any frame is decoded
        argv = tmp_path / "work" / "fake_vllm.argv.json"
        deadline = time.monotonic() + 30
        while not argv.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        seen.append(argv.exists())
        return real_decode(*args)
    monkeypatch.setattr(locate_vllm, "decode", decode)
    got = bridge_here(tmp_path, monkeypatch, [tmp_path / "a.mp4"])
    assert seen == [True] and [m["ok"] for m in got if m["type"] == "item"] == [True]


def test_a_clip_that_cannot_be_decoded_is_its_error_and_the_server_still_stops(tmp_path, monkeypatch):
    make_mp4(tmp_path / "a.mp4", seconds=1.0, fps=24, width=320, height=180)
    monkeypatch.setattr(locate_vllm, "decode", lambda *a: (_ for _ in ()).throw(ValueError("cannot decode")))
    got = bridge_here(tmp_path, monkeypatch, [tmp_path / "a.mp4"])
    [item] = [m for m in got if m["type"] == "item"]
    assert not item["ok"] and item["error"] == "ValueError: cannot decode" and "cannot decode" in item["traceback"]
    assert got[-1]["type"] == "job_done"


def test_a_file_that_is_not_a_video_is_refused_at_submit(env, tmp_path):
    q, caller, run = env
    (tmp_path / "ws" / "c.mp4").write_text("not a video")
    with pytest.raises(ToolError, match="not a readable video"):
        le.submit(q, caller, [{"video": "c.mp4", "events": ["Rain."]}])


def test_a_cancel_before_the_server_is_ready_asks_nothing(tmp_path):
    reg = TokenRegistry()
    q = JobQueue(tmp_path / "run", threading.Lock(), wait_cap_s=120)
    q.register(FakeServer(fake_cfg(captioner={"extra_args": ["--fake-mode", "hang"]}), tmp_path / "run", [0, 1, 2, 3],
                          reg, gpu_memory=lambda gpus: {g: 100 for g in gpus}))
    ws = tmp_path / "ws"
    make_mp4(ws / "a.mp4", seconds=1.0, fps=24, width=320, height=180)
    (tmp_path / "st").mkdir()
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=tmp_path / "st")
    try:
        job_id = le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"]
        assert q.wait(caller, job_id, 6)["state"] == "running"
        q.cancel(caller, job_id)
        out = q.wait(caller, job_id, 120)
    finally:
        q.shutdown()
    assert out["state"] == "canceled"
    assert job_result(out)["clips"] == {"/workspace/a.mp4": {"error": "canceled before this item finished"}}
    assert not [m for m in messages(tmp_path / "run", job_id) if m["type"] == "item"]
    assert _server_gone(job_folder(tmp_path / "run", job_id))


def test_a_server_that_dies_while_loading_fails_every_clip_with_its_log(tmp_path):
    reg = TokenRegistry()
    q = JobQueue(tmp_path / "run", threading.Lock(), wait_cap_s=120)
    q.register(FakeServer(fake_cfg(captioner={"extra_args": ["--fake-mode", "exit"]}), tmp_path / "run", [0, 1, 2, 3],
                          reg, gpu_memory=lambda gpus: {g: 100 for g in gpus}))
    ws = tmp_path / "ws"
    make_mp4(ws / "a.mp4", seconds=1.0, fps=24, width=320, height=180)
    (tmp_path / "st").mkdir()
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=tmp_path / "st")
    try:
        out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"], 120)
    finally:
        q.shutdown()
    assert out["state"] == "done"
    error = job_result(out)["clips"]["/workspace/a.mp4"]["error"]
    assert error.startswith("not run: worker 0 ended before starting any item (exit code 1); log tail:")
    assert "the model server failed to start" in error
