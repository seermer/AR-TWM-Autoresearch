"""locate_events: the frames, the span rule, and the job against a fake model server
(tests/fixtures/fake_vllm.py, no GPU)."""
import json
import os
import threading
import time
from pathlib import Path

import pytest

from conftest import job_result, make_mp4, job_folder
from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import locate_events as le
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError

FAKE = Path(__file__).parent / "fixtures" / "fake_vllm.py"
REAL = KernelConfig.load()


def test_a_clip_is_decoded_at_24_frames_a_second(tmp_path):
    make_mp4(tmp_path / "a.mp4", seconds=2.5, fps=24, width=320, height=180)
    make_mp4(tmp_path / "b.mp4", seconds=1.0, fps=30, width=320, height=180)
    frames = le.decode(tmp_path / "a.mp4", 160)
    assert len(frames) == 60 and frames[0].size == (160, 90)
    assert len(le.decode(tmp_path / "b.mp4", 160)) == 24


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
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    q.register(FakeServer(fake_cfg(), tmp_path / "run", [0, 1, 2, 3], reg, rec,
                          gpu_memory=lambda gpus: {g: 100 for g in gpus}, poll_s=0.2))
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2.5, fps=24, width=320, height=180)
    make_mp4(ws / "b.mp4", seconds=1.0, fps=24, width=320, height=180)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, rec, tmp_path / "run"
    q.shutdown()


def test_a_job_asks_about_every_event_in_every_frame_and_returns_spans_only(env):
    q, caller, rec, run = env
    items = [{"video": "a.mp4", "events": ["A dog  walks in.", "NEVER a cat."]}, {"video": "b.mp4", "events": ["Rain."]}]
    out = q.wait(caller, le.submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    clips = job_result(out)["clips"]
    assert clips == {
        "/workspace/a.mp4": {"events": [
            {"event": "A dog walks in.", "start_s": 0.0, "end_s": 2.5, "disagreeing_s": 0.0},
            {"event": "NEVER a cat.", "start_s": None, "end_s": None, "disagreeing_s": 0}]},
        "/workspace/b.mp4": {"events": [{"event": "Rain.", "start_s": 0.0, "end_s": 1.0, "disagreeing_s": 0.0}]}}
    asked = [e for e in rec.read_events("n1") if e["type"] == "locate.event"]
    got = [rec.load_payload(e["payload"]) for e in asked]
    assert sorted((p["video"], p["event"], len(p["answers"])) for p in got) == [  # each event on its own, every frame
        ("/workspace/a.mp4", 0, 60), ("/workspace/a.mp4", 1, 60), ("/workspace/b.mp4", 0, 24)]
    assert all(e["ok"] for e in asked)
    assert all(r == "looked at one frame" for p in got for r in p["reasonings"])  # recorded, not returned
    argv = json.loads((job_folder(run, out["id"]) / "fake_vllm.argv.json").read_text())
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 0}
    assert not (job_folder(run, out["id"]) / "media").exists()


def test_an_answer_off_the_asked_shape_is_that_clips_error_and_recorded(env):
    q, caller, rec, _ = env
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["BROKEN."]},
                                               {"video": "b.mp4", "events": ["Rain."]}])["job_id"], 120)
    clips = job_result(out)["clips"]
    assert "error" in clips["/workspace/a.mp4"] and "events" in clips["/workspace/b.mp4"]
    asked = {rec.load_payload(e["payload"])["video"]: (e["ok"], rec.load_payload(e["payload"]))
             for e in rec.read_events("n1") if e["type"] == "locate.event"}
    ok, broken = asked["/workspace/a.mp4"]
    assert not ok and broken["answers"] == [None] * 60
    assert all(r.startswith("ValidationError") for r in broken["reasonings"])         # each frame's error
    assert asked["/workspace/b.mp4"][0]


@pytest.mark.parametrize("items, message", [
    ([], "items is empty"),
    ([{"video": "a.mp4"}], "an item is"),
    ([{"video": "a.mp4", "events": []}], "non-empty list of sentences"),
    ([{"video": "a.mp4", "events": ["x", " "]}], "non-empty list of sentences"),
    ([{"video": "missing.mp4", "events": ["x"]}], "is not a file"),
    ([{"video": "a.mp4", "events": ["x"]}, {"video": "/workspace/a.mp4", "events": ["y"]}], "listed twice"),
])
def test_bad_items_are_refused_before_a_job_is_queued(env, items, message):
    q, caller, _, _ = env
    with pytest.raises(ToolError, match=message):
        le.submit(q, caller, items)


def _server_gone(run, job_id):
    pid_file = job_folder(run, job_id) / "fake_vllm.pid"
    if not pid_file.exists():                   # stopped before it wrote its pid (under load), or never started
        return True
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_the_server_loads_while_the_frames_are_decoded(env, monkeypatch):
    q, caller, _, run = env
    real_decode, seen = le.decode, []

    def decode(path, width):                             # the server process starts before any frame is decoded
        argv = path.parent.parent / "fake_vllm.argv.json"
        deadline = time.monotonic() + 30
        while not argv.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        seen.append(argv.exists())
        return real_decode(path, width)
    monkeypatch.setattr(le, "decode", decode)
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"], 120)
    assert out["state"] == "done" and seen == [True]


def test_a_file_that_is_not_a_video_is_refused_at_submit(env, tmp_path):
    q, caller, rec, run = env
    (tmp_path / "ws" / "c.mp4").write_text("not a video")
    with pytest.raises(ToolError, match="not a readable video"):
        le.submit(q, caller, [{"video": "c.mp4", "events": ["Rain."]}])


def test_a_job_whose_clips_all_fail_prep_still_stops_the_server(env, tmp_path, monkeypatch):
    q, caller, rec, run = env
    monkeypatch.setattr(le, "decode", lambda path, width: (_ for _ in ()).throw(ValueError("cannot decode")))
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"], 120)
    assert out["state"] == "done" and "cannot decode" in job_result(out)["clips"]["/workspace/a.mp4"]["error"]
    assert [e for e in rec.read_events("n1") if e["type"] == "locate.server_stopped"]
    assert not [e for e in rec.read_events("n1") if e["type"] == "locate.event"]
    assert _server_gone(run, out["id"])


def test_a_cancel_while_the_frames_are_decoded_asks_nothing(env, monkeypatch):
    q, caller, rec, run = env
    real_decode, drawing = le.decode, threading.Event()

    def decode(path, width):
        drawing.set()
        time.sleep(3)
        return real_decode(path, width)
    monkeypatch.setattr(le, "decode", decode)
    job_id = le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"]
    assert drawing.wait(60)
    q.cancel(caller, job_id)
    out = q.wait(caller, job_id, 120)
    assert out["state"] == "canceled"
    assert not [e for e in rec.read_events("n1") if e["type"] == "locate.event"]
    assert _server_gone(run, job_id)


def test_a_server_that_dies_while_loading_fails_the_job(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    q.register(FakeServer(fake_cfg(captioner={"extra_args": ["--fake-mode", "exit"]}), tmp_path / "run", [0, 1, 2, 3],
                          reg, rec, gpu_memory=lambda gpus: {g: 100 for g in gpus}, poll_s=0.2))
    ws = tmp_path / "ws"
    make_mp4(ws / "a.mp4", seconds=1.0, fps=24, width=320, height=180)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=tmp_path / "st")
    try:
        out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"], 120)
    finally:
        q.shutdown()
    assert out["state"] == "failed" and "exited during startup" in out["error"]
