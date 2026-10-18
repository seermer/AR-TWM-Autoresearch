"""locate_events: the pieces and sheets, the answer shape, the span rule, and the job against a fake
model server (tests/fixtures/fake_vllm.py, no GPU)."""
import json
import os
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from conftest import job_result, make_mp4
from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import locate_events as le
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError

FAKE = Path(__file__).parent / "fixtures" / "fake_vllm.py"
REAL = KernelConfig.load()


@pytest.mark.parametrize("frames, fps, last, count", [
    (240, 24, (9.0, 10.0), 10), (243, 24, (9.125, 10.125), 11), (12, 24, (0.0, 0.5), 1), (24, 24, (0.0, 1.0), 1)])
def test_a_clip_is_cut_into_seconds_and_a_short_last_piece_moves_back(frames, fps, last, count):
    cut = le.pieces(frames, fps)
    assert len(cut) == count and cut[-1] == last
    assert all(end - start == 1.0 for start, end in cut[:-1])


def test_a_sheet_shows_sixteen_frames_a_second_with_times_from_zero():
    times = le.sheet_times(9.125, 10.125)
    assert times == [i / 16 for i in range(16)]
    assert len(le.sheet_times(0.0, 0.5)) == 8                      # a clip shorter than a second
    frames = [Image.new("RGB", (64, 36), (i, 0, 0)) for i in range(48)]
    sheet = le.sheet(frames, 24.0, 1.0, times)
    assert sheet.size == (4 * 64 + 5 * le.GAP, 4 * (36 + le.CAPTION_H) + 5 * le.GAP)
    assert sheet.getpixel((1, 1)) == (255, 255, 255)               # white between the frames
    # the nearest source frame for each time: 1.0 s is frame 24, 1.0625 s is frame 26 (25.5 rounds to even)
    assert sheet.getpixel((le.GAP + 5, le.GAP + 5))[0] == 24
    assert sheet.getpixel((2 * le.GAP + 64 + 5, le.GAP + 5))[0] == 26


def test_an_answer_takes_only_the_sheets_own_times_or_none():
    model = le.answer_model([0.0, 0.0625, 0.125])
    ok = {"happening_now": True, "already_happened": False, "start": "0.06"}
    assert model.model_validate(ok).start == "0.06" and model.model_validate({**ok, "start": None}).start is None
    for bad in ({**ok, "start": "0.50"}, {"happening_now": True, "start": None}):
        with pytest.raises(ValueError):
            model.model_validate(bad)


@pytest.mark.parametrize("there, span", [
    ([0, 0, 0], None),
    ([0, 1, 1, 1], (1, 3, 0)),
    ([0, 1, 0, 1, 1, 1], (1, 5, 1)),             # one piece without it inside the run
    ([1, 0, 0, 0], (0, 0, 0)),
    ([1, 0, 1], (0, 2, 1)),                      # a tie: the earliest start, then the longest run
    ([1, 0, 0, 1], (0, 0, 1)),
])
def test_the_span_is_the_run_of_pieces_that_fits_best(there, span):
    assert le.best_span([bool(x) for x in there]) == span


def test_a_moved_back_last_piece_counts_only_its_own_seconds():
    cut = [(0.0, 1.0), (1.0, 2.0), (1.5, 2.5)]
    assert le.own_seconds(cut) == [1.0, 1.0, 0.5]
    assert le.best_span([True, False, True]) == (0, 2, 1)                              # one each: a tie
    assert le.best_span([True, False, True], le.own_seconds(cut)) == (0, 0, 0.5)      # the last piece is half as heavy
    no_event = {"happening_now": False, "already_happened": False, "start": None}
    here = {**no_event, "happening_now": True}
    got = le.spans(["A dog walks in."], cut, [[here, no_event, here]])
    assert got[0]["end_s"] == 1.0 and got[0]["disagreeing_s"] == 0.5


def test_spans_use_the_picked_start_and_end_with_the_last_piece():
    def a(now, done, start=None):
        return {"happening_now": now, "already_happened": done, "start": start}
    cut = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (2.5, 3.5)]
    none = [a(False, False)] * 4
    got = le.spans(["A dog walks in.", "Rain begins to fall."], cut,
                   [[a(False, False), a(True, False, "0.44"), a(False, True), a(False, True)], none])
    assert got == [{"event": "A dog walks in.", "start_s": 1.4375, "end_s": 3.5, "disagreeing_s": 0},
                   {"event": "Rain begins to fall.", "start_s": None, "end_s": None, "disagreeing_s": 0}]
    no_pick = le.spans(["A dog walks in."], cut[:2], [[a(False, False), a(False, True)]])
    assert no_pick[0]["start_s"] == 1.0 and no_pick[0]["end_s"] == 2.0       # the piece's own start


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


def test_a_job_asks_about_every_event_in_every_piece_and_returns_spans_only(env):
    q, caller, rec, run = env
    items = [{"video": "a.mp4", "events": ["A dog  walks in.", "NEVER a cat."]}, {"video": "b.mp4", "events": ["Rain."]}]
    out = q.wait(caller, le.submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    clips = job_result(out)["clips"]
    assert clips == {
        "/workspace/a.mp4": {"events": [
            {"event": "A dog walks in.", "start_s": 0.125, "end_s": 2.5, "disagreeing_s": 0},
            {"event": "NEVER a cat.", "start_s": None, "end_s": None, "disagreeing_s": 0}]},
        "/workspace/b.mp4": {"events": [{"event": "Rain.", "start_s": 0.125, "end_s": 1.0, "disagreeing_s": 0}]}}
    pieces = [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "locate.piece"]
    assert sorted((p["video"], p["event"], p["piece"]) for p in pieces) == [      # each event on its own
        ("/workspace/a.mp4", n, k) for n in (0, 1) for k in (0, 1, 2)] + [("/workspace/b.mp4", 0, 0)]
    assert all(p["reasoning"] == "looked at 16 frames" for p in pieces)          # recorded, not returned
    argv = json.loads((run / "jobs" / out["id"] / "fake_vllm.argv.json").read_text())
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 0}
    assert not (run / "jobs" / out["id"] / "media").exists()


def test_an_answer_off_the_asked_shape_is_that_clips_error(env):
    q, caller, _, _ = env
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["BROKEN."]},
                                               {"video": "b.mp4", "events": ["Rain."]}])["job_id"], 120)
    clips = job_result(out)["clips"]
    assert "error" in clips["/workspace/a.mp4"] and "events" in clips["/workspace/b.mp4"]


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
    pid = int((run / "jobs" / job_id / "fake_vllm.pid").read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_the_server_loads_while_the_sheets_are_drawn(env, monkeypatch):
    q, caller, _, run = env
    real_decode, seen = le.decode, []

    def decode(path, width):                             # the server process starts before any sheet is drawn
        argv = path.parent.parent / "fake_vllm.argv.json"
        deadline = time.monotonic() + 30
        while not argv.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        seen.append(argv.exists())
        return real_decode(path, width)
    monkeypatch.setattr(le, "decode", decode)
    out = q.wait(caller, le.submit(q, caller, [{"video": "a.mp4", "events": ["Rain."]}])["job_id"], 120)
    assert out["state"] == "done" and seen == [True]


def test_a_job_whose_clips_all_fail_prep_still_stops_the_server(env, tmp_path):
    q, caller, rec, run = env
    (tmp_path / "ws" / "c.mp4").write_text("not a video")
    out = q.wait(caller, le.submit(q, caller, [{"video": "c.mp4", "events": ["Rain."]}])["job_id"], 120)
    assert out["state"] == "done" and "error" in job_result(out)["clips"]["/workspace/c.mp4"]
    assert [e for e in rec.read_events("n1") if e["type"] == "locate.server_stopped"]
    assert not [e for e in rec.read_events("n1") if e["type"] == "locate.piece"]
    assert _server_gone(run, out["id"])


def test_a_cancel_while_the_sheets_are_drawn_asks_nothing(env, monkeypatch):
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
    assert not [e for e in rec.read_events("n1") if e["type"] == "locate.piece"]
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
