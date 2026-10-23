"""telemetry/files.py: every file a kernel tool reads or writes, stored once by content, with a `file` event."""
import hashlib
import os
import threading

import pytest

from ar_kernel.archive.blobs import BlobStore
from ar_kernel.archive.db import open_db
from ar_kernel.telemetry import files
from ar_kernel.telemetry.recorder import Recorder

AT = dict(node="n1", phase="improve_recipe", attempt=2)


@pytest.fixture
def env(tmp_path):
    return Recorder(tmp_path / "run"), tmp_path / "run" / "telemetry" / "files"


def file_events(rec):
    return [e for e in rec.read_events("n1") if e["type"] == "file"]


def test_the_same_bytes_are_stored_once_with_an_event_each(env):
    rec, store = env
    with files.scope(rec, tool="video_probe", parent_span_id="s1", **AT):
        first = files.record(b"clip bytes", direction="in", path="/workspace/a.mp4")
        again = files.record(b"clip bytes", direction="in", path="/workspace/b.mp4")
    digest = hashlib.sha256(b"clip bytes").hexdigest()
    assert first == again == digest
    assert [p.name for p in store.iterdir()] == [digest] and (store / digest).read_bytes() == b"clip bytes"
    events = file_events(rec)
    assert [e["path"] for e in events] == ["/workspace/a.mp4", "/workspace/b.mp4"]
    e = events[0]
    assert (e["direction"], e["digest"], e["bytes"], e["tool"], e["parent_span_id"]) == ("in", digest, 10, "video_probe", "s1")
    assert (e["node"], e["phase"], e["attempt"]) == ("n1", "improve_recipe", 2)


def test_a_file_the_clip_store_holds_is_a_link_to_its_blob_and_outlives_it(env, tmp_path):
    rec, store = env
    conn = open_db(tmp_path / "run")
    blobs = BlobStore(tmp_path / "run", conn)
    (tmp_path / "c.mp4").write_bytes(b"archived")
    digest = blobs.put(tmp_path / "c.mp4", "video")
    with files.scope(rec, tool="data_fetch", parent_span_id="s1", **AT):
        files.record(b"archived", direction="out", path="/workspace/staging/pool_clips/videos/c.mp4")
    assert os.stat(store / digest).st_ino == blobs.path(digest, "video").stat().st_ino
    assert [p.name for p in store.iterdir()] == [digest]
    assert blobs.remove_unreferenced() == 1 and not blobs.path(digest, "video").exists()   # an abandoned attempt's clip
    assert (store / digest).read_bytes() == b"archived"


def test_rewriting_the_source_later_leaves_the_record(env, tmp_path):
    rec, store = env
    src = tmp_path / "a.mp4"
    src.write_bytes(b"first")
    with files.scope(rec, tool="video_probe", parent_span_id="s1", **AT), src.open("rb") as handle:
        digest = files.record(handle, direction="in", path="/workspace/a.mp4")
    src.write_bytes(b"second")                         # rewritten in place, as ffmpeg -y does
    assert (store / digest).read_bytes() == b"first"


def test_threads_recording_the_same_bytes_leave_one_whole_file(env):
    rec, store = env
    data = os.urandom(3 << 20)

    def one():
        with files.scope(rec, tool="caption_videos", job_id="j1", **AT):
            files.record(data, direction="in", path="/workspace/a.mp4")
    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [p.name for p in store.iterdir()] == [hashlib.sha256(data).hexdigest()]
    assert len(file_events(rec)) == 8
    assert all(e["job_id"] == "j1" for e in file_events(rec))


def test_a_file_recorded_outside_a_tool_call_or_job_raises(env):
    def outside():                             # a new thread starts without the tests' own scope
        try:
            files.record(b"x", direction="in", path="/workspace/a.mp4")
        except RuntimeError as exc:
            errors.append(str(exc))
    errors = []
    thread = threading.Thread(target=outside)
    thread.start()
    thread.join()
    assert errors and "outside a tool call or job" in errors[0]
