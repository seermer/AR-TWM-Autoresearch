"""The telemetry writer: records, payloads, the file store, redaction and a failed write."""
import importlib
import json
import os
import signal
import threading

import pytest
import zstandard

from ar_kernel import telemetry

writer_mod = importlib.import_module("ar_kernel.telemetry.writer")     # the package's `writer` is the function


@pytest.fixture(autouse=True)
def _no_writer(monkeypatch):
    monkeypatch.setattr(writer_mod, "_writer", None)


def _records(run):
    out = []
    for path in sorted((run / "telemetry" / "records").glob("*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines()]
    return out


def _payload(run, digest):
    path = run / "telemetry" / "payloads" / f"{digest}.json.zst"
    return json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))


def test_the_first_record_names_the_writer_and_every_record_has_the_common_fields(tmp_path):
    w = telemetry.open(tmp_path)
    w.write("log", "loop", "s1", stream="stdout", text="hello")
    first, second = _records(tmp_path)
    assert first["kind"] == "writer" and first["run_id"] == tmp_path.name and first["pid"] == os.getpid()
    assert first["boot_id"] and first["process_start"] and first["argv"] and first["host"]
    assert {k: second[k] for k in ("v", "writer", "seq", "component", "kind", "span", "text")} == {
        "v": 1, "writer": w.id, "seq": 2, "component": "loop", "kind": "log", "span": "s1", "text": "hello"}
    assert second["thread"].startswith("MainThread:") and second["task"] is None and second["ts"] and second["mono"]
    assert telemetry.writer() is w


def test_seq_follows_the_file_order_across_threads(tmp_path):
    w = telemetry.open(tmp_path)
    threads = [threading.Thread(target=lambda: [w.write("log", "loop", None, text="x") for _ in range(200)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["seq"] for r in _records(tmp_path)] == list(range(1, 1602))


def test_a_full_segment_rolls_over_to_the_next_file(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_mod, "SEGMENT_BYTES", 2000)
    w = telemetry.open(tmp_path)
    for i in range(30):
        w.write("log", "loop", None, text=f"line {i:03d} " + "x" * 100)
    files = sorted((tmp_path / "telemetry" / "records").iterdir())
    assert len(files) > 1 and files[0].name == f"{w.id}-0001.jsonl"
    assert all(p.stat().st_size <= 2000 for p in files)
    assert [r["seq"] for r in _records(tmp_path)] == list(range(1, 32))


def test_nothing_is_fsynced(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    w = telemetry.open(tmp_path)
    w.write("start", "loop", "s1", payloads={"inputs": {"a": 1}}, name="loop")
    assert w.store_file(b"bytes") and calls == []


def test_payloads_are_stored_once_by_content(tmp_path):
    w = telemetry.open(tmp_path)
    w.write("start", "loop", "s1", payloads={"inputs": {"a": [1, 2]}}, name="loop")
    w.write("start", "loop", "s2", payloads={"inputs": {"a": [1, 2]}}, name="loop")
    _, one, two = _records(tmp_path)
    assert one["inputs"] == two["inputs"] and _payload(tmp_path, one["inputs"]) == {"a": [1, 2]}
    assert len(list((tmp_path / "telemetry" / "payloads").iterdir())) == 1


def test_secrets_are_redacted_in_records_and_payloads_but_not_in_stored_files(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-SECRET123\n")
    w = telemetry.open(tmp_path / "run", [env], {})
    w.add_redaction("ar-tok\"en")
    w.write("start", "loop", "s1", payloads={"inputs": {"cmd": "use sk-SECRET123", "path": tmp_path / "sk-SECRET123"}},
            name="loop", note="token ar-tok\"en")
    record = _records(tmp_path / "run")[1]
    assert record["note"] == "token [REDACTED]"
    assert _payload(tmp_path / "run", record["inputs"]) == {"cmd": "use [REDACTED]",
                                                            "path": f"{tmp_path}/[REDACTED]"}
    digest, size = w.store_file(b"key sk-SECRET123")
    assert (tmp_path / "run" / "telemetry" / "files" / digest).read_bytes() == b"key sk-SECRET123" and size == 16
    assert w.redact_env({"OPENAI_API_KEY": "sk-SECRET123", "HOME": "/h"}) == {"OPENAI_API_KEY": "<redacted>",
                                                                              "HOME": "/h"}


def test_a_stored_file_matching_a_clip_blob_is_a_hard_link(tmp_path):
    w = telemetry.open(tmp_path)
    digest, _ = w.store_file(b"video")
    (tmp_path / "telemetry" / "files" / digest).unlink()
    blob = tmp_path / "store" / "blobs" / "video" / f"{digest}.mp4"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"video")
    w.store_file(b"video")
    assert os.path.samefile(blob, tmp_path / "telemetry" / "files" / digest)
    assert not list((tmp_path / "telemetry" / "files").glob("*.tmp"))


def test_redaction_names_come_from_the_env_files_only(tmp_path):
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    a.write_text("HF_TOKEN=hf-file\nPROXY_URL=http://proxy:1\n# OLD_KEY=commented\nEMPTY_KEY=\nBLANK_KEY='  '\n"
                 "HFK=hfk-value\nPLAIN=not-a-secret\n")
    b.write_text("VLM_API_KEY=vlm-file\n")
    environ = {"HF_TOKEN": "hf-shell", "OTHER_TOKEN": "outside-env-files", "PLAIN": "x"}
    assert set(telemetry.env_secrets([a, b, tmp_path / "missing.env"], environ)) == {
        "hf-shell", "hf-file", "http://proxy:1", "hfk-value", "vlm-file"}


def test_a_redacted_value_never_breaks_a_json_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text('X_KEY="a\\"b"\n')
    w = telemetry.open(tmp_path / "run", [env], {})
    w.write("log", "loop", None, text='say a"b and a\\"b')
    assert _records(tmp_path / "run")[1]["text"] == 'say [REDACTED] and a\\"b'


def test_a_failed_write_raises_nothing_prints_once_stops_the_run_and_goes_quiet(tmp_path, monkeypatch, capfd):
    kills = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    w = telemetry.open(tmp_path)

    def broken(line):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(w, "_append", broken)
    for kind in ("log", "log"):
        w.write(kind, "loop", None, text="x")
    w.store_file(b"y")
    err = capfd.readouterr().err
    assert err.count("telemetry: cannot write") == 1 and "No space left on device" in err
    assert kills == [(os.getpid(), signal.SIGTERM)] and w.failed and telemetry.writer() is None


def test_a_failure_in_a_side_thread_stops_the_run_once(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(sig))
    w = telemetry.open(tmp_path)
    (tmp_path / "telemetry" / "payloads").rmdir()
    threads = [threading.Thread(target=w.write, args=("start", "loop", "s"), kwargs={"payloads": {"inputs": i}})
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert kills == [signal.SIGTERM]


def test_with_no_writer_there_is_none(tmp_path):
    assert telemetry.writer() is None


def test_a_string_no_utf8_can_hold_is_still_written(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("the run was stopped"))
    w = telemetry.open(tmp_path)
    name = b"clip-\xff.mp4".decode(errors="surrogateescape")             # a non-UTF-8 file name
    w.write("start", "loop", "s1", payloads={"inputs": {"path": name}}, name="loop", path=name)
    assert _records(tmp_path)[1]["path"] == name
