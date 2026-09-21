import json, pytest
from ar_kernel.telemetry.recorder import Recorder, TelemetryError

def test_event_is_appended_with_identity_fields(tmp_path):
    rec = Recorder(tmp_path)
    rec.event("train.start", node="n1", phase="train", attempt=1, payload={"lr": 5e-5})
    lines = [json.loads(l) for l in rec.events_path("n1").read_text().splitlines()]
    assert len(lines) == 1
    e = lines[0]
    assert e["type"] == "train.start" and e["node"] == "n1" and e["phase"] == "train"
    assert e["attempt"] == 1 and e["ts_wall"] and e["ts_mono"] and e["span_id"]
    assert rec.load_payload(e["payload"]) == {"lr": 5e-5}

def test_large_payload_is_content_addressed_and_deduped(tmp_path):
    rec = Recorder(tmp_path)
    a = rec.store_payload({"prompt": "x" * 10_000})
    b = rec.store_payload({"prompt": "x" * 10_000})
    assert a == b
    assert rec.load_payload(a)["prompt"].startswith("xxx")

def test_secrets_are_redacted(tmp_path):
    rec = Recorder(tmp_path, redact=["sk-supersecret"])
    digest = rec.store_payload({"headers": {"authorization": "Bearer sk-supersecret"}})
    assert "sk-supersecret" not in json.dumps(rec.load_payload(digest))
    assert "[REDACTED]" in json.dumps(rec.load_payload(digest))

def test_span_emits_start_and_end(tmp_path):
    rec = Recorder(tmp_path)
    with rec.span("merge", node="n1", phase="eval"):
        pass
    types = [json.loads(l)["type"] for l in rec.events_path("n1").read_text().splitlines()]
    assert types == ["merge.start", "merge.end"]

def test_span_records_error_and_reraises(tmp_path):
    rec = Recorder(tmp_path)
    with pytest.raises(RuntimeError):
        with rec.span("merge", node="n1", phase="eval"):
            raise RuntimeError("boom")
    events = [json.loads(l) for l in rec.events_path("n1").read_text().splitlines()]
    assert events[-1]["type"] == "merge.error"
    assert "boom" in json.dumps(rec.load_payload(events[-1]["payload"]))

def test_fail_closed_when_events_cannot_be_written(tmp_path):
    rec = Recorder(tmp_path)
    (tmp_path / "telemetry" / "events").chmod(0o500)
    try:
        with pytest.raises(TelemetryError):
            rec.event("train.start", node="n2")
    finally:
        (tmp_path / "telemetry" / "events").chmod(0o700)


def test_secret_inside_a_tuple_is_redacted(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path, redact=["sk-SECRET123"])
    digest = rec.store_payload({"argv": ("--key", "sk-SECRET123")})
    assert "sk-SECRET123" not in (tmp_path / "telemetry" / "payloads" / f"{digest}.json").read_text()


def test_secret_rendered_through_str_of_an_object_is_redacted(tmp_path):
    """json.dumps(default=str) stringifies objects AFTER structural scrubbing."""
    from pathlib import PurePosixPath
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path, redact=["sk-SECRET123"])
    digest = rec.store_payload({"path": PurePosixPath("/tmp/sk-SECRET123/x")})
    assert "sk-SECRET123" not in (tmp_path / "telemetry" / "payloads" / f"{digest}.json").read_text()
