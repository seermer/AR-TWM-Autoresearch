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
    assert "sk-SECRET123" not in json.dumps(rec.load_payload(digest))
    assert b"sk-SECRET123" not in _raw_payload(tmp_path, digest)


def test_secret_rendered_through_str_of_an_object_is_redacted(tmp_path):
    """json.dumps(default=str) stringifies objects AFTER structural scrubbing."""
    from pathlib import PurePosixPath
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path, redact=["sk-SECRET123"])
    digest = rec.store_payload({"path": PurePosixPath("/tmp/sk-SECRET123/x")})
    assert "sk-SECRET123" not in json.dumps(rec.load_payload(digest))
    assert b"sk-SECRET123" not in _raw_payload(tmp_path, digest)


import json
import zstandard


def _raw_payload(run_dir, digest) -> bytes:
    """Decompressed on-disk bytes, so a test proves the secret never reached the disk."""
    path = run_dir / "telemetry" / "payloads" / f"{digest}.json.zst"
    return zstandard.ZstdDecompressor().decompress(path.read_bytes())


def test_every_event_carries_run_id_and_component(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path / "run42")
    rec.event("x.happened", node="n1", component="gateway")
    event = rec.read_events("n1")[0]
    assert event["run_id"] == "run42"
    assert event["component"] == "gateway"


def test_component_defaults_to_kernel(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.event("y")
    assert rec.read_events()[0]["component"] == "kernel"


def test_payloads_are_zstd_and_round_trip(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    big = {"messages": ["the same long prompt " * 200] * 20}
    digest = rec.store_payload(big)
    path = tmp_path / "telemetry" / "payloads" / f"{digest}.json.zst"
    assert path.exists()
    assert path.stat().st_size < len(json.dumps(big)) / 5
    assert rec.load_payload(digest) == big


def test_legacy_uncompressed_payloads_stay_readable(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    legacy = tmp_path / "telemetry" / "payloads" / "abc123.json"
    legacy.write_text('{"old": true}')
    assert rec.load_payload("abc123") == {"old": True}


def test_redaction_added_after_construction_applies(tmp_path):
    """Container tokens are issued after the recorder exists."""
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.add_redaction("tok-LATE-ISSUED")
    digest = rec.store_payload({"auth": "Bearer tok-LATE-ISSUED"})
    assert b"tok-LATE-ISSUED" not in _raw_payload(tmp_path, digest)
