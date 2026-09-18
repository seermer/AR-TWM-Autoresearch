from __future__ import annotations
import hashlib, json, os, time, traceback, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

class TelemetryError(RuntimeError):
    """Telemetry could not be persisted; the caller must not proceed."""

class Recorder:
    def __init__(self, run_dir: Path, redact: Iterable[str] = ()) -> None:
        self.run_dir = Path(run_dir)
        self.redact = [s for s in redact if s]
        self._events = self.run_dir / "telemetry" / "events"
        self._payloads = self.run_dir / "telemetry" / "payloads"
        for directory in (self._events, self._payloads):
            directory.mkdir(parents=True, exist_ok=True)

    def events_path(self, node: str = "run") -> Path:
        return self._events / f"{node}.jsonl"

    def _scrub(self, obj: Any) -> Any:
        if isinstance(obj, str):
            for secret in self.redact:
                obj = obj.replace(secret, "[REDACTED]")
            return obj
        if isinstance(obj, dict):
            return {k: self._scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._scrub(v) for v in obj]
        return obj

    def store_payload(self, obj: dict) -> str:
        blob = json.dumps(self._scrub(obj), sort_keys=True, default=str).encode()
        digest = hashlib.sha256(blob).hexdigest()
        target = self._payloads / f"{digest}.json"
        if not target.exists():
            try:
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(blob)
                os.replace(tmp, target)
            except OSError as exc:
                raise TelemetryError(f"cannot write payload {digest}: {exc}") from exc
        return digest

    def load_payload(self, digest: str) -> dict:
        return json.loads((self._payloads / f"{digest}.json").read_text())

    def event(self, type: str, *, node: str = "run", phase: str = "-", attempt: int = 0,
              payload: dict | None = None, span_id: str | None = None,
              parent_span_id: str | None = None, **fields: Any) -> str:
        record = {
            "ts_wall": time.time(),
            "ts_mono": time.monotonic(),
            "run_dir": str(self.run_dir),
            "node": node,
            "phase": phase,
            "attempt": attempt,
            "span_id": span_id or uuid.uuid4().hex,
            "parent_span_id": parent_span_id,
            "type": type,
            "payload": self.store_payload(payload) if payload is not None else None,
            **self._scrub(fields),
        }
        try:
            with self.events_path(node).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TelemetryError(f"cannot append event {type}: {exc}") from exc
        return record["span_id"]

    def read_events(self, node: str = "run") -> list[dict]:
        path = self.events_path(node)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @contextmanager
    def span(self, type: str, *, node: str = "run", phase: str = "-", attempt: int = 0,
             payload: dict | None = None, **fields: Any):
        span_id = self.event(f"{type}.start", node=node, phase=phase, attempt=attempt,
                             payload=payload, **fields)
        started = time.monotonic()
        try:
            yield span_id
        except BaseException as exc:
            self.event(f"{type}.error", node=node, phase=phase, attempt=attempt,
                       span_id=span_id,
                       payload={"error": repr(exc), "traceback": traceback.format_exc()},
                       duration_s=time.monotonic() - started)
            raise
        self.event(f"{type}.end", node=node, phase=phase, attempt=attempt, span_id=span_id,
                   duration_s=time.monotonic() - started)
