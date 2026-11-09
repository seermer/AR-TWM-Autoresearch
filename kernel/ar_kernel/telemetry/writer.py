"""The process's one telemetry writer: records, payloads and the file store under <run>/telemetry."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import zstandard
from dotenv import dotenv_values

VERSION = 1
SEGMENT_BYTES = 64 << 20
REDACTED = "[REDACTED]"

_writer: Writer | None = None


def env_secrets(env_files: Iterable[Path], environ) -> list[str]:
    """Values of the .env variables named *TOKEN*, *URL*, *KEY* or HFK: the shell's and the file's."""
    out = set()
    for path in env_files:
        for name, value in dotenv_values(path, interpolate=False).items() if Path(path).exists() else ():
            if any(k in name for k in ("TOKEN", "URL", "KEY")) or name == "HFK":
                out.update(v for v in (environ.get(name), value) if v and v.strip())
    return sorted(out, key=len, reverse=True)


def open(run_dir: Path, env_files: Iterable[Path] = (), environ=os.environ) -> Writer:      # noqa: A001
    global _writer
    _writer = Writer(Path(run_dir), env_secrets(env_files, environ))
    return _writer


def writer() -> Writer | None:
    """The open writer, or None when none is open or it has failed."""
    return _writer if _writer is not None and not _writer.failed else None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _start_time() -> str:
    """Field 22 of /proc/self/stat, as `subproc.proc_start_time` reads it."""
    return Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]


def _task() -> str | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:                         # no event loop in this thread
        return None
    return task.get_name() if task else None


class Writer:
    def __init__(self, run_dir: Path, secrets: list[str]) -> None:
        self.run_dir, self.secrets, self.failed = run_dir, list(secrets), False
        self.id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        self._stderr = sys.__stderr__
        self._records, self._payloads = run_dir / "telemetry" / "records", run_dir / "telemetry" / "payloads"
        self._files = run_dir / "telemetry" / "files"
        self._lock, self._seq, self._segment, self._out, self._size = threading.Lock(), 0, 0, None, 0
        self._fail_lock = threading.Lock()
        self.pending: list[tuple] = []           # records a signal handler left: it must not take the lock
        try:
            for d in (self._records, self._payloads, self._files):
                d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._fail(exc)
        self.write("writer", "cli", None, run_id=run_dir.name, host=socket.gethostname(), boot_id=_boot_id(),
                   pid=os.getpid(), process_start=_start_time(), argv=sys.argv,
                   python=sys.version, executable=sys.executable)

    def add_redaction(self, secret: str) -> None:
        if secret and secret.strip() and secret not in self.secrets:
            self.secrets = sorted([*self.secrets, secret], key=len, reverse=True)

    def redact_env(self, env: dict) -> dict:
        """An environment to record: each secret value shown as <redacted>."""
        return {k: "<redacted>" if v in self.secrets else v for k, v in env.items()}

    def scrub(self, obj: Any) -> Any:
        """`obj` as plain JSON values, each secret in its strings replaced."""
        if isinstance(obj, dict):
            return {self.scrub(str(k)): self.scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [self.scrub(v) for v in obj]
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        text = obj if isinstance(obj, str) else str(obj)
        for secret in self.secrets:
            text = text.replace(secret, REDACTED)
        return text

    def write(self, kind: str, component: str, span: str | None, payloads: dict | None = None,
              **fields: Any) -> None:
        """One record, after those a signal handler left in `pending`; each value in `payloads` is stored
        and recorded by its digest."""
        while True:
            try:
                left = self.pending.pop(0)
            except IndexError:
                break
            self._write(*left)
        self._write(kind, component, span, payloads, fields)

    def _write(self, kind: str, component: str, span: str | None, payloads: dict | None, fields: dict) -> None:
        if self.failed:
            return
        try:
            fields = self.scrub(fields)
            fields.update({k: self._payload(v) for k, v in (payloads or {}).items()})
            t = threading.current_thread()
            with self._lock:
                self._seq += 1
                record = {"v": VERSION, "writer": self.id, "seq": self._seq, "ts": time.time(),
                          "mono": time.monotonic(), "thread": f"{t.name}:{t.native_id}", "task": _task(),
                          "component": component, "kind": kind, "span": span, **fields}
                self._append((json.dumps(record) + "\n").encode())
        except Exception as exc:                 # noqa: BLE001 -- any failure to record stops the run
            self._fail(exc)

    def _append(self, line: bytes) -> None:
        if self._out is None or (self._size and self._size + len(line) > SEGMENT_BYTES):
            if self._out is not None:
                self._out.close()
            self._segment += 1
            path = self._records / f"{self.id}-{self._segment:04d}.jsonl"
            self._out, self._size = path.open("ab", buffering=0), 0
        if self._out.write(line) != len(line):
            raise OSError(f"short write to {self._out.name}")
        self._size += len(line)

    def _payload(self, obj: Any) -> str:
        blob = json.dumps(self.scrub(obj), sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()
        target = self._payloads / f"{digest}.json.zst"
        if not target.exists():
            tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(zstandard.ZstdCompressor(level=10).compress(blob))
            os.replace(tmp, target)
        return digest

    def store_file(self, source: BinaryIO | bytes) -> tuple[str, int] | None:
        """The bytes of `source` under telemetry/files/<sha256>, unredacted; a clip-store blob with
        the same digest is hard-linked instead."""
        if self.failed:
            return None
        source = io.BytesIO(source) if isinstance(source, bytes) else source
        tmp, digest, size = self._files / f"{uuid.uuid4().hex}.tmp", hashlib.sha256(), 0
        try:
            with tmp.open("wb") as out:
                while chunk := source.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            target = self._files / digest.hexdigest()
            if not target.exists():
                blob = next(self.run_dir.glob(f"store/blobs/*/{target.name}.*"), None)
                if blob is None:
                    os.replace(tmp, target)
                else:
                    try:
                        os.link(blob, target)
                    except FileExistsError:      # stored meanwhile by another thread
                        pass
            return target.name, size
        except Exception as exc:                 # noqa: BLE001 -- any failure to record stops the run
            self._fail(exc)
            return None
        finally:
            tmp.unlink(missing_ok=True)

    def _fail(self, exc: BaseException) -> None:
        """Print to the original stderr once, go quiet, and stop the run as `ar stop --force` does."""
        with self._fail_lock:
            if self.failed:
                return
            self.failed = True
        print(f"telemetry: cannot write to {self.run_dir / 'telemetry'}: {type(exc).__name__}: {exc}; "
              f"stopping the run", file=self._stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
