"""Every file a kernel tool reads from or writes into a caller's workspace, stored once by content under
telemetry/<folder>/<sha256>, with a `file` event."""
from __future__ import annotations

import contextvars
import hashlib
import io
import os
import uuid
from contextlib import contextmanager
from typing import BinaryIO

_SCOPE = contextvars.ContextVar("file_scope", default=None)     # (recorder, event fields)


@contextmanager
def scope(recorder, **fields):
    """Files recorded inside belong to one tool call or job: `fields` (node, phase, attempt, tool, and
    parent_span_id or job_id) go on their events."""
    token = _SCOPE.set((recorder, fields))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def record(source: BinaryIO | bytes, *, direction: str, path: str, folder: str = "files") -> str:
    """Store the bytes of `source` and emit a `file` event; returns the digest. A digest the clip store
    holds becomes a hard link to that blob, which the kernel never changes."""
    if (scope := _SCOPE.get()) is None:
        raise RuntimeError(f"{path}: a file recorded outside a tool call or job")
    recorder, fields = scope
    root = recorder.run_dir / "telemetry" / folder
    root.mkdir(parents=True, exist_ok=True)
    source = io.BytesIO(source) if isinstance(source, bytes) else source
    tmp, digest, size = root / f"{uuid.uuid4().hex}.tmp", hashlib.sha256(), 0
    try:
        with tmp.open("wb") as out:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        target = root / digest.hexdigest()
        if not target.exists():
            blob = next(recorder.run_dir.glob(f"store/blobs/*/{target.name}.*"), None)
            if blob is None:
                os.replace(tmp, target)
            else:
                try:
                    os.link(blob, target)
                except FileExistsError:                 # recorded meanwhile by another thread
                    pass
    finally:
        tmp.unlink(missing_ok=True)
    recorder.event("file", direction=direction, path=path, digest=target.name, bytes=size, **fields)
    return target.name
