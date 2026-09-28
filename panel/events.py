"""The run's event logs (telemetry/events/*.jsonl), read incrementally, and their payloads."""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict

import zstandard

from .runfiles import RunFiles, fmt_ts

SUMMARY_KEYS = ("tool", "model", "status", "kind", "level", "message", "child", "returncode",
                "exit_code", "turn_index", "duration_s", "cost_usd")


class EventLog:
    """Every event of the run. Gradio calls this from several threads, so all shared state
    (offsets, events, the payload cache) is behind one lock."""

    def __init__(self, files: RunFiles, cache_size: int = 256) -> None:
        self.files = files
        self.dir = files.root / "telemetry" / "events"
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}
        self._marks: dict[str, tuple[int, bytes]] = {}   # per file: inode, last line read (with its newline)
        self._events: dict[str, list[dict]] = {}
        self._by_seq: dict[int, dict] = {}
        self._seq = 0
        self._payloads: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size

    def refresh(self) -> None:
        """Read what was appended since the last call. The folder is rescanned each time (a
        node's file appears when the node starts); a line without its newline yet is left
        for the next call. A file that was deleted, replaced or rewritten since (a run reset to
        its root rewrites run.jsonl and removes node files) is dropped and read again from the
        start: detected by its inode, or by the last line read no longer sitting before the offset."""
        with self._lock:
            if not self.dir.is_dir():
                return
            paths = {p.stem: p for p in sorted(self.dir.glob("*.jsonl"))}
            for name in [n for n in self._offsets if n not in paths]:
                self._forget(name)
            for name, path in paths.items():
                offset = self._offsets.get(name, 0)
                with path.open("rb") as handle:
                    inode = os.fstat(handle.fileno()).st_ino
                    if offset and not self._still_there(handle, name, inode, offset):
                        self._forget(name)
                        offset = 0
                    handle.seek(offset)
                    chunk = handle.read()
                end = chunk.rfind(b"\n")
                if end < 0:
                    continue
                lines = chunk[:end].splitlines()
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:                 # a line torn by kill -9 or a full disk
                        continue
                    event["_file"], event["_seq"] = name, self._seq
                    self._by_seq[self._seq] = event
                    self._seq += 1
                    self._events.setdefault(name, []).append(event)
                self._offsets[name] = offset + end + 1
                self._marks[name] = (inode, lines[-1] + b"\n")

    def _still_there(self, handle, name: str, inode: int, offset: int) -> bool:
        mark_inode, last = self._marks.get(name, (None, b""))
        if inode != mark_inode or offset < len(last):
            return False
        handle.seek(offset - len(last))
        return handle.read(len(last)) == last

    def _forget(self, name: str) -> None:
        for event in self._events.pop(name, []):
            self._by_seq.pop(event["_seq"], None)
        self._offsets.pop(name, None)
        self._marks.pop(name, None)

    def events(self, files: list[str] | None = None, include_gpu: bool = False) -> list[dict]:
        self.refresh()
        with self._lock:
            names = files if files is not None else [n for n in self._events if include_gpu or n != "gpu"]
            out = [e for n in names for e in self._events.get(n, [])]
        return sorted(out, key=lambda e: (e.get("ts_wall", 0.0), e["_seq"]))

    def by_seq(self, seq: int) -> dict | None:
        with self._lock:
            return self._by_seq.get(seq)

    def payload(self, digest: str | None) -> dict | None:
        if not digest:
            return None
        with self._lock:
            if digest in self._payloads:
                self._payloads.move_to_end(digest)
                return self._payloads[digest]
        folder = self.files.root / "telemetry" / "payloads"
        try:
            compressed = folder / f"{digest}.json.zst"
            if compressed.exists():
                data = json.loads(zstandard.ZstdDecompressor().decompress(compressed.read_bytes()))
            else:
                data = json.loads((folder / f"{digest}.json").read_text())      # pre-zstd runs
        except (OSError, ValueError, zstandard.ZstdError):
            return None
        with self._lock:
            self._payloads[digest] = data
            while len(self._payloads) > self._cache_size:
                self._payloads.popitem(last=False)
        return data


def filter_events(events: list[dict], *, node: str | None = None, phase: str | None = None,
                  attempt: int | None = None, types: list[str] | None = None,
                  component: str | None = None, text: str | None = None) -> list[dict]:
    """Free text matches the event record itself, not its payload."""
    needle = (text or "").strip().lower()
    out = []
    for e in events:
        if (node and e.get("node") != node) or (phase and e.get("phase") != phase) \
                or (attempt is not None and e.get("attempt") != attempt) \
                or (types and e.get("type") not in types) or (component and e.get("component") != component):
            continue
        if needle and needle not in json.dumps(e, default=str).lower():
            continue
        out.append(e)
    return out


def event_row(e: dict) -> dict:
    summary = " ".join(f"{k}={e[k]}" for k in SUMMARY_KEYS if e.get(k) is not None)
    return {"seq": e["_seq"], "time": fmt_ts(e.get("ts_wall", 0.0)), "node": e.get("node"),
            "phase": e.get("phase"), "attempt": e.get("attempt"), "type": e.get("type"),
            "component": e.get("component"), "summary": summary[:200]}
