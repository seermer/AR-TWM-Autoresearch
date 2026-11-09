"""A run's records as invocation trees, and the telemetry checks (`ar trace --check`)."""
from __future__ import annotations

import functools
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import zstandard

from ..subproc import proc_running
from . import registry


@dataclass
class Node:
    start: dict
    end: dict | None = None
    ends: int = 0
    children: list = field(default_factory=list)          # Nodes and other records, in seq order

    @property
    def id(self) -> str:
        return self.start["span"]


@dataclass
class Trace:
    """A run's records, read by `refresh` as they are appended."""
    run_dir: Path
    writers: list[dict] = field(default_factory=list)              # writer records, in time order
    roots: dict[str, list] = field(default_factory=dict)           # writer id -> its top-level Nodes and records
    spans: dict[str, Node] = field(default_factory=dict)
    unreadable: list[tuple[str, int]] = field(default_factory=list)   # (file, line)
    last: dict[str, dict] = field(default_factory=dict)            # writer id -> its last record
    by_writer: dict[str, list[dict]] = field(default_factory=dict)
    read: dict[str, tuple[int, int]] = field(default_factory=dict)  # file -> (bytes, lines) read
    _records: list[dict] | None = None

    @property
    def records(self) -> list[dict]:
        """In writer order, then seq."""
        if self._records is None:
            self._records = [r for w in self.writers for r in self.by_writer[w["writer"]]]
        return self._records

    def payload(self, digest: str | None):
        if digest is None:
            return None
        path = self.run_dir / "telemetry" / "payloads" / f"{digest}.json.zst"
        return json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))

    def refresh(self, whole: bool = False) -> bool:
        """Read the lines appended since the last call; `whole` also reads a last line without its newline (a torn
        write; otherwise it waits for the rest). Returns whether there was anything new."""
        new: dict[str, list[dict]] = {}
        for path in sorted((self.run_dir / "telemetry" / "records").glob("*.jsonl")):
            done, lines = self.read.get(path.name, (0, 0))
            with path.open("rb") as f:
                f.seek(done)
                chunk = f.read()
            size = len(chunk) if whole else chunk.rfind(b"\n") + 1
            if size <= 0:
                continue
            for i, line in enumerate(chunk[:size].decode(errors="replace").splitlines(), lines + 1):
                try:
                    record = json.loads(line)
                    record["seq"], record["kind"]
                    new.setdefault(record["writer"], []).append(record)
                except (json.JSONDecodeError, KeyError, TypeError):
                    self.unreadable.append((str(path.relative_to(self.run_dir)), i))
                lines = i
            self.read[path.name] = (done + size, lines)
        for w, records in new.items():
            records.sort(key=lambda r: r["seq"])
            self.by_writer.setdefault(w, []).extend(records)
            self.last[w] = self.by_writer[w][-1]
        firsts = {w: next((r for r in rs if r["kind"] == "writer"), rs[0]) for w, rs in self.by_writer.items()}
        self.writers = [firsts[w] for w in sorted(firsts, key=lambda w: functools.cmp_to_key(_before)(firsts[w]))]
        for w in [x["writer"] for x in self.writers if x["writer"] in new]:
            self._link(self.roots.setdefault(w, []), new[w])
        if new:
            self._records = None
        return bool(new)

    def _link(self, top: list, records: list[dict]) -> None:
        for r in records:
            if r["kind"] == "writer":
                continue
            if r["kind"] == "start":
                node = self.spans[r["span"]] = Node(r)
                parent = self.spans.get(r.get("parent"))
                (parent.children if parent else top).append(node)
            elif r["kind"] == "end":
                if (node := self.spans.get(r["span"])) is not None:
                    node.ends += 1
                    node.end = node.end or r
            else:
                parent = self.spans.get(r.get("span"))
                (parent.children if parent else top).append(r)


def _before(a: dict, b: dict) -> int:
    """Writer order: the monotonic clock on the same boot, else wall time."""
    if a.get("boot_id") and a.get("boot_id") == b.get("boot_id"):
        return -1 if a["mono"] < b["mono"] else 1
    return -1 if a["ts"] < b["ts"] else 1


def load(run_dir: Path) -> Trace:
    trace = Trace(Path(run_dir))
    trace.refresh(whole=True)
    return trace


def check(trace: Trace) -> list[str]:
    """Telemetry bugs: blank ends, records without a parent, unregistered names, double ends, torn lines."""
    out = [f"unreadable line: {f}:{n}" for f, n in trace.unreadable]
    tables = {"start": registry.SPANS, "end": registry.SPANS, "decision": registry.DECISIONS,
              "alert": registry.ALERTS}
    for r in trace.records:
        where = f"{r['writer']}#{r['seq']} {r['kind']} {r.get('name') or ''}".rstrip()
        if r["kind"] in tables and registry.entry(tables[r["kind"]], r.get("name")) is None:
            out.append(f"unregistered name: {where}")
        if r["kind"] == "start":
            if r.get("parent") is None and r.get("name") not in registry.ROOTS:
                out.append(f"no parent: {where}")
            elif r.get("parent") is not None and r["parent"] not in trace.spans:
                out.append(f"parent never started: {where}")
        elif r["kind"] == "end":
            if r["span"] not in trace.spans:
                out.append(f"end without a start: {where}")
            if r.get("output") is None and r.get("error") is None:
                out.append(f"end with neither output nor error: {where}")
        elif r["kind"] != "writer" and (r.get("span") is None or r["span"] not in trace.spans):
            out.append(f"no parent: {where}")
    out += [f"{n.ends} ends: {n.start['writer']}#{n.start['seq']} {n.start['name']}"
            for n in trace.spans.values() if n.ends > 1]
    return out


def writer_alive(trace: Trace, writer: str) -> bool | None:
    """Whether the process that wrote `writer`'s records still runs; None on another host or boot."""
    w = next(x for x in trace.writers if x["writer"] == writer)
    try:
        here = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        here = None
    if w.get("host") != socket.gethostname() or w.get("boot_id") != here:
        return None
    return proc_running(w["pid"], w.get("process_start"))


def open_span_facts(trace: Trace, node: Node) -> str:
    """What is known of a span with no end: never a cause."""
    w = next(x for x in trace.writers if x["writer"] == node.start["writer"])
    alive = {True: "alive", False: "not running", None: "not checkable here"}[writer_alive(trace, w["writer"])]
    last = trace.last[w["writer"]]
    return f"no end recorded; the writer's last record at {_time(last['ts'])}, host {w.get('host')}, process {alive}"


def _time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
