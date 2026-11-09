"""The run's telemetry records (telemetry.tree), read as they are appended, each placed in its phase attempt."""
from __future__ import annotations

import itertools
import json
import threading
from collections import OrderedDict

import zstandard

from ar_kernel.telemetry import tree

from .chat import Call
from .runfiles import RunFiles, fmt_ts

AGENT_PHASES = ("edit_self", "improve_recipe")


class Records:
    """The FastAPI thread pool calls this from several threads: refresh and the caches are behind one lock."""

    def __init__(self, files: RunFiles, cache_size: int = 256) -> None:
        self.files = files
        self.trace = tree.Trace(files.root)
        self._lock = threading.RLock()
        self._payloads: OrderedDict[str, object] = OrderedDict()
        self._cache_size = cache_size
        self._where: dict[str, tuple] = {}
        self._calls: dict[str, Call] = {}            # llm.request span -> its ended call
        self._named: dict[str, list[tree.Node]] = {}  # span name -> its spans, in start order
        self._indexed = 0                            # trace.spans (append-only) indexed so far

    def refresh(self) -> None:
        with self._lock:
            self.trace.refresh()

    def all(self) -> list[dict]:
        """Every record but the writers', in writer order, then seq."""
        self.refresh()
        with self._lock:
            return [r for r in self.trace.records if r["kind"] != "writer"]

    def spans(self) -> list[tree.Node]:
        """Every span, in the order its start was read."""
        self.refresh()
        with self._lock:
            return list(self.trace.spans.values())

    def named(self, name: str) -> list[tree.Node]:
        """Every span called `name`, in the order its start was read."""
        self.refresh()
        with self._lock:
            spans = self.trace.spans
            for n in itertools.islice(spans.values(), self._indexed, None):
                self._named.setdefault(n.start["name"], []).append(n)
            self._indexed = len(spans)
            return list(self._named.get(name, ()))

    def span(self, span_id: str | None) -> tree.Node | None:
        return self.trace.spans.get(span_id)

    def payload(self, digest: str | None):
        if not digest:
            return None
        with self._lock:
            if digest in self._payloads:
                self._payloads.move_to_end(digest)
                return self._payloads[digest]
        path = self.files.root / "telemetry" / "payloads" / f"{digest}.json.zst"
        try:
            data = json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))
        except (OSError, ValueError, zstandard.ZstdError):
            return None
        with self._lock:
            self._payloads[digest] = data
            while len(self._payloads) > self._cache_size:
                self._payloads.popitem(last=False)
        return data

    def kept(self, sha256: str | None):
        """A JSON file the file store kept, or None."""
        if not sha256:
            return None
        try:
            return json.loads((self.files.root / "telemetry" / "files" / sha256).read_text())
        except (OSError, ValueError):
            return None

    def where(self, span_id: str | None) -> tuple[str | None, str | None, int | None]:
        """(node, phase, attempt) of the phase attempt holding `span_id`; a node span gives only the node."""
        with self._lock:
            chain, node = [], self.trace.spans.get(span_id)
            found = (None, None, None)
            while node is not None:
                if node.id in self._where:
                    found = self._where[node.id]
                    break
                chain.append(node.id)
                name = node.start["name"]
                if name in ("phase.attempt", "node"):
                    inputs = self.payload(node.start.get("inputs")) or {}
                    found = (inputs.get("node"), inputs.get("phase") if name == "phase.attempt" else None,
                             inputs.get("attempt") if name == "phase.attempt" else None)
                    break
                node = self.trace.spans.get(node.start.get("parent"))
            for sid in chain:
                self._where[sid] = found
            return found

    def place(self, r: dict) -> tuple[str | None, str | None, int | None]:
        return self.where(r["span"])

    def attempts(self, node_id: str) -> list[tuple[str, int, tree.Node]]:
        """(phase, attempt, span) of each agent phase attempt of the node, oldest first."""
        out = []
        for n in self.named("phase.attempt"):
            node, phase, attempt = self.where(n.id)
            if node == node_id and phase in AGENT_PHASES:
                out.append((phase, attempt, n))
        return out

    def call(self, n: tree.Node) -> Call | None:
        """An LLM request span as a call: None when it never reached the model (refused before linking)."""
        with self._lock:
            if n.id in self._calls:
                return self._calls[n.id]
        link = next((c for c in n.children if isinstance(c, dict) and c["kind"] == "decision"
                     and c["name"] == "llm.link"), None)
        if link is None:
            return None
        meta = self.payload(link.get("inputs")) or {}
        censor = next((c for c in n.children if isinstance(c, tree.Node) and c.start["name"] == "llm.censor"), None)
        call = Call(meta.get("call_id") or n.id, meta.get("conversation_id") or n.id, meta.get("turn_index") or 0,
                    n.start["ts"], censor.end.get("output") if censor and censor.end else None,
                    role=meta.get("role"), tool=meta.get("tool"), span=n.id)
        if n.end is not None:
            out = self.payload(n.end.get("output")) or {}
            body = out.get("body") if isinstance(out.get("body"), dict) else {}
            call.response, call.status, call.usage = n.end.get("output"), out.get("status"), body.get("usage") or {}
            call.cost_usd, call.done_ts = out.get("cost_usd"), n.end["ts"]
            call.latency_s = n.end["ts"] - n.start["ts"]
            with self._lock:
                self._calls[n.id] = call
        return call

    def calls(self, node_id: str | None = None, phase: str | None = None, attempt: int | None = None) -> list[Call]:
        """Every call that reached the model, in time order; with `node_id`, only that phase attempt's agent calls
        (a kernel tool's own call, ask, is no conversation)."""
        out = []
        for n in self.named("llm.request"):
            if node_id is not None and self.where(n.id) != (node_id, phase, attempt):
                continue
            c = self.call(n)
            if c is not None and (node_id is None or not c.tool):
                out.append(c)
        return sorted(out, key=lambda c: c.ts)

    def running(self, n: tree.Node) -> bool:
        """The span has no end and the process that started it still runs."""
        if n.end is not None:
            return False
        with self._lock:
            return tree.writer_alive(self.trace, n.start["writer"]) is True

    def open_facts(self, n: tree.Node) -> str:
        with self._lock:
            return tree.open_span_facts(self.trace, n)

    def subtree(self, span_id: str) -> set[str]:
        """The span and every span under it."""
        out, todo = set(), [span_id]
        while todo:
            sid = todo.pop()
            out.add(sid)
            n = self.trace.spans.get(sid)
            if n is not None:
                todo += [c.id for c in n.children if isinstance(c, tree.Node)]
        return out


def name_of(r: dict) -> str:
    """A record's kind and name, e.g. "start tool.data_ingest", "alert monitor.stall", "log"."""
    return f"{r['kind']} {r['name']}" if r.get("name") else r["kind"]


SUMMARY_KEYS = ("status", "chosen", "reason", "level", "message", "handled_as", "stream", "text")


def record_row(records: Records, r: dict, n: int) -> dict:
    node, phase, attempt = records.place(r)
    summary = " ".join(f"{k}={r[k]}" for k in SUMMARY_KEYS if r.get(k) is not None)
    return {"seq": n, "time": fmt_ts(r.get("ts", 0.0)), "ts": r.get("ts"), "node": node, "phase": phase,
            "attempt": attempt, "type": name_of(r), "component": r.get("component"), "span": r.get("span"),
            "summary": summary[:200]}
