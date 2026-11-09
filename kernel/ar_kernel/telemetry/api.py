"""Spans and records under the current span. With no writer every call records nothing."""
from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

from . import registry
from .errors import describe
from .writer import writer

_current: ContextVar[Span | None] = ContextVar("telemetry_span", default=None)
_root: Span | None = None                        # the invocation's root span: parent of records with no current span


class Span:
    def __init__(self, name: str, component: str) -> None:
        self.id, self.name, self.component = uuid.uuid4().hex, name, component
        self.output: Any = None
        self.status: str | None = None

    def end(self, status: str, output: Any = None, error: BaseException | None = None) -> None:
        if (w := writer()) is not None:
            payloads = {"output": output} if output is not None else {}
            if error is not None:
                payloads["error"] = describe(error)
            w.write("end", self.component, self.id, payloads, name=self.name, status=status)

    def fail(self, exc: BaseException) -> None:
        """End by `exc`: cancelled for a stop, otherwise error, unless the body set a status."""
        self.end(self.status or _status(exc), self.output, exc)


def current() -> Span | None:
    return _current.get()


def _parent() -> Span | None:
    return _current.get() or _root


def start(name: str, inputs: Any = None, parent: Span | None = None) -> Span:
    """A span that ends by an explicit `end`, in this thread or another."""
    global _root
    component = registry.component(registry.SPANS, name)
    parent = None if name in registry.ROOTS else parent or _parent()
    sp = Span(name, component or (parent.component if parent else "cli"))
    if name in registry.ROOTS:
        _root = sp
    if (w := writer()) is not None:
        w.write("start", sp.component, sp.id, {"inputs": inputs} if inputs is not None else None,
                name=name, parent=parent.id if parent else None)
    return sp


def _status(exc: BaseException) -> str:
    from ..control import ForceStop
    return "cancelled" if isinstance(exc, (ForceStop, KeyboardInterrupt, asyncio.CancelledError)) else "error"


@contextmanager
def span(name: str, inputs: Any = None, parent: Span | None = None):
    """The body sets `sp.output` and may set `sp.status`; an exception ends the span and is re-raised."""
    sp = start(name, inputs, parent)
    token = _current.set(sp)
    try:
        yield sp
    except BaseException as exc:
        sp.fail(exc)
        raise
    else:
        sp.end(sp.status or "ok", sp.output)
    finally:
        _current.reset(token)


@contextmanager
def use(sp: Span | None):
    """Make an existing span current, ending nothing."""
    token = _current.set(sp)
    try:
        yield sp
    finally:
        _current.reset(token)


def thread(target: Callable, name: str, *, args: tuple = (), daemon: bool = True) -> threading.Thread:
    """An unstarted thread that runs `target` under the creator's current span."""
    parent = _current.get()

    def run() -> None:
        with use(parent):
            target(*args)
    return threading.Thread(target=run, name=name, daemon=daemon)


def submit(pool, fn: Callable, *args):
    """`pool.submit(fn, *args)`, run under the submitter's current span."""
    parent = _current.get()

    def run():
        with use(parent):
            return fn(*args)
    return pool.submit(run)


def _record(kind: str, component: str | None, payloads: dict | None, **fields: Any) -> None:
    if (w := writer()) is not None:
        sp = _parent()
        w.write(kind, component or (sp.component if sp else "cli"), sp.id if sp else None, payloads, **fields)


def decision(name: str, inputs: Any, options: Any, chosen: Any, reason: str) -> None:
    _record("decision", registry.component(registry.DECISIONS, name), {"inputs": inputs},
            name=name, options=options, chosen=chosen, reason=reason)


def deferred_decision(name: str, inputs: Any, options: Any, chosen: Any, reason: str) -> None:
    """A decision under the root, written before the next record: for signal handlers, which must not write."""
    if (w := writer()) is not None:
        w.pending.append(("decision", registry.component(registry.DECISIONS, name), _root.id if _root else None,
                          {"inputs": inputs}, {"name": name, "options": options, "chosen": chosen, "reason": reason}))


def error(exc: BaseException, handled_as: str) -> None:
    """An exception handled without ending a span with it, and what it became."""
    _record("error", None, {"error": describe(exc)}, handled_as=handled_as)


def log(stream: str, text: str, level: str | None = None, logger: str | None = None) -> None:
    _record("log", None, None, stream=stream, level=level, logger=logger, text=text)


def alert(name: str, level: str, message: str, details: Any = None) -> None:
    _record("alert", registry.component(registry.ALERTS, name), None,
            name=name, level=level, message=message, details=details)


def keep(path: str, source, how: str) -> None:
    """The bytes of `source` (bytes or a binary file) in the file store: a file a kernel tool read from an agent
    (`how` "read") or handed to it ("handed"), `path` as the agent names it."""
    if (w := writer()) is None:
        return
    with span("file.kept", {"path": path, "how": how}) as sp:
        stored = w.store_file(source)
        sp.output = {"sha256": stored[0], "size": stored[1]} if stored else {}


def sample(**readings: Any) -> None:
    _record("sample", None, None, **readings)
