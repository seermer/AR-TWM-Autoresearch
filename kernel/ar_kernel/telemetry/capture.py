"""Catch-alls: uncaught and unraisable exceptions, library logging, warnings and printed lines become records."""
from __future__ import annotations

import logging
import sys
import threading
import warnings

from .api import error, log

_busy = threading.local()                        # a record being written from inside a capture: no second one


def _guarded(fn, *args, **kwargs) -> None:
    if getattr(_busy, "on", False):
        return
    _busy.on = True
    try:
        fn(*args, **kwargs)
    finally:
        _busy.on = False


class _Lines:
    """A text stream that passes everything on and records each complete line, per thread."""

    def __init__(self, raw, stream: str) -> None:
        self.raw, self.stream, self._partial = raw, stream, threading.local()

    def write(self, text: str) -> int:
        n = self.raw.write(text)
        lines = (getattr(self._partial, "text", "") + text).split("\n")
        self._partial.text = lines.pop()
        for line in lines:
            _guarded(log, self.stream, line)
        return n

    def flush(self) -> None:
        self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)


class _Handler(logging.Handler):
    """Every log record; WARNING and above also go to the terminal when no other handler would print them."""

    def __init__(self, terminal, alone: bool) -> None:
        super().__init__(logging.NOTSET)
        self.terminal, self.alone = terminal, alone

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        _guarded(log, "logging", text, record.levelname, record.name)
        if self.alone and record.levelno >= logging.WARNING:
            self.terminal.write(text + "\n")


def install() -> None:
    """Called once, right after the writer opens; each hook records, then does what it did before."""
    terminal = sys.stderr
    root = logging.getLogger()
    root.addHandler(_Handler(terminal, alone=not root.handlers))
    root.setLevel(logging.NOTSET)

    previous = sys.excepthook, threading.excepthook, sys.unraisablehook

    def excepthook(kind, value, tb):
        _guarded(error, value, "uncaught: the main thread ends")
        previous[0](kind, value, tb)

    def thread_hook(args):
        if args.exc_value is not None:
            _guarded(error, args.exc_value, f"uncaught: thread {args.thread.name if args.thread else '?'} ends")
        previous[1](args)

    def unraisable(args):
        if args.exc_value is not None:
            _guarded(error, args.exc_value, f"unraisable: {args.err_msg or 'ignored'} {args.object!r}")
        previous[2](args)

    sys.excepthook, threading.excepthook, sys.unraisablehook = excepthook, thread_hook, unraisable

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = warnings.formatwarning(message, category, filename, lineno, line)
        _guarded(log, "warnings", text.rstrip("\n"), "WARNING", category.__name__)
        terminal.write(text)
    warnings.showwarning = showwarning
    sys.stdout, sys.stderr = _Lines(sys.stdout, "stdout"), _Lines(terminal, "stderr")
