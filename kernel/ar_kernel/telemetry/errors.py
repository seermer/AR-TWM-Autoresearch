"""An exception as a record: type, message, OSError details, frames with their locals, cause and context."""
from __future__ import annotations

import traceback


def describe(exc: BaseException, _seen: set | None = None) -> dict:
    seen = _seen if _seen is not None else set()
    seen.add(id(exc))
    frames = traceback.StackSummary.extract(traceback.walk_tb(exc.__traceback__), capture_locals=True)
    out = {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": str(exc),
           "frames": [{"file": f.filename, "line": f.lineno, "function": f.name, "code": f.line,
                       "locals": f.locals} for f in frames]}
    if isinstance(exc, OSError):
        out.update(errno=exc.errno, filename=exc.filename)
    for key, linked in (("cause", exc.__cause__), ("context", exc.__context__)):
        if linked is not None and id(linked) not in seen:
            out[key] = describe(linked, seen)
    return out
