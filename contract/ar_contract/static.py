"""The checks of the agent's code that need no container."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .metrics import METRICS

ENTRY_POINTS = ("edit_self", "improve_recipe")
_MENTION = re.compile(r"\b(node\d{4}|" + "|".join(METRICS) + r")\b")


def static_check(entry_source: str) -> str | None:
    """The problem with agent/entry.py's entry points, or None."""
    try:
        tree = ast.parse(entry_source)
    except SyntaxError as exc:
        return f"syntax error in agent/entry.py: {exc}"
    except (ValueError, RecursionError, MemoryError) as exc:   # e.g. deeply nested agent source
        return f"cannot parse agent/entry.py: {type(exc).__name__}: {exc}"
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ENTRY_POINTS:
        fn = top.get(name)
        if fn is None:
            return f"agent/entry.py defines no top-level {name}"
        a = fn.args
        if len(a.posonlyargs + a.args) != 1 or a.vararg or a.kwarg or a.kwonlyargs:
            return f"{name} must take exactly one positional parameter (ctx)"
    return None


def mentions(root: str | Path, files: list[str] | None = None) -> list[str]:
    """Every line of a text file under `root` (or of `files`, relative to it) that mentions a node or a metric."""
    root, hits = Path(root), []
    if files is None:
        files = sorted(str((Path(top) / name).relative_to(root)) for top, _, names in os.walk(root) for name in names)
    for rel in files:
        path = root / rel
        if path.is_symlink() or not path.is_file():         # regular files only: reading a FIFO blocks
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:                  # not a text file
            continue
        except OSError as exc:
            hits.append(f"{rel}: cannot read: {exc}")
            continue
        for i, line in enumerate(text.splitlines(), 1):
            hits += [f"{rel}:{i} mentions {m}" for m in sorted(set(_MENTION.findall(line)))]
    return hits
