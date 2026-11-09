"""`ar trace`: a run's invocation trees, printed in order and indented."""
from __future__ import annotations

import json

from .tree import Node, Trace, _time, open_span_facts


def _short(value, full: bool) -> str:
    text = json.dumps(value, indent=2 if full else None, ensure_ascii=False, default=str)
    return text if full or len(text) <= 160 else text[:157] + "..."


def _selected(trace: Trace, node: str | None, attempt: int | None) -> list[Node] | None:
    """The spans to print from: a node's span, or its phase attempts numbered `attempt`."""
    if node is None:
        return None
    picked = [n for n in trace.spans.values()
              if n.start["name"] == "node" and (trace.payload(n.start.get("inputs")) or {}).get("node") == node]
    if attempt is not None:
        picked = [c for n in picked for c in n.children if isinstance(c, Node) and c.start["name"] == "phase.attempt"
                  and (trace.payload(c.start.get("inputs")) or {}).get("attempt") == attempt]
    return picked


def render(trace: Trace, *, node: str | None = None, attempt: int | None = None, since: int | None = None,
           depth: int | None = None, full: bool = False) -> str:
    lines: list[tuple[int, str]] = []                      # (seq, text)

    def payload_line(pad: str, label: str, digest) -> None:
        if digest is not None:
            text = _short(trace.payload(digest), full)
            lines.append((-1, f"{pad}  {label}: " + text.replace("\n", "\n" + pad + "    ")))

    def span(n: Node, level: int, parallel: bool) -> None:
        pad = "  " * level
        s = n.start
        mark = "∥ " if parallel else ""
        if n.end is None:
            head = f"{pad}{mark}{s['name']}  [{s['thread']}]  {open_span_facts(trace, n)}"
        else:
            head = f"{pad}{mark}{s['name']} {n.end['status']} {n.end['mono'] - s['mono']:.1f}s  [{s['thread']}]"
        lines.append((s["seq"], f"{s['seq']:>7} {head}"))
        payload_line(" " * 8 + pad, "inputs", s.get("inputs"))
        if depth is None or level < depth:
            items(n.children, level + 1)
        if n.end is not None:
            payload_line(" " * 8 + pad, "output", n.end.get("output"))
            if (err := trace.payload(n.end.get("error"))) is not None:
                text = _short(err, True) if full else f"{err['type']}: {err['message']}"
                lines.append((-1, f"{' ' * 8}{pad}  error: " + text.replace("\n", "\n" + " " * 8 + pad + "    ")))

    def record(r: dict, level: int) -> None:
        pad = "  " * level
        k = r["kind"]
        if k == "log":
            text = f"log {r['stream']}{' ' + r['level'] if r.get('level') else ''}: {r['text']}"
        elif k == "decision":
            text = f"decision {r['name']}: {r['chosen']} ({r['reason']})"
        elif k == "error":
            err = trace.payload(r["error"])
            text = f"error, {r['handled_as']}: {err['type']}: {err['message']}"
        elif k == "alert":
            text = f"alert {r['level']} {r['name']}: {r['message']}"
        else:
            text = f"{k}: " + _short({x: v for x, v in r.items() if x not in _COMMON}, full)
        lines.append((r["seq"], f"{r['seq']:>7} {pad}· {text}"))
        if k == "decision":
            payload_line(" " * 8 + pad, "inputs", r.get("inputs"))

    def items(children: list, level: int) -> None:
        prev_end = 0                                       # where the previous sibling span ended; None: never
        for c in children:
            if isinstance(c, Node):
                span(c, level, prev_end is None or prev_end > c.start["seq"])
                prev_end = c.end["seq"] if c.end else None
            else:
                record(c, level)

    picked = _selected(trace, node, attempt)
    for w in trace.writers:
        lines.append((0, f"writer {w['writer']}: pid {w.get('pid')} on {w.get('host')} at {_time(w['ts'])}, "
                         f"argv {' '.join(w.get('argv') or [])}"))
        if picked is None:
            items(trace.roots.get(w["writer"], []), 1)
    for n in picked or []:
        span(n, 1, False)
    out, show = [], True
    for seq, text in lines:                                # seq 0: a writer line; -1: a payload of the line above
        show = seq == 0 or (show if seq == -1 else since is None or seq >= since)
        if show:
            out.append(text)
    return "\n".join(out)


_COMMON = {"v", "writer", "seq", "ts", "mono", "thread", "task", "component", "kind", "span"}
