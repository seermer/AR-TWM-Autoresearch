"""A scrubbed snapshot of one run: the only thing the simple panel serves."""
from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

from ar_kernel.budget import usage_tokens
from ar_kernel.telemetry import env_secrets
from panel import media, views
from panel.runfiles import fmt_ts

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
MAX_VIDEOS = 10
TOOL_TEXT = 2000
TOKENS = ("cached", "uncached", "output")

PATH = re.compile(r"(?<![\w.:/<>-])/(?!workspace\b|agent\b|dev/null\b)[\w.\-@+]+(?:/[\w.\-@+]*)+")
SECRETS = [re.compile(p) for p in (r"\bsk-[\w-]{16,}", r"\bhf_[A-Za-z0-9]{20,}", r"\bgh[pousr]_[A-Za-z0-9]{20,}",
                                   r"\bAKIA[0-9A-Z]{16}\b", r"(?i)\bbearer\s+[\w.~+/=-]{8,}")]
BLOB = re.compile(r"[A-Za-z0-9+_-]{40,}={0,2}")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


class Scrubber:
    def __init__(self) -> None:
        self.secrets = env_secrets([REPO / ".env"], os.environ)
        host = socket.gethostname()
        self.words = [(re.compile(rf"\b{re.escape(w)}\b", re.I), tag)
                      for w, tag in ((getpass.getuser(), "<user>"), (host, "<host>"), (host.split(".")[0], "<host>"))
                      if len(w) >= 3]

    def __call__(self, text):
        if not isinstance(text, str):
            return text
        for s in self.secrets:
            text = text.replace(s, "<secret>")
        for p in SECRETS:
            text = p.sub("<secret>", text)
        text = BLOB.sub(lambda m: "<secret>" if _mixed(m.group()) else m.group(), text)
        text = text.replace(str(PROJECT), "<project>")
        text = PATH.sub("<path>", text)
        text = EMAIL.sub("<email>", text)
        text = IPV4.sub("<ip>", text)
        for p, tag in self.words:
            text = p.sub(tag, text)
        return text


def _mixed(s: str) -> bool:
    """A key or token, not a word, a hex digest or a path: upper, lower and digits all present."""
    return any(c.isupper() for c in s) and any(c.islower() for c in s) and any(c.isdigit() for c in s)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} characters not shown]"


def _copy(run: views.Run, rel: str, root: Path, dest: str) -> str | None:
    """`dest` relative to the snapshot, or None when the run has no such file."""
    src = run.files.path(rel)
    if not src.is_file():
        return None
    (root / dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, root / dest)
    return dest


def _nodes(run: views.Run, scrub: Scrubber) -> list[dict]:
    return [{"node": r["node"], "parent": r["parent"], "status": r["status"], "score": r["score"],
             "vs parent": r["vs parent"], "summary": scrub((run.files.read_text(f"nodes/{r['node']}/summary.md") or "").strip()),
             "error": scrub(r["error"]), "created": r["created"]} for r in views.overview(run)["nodes"]]


def _alerts(run: views.Run, scrub: Scrubber) -> list[dict]:
    out = []
    for r in run.records.all():
        if r["kind"] == "alert":
            node, phase, attempt = run.records.place(r)
            out.append({"time": fmt_ts(r["ts"]), "level": r.get("level"), "name": r.get("name"),
                        "node": node, "phase": f"{phase}-{attempt}" if phase else None, "message": scrub(r.get("message"))})
    return out[::-1]


def _training_videos(run: views.Run, node: str, root: Path, folder: str, scrub: Scrubber) -> list[dict]:
    out = []
    for ds in media.training_data(run, node)["datasets"]:
        for c in media.clips(run, node, ds["dataset"], page_size=MAX_VIDEOS - len(out))["rows"]:
            caption = (c["caption"] or {}).get("caption") if isinstance(c["caption"], dict) else None
            out.append({"file": _copy(run, c["video"], root, f"{folder}/train_{len(out)}.mp4"),
                        "label": f"{ds['dataset']} · clip {len(out) + 1}", "caption": scrub(caption or "")})
        if len(out) >= MAX_VIDEOS:
            break
    return out


def _eval_videos(run: views.Run, node: str, root: Path, folder: str, scrub: Scrubber) -> list[dict]:
    out = []
    for c in media.eval_view(run, node)["cases"][:MAX_VIDEOS]:
        turns = {}
        for p in c["prompt_schedule"] or []:
            turns.setdefault(p.get("turn", len(turns)), p.get("prompt"))
        actions = c["actions"] or []
        prompt = "\n".join(f"turn {t} ({actions[t] if isinstance(t, int) and t < len(actions) else '?'}): {p}"
                           for t, p in turns.items())
        out.append({"file": _copy(run, c["video"], root, f"{folder}/eval_{len(out)}.mp4"), "label": f"case {c['case']}",
                    "perspective": c["perspective"], "prompt": scrub(prompt), "scores": c["scores"]})
    return out


def tokens_text(t: dict) -> str:
    return " · ".join(f"{t[k]:,} {k}" for k in TOKENS)


def _split(usage: dict) -> dict:
    """The ledger's split (budget.usage_tokens) of one call's usage."""
    uncached, cached, output = usage_tokens(usage)
    return {"cached": cached, "uncached": uncached, "output": output}


def _item(it: dict, calls: dict, scrub: Scrubber) -> dict:
    if it["kind"] == "usage":                    # tokens only: its cost and latency are left out
        return {"kind": "tokens", "title": None, "text": f"turn {it['turn']}: {tokens_text(_split(calls[it['call_id']].usage))}"}
    text = _clip(it["text"], TOOL_TEXT) if it["kind"] in ("tool_output", "tool_error") else it["text"]
    return {"kind": it["kind"], "title": scrub(it["title"]), "text": scrub(text)}


def _conversations(run: views.Run, node: str, scrub: Scrubber) -> list[dict]:
    out = []
    for phase, attempt in views.agent_attempts(run, node):
        for conv in views.conversations(run, node, phase, attempt):
            calls = {c.call_id: c for s in conv["chain"] for c in s.calls}
            total = {k: sum(_split(c.usage)[k] for c in calls.values()) for k in TOKENS}
            items = [_item(it, calls, scrub) for it in views.conversation_chat(run, node, phase, attempt, conv["index"])]
            out.append({"label": f"{phase}-{attempt} · {conv['role']} · {conv['index']} · {tokens_text(total)}",
                        "tokens": total, "items": items})
    return out


def _edits(run: views.Run, node: str, scrub: Scrubber) -> dict:
    d = views.code_edits(run, node)
    return {"node_diff": scrub(d["node_diff"]),
            "attempts": [{"attempt": a["attempt"], "outcome": a["outcome"], "detail": scrub(a["detail"]),
                          "diff": scrub(a["diff"])} for a in d["attempts"]]}


def export(run_dir: Path, out: Path) -> Path:
    """Writes the snapshot to a fresh folder, then swaps it in for `out`."""
    run, scrub = views.Run(run_dir), Scrubber()
    new = out.with_name(out.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    nodes = _nodes(run, scrub)
    matrix = views.eval_matrix(run)
    summary = {"run": run.name, "exported": fmt_ts(time.time()), "nodes": nodes, "alerts": _alerts(run, scrub),
               "metrics": matrix["metrics"],
               "scores": [{"node": n["node"], "score": n["score"], **n["metrics"]} for n in matrix["nodes"]]}
    for i, n in enumerate(nodes):
        node, folder = n["node"], f"videos/{i}"
        detail = {"training_videos": _training_videos(run, node, new, folder, scrub),
                  "eval_videos": _eval_videos(run, node, new, folder, scrub),
                  "conversations": _conversations(run, node, scrub), "edits": _edits(run, node, scrub)}
        n.update({k: sum(c["tokens"][k] for c in detail["conversations"]) for k in TOKENS}, file=f"node_{i}.json")
        (new / n["file"]).write_text(json.dumps(detail, ensure_ascii=False, default=str))
    summary["tokens"] = {k: sum(n[k] for n in nodes) for k in TOKENS}
    (new / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, default=str))
    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    new.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    return out
