"""Keeping the evaluation set out of agents' reach: names the kernel refuses, text that copies an
evaluation prompt, and the scrub of kernel tool output. Nothing here is shown to an agent."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .process_digest import _payload

RUN_WORDS = 8               # this many consecutive words shared with an evaluation prompt is a copy
SCRUBBED = "render"         # what a blocked name is replaced with: run_<name> reads run_render
EXCLUDED_CLIP = ("this clip is on the kernel's exclusion list and cannot be used as training data. "
                 "Use a different clip.")
EXCLUDED_PROMPT = "this prompt is on the kernel's exclusion list. Write a different one."


def blocked(cfg, text: str | None) -> bool:
    text = (text or "").lower()
    return any(name.lower() in text for name in cfg.get("isolation.blocked_names") or [])


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def _runs(text: str) -> set[tuple]:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(words[i:i + RUN_WORDS]) for i in range(len(words) - RUN_WORDS + 1)}


@lru_cache(maxsize=None)
def _held_out_runs(cases_dir: Path) -> frozenset:
    runs: set[tuple] = set()
    for path in sorted(cases_dir.glob("case_*.json")):
        for text in strings(json.loads(path.read_text(encoding="utf-8"))):
            runs |= _runs(text)
    return frozenset(runs)


def copies_held_out(cfg, *texts) -> bool:
    """A word run survives an added prefix, suffix or a split sentence; a paraphrase or a short
    common phrase does not match, on purpose."""
    held = _held_out_runs(cfg.wbench / "data" / "cases")
    return any(not held.isdisjoint(_runs(text)) for text in texts if isinstance(text, str))


def scrub(value, names: list[str]):
    if isinstance(value, str):
        for name in names:
            value = re.sub(re.escape(name), SCRUBBED, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, dict):
        return {key: scrub(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, names) for item in value]
    return value


def censor_names(cfg) -> list[str]:
    """Every name the gateway replaces in what a tool returned to the model."""
    return list(dict.fromkeys([*(cfg.get("isolation.blocked_names") or []), *(cfg.get("isolation.audit_patterns") or [])]))


def censor_tool_results(body: dict, names: list[str]) -> dict:
    """An LLM request body with `names` replaced in every tool result (Chat Completions `tool` messages,
    Responses `function_call_output` items). What the model itself wrote is left as it is: the audit
    reads that. So a mention the agent only came across never reaches the model or a transcript."""
    out = dict(body)
    if isinstance(body.get("messages"), list):
        out["messages"] = [{**m, "content": scrub(m.get("content"), names)}
                           if isinstance(m, dict) and m.get("role") == "tool" else m for m in body["messages"]]
    if isinstance(body.get("input"), list):
        out["input"] = [{**item, "output": scrub(item.get("output"), names)}
                        if isinstance(item, dict) and item.get("type") == "function_call_output" else item
                        for item in body["input"]]
    return out


def audit(cfg, recorder, node: str, phase: str, attempt: int) -> list[str]:
    """What the model wrote in one attempt that names an audit pattern: its reasoning, its replies and
    every tool-call argument (commands, files written through tools, queries, plans). One line per
    place and pattern; empty when clean. Tool results are not read here: the gateway censors them."""
    patterns = [p.lower() for p in cfg.get("isolation.audit_patterns") or []]
    hits = set()
    for e in recorder.read_events(node):
        if e.get("type") != "llm.response" or e.get("phase") != phase or e.get("attempt") != attempt or e.get("tool"):
            continue
        for choice in (_payload(recorder, e).get("body") or {}).get("choices") or []:
            message = choice.get("message") or {}
            written = [("reply", message.get("content")),
                       ("reasoning", message.get("reasoning") or message.get("reasoning_content")),
                       *[((call.get("function") or {}).get("name"), (call.get("function") or {}).get("arguments"))
                         for call in message.get("tool_calls") or []]]
            for where, text in written:
                text = json.dumps(text).lower() if not isinstance(text, str) else text.lower()
                hits |= {f"{where} names {p}" for p in patterns if p in text}
    return sorted(hits)
