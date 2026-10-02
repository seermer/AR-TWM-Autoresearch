"""Keeping the evaluation set out of agents' reach: names the kernel refuses, text that copies an
evaluation prompt, and the scrub of kernel tool output. Nothing here is shown to an agent."""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .process_digest import _payload

RUN_WORDS = 8               # this many consecutive words shared with an evaluation prompt is a copy
SCRUBBED = "render"         # what a blocked name is replaced with: run_<name> reads run_render
EXCLUDED_CLIP = ("this clip is on the kernel's exclusion list and cannot be used as training data. "
                 "Use a different clip.")
EXCLUDED_PROMPT = "this prompt is on the kernel's exclusion list. Write a different one."


@lru_cache(maxsize=None)
def _matcher(names: tuple[str, ...]) -> re.Pattern:
    """Any of `names`, in any case, not preceded by a letter or digit: `run_<name>` matches,
    a longer word that merely ends in the name (DrawBench) does not."""
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, names)) + ")", re.IGNORECASE)


def _found(names, text: str) -> list[str]:
    return [m.lower() for m in _matcher(tuple(names)).findall(text)] if names else []


def blocked(cfg, text: str | None) -> bool:
    return bool(_found(cfg.get("isolation.blocked_names") or [], text or ""))


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
    """The word runs that belong to exactly one evaluation prompt set. A run several of them share
    ("first person view at eye level from the") is boilerplate anyone might write, not a copy."""
    seen: Counter = Counter()
    for path in sorted(cases_dir.glob("case_*.json")):
        runs: set[tuple] = set()
        for text in strings(json.loads(path.read_text(encoding="utf-8"))):
            runs |= _runs(text)
        seen.update(runs)
    return frozenset(run for run, cases in seen.items() if cases == 1)


def copies_held_out(cfg, *texts) -> bool:
    """A word run survives an added prefix, suffix or a split sentence; a paraphrase or a short
    common phrase does not match, on purpose."""
    held = _held_out_runs(cfg.wbench / "data" / "cases")
    return any(not held.isdisjoint(_runs(text)) for text in texts if isinstance(text, str))


def scrub(value, names: list[str]):
    if isinstance(value, str):
        return _matcher(tuple(names)).sub(SCRUBBED, value) if names else value
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


def audit(cfg, recorder, node: str, phase: str | None = None, attempt: int | None = None) -> list[str]:
    """The audit patterns the model itself wrote in one attempt (or, without phase and attempt, anywhere
    in the node): every string of every model response, whatever its shape, so reasoning, replies and
    tool-call arguments (commands, files written through tools, queries, plans) are all read. Empty when
    clean. Tool results are not read here: the gateway censors them before the model sees them."""
    patterns = cfg.get("isolation.audit_patterns") or []
    hits: set[str] = set()
    for e in recorder.read_events(node):
        if e.get("type") != "llm.response" or e.get("tool"):
            continue
        if phase is not None and (e.get("phase") != phase or e.get("attempt") != attempt):
            continue
        hits.update(_found(patterns, " ".join(strings(_payload(recorder, e).get("body")))))
    return sorted(hits)
