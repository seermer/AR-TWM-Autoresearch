"""Keeping the evaluation set out of agents' reach: names the kernel refuses, text that copies an
evaluation prompt, and the scrub of kernel tool output. Nothing here is shown to an agent."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

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
