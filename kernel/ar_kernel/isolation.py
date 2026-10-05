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
SCRUBBED = "render"         # what a blocked name becomes in the kernel's own output, where it only occurs
                            # inside the kernel's file names: run_<name>.py reads run_render.py
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


def host_roots(cfg) -> tuple[tuple[str, str], ...]:
    """(folder on this machine, what an agent reads instead) for the folders a kernel message may name:
    command lines and log tails quoted in errors hold them. Read from the config and the environment,
    so nothing here depends on where the project sits; a folder inside another comes first."""
    folders = {"<host>/" + p.name: p for p in (cfg.repo_root, cfg.worldmodel, cfg.wbench, cfg.runs_dir)}
    folders["<host>/home"] = Path.home()
    pairs = {(str(p), shown) for shown, path in folders.items() for p in (path, path.resolve())}
    return tuple(sorted(pairs, key=lambda pair: -len(pair[0])))


def scrub(value, names: list[str], roots=()):
    """`value` as an agent may read it: without the blocked `names` and without the host folders `roots`."""
    if isinstance(value, str):
        for root, shown in roots:
            value = value.replace(root, shown)
        return _matcher(tuple(names)).sub(SCRUBBED, value) if names else value
    if isinstance(value, dict):
        return {key: scrub(item, names, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, names, roots) for item in value]
    return value


def scrub_train_files(node_dir: Path, names: list[str], roots) -> None:
    """Scrub, in place, the training config and logs of a finished node: the trainer wrote this
    machine's folders into them, and later agents read them under /nodes."""
    for attempt in (Path(node_dir) / "attempts").glob("improve_recipe-*"):
        for path in (attempt / "train_config.yaml", attempt / "train" / "train.log",
                     *(attempt / "train" / "logs").rglob("*.log")):
            if not path.is_file() or path.is_symlink():
                continue
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            clean = scrub(text, names, roots)
            if clean != text:
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(clean, encoding="utf-8", errors="surrogateescape")
                tmp.replace(path)


def censor_names(cfg) -> list[str]:
    """Every name whose sentence the gateway drops from what the model did not write."""
    return list(dict.fromkeys([*(cfg.get("isolation.blocked_names") or []), *(cfg.get("isolation.audit_patterns") or [])]))


_MARK = "\x00"
_PART = r'[^.!?\n"' + _MARK + "]*"                  # text inside one sentence, line or quoted string
_SENTENCE = f"{_PART}{_MARK}(?:{_PART}{_MARK})*{_PART}[.!?]?"
_MARKED = re.compile(rf"(?:\A|(?<=\n)){_SENTENCE}[ \t]*|{_SENTENCE}")


def drop_sentences(value, names: list[str]):
    """`value` without any sentence that holds one of `names`. For text from outside (search results,
    papers, web pages): no word could stand in for the name there, and the sentence around it is about
    what the agent must not read. A sentence ends at . ! ? a newline or a double quote."""
    if isinstance(value, str):
        marked = _matcher(tuple(names)).sub(_MARK, value) if names else value
        # only where a name was found: on a long text without sentence ends the pattern is quadratic
        return _MARKED.sub("", marked) if _MARK in marked else value
    if isinstance(value, dict):
        return {key: drop_sentences(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [drop_sentences(item, names) for item in value]
    return value


MODEL_ITEMS = ("function_call", "reasoning")       # Responses input items the model itself wrote


def censor_request(body: dict, names: list[str]) -> dict:
    """An LLM request body without the sentences that hold one of `names`, in everything the model did
    not write: tool results and user and system messages (Chat Completions `messages`, Responses
    `input`). What the model wrote (assistant messages, its tool calls and reasoning) is left as it is:
    the audit reads that. So a mention the agent only came across never reaches the model or a transcript."""
    def own(item) -> bool:
        return not isinstance(item, dict) or item.get("role") == "assistant" or item.get("type") in MODEL_ITEMS

    def censored(item: dict) -> dict:
        return {key: value if key in ("role", "type", "name", "tool_call_id", "call_id") else drop_sentences(value, names)
                for key, value in item.items()}
    out = dict(body)
    for key in ("messages", "input"):
        if isinstance(body.get(key), list):
            out[key] = [item if own(item) else censored(item) for item in body[key]]
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
