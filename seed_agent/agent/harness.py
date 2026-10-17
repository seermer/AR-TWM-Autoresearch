"""The agent loop, compaction, tool output files and the write guard."""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from pydantic import ValidationError

COMPACT_PROMPT = Path(__file__).resolve().parent / "prompts" / "compact.md"
TOOL_BLOCK_TYPES = {"text", "image_url", "image", "json", "search_result", "custom_tool_call_output",
                    "document", "file"}
SHOW_WHOLE, SHOWN = 100_000, 10_000        # chars of a tool result
CHARS_PER_TOKEN = 4
IMAGE_TOKENS = 1_500                       # tokens per image
IMAGE_BLOCK_TYPES = {"image_url", "image", "input_image"}
CONTINUATION = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
    "<summary>\n{summary}\n</summary>\n\n"
    "Continue the work from where it left off without asking any further questions. "
    "Resume directly: do not acknowledge the summary or recap what was happening."
)
GUARDED_TOOLS = ("run_command", "write_file", "edit_file")
GUARD_TIMEOUT_S = 3600


def _content(output: Any) -> str | list:
    if isinstance(output, str) or (isinstance(output, list) and all(
            isinstance(x, dict) and x.get("type") in TOOL_BLOCK_TYPES for x in output)):
        return output
    return json.dumps(output, ensure_ascii=False, default=str)


def error_text(name: str, exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        fields = "\n".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'invalid')}" for e in exc.errors())
        return f"Error: invalid arguments for {name}:\n{fields}"
    if isinstance(exc, ToolException):
        lines = [re.sub(r"\s*\[type=.*\]$", "", line)
                 for line in str(exc).removeprefix(f"Error executing tool {name}: ").splitlines()
                 if "errors.pydantic.dev" not in line]
        return "Error: " + "\n".join(lines)
    return f"Error: {type(exc).__name__}: {exc}"


_spilled = itertools.count(1)


def spill(role: str, name: str, args: dict, content: str | list) -> str | list:
    """Saves the result to a file; a long text is shown as its head and tail."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    header = f"$ {args.get('command', '')}" if name == "run_command" else f"$ {name} {json.dumps(args, ensure_ascii=False, default=str)}"
    path = (Path(os.environ.get("AR_WORKSPACE", "/workspace")) / "tool_output"
            / f"{role}-{name}-{time.strftime('%Y%m%d-%H%M%S')}-{next(_spilled)}.txt")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{header}\n{text}", encoding="utf-8")
        saved = f"the full result is in {path}"
    except OSError as exc:
        saved = f"the full result could not be saved to {path}: {exc}"
        if not isinstance(content, str):
            return [*content, {"type": "text", "text": f"[Error: {saved}]"}]
        if len(content) <= SHOW_WHOLE:
            return f"{content}\n[Error: {saved}]"
    if not isinstance(content, str) or len(content) <= SHOW_WHOLE:
        return content
    first, _, rest = content.partition("\n") if name == "run_command" else ("", "", content)
    note = f"[Long result: {len(rest)} chars. Only the first and last {SHOWN} are shown; {saved}.]"
    return "\n".join(filter(None, [first, note, rest[:SHOWN], "[...]", rest[-SHOWN:]]))


class Guard:
    """Reverts and reports what a non-owner's call changes in a tree."""

    def __init__(self, tree: str, owner: str, git_dir: str) -> None:
        self.tree, self.owner, self.git_dir = tree, owner, git_dir
        self.stale = True
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=guard", "-c", "user.email=guard@agent", "--git-dir",
                               self.git_dir, "--work-tree", self.tree, *args], capture_output=True, text=True,
                              check=True, timeout=GUARD_TIMEOUT_S).stdout

    def _changed(self) -> list[str]:
        return [entry[3:] for entry in self._git("status", "--porcelain", "-z", "--untracked-files=all").split("\0")
                if entry]

    def _failed(self, exc: Exception) -> str:
        return f"Error: the write guard of {self.tree} failed: {type(exc).__name__}: {getattr(exc, 'stderr', '') or exc}"

    def baseline(self) -> str | None:
        try:
            if self.stale:
                if not os.path.isdir(self.git_dir):
                    Path(self.git_dir).mkdir(parents=True)
                    self._git("init", "-q")
                self._git("add", "-A")
                self._git("commit", "-q", "--allow-empty", "--no-verify", "-m", "baseline")
                self.stale = False
        except (OSError, subprocess.SubprocessError) as exc:
            return self._failed(exc)
        return None

    def check(self, name: str) -> str | None:
        try:
            changed = self._changed()
            if not changed:
                return None
            why = ""
            try:
                self._git("reset", "-q", "--hard")
                self._git("clean", "-fdq")
            except subprocess.SubprocessError as exc:
                why = f": {getattr(exc, 'stderr', '') or exc}".rstrip()
            left = set(self._changed())
        except (OSError, subprocess.SubprocessError) as exc:
            return self._failed(exc)
        files = "; ".join(f"{p} ({f'could not revert{why}' if p in left else 'reverted'})" for p in changed)
        return f"Error: {name} changed files in {self.tree}, which belongs to {self.owner}: {files}"

    def remove(self) -> None:
        remove_tree(self.git_dir)


def remove_tree(path) -> None:
    """Delete a folder; what could not be removed goes to stderr."""
    def report(_, p, exc):
        if not isinstance(exc, FileNotFoundError):
            print(f"could not remove {p}: {exc}", file=sys.stderr)
    shutil.rmtree(path, onexc=report)


async def _invoke(tool: BaseTool, call: dict) -> tuple[str | list, str]:
    try:
        return _content(await tool.ainvoke(call["args"])), "success"
    except Exception as exc:  # noqa: BLE001
        return error_text(call["name"], exc), "error"


async def run_tool(tools: dict[str, BaseTool], call: dict, role: str, guard: Guard | None) -> ToolMessage:
    name, tool = call["name"], tools.get(call["name"])
    guarded = guard is not None and role != guard.owner and name in GUARDED_TOOLS
    if tool is None:
        content, status = f"Error: {name} is not a valid tool, try one of [{', '.join(tools)}].", "error"
    elif guarded and (problem := await asyncio.to_thread(guard.baseline)):
        content, status = f"{problem}, so {name} was not run", "error"
    else:
        content, status = await _invoke(tool, call)
        if guarded and (problem := await asyncio.to_thread(guard.check, name)):
            shown = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            content, status = f"{problem}\n\nWhat {name} returned:\n{shown}", "error"
    if guard is not None and role == guard.owner:
        guard.stale = True
    return ToolMessage(spill(role, name, call["args"], content), name=name, tool_call_id=call["id"], status=status)


async def run_batch(tools: dict[str, BaseTool], calls: list[dict], role: str,
                    guard: Guard | None) -> tuple[list[ToolMessage], bool]:
    """Returns (results, whether a submit was accepted)."""
    if all(name in tools and (tools[name].metadata or {}).get("parallel") for name in (c["name"] for c in calls)):
        return list(await asyncio.gather(*(run_tool(tools, c, role, guard) for c in calls))), False
    out, skip, submitted = [], None, False
    for call in calls:
        if skip:
            out.append(ToolMessage(skip, name=call["name"], tool_call_id=call["id"]))
            continue
        message = await run_tool(tools, call, role, guard)
        out.append(message)
        tool = tools.get(call["name"])
        if tool is not None and tool.return_direct and message.status != "error":
            skip, submitted = f"Not run: the turn ended when {call['name']} was accepted.", True
        elif tool is not None and (tool.metadata or {}).get("ends_batch"):
            skip = f"Not run: {call['name']} ran in this turn; read its result first."
    return out, submitted


def _chars(message: AnyMessage) -> int:
    if isinstance(message.content, str):
        size = len(message.content)
    else:
        size = sum(IMAGE_TOKENS * CHARS_PER_TOKEN
                   if isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES
                   else len(block if isinstance(block, str) else json.dumps(block, default=str))
                   for block in message.content)
    calls = getattr(message, "tool_calls", None) or []
    return size + (len(json.dumps(calls, default=str)) if calls else 0)


def estimate_tokens(messages: list[AnyMessage], system_prompt: str | None = None) -> int:
    """The last reported usage plus an estimate of what came after."""
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AIMessage) and message.usage_metadata:
            tail = sum(_chars(m) for m in messages[i + 1:])
            return message.usage_metadata["total_tokens"] + tail // CHARS_PER_TOKEN
    head = len(system_prompt or "")
    return (head + sum(_chars(m) for m in messages)) // CHARS_PER_TOKEN


def needs_compaction(messages: list[AnyMessage], system_prompt: str | None, context_window: int,
                     compact_at: float) -> bool:
    return len(messages) > 1 and estimate_tokens(messages, system_prompt) >= compact_at * context_window


async def summarize(model, bound, tools: list[BaseTool], prompt: list[AnyMessage]) -> str:
    """The compaction summary."""
    prompt = [*prompt, HumanMessage(content=COMPACT_PROMPT.read_text())]
    reply = await bound.ainvoke(prompt)
    if reply.tool_calls and not reply.text.strip():
        reply = await (model.bind_tools(tools, tool_choice="none") if tools else model.bind()).ainvoke(prompt)
    if not reply.text.strip():
        raise RuntimeError("compaction produced no summary")
    return reply.text.strip()


async def run_loop(model, tools: list[BaseTool], system: str | None, messages: list[AnyMessage], *, role: str,
                   pinned: str, compact_at: float, context_window: int, guard: Guard | None = None) -> list[AnyMessage]:
    """Runs until a reply has no tool call or a submit is accepted; extends `messages` in place."""
    by_name = {t.name: t for t in tools}
    head = [SystemMessage(content=system)] if system is not None else []
    while True:
        bound = model.bind_tools(tools, tool_choice=None) if tools else model.bind()
        if needs_compaction(messages, system, context_window, compact_at):
            summary = await summarize(model, bound, tools, [*head, *messages])
            messages[:] = [HumanMessage(content=f"{pinned}\n\n{CONTINUATION.format(summary=summary)}")]
        reply = await bound.ainvoke([*head, *messages])
        messages.append(reply)
        if not tools or not reply.tool_calls:
            return messages
        results, submitted = await run_batch(by_name, reply.tool_calls, role, guard)
        messages += results
        if submitted:
            return messages
