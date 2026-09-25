"""The agent's tools: file/shell tools bound to one root, the timed-prompt helper, the kernel
tools (the MCP tool server) as LangChain tools, and submit_<x> result tools. Captioning is the
kernel's caption_videos GPU job."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.tools import StructuredTool, ToolException, tool
from pydantic import BaseModel

MAX_READ = 200_000
MAX_OUTPUT = 8_000
FIRST_ROUND_END = 25 / 24          # the 25 history frames
ROUND = 32 / 24                    # one rollout round


# ---- files and shell (root is /agent for edit_self, /workspace for improve_recipe) ----

def resolve_inside(root: str, path: str) -> Path:
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{path} is outside {root}")
    return target


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old text must appear exactly once, found {count} matches")
    return text.replace(old, new, 1)


def make_file_tools(root: str, *, writable: bool = True) -> list:
    @tool
    def read_file(path: str) -> str:
        """Read a text file (path relative to the tool root)."""
        return resolve_inside(root, path).read_text(encoding="utf-8", errors="replace")[:MAX_READ]

    @tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory (path relative to the tool root); directories end with '/'."""
        return sorted(p.name + ("/" if p.is_dir() else "") for p in resolve_inside(root, path).iterdir())

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file (path relative to the tool root)."""
        target = resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a text file. `old` must appear exactly once."""
        target = resolve_inside(root, path)
        try:        # strict: writing back text with replaced undecodable bytes would corrupt the file
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path} is not UTF-8 text (byte {exc.start}); "
                             f"change it with run_command instead") from None
        target.write_text(replace_once(text, old, new), encoding="utf-8")
        return f"edited {path}"

    @tool
    def run_command(command: str, timeout_s: int = 600) -> str:
        """Run a shell command in the tool root (ffmpeg, ffprobe, python, ...). Returns the exit code and the output tail."""
        try:
            r = subprocess.run(["bash", "-lc", command], cwd=root, capture_output=True, text=True,
                               timeout=min(int(timeout_s), 3600))
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout_s}s"
        return f"exit {r.returncode}\n{(r.stdout + r.stderr)[-MAX_OUTPUT:]}"

    if not writable:
        return [read_file, list_dir]
    return [read_file, list_dir, write_file, edit_file, run_command]


# ---- timed prompts ----

def snap_segments(segments: list[dict], duration: float) -> list[dict]:
    """Snap internal boundaries to 25/24 + k*32/24 s (per_chunk rule), keep the ends,
    make segments contiguous, and drop any that collapse to zero length."""
    bounds, t = [], FIRST_ROUND_END
    while t < duration:
        bounds.append(t)
        t += ROUND
    ordered = sorted(segments, key=lambda s: s["time_range_s"][0])
    cuts = [0.0]
    for seg in ordered[:-1]:
        end = seg["time_range_s"][1]
        cuts.append(min(bounds, key=lambda b: abs(b - end)) if bounds else end)
    cuts.append(float(duration))
    return [{"time_range_s": [start, end], "prompt": seg["prompt"]}
            for seg, start, end in zip(ordered, cuts, cuts[1:]) if end - start > 1e-9]


@tool
def snap_timed_prompts(segments_json: str, duration: float) -> str:
    """Snap timed-prompt segment boundaries to rollout-round boundaries. Input and output: a JSON list of
    {"time_range_s": [start, end], "prompt": str}."""
    return json.dumps(snap_segments(json.loads(segments_json), duration))


# ---- kernel tools over one MCP session (MCP 2.x: is_error, structured_content, input_schema) ----

def result_text(result) -> str:
    """A kernel tool result as text: structured content JSON-encoded (MCP splits a list result
    into one text block per item), else the text blocks. A failed call raises ToolException,
    which the harness reports to the model."""
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if result.is_error:
        raise ToolException(text)
    return text if result.structured_content is None else json.dumps(result.structured_content)


async def kernel_tools(session) -> list[StructuredTool]:
    tools = []
    for spec in (await session.list_tools()).tools:
        async def call(_name: str = spec.name, **kwargs: Any) -> str:
            return result_text(await session.call_tool(_name, kwargs))
        tools.append(StructuredTool(name=spec.name, description=spec.description or "",
                                    args_schema=spec.input_schema, coroutine=call))
    return tools


# ---- result tools: a role finishes by calling submit_<x>; invalid arguments come back
# to the model as a tool error (the harness), so it can correct them ----

def submit_tool(name: str, description: str, schema: type[BaseModel]) -> tuple[StructuredTool, SimpleNamespace]:
    box = SimpleNamespace(value=None, name=name)       # box.value: the validated submission, once submitted

    async def submit(**kwargs) -> str:
        box.value = schema.model_validate(kwargs)
        return "submitted"

    return StructuredTool(name=name, description=description, args_schema=schema, coroutine=submit), box
