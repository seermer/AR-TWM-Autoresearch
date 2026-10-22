"""read_knowledge: the kernel's reference notes for agents (kernel/ar_kernel/knowledge/*.md). Agents read
them through this tool and cannot change them."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from .server import ToolError

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def arguments(schema: dict, depth: int = 0) -> list[str]:
    """A tool's arguments, one line each, with the fields of its items indented under them."""
    lines = []
    for name, spec in (schema.get("properties") or {}).items():
        optional = "" if name in (schema.get("required") or []) else " (optional)"
        choices = f" (one of: {', '.join(map(str, spec['enum']))})" if spec.get("enum") else ""
        lines.append(f"{'  ' * depth}- `{name}`{optional}: {spec.get('description', '')}{choices}")
        for part in (spec, *spec.get("anyOf", [])):
            lines += arguments(part.get("items", part), depth + 1)
    return lines


def load_knowledge(tools: dict | None = None) -> dict[str, dict]:
    """name -> {description, text}. A knowledge file starts with front matter: `name:` and `description:`.
    One with `tool:` is that kernel tool's file: it exists only when the tool is in `tools`
    (name -> the registered tool), and its text starts with the tool's own description and arguments."""
    files = {}
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        head, text = path.read_text(encoding="utf-8").removeprefix("---\n").split("\n---\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        text = text.strip()
        if "tool" in meta:
            if meta["tool"] not in (tools or {}):
                continue
            tool = tools[meta["tool"]]
            text = "\n\n".join(["# The tool", tool.description, "# Its arguments",
                                "\n".join(arguments(tool.input_schema)), text])
        files[meta["name"]] = {"description": meta["description"], "text": text}
    return files


def register_knowledge_tool(mcp, kit) -> None:
    """Call after the data-source tools are registered: a tool's file is listed only when the tool is."""
    files = load_knowledge({t.name: t for t in asyncio.run(mcp.list_tools())})
    index = "\n".join(f"- {name}: {file['description']}" for name, file in files.items())

    @mcp.tool(name="read_knowledge", description="Read one of the kernel's knowledge files. Read a file at the "
              "moment you are about to do what it covers, not up front. The files:\n" + index)
    async def read_knowledge(name: Annotated[str, Field(description="a file's name, from the list above")],
                         ctx: Context) -> str:
        def read(_caller) -> str:
            if name not in files:
                raise ToolError(f"no knowledge file {name!r}; the files are: {', '.join(files)}")
            return files[name]["text"]
        return await kit.call(ctx, "read_knowledge", {"name": name}, read)
