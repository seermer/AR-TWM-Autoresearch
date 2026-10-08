"""read_skill: the kernel's reference notes for agents (kernel/ar_kernel/skills/*.md). Agents read
them through this tool and cannot change them."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from .server import ToolError

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


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


def load_skills(tools: dict | None = None) -> dict[str, dict]:
    """name -> {description, text}. A skill file starts with front matter: `name:` and `description:`.
    One with `tool:` is that kernel tool's skill: it exists only when the tool is in `tools`
    (name -> the registered tool), and its text starts with the tool's own description and arguments."""
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        head, text = path.read_text(encoding="utf-8").removeprefix("---\n").split("\n---\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        text = text.strip()
        if "tool" in meta:
            if meta["tool"] not in (tools or {}):
                continue
            tool = tools[meta["tool"]]
            text = "\n\n".join(["# The tool", tool.description, "# Its arguments",
                                "\n".join(arguments(tool.input_schema)), text])
        skills[meta["name"]] = {"description": meta["description"], "text": text}
    return skills


def register_skill_tool(mcp, kit) -> None:
    """Call after the data-source tools are registered: a tool's skill is listed only when the tool is."""
    skills = load_skills({t.name: t for t in asyncio.run(mcp.list_tools())})
    index = "\n".join(f"- {name}: {skill['description']}" for name, skill in skills.items())

    @mcp.tool(name="read_skill", description="Read one of the kernel's reference notes. Read a skill at the "
              "moment you are about to do what it covers, not up front. The skills:\n" + index)
    async def read_skill(name: Annotated[str, Field(description="a skill's name, from the list above")],
                         ctx: Context) -> str:
        def read(_caller) -> str:
            if name not in skills:
                raise ToolError(f"no skill {name!r}; the skills are: {', '.join(skills)}")
            return skills[name]["text"]
        return await kit.call(ctx, "read_skill", {"name": name}, read)
