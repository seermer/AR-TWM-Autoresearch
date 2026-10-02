"""read_skill: the kernel's reference notes for agents (kernel/ar_kernel/skills/*.md). Agents read
them through this tool and cannot change them."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from .server import ToolError

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def load_skills() -> dict[str, dict]:
    """name -> {description, text}. A skill file starts with front matter: `name:` and `description:`."""
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        head, text = path.read_text(encoding="utf-8").removeprefix("---\n").split("\n---\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        skills[meta["name"]] = {"description": meta["description"], "text": text.strip()}
    return skills


def register_skill_tool(mcp, kit) -> None:
    skills = load_skills()
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
