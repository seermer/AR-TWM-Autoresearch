"""Uses the real MCP tools from inside the container: queries the pool and commits it."""
import asyncio
import json

from ar_contract.client import mcp_session
from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(summary="unused")


async def improve_recipe(ctx):
    async with mcp_session() as tools:
        pool = json.loads((await tools.call_tool("data_query",
                                                  {"filter": {"format": "video_caption_camera"}})).content[0].text)
        ids = [c["clip_id"] for c in pool["clips"]][:4]
        made = json.loads((await tools.call_tool("data_commit", {
            "parent": None, "message": "from inside the sandbox",
            "datasets": {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                 "weight": 1.0, "clips": ids}}})).content[0].text)
    return RecipeResult(data_commit=made["commit_id"], recipe={}, rationale=f"{len(ids)} clips")
