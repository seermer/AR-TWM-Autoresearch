"""Uses the real MCP tools from inside the container: queries the pool, commits it and snapshots the commit."""
import asyncio
import json

from ar_contract.client import mcp_session
from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(snapshot_id="unused", summary="unused")


async def improve_recipe(ctx):
    async with mcp_session() as tools:
        pool = json.loads((await tools.call_tool("data_query", {})).content[0].text)
        ids = [c["clip_id"] for c in pool["clips"]][:4]
        made = json.loads((await tools.call_tool("data_commit", {
            "parent": None, "message": "from inside the sandbox",
            "datasets": {"cam": {"weight": 1.0, "clips": ids}}})).content[0].text)
        snapshot = json.loads((await tools.call_tool("snapshot", {
            "data_commit": made["commit_id"], "recipe": {}, "rationale": f"{len(ids)} clips"})).content[0].text)
    return RecipeResult(snapshot_id=snapshot["snapshot_id"], summary="the pool's first clips")
