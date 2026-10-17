"""snapshot: the kernel's record of a phase's work, per (node, phase, attempt)."""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable

from mcp.server.mcpserver import Context
from pydantic import Field

from .server import ToolError

DESCRIPTION = ("Store the work as it is now as an immutable snapshot and return its id. A phase's result names "
               "the snapshot it hands back by this id (snapshot_id); later changes cannot alter a snapshot. In "
               "edit_self it takes no arguments and records the /agent tree. In improve_recipe it takes "
               "data_commit, recipe and rationale, all three or none; with none, the snapshot records that there "
               "is no result.")


def folder(run_dir: Path, node: str, phase: str, attempt: int) -> Path:
    return Path(run_dir) / "snapshots" / node / f"{phase}-{attempt}"


def ref_prefix(node: str, phase: str, attempt: int) -> str:
    return f"refs/snapshots/{node}/{phase}-{attempt}/"


def clear(run_dir: Path, repo, node: str, phase: str, attempt: int) -> None:
    """Removes the attempt's snapshots from an earlier run."""
    path = folder(run_dir, node, phase, attempt)
    if path.exists():
        shutil.rmtree(path)
    repo.delete_refs(ref_prefix(node, phase, attempt))


def recipe_snapshot(run_dir: Path, node: str, phase: str, attempt: int, snapshot_id: str) -> dict | None:
    """The stored body, or None when this attempt has no such snapshot."""
    path = folder(run_dir, node, phase, attempt) / f"{snapshot_id}.json"
    if not snapshot_id.isalnum() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def edit_snapshot(repo, node: str, phase: str, attempt: int, snapshot_id: str) -> str | None:
    """The commit, or None when this attempt has no such snapshot."""
    return repo.resolve(ref_prefix(node, phase, attempt) + snapshot_id) if snapshot_id.isalnum() else None


class SnapshotStore:
    def __init__(self, run_dir: Path, repo, commit_exists: Callable[[str], bool]) -> None:
        self.run_dir, self.repo, self.commit_exists = Path(run_dir), repo, commit_exists

    def take(self, caller, data_commit: str | None, recipe: dict | None, rationale: str | None) -> dict:
        given = [v is not None for v in (data_commit, recipe, rationale)]
        if caller.agent_host is not None:
            if any(given):
                raise ToolError("in edit_self, snapshot takes no arguments: it records the /agent tree as it is now")
            try:
                sha = self.repo.commit_tree(caller.agent_host, caller.base_commit,
                                            f"{caller.node} {caller.phase} attempt {caller.attempt} snapshot")
            except subprocess.CalledProcessError as exc:
                raise ToolError(f"cannot commit the /agent tree: {exc.stderr.decode(errors='replace')}") from exc
            self.repo.set_ref(ref_prefix(caller.node, caller.phase, caller.attempt) + sha, sha, push=False)
            return {"snapshot_id": sha}
        if any(given) and not all(given):
            raise ToolError("give data_commit, recipe and rationale together, or none of them")
        if all(given) and not self.commit_exists(data_commit):
            raise ToolError(f"data commit {data_commit!r} does not exist")
        body = {"has_result": all(given), "data_commit": data_commit, "recipe": recipe, "rationale": rationale}
        snapshot_id = uuid.uuid4().hex
        path = folder(self.run_dir, caller.node, caller.phase, caller.attempt)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{snapshot_id}.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
        return {"snapshot_id": snapshot_id}


def register_snapshot_tool(mcp, kit, store: SnapshotStore) -> None:
    @mcp.tool(name="snapshot", description=DESCRIPTION)
    async def snapshot(
            ctx: Context,
            data_commit: Annotated[str | None, Field(description="improve_recipe: the data commit id to train on")] = None,
            recipe: Annotated[dict[str, Any] | None, Field(description="improve_recipe: tunable key -> value")] = None,
            rationale: Annotated[str | None, Field(description="improve_recipe: why the recipe and data are as they are")] = None) -> dict:
        args = {k: v for k, v in dict(data_commit=data_commit, recipe=recipe, rationale=rationale).items()
                if v is not None}
        return await kit.call(ctx, "snapshot", args, lambda c: store.take(c, data_commit, recipe, rationale))
