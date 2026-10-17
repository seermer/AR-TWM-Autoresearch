"""The snapshot tool over the socket, as a container calls it."""
import asyncio
import json

import pytest
from fastapi import FastAPI

from ar_kernel.services import RunServices, socket_dir_for
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import snapshot
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
from ar_kernel.tools.snapshot import SnapshotStore, register_snapshot_tool
from ar_kernel.vcs.agents_repo import AgentsRepo

KNOWN = "c" * 64


@pytest.fixture
def live(tmp_path):
    run = tmp_path / "run"
    seed = tmp_path / "seed" / "agent"
    seed.mkdir(parents=True)
    (seed / "entry.py").write_text("v1\n")
    repo = AgentsRepo(run / "agents.git")
    root = repo.init(tmp_path / "seed")
    rec = Recorder(run)
    reg = TokenRegistry(rec)
    mcp = new_mcp()
    register_snapshot_tool(mcp, ToolKit(reg, rec), SnapshotStore(run, repo, lambda c: c == KNOWN))
    services = RunServices(socket_dir_for(run))
    services.start(FastAPI(), build_tool_app(mcp))
    agent = tmp_path / "agent"
    repo.checkout(root, agent)
    edit = reg.issue(node="n1", phase="edit_self", attempt=2, workspace_host=tmp_path, staging_host=tmp_path,
                     agent_host=agent, base_commit=root)
    recipe = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    yield services, repo, run, rec, edit, recipe, agent, root
    services.stop()


def _call(services, caller, args):
    from ar_contract.client import mcp_session

    async def go():
        async with mcp_session(services.socket_dir, caller.token) as session:
            return await session.call_tool("snapshot", args)
    result = asyncio.run(go())
    return result.is_error, result.content[0].text


def test_an_edit_snapshot_commits_the_tree_under_an_attempt_ref(live):
    services, repo, run, rec, edit, _, agent, root = live
    (agent / "agent" / "entry.py").write_text("v2\n")
    error, text = _call(services, edit, {})
    sha = json.loads(text)["snapshot_id"]
    assert not error and repo.read_file(sha, "agent/entry.py") == "v2\n"
    assert snapshot.edit_snapshot(repo, "n1", "edit_self", 2, sha) == sha
    assert snapshot.edit_snapshot(repo, "n1", "edit_self", 1, sha) is None
    assert [e["type"] for e in rec.read_events("n1")] == ["tool.call", "tool.result"]


def test_an_edit_snapshot_takes_no_arguments(live):
    services, _, _, _, edit, *_ = live
    error, text = _call(services, edit, {"rationale": "r"})
    assert error and "no arguments" in text


def test_a_recipe_snapshot_is_stored_for_its_attempt(live):
    services, _, run, _, _, recipe, *_ = live
    error, text = _call(services, recipe, {"data_commit": KNOWN, "recipe": {"lr": 0.5}, "rationale": "why"})
    sid = json.loads(text)["snapshot_id"]
    assert not error
    assert snapshot.recipe_snapshot(run, "n1", "improve_recipe", 1, sid) == {
        "has_result": True, "data_commit": KNOWN, "recipe": {"lr": 0.5}, "rationale": "why"}
    assert snapshot.recipe_snapshot(run, "n1", "improve_recipe", 2, sid) is None
    error, text = _call(services, recipe, {})
    assert not error and snapshot.recipe_snapshot(
        run, "n1", "improve_recipe", 1, json.loads(text)["snapshot_id"])["has_result"] is False


@pytest.mark.parametrize("args,fragment", [
    ({"data_commit": KNOWN}, "together, or none"),
    ({"recipe": {}, "rationale": "r"}, "together, or none"),
    ({"data_commit": "d" * 64, "recipe": {}, "rationale": "r"}, "does not exist"),
])
def test_a_recipe_snapshot_is_refused_with_an_error(live, args, fragment):
    services, _, run, rec, _, recipe, *_ = live
    error, text = _call(services, recipe, args)
    assert error and fragment in text
    assert not (run / "snapshots").exists()
    assert [e["type"] for e in rec.read_events("n1")][-1] == "tool.error"


def test_clear_removes_an_attempts_snapshots_and_refs(live):
    services, repo, run, _, edit, recipe, *_ = live
    sha = json.loads(_call(services, edit, {})[1])["snapshot_id"]
    sid = json.loads(_call(services, recipe, {})[1])["snapshot_id"]
    snapshot.clear(run, repo, "n1", "edit_self", 2)
    snapshot.clear(run, repo, "n1", "improve_recipe", 1)
    assert snapshot.edit_snapshot(repo, "n1", "edit_self", 2, sha) is None
    assert snapshot.recipe_snapshot(run, "n1", "improve_recipe", 1, sid) is None
