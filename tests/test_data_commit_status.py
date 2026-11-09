"""Data commit status (draft, accepted, obsolete, failed) and the limit on a commit's new clips."""
import asyncio
import threading

import pytest

from ar_kernel.archive.clips import ClipStore
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig, merged
from ar_kernel.data.ingest import IngestResult
from ar_kernel.tools import data_tools
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import DataTools, register_data_tools
from ar_kernel.tools.server import ToolError, ToolKit, new_mcp
from ar_kernel.tools.snapshot import DESCRIPTION, SnapshotStore, register_snapshot_tool

BASE = KernelConfig.load()
CFG = KernelConfig(raw=merged(BASE.raw, {"improve_recipe": {"new_clips_per_commit": 3}}), repo_root=BASE.repo_root)


def _clip(clip_id, node="n1", attempt=1):
    return {"clip_id": clip_id, "video_digest": "v", "caption_digest": "c", "annotation_digest": "a",
            "pose_digest": "p", "metadata": {}, "warnings": [], "derived_from": {"kind": "model", "name": "x"},
            "ingested_by": node, "attempt": attempt}


@pytest.fixture
def env(tmp_path):
    conn = open_db(tmp_path)
    tools = DataTools(CFG, tmp_path, [0], threading.Lock())
    reg = TokenRegistry()

    def caller(attempt=2, node="n1"):
        ws = tmp_path / "nodes" / node / "attempts" / f"improve_recipe-{attempt}" / "workspace"
        st = tmp_path / "staging" / node / f"improve_recipe-{attempt}"
        ws.mkdir(parents=True, exist_ok=True), st.mkdir(parents=True, exist_ok=True)
        return reg.issue(node=node, phase="improve_recipe", attempt=attempt, workspace_host=ws, staging_host=st)

    def commit(clips, node="n1", attempt=2, status="draft"):
        """A stored commit of `clips`, then set to `status`."""
        for c in clips:
            ClipStore(conn).add(_clip(c, node, attempt))
        cid, _ = CommitStore(conn, None, ClipStore(conn)).commit({"d": {"clips": list(clips)}}, "m", node, attempt)
        conn.execute("UPDATE data_commits SET status=? WHERE commit_id=?", (status, cid))
        return cid
    yield tools, caller, commit, conn
    conn.close()


def test_a_clip_is_new_unless_an_accepted_commit_holds_it(env):
    _, _, commit, conn = env
    commit(["kept_here"], attempt=1, status="accepted")                 # an earlier subphase's kept commit
    commit(["kept_elsewhere"], node="n0", status="accepted")            # another node's
    commit(["obsolete_elsewhere"], node="n0", status="obsolete")
    commit(["failed"], attempt=1, status="failed")
    commit(["draft_here"])
    ClipStore(conn).add(_clip("never_committed", attempt=1))
    held = ["kept_here", "kept_elsewhere", "obsolete_elsewhere", "failed", "draft_here", "never_committed"]
    assert CommitStore(conn, None, None).new_clips({"datasets": {"d": {"clips": held}}}) == 4


def test_data_commit_refuses_more_new_clips_than_the_limit_and_stores_nothing(env):
    tools, caller, commit, conn = env
    kept = commit(["k1", "k2"], attempt=1, status="accepted")
    for c in ("a", "b", "c", "d"):
        ClipStore(conn).add(_clip(c))
    rows = conn.execute("SELECT COUNT(*) FROM data_commits").fetchone()[0]
    with pytest.raises(ToolError, match="^4 new clips, more than the 3 a data commit may hold; a clip is new unless "
                                        "a kept data commit holds it$"):
        tools.commit(caller(), {"new": {"weight": 1.0, "clips": ["a", "b", "c", "d"]}}, "too many")
    assert conn.execute("SELECT COUNT(*) FROM data_commits").fetchone()[0] == rows
    out = tools.commit(caller(), {"new": {"weight": 1.0, "clips": ["a", "b", "c"]}}, "kept and three new", include=[kept])
    assert out["new_clips"] == 3 and set(out["datasets"]) == {"d", "new"}


def test_clips_of_a_draft_count_as_new_even_when_included(env):
    tools, caller, commit, conn = env
    draft = commit(["a", "b"])
    for c in ("c", "d"):
        ClipStore(conn).add(_clip(c))
    with pytest.raises(ToolError, match="^4 new clips"):
        tools.commit(caller(), {"more": {"weight": 1.0, "clips": ["c", "d"]}}, "draft and two", include=[draft])
    assert tools.commit(caller(), {"more": {"weight": 1.0, "clips": ["c"]}}, "draft and one", include=[draft])["new_clips"] == 3


class FakeIngestor:
    def __init__(self, cfg, run_dir, conn, leakage):
        self.clips = ClipStore(conn)

    def ingest(self, candidates, node_id, attempt):
        for c in candidates:
            self.clips.add(_clip(c.video.parent.name, node_id, attempt))
        return [IngestResult(accepted=True, clip_id=c.video.parent.name) for c in candidates]


def test_ingest_takes_more_clips_than_the_limit(env, monkeypatch):
    tools, caller, _, _ = env
    monkeypatch.setattr(data_tools, "Ingestor", FakeIngestor)
    tools._leakage = object()
    c = caller()
    names = ["a", "b", "c", "d", "e"]
    for n in names:
        (c.staging_host / n).mkdir()
        for f in ("v.mp4", "a.json", "p.npz"):
            (c.staging_host / n / f).write_text("x")
    cands = [{"video": f"/workspace/staging/{n}/v.mp4", "annotation": f"/workspace/staging/{n}/a.json",
              "pose": f"/workspace/staging/{n}/p.npz", "derived_from": {"kind": "model", "name": "x"}} for n in names]
    assert tools.ingest(c, cands)["accepted"] == 5


@pytest.mark.parametrize("status,error", [
    ("obsolete", "can no longer be used: a result was submitted that did not name it. Its clips are still in the pool"),
    ("failed", "can no longer be used: the work that made it failed")])
def test_obsolete_and_failed_commits_are_refused_everywhere(env, status, error):
    tools, caller, commit, _ = env
    gone = commit(["a"], attempt=1, status=status)
    c = caller()
    store = SnapshotStore(c.workspace_host, None, tools.usable)
    for call in (lambda: tools.commit(c, {"x": {"weight": 1.0, "clips": ["a"]}}, "m", include=[gone]),
                 lambda: tools.query(c, {"data_commit": gone}),
                 lambda: tools.recipe_check(c, {}, gone),
                 lambda: store.take(c, gone, {}, "why")):
        with pytest.raises(ToolError, match=f"^data commit {gone} {error}$"):
            call()


@pytest.mark.parametrize("status", ["draft", "accepted"])
def test_draft_and_accepted_commits_can_be_included_and_read(env, status):
    tools, caller, commit, _ = env
    usable = commit(["a"], node="n0" if status == "accepted" else "n1", status=status)
    c = caller()
    assert tools.query(c, {"data_commit": usable})["total"] == 1
    assert tools.commit(c, {}, "m", include=[usable])["datasets"]["d"]["clips"] == 1


def test_settling_an_unknown_commit_raises(env):
    _, _, _, conn = env
    with pytest.raises(KeyError):
        CommitStore(conn, None, None).settle("n1", 2, "dc99")


def test_only_data_commit_states_the_limit():
    mcp = new_mcp()
    register_data_tools(mcp, ToolKit(None, CFG.repo_root), DataTools(CFG, CFG.repo_root, [0], threading.Lock()))
    register_snapshot_tool(mcp, ToolKit(None, CFG.repo_root), SnapshotStore(CFG.repo_root, None, lambda c: None))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert tools["data_commit"].description == (
        "Create an immutable data commit: the tentative whole training set of this node. It holds the datasets of the "
        "commits in `include` plus the ones in `datasets`. Every node's improve_recipe phase runs in subphases. When a "
        "subphase ends, the data commit of the snapshot its result names is kept, and the subphase's other data commits "
        "can no longer be used. A data commit may hold at most 3 new clips: clips that no kept data commit holds. "
        "Returns the commit id, each dataset's weight and clip counts, and `new_clips`: how many of its clips are new.")
    assert "parent" not in tools["data_commit"].input_schema["properties"]
    assert tools["data_commit"].input_schema["properties"]["include"]["description"].startswith(
        "kept data commits of earlier nodes (the parent's, or any other finished node's) or data commits of this node "
        "that can still be used, whose datasets")
    for name in ("data_ingest", "recipe_check", "snapshot"):
        assert "new clips" not in tools[name].description, name
    assert tools["snapshot"].description == DESCRIPTION and DESCRIPTION.startswith(
        "Store the work as it is now as an immutable snapshot and return its id. A result names the snapshot")


def test_without_a_limit_data_commit_states_none():
    mcp = new_mcp()
    register_data_tools(mcp, ToolKit(None, CFG.repo_root), None)
    text = {t.name: t.description for t in asyncio.run(mcp.list_tools())}["data_commit"]
    assert "new clips" not in text and text.endswith("Returns the commit id, each dataset's weight and clip counts.")
