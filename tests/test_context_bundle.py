import json

import pytest
import yaml

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.context_bundle import (archive_summary, build_edit_context, build_recipe_context, lineage,
                                      siblings, write_bundle)
from ar_kernel.eval.score import agent_aggregates
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
METRIC_SET = ["aesthetic_quality", "event_edit_adherence", "causal_fidelity"]
# The shape score.aggregates() returns and the loop writes to eval/aggregates.json.
AGGREGATES = {"metrics": {"aesthetic_quality": 0.78}, "dimensions": {"quality": 0.78},
              "strata": {"interaction_type": {"navigation": {"quality": 0.8}}, "category": {"Nature": {"quality": 0.8}},
                         "perspective": {"first_person": {"quality": 0.8}}}}


@pytest.fixture
def world(tmp_path):
    seed = tmp_path / "seed" / "agent"
    seed.mkdir(parents=True)
    (seed / "entry.py").write_text("x = 1\n")
    repo = AgentsRepo(tmp_path / "agents.git")
    root_commit = repo.init(tmp_path / "seed")
    conn = open_db(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root_commit)
    nodes.record_score("root", 0.78, METRIC_SET, {"aesthetic_quality": 0.78})
    nodes.create("n1", "root", 1)
    (tmp_path / "nodes" / "root" / "eval").mkdir(parents=True)
    (tmp_path / "nodes" / "root" / "eval" / "aggregates.json").write_text(
        json.dumps(AGGREGATES))
    (tmp_path / "nodes" / "root" / "rationale.md").write_text("released checkpoint")
    (tmp_path / "nodes" / "root" / "edit.json").write_text(json.dumps({"summary": "s"}))
    return conn, repo, tmp_path


def test_edit_context_validates_and_round_trips(world, tmp_path):
    conn, repo, run = world
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=1,
                             max_attempts=3, retry=None, nodes_remaining=9)
    path = write_bundle(ctx, tmp_path / "ctx")
    loaded = json.loads(path.read_text())
    assert loaded["nodes_remaining"] == 9 and loaded["lineage"][0]["node_id"] == "root"
    assert not (tmp_path / "ctx" / "retry.json").exists()


def test_recipe_context_carries_rules_allowlists_and_retry(world, tmp_path):
    conn, repo, run = world
    retry = {"attempt": 1, "failures": ["dataset cam has 2 clips, fewer than 4 GPUs"]}
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1",
                               parent_id="root", attempt=2, max_attempts=3, retry=retry,
                               nodes_remaining=9, n_gpus=4, tools=["data_query", "data_commit"])
    assert "optimizer.max_steps" in ctx.tunable_rules
    assert ctx.resolution_allowlist == [[544, 960], [416, 736]]
    assert ctx.base_recipe["optimizer"]["batch_size"] == 1
    assert not hasattr(ctx, "format_rules")
    assert ctx.metric_guide["cause_and_effect"]["weight"] == 4.5 and len(ctx.metric_guide) == len(METRIC_SET)
    write_bundle(ctx, tmp_path / "ctx2")
    assert json.loads((tmp_path / "ctx2" / "retry.json").read_text()) == retry


def test_parent_recipe_is_read_from_the_node_artifact(world):
    conn, repo, run = world
    (run / "nodes" / "root" / "recipe.yaml").write_text(yaml.safe_dump({"optimizer.lr": 2e-5}))
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1",
                               parent_id="root", attempt=1, max_attempts=3, retry=None,
                               nodes_remaining=1, n_gpus=4, tools=[])
    assert ctx.parent_recipe == {"optimizer.lr": 2e-5}


def test_recipe_context_carries_the_guide(world):
    conn, repo, run = world
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1", parent_id="root", attempt=1,
                               max_attempts=3, retry=None, nodes_remaining=1, n_gpus=4, tools=[])
    assert set(ctx.recipe_guide) == set(ctx.tunable_rules)


def test_siblings_are_the_parents_finished_children(world):
    from ar_kernel.context_bundle import siblings
    conn, repo, run = world
    nodes = NodeStore(conn)                                    # n1 is the node being built (running)
    nodes.create("n2", "root", 1), nodes.record_score("n2", 0.8, ["aesthetic_quality"], {"aesthetic_quality": 0.8})
    nodes.create("n3", "root", 1), nodes.set_status("n3", "train_failed")
    nodes.create("n4", "root", 1), nodes.set_status("n4", "interrupted")
    nodes.create("n5", "n2", 2), nodes.set_status("n5", "scored")
    assert [(s["node_id"], s["status"], s["score"]) for s in siblings(conn, run, repo, "root", "improve_recipe")] == [
        ("n2", "scored", 0.8), ("n3", "train_failed", None)]
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=1, max_attempts=3,
                             retry=None, nodes_remaining=3)
    assert [s["node_id"] for s in ctx.siblings] == ["n2", "n3"]
    assert sorted(ctx.siblings[0]) == sorted(ctx.lineage[0])
    assert [n["node_id"] for n in ctx.lineage] == ["root"]


def test_every_finished_node_is_mounted_but_not_the_one_being_built(world):
    from ar_kernel.agent_phase import finished_node_dirs
    conn, _, run = world
    nodes = NodeStore(conn)
    nodes.create("n2", "root", 1), nodes.set_status("n2", "eval_failed")
    nodes.create("n3", "n2", 2), nodes.set_status("n3", "interrupted")
    for node in ("n1", "n2", "n3"):
        (run / "nodes" / node).mkdir(parents=True)
    assert finished_node_dirs(conn, run) == {"root": run / "nodes" / "root", "n2": run / "nodes" / "n2"}


def test_a_data_entry_carries_scores_under_agent_names_and_no_code_or_process(world):
    conn, repo, run = world
    [entry] = lineage(conn, run, repo, "root", "improve_recipe")
    assert sorted(entry) == ["aggregates", "data", "data_commit", "error", "metrics", "node_id", "rationale", "recipe",
                             "score", "status"]
    assert entry["score"] == 0.78 and entry["metrics"] == {"frame_aesthetics": 0.78}
    assert entry["aggregates"] == agent_aggregates(AGGREGATES) and "instruction_kind" in entry["aggregates"]["groups"]
    assert entry["rationale"] == "released checkpoint"
    assert json.loads((run / "nodes" / "root" / "eval" / "aggregates.json").read_text()) == AGGREGATES   # disk unchanged


def test_an_edit_entry_carries_code_and_process_and_no_score(world):
    conn, repo, run = world
    NodeStore(conn).add_attempt("root", "edit_self", 1, "contract_failed", {})
    [entry] = lineage(conn, run, repo, "root", "edit_self")
    assert sorted(entry) == ["code_diff_stats", "edit", "error", "node_id", "process", "status"]
    assert entry["edit"] == {"summary": "s"}
    assert entry["process"] == {"attempts": [{"phase": "edit_self", "attempt": 1, "outcome": "contract_failed"}]}
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=1, max_attempts=3,
                             retry=None, nodes_remaining=9)
    assert "0.78" not in ctx.model_dump_json()                  # no score anywhere in an edit context
    assert ctx.archive == {"nodes": [{"node_id": "root", "parent_id": None, "status": "scored", "depth": 0},
                                     {"node_id": "n1", "parent_id": "root", "status": "running", "depth": 1}]}


def test_archive_summary_names_the_best_node(world):
    conn, _, _ = world
    s = archive_summary(conn, "improve_recipe")
    assert s["best"] == {"node_id": "root", "score": 0.78} and s["n_scored"] == 1
    assert "error" not in s["nodes"][0]


def test_kernel_failures_show_a_fixed_error_and_agent_failures_their_own(world):
    conn, repo, run = world
    nodes = NodeStore(conn)
    nodes.set_status("n1", "invalid_code")
    nodes.set_fields("n1", error="contract import failed")
    assert lineage(conn, run, repo, "n1", "edit_self")[-1]["error"] == "contract import failed"
    nodes.set_status("n1", "eval_failed")
    nodes.set_fields("n1", error="RuntimeError: wbench gpu failed (rc=1)")
    for phase in ("edit_self", "improve_recipe"):
        entry = lineage(conn, run, repo, "n1", phase)[-1]
        assert entry["error"] == "the kernel's evaluation of this node failed"
        if phase == "improve_recipe":                          # failed before scoring: nothing to alias
            assert entry["aggregates"] is None and entry["metrics"] == {}


def test_a_quarantined_node_is_in_no_context_and_its_clips_leave_the_pool(world):
    from ar_kernel.archive.clips import ClipStore
    from ar_kernel.tools.data_tools import pool_clips
    conn, repo, run = world
    nodes = NodeStore(conn)
    nodes.create("n2", "root", 1), nodes.set_status("n2", "quarantined")
    clip = {"clip_id": "a" * 64, "video_digest": "v", "caption_digest": "c", "pose_digest": None,
            "camera_motion": "static", "metadata": {}, "formats": [], "warnings": [],
            "provenance": {"kind": "derived"}, "license": None, "derived_from": []}
    ClipStore(conn).add({**clip, "ingested_by": "n2"})
    ClipStore(conn).add({**clip, "clip_id": "b" * 64, "ingested_by": "n1"})
    assert [c["clip_id"] for c in pool_clips(conn)] == ["b" * 64]
    for phase in ("edit_self", "improve_recipe"):
        assert siblings(conn, run, repo, "root", phase) == []
        assert "n2" not in [n["node_id"] for n in archive_summary(conn, phase)["nodes"]]
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1", parent_id="root",
                               attempt=1, max_attempts=3, retry=None, nodes_remaining=1, n_gpus=4, tools=[])
    assert ctx.clip_pool_size == 1 and not hasattr(ctx, "clip_pool")      # clips are listed by data_query
    assert ctx.clip_pool_stats == {"by_node": {"n1": 1}, "by_source": {"derived": 1}, "by_format": {}}
    assert "/workspace/staging" in ctx.folders and "separate mount" in ctx.folders["/workspace/staging"]
    nodes = NodeStore(conn)
    nodes.create("n9", "root", 1), nodes.set_status("n9", "interrupted")
    for phase in ("edit_self", "improve_recipe"):          # live-10-02 listed the interrupted n1
        assert "n9" not in [n["node_id"] for n in archive_summary(conn, phase)["nodes"]]


def test_the_written_bundle_has_blocked_names_replaced(world, tmp_path):
    conn, repo, run = world
    retry = {"kind": "train", "log_tail": "Traceback ... scripts/tools/run_wbench.py line 3"}
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=2, max_attempts=3,
                             retry=retry, nodes_remaining=9)
    write_bundle(ctx, tmp_path / "ctx", ["wbench"])
    for name in ("context.json", "retry.json"):
        text = (tmp_path / "ctx" / name).read_text()
        assert "wbench" not in text.lower() and "run_render.py" in text
