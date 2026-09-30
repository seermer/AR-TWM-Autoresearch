import json

import pytest
import yaml

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.context_bundle import (FORMAT_RULES, archive_summary, build_edit_context,
                                      build_recipe_context, lineage, write_bundle)
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
# The shape score.aggregates() returns and Plan 4 writes to eval/aggregates.json.
AGGREGATES = {"metrics": {"aesthetic_quality": 0.78}, "dimensions": {"quality": 0.78},
              "strata": {"interaction_type": {"navigation": 0.8}, "category": {"Nature": 0.8},
                         "perspective": {"first_person": 0.8}}}


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
    nodes.record_score("root", 0.78, ["aesthetic_quality"], {"aesthetic_quality": 0.78})
    nodes.create("n1", "root", 1)
    (tmp_path / "nodes" / "root" / "eval").mkdir(parents=True)
    (tmp_path / "nodes" / "root" / "eval" / "aggregates.json").write_text(
        json.dumps(AGGREGATES))
    (tmp_path / "nodes" / "root" / "rationale.md").write_text("released checkpoint")
    (tmp_path / "nodes" / "root" / "edit.json").write_text(json.dumps({"summary": "s", "component": "tools"}))
    return conn, repo, tmp_path


def test_lineage_is_root_first_and_carries_artifacts(world):
    conn, repo, run = world
    lin = lineage(conn, run, repo, "root")
    assert [e["node_id"] for e in lin] == ["root"]
    assert lin[0]["score"] == 0.78 and lin[0]["aggregates"] == AGGREGATES
    assert lin[0]["rationale"] == "released checkpoint"
    assert lin[0]["edit"] == {"summary": "s", "component": "tools"}


def test_archive_summary_names_the_best_node(world):
    conn, _, _ = world
    s = archive_summary(conn)
    assert s["best"] == {"node_id": "root", "score": 0.78} and s["n_scored"] == 1


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
    assert ctx.resolution_allowlist == [[416, 736], [352, 608]]
    assert ctx.base_recipe["optimizer"]["batch_size"] == 1
    assert "57 frames" in ctx.format_rules and ctx.format_rules == FORMAT_RULES
    write_bundle(ctx, tmp_path / "ctx2")
    assert json.loads((tmp_path / "ctx2" / "retry.json").read_text()) == retry


def test_lineage_and_archive_summary_carry_component_and_error(world):
    conn, repo, run = world
    nodes = NodeStore(conn)
    nodes.set_status("n1", "invalid_code")
    nodes.set_fields("n1", edit_component="tools", error="contract import failed")
    lin = lineage(conn, run, repo, "n1")
    assert lin[-1]["component"] == "tools" and lin[-1]["error"] == "contract import failed"
    summary_nodes = {n["node_id"]: n for n in archive_summary(conn)["nodes"]}
    assert summary_nodes["n1"]["error"] == "contract import failed"
    assert summary_nodes["root"]["error"] is None


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


def test_lineage_entries_carry_the_process_digest(world):
    conn, repo, run = world
    assert lineage(conn, run, repo, "n1")[-1]["process"] == {}        # no events: empty, not an error
