import math

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.selection import candidates, select_parent, selection_seed, update_values
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig(raw={"selection": {"decay": 0.5, "prior_weight": 1.0, "subtree_share": 0.3,
                                      "size_scale": 4, "temperature": 2.0, "noise_floor": 4.4e-4,
                                      "epsilon": 0.2}}, repo_root=None)


def node(nid, parent, status="scored", score=None):
    return {"node_id": nid, "parent_id": parent, "status": status, "score": score}


def by_id(rows):
    return {r["node_id"]: r for r in rows}


def roots(*scores):                       # independent candidates with no descendants
    return [node(f"r{i}", None, score=s) for i, s in enumerate(scores)]


def test_single_candidate_is_chosen_with_certainty():
    assert [(r["node_id"], r["P"]) for r in candidates([node("root", None, score=0.78)], CFG)] == [("root", 1.0)]


def test_probabilities_sum_to_one():
    tree = [node("root", None, score=0.78), node("a", "root", score=0.80), node("b", "root", score=0.70),
            node("c", "a", score=0.82)]
    assert math.isclose(sum(r["P"] for r in candidates(tree, CFG)), 1.0)


def test_worked_example_is_not_too_sharp():
    ps = [r["P"] for r in candidates(roots(0.78, 0.79, 0.80, 0.83), CFG)]
    assert ps == pytest.approx([0.152, 0.184, 0.225, 0.440], abs=2e-3)


def test_probabilities_are_continuous_in_scores():
    before = candidates(roots(0.78, 0.79, 0.80, 0.83), CFG)[2]["P"]
    after = candidates(roots(0.78, 0.79, 0.80001, 0.83), CFG)[2]["P"]
    assert 0 < after - before < 1e-3


def test_values_within_the_noise_floor_are_near_uniform():
    ps = [r["P"] for r in candidates(roots(0.7500, 0.7502, 0.7503), CFG)]
    assert max(ps) / min(ps) < 1.5


def test_one_outlier_child_is_not_catastrophic():
    rows = by_id(candidates([node("a", None, score=0.80), node("a1", "a", score=0.40)], CFG))
    assert rows["a"]["value"] == pytest.approx(0.7 * 0.80 + 0.3 * (0.80 + 0.5 * 0.40) / 1.5)   # 0.76


def test_own_score_matters_more_than_the_subtree():
    tree = [node("a", None, score=0.80), node("a1", "a", score=0.70),
            node("b", None, score=0.76), node("b1", "b", score=0.86)]
    rows = by_id(candidates(tree, CFG))
    assert rows["a"]["value"] > rows["b"]["value"]


def test_large_subtree_is_down_weighted():
    tree = [node("a", None, score=0.8), node("b", None, score=0.8),
            *[node(f"a{i}", "a", score=0.8) for i in range(4)]]
    rows = by_id(candidates(tree, CFG))
    assert rows["a"]["value"] == pytest.approx(rows["b"]["value"])
    assert rows["a"]["w"] / rows["b"]["w"] == pytest.approx(0.5)              # size 4, scale 4


def test_failed_children_count_in_size_but_not_the_mean_and_interrupted_counts_nowhere():
    tree = [node("a", None, score=0.8), node("a1", "a", status="invalid_code"),
            node("a2", "a", status="crashed"), node("a3", "a", status="interrupted")]
    (row,) = candidates(tree, CFG)
    assert row["value"] == pytest.approx(0.8) and row["size"] == 2 and row["penalty"] == pytest.approx(2 / 3)


def test_select_parent_is_seeded_and_recorded(tmp_path):
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    for nid, parent, score in (("root", None, 0.7), ("n1", "root", 0.8), ("n2", "root", 0.6)):
        nodes.create(nid, parent, 0 if parent is None else 1)
        nodes.record_score(nid, score, ["m"], {"m": score})
    seed = selection_seed("run1", "n3")
    first = select_parent(conn, CFG, "n3", seed, rec)
    assert all(select_parent(conn, CFG, "n3", seed, rec) == first for _ in range(5))
    row = conn.execute("SELECT * FROM selection_events ORDER BY id LIMIT 1").fetchone()
    assert (row["child_id"], row["chosen"], row["seed"]) == ("n3", first, seed)
    assert [e["type"] for e in rec.read_events()].count("select") == 6


def test_update_values_writes_the_value(tmp_path):
    conn = open_db(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.7, ["m"], {"m": 0.7})
    nodes.create("n1", "root", 1), nodes.record_score("n1", 0.9, ["m"], {"m": 0.9})
    update_values(conn, CFG)
    assert nodes.get("root")["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.45) / 1.5)
    assert nodes.get("n1")["subtree_value"] == pytest.approx(0.9)


def test_descendants_of_an_interrupted_node_count_nowhere():
    tree = [node("a", None, score=0.8), node("a1", "a", status="interrupted"),
            node("a11", "a1", score=0.2), node("a12", "a1", status="crashed")]
    row = by_id(candidates(tree, CFG))["a"]
    assert row["value"] == pytest.approx(0.8) and row["size"] == 0


# Simulation behind these tests (2026-09-28, the shipped `candidates`, 6-node chain): with pure-noise
# scores (SD 0.008) the newest node holds P = 0.206 against a uniform 0.167, because the size penalty
# gives the newest leaf weight 1; a node that is truly +0.02 better receives P = 0.35. The acceptance
# run's "always the newest" picks were luck (joint probability about 0.09).
import random
from collections import Counter


def chain(scores):
    return [node(f"n{i}", None if i == 0 else f"n{i-1}", score=s) for i, s in enumerate(scores)]


def test_truly_better_node_beats_uniform_in_a_chain():
    ps = [r["P"] for r in candidates(chain([0.787, 0.807, 0.787, 0.787, 0.787, 0.787]), CFG)]
    assert ps[1] == max(ps) and ps[1] > 2 * (1 / 6)


def test_pure_noise_does_not_strongly_prefer_the_newest():
    rng, newest = random.Random(0), 0.0
    for _ in range(500):
        newest += candidates(chain([0.79 + rng.gauss(0, 0.008) for _ in range(6)]), CFG)[-1]["P"]
    assert newest / 500 < 0.24                  # measured 0.206; uniform would be 0.167


def test_draw_frequencies_follow_the_recorded_probabilities(tmp_path):
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    for i, score in enumerate((0.7871, 0.7945, 0.7917, 0.7907)):
        nid = "root" if i == 0 else f"n{i}"
        nodes.create(nid, None if i == 0 else ("root" if i == 1 else f"n{i-1}"), i)
        nodes.record_score(nid, score, ["m"], {"m": score})
    expected = {r["node_id"]: r["P"] for r in candidates(NodeStore(conn).all(), CFG)}
    picks = Counter(select_parent(conn, CFG, f"c{seed}", seed, rec) for seed in range(4000))
    for nid, p in expected.items():
        assert abs(picks[nid] / 4000 - p) < 0.03


def test_all_scores_equal_gives_a_valid_distribution():
    ps = [r["P"] for r in candidates(chain([0.79] * 6), CFG)]
    assert math.isclose(sum(ps), 1.0) and min(ps) > 0
