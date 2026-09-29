import pytest
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore, run_rel

def test_create_and_read_node(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    node = store.get("root")
    assert node["status"] == "running" and node["parent_id"] is None and node["depth"] == 0

def test_score_roundtrip_preserves_metric_set_order(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    store.record_score("root", 0.8123, ["aesthetic_quality", "hpsv3_quality"],
                       {"aesthetic_quality": 0.80, "hpsv3_quality": 0.82})
    node = store.get("root")
    assert node["score"] == pytest.approx(0.8123)
    assert node["metric_set"] == ["aesthetic_quality", "hpsv3_quality"]
    assert node["metrics"]["hpsv3_quality"] == pytest.approx(0.82)
    assert node["status"] == "scored"

def test_children_and_status_updates(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    store.create("n1", "root", 1)
    store.create("n2", "root", 1)
    store.set_status("n2", "invalid_code")
    assert {n["node_id"] for n in store.all() if n["parent_id"] == "root"} == {"n1", "n2"}
    assert store.get("n2")["status"] == "invalid_code"

def test_unknown_status_is_rejected(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("root", None, 0)
    with pytest.raises(ValueError, match="unknown status"):
        store.set_status("root", "finished")

def test_attempts_are_ordered_per_phase(tmp_path):
    store = NodeStore(open_db(tmp_path))
    store.create("n1", None, 0)
    store.add_attempt("n1", "improve_recipe", 1, "failed", {"reason": "gate"})
    store.add_attempt("n1", "improve_recipe", 2, "ok", {})
    outcomes = [a["outcome"] for a in store.attempts("n1", "improve_recipe")]
    assert outcomes == ["failed", "ok"]


def test_updating_an_unknown_node_raises_instead_of_silently_doing_nothing(tmp_path):
    import pytest
    from ar_kernel.archive.db import open_db
    from ar_kernel.archive.nodes import NodeStore
    store = NodeStore(open_db(tmp_path))
    with pytest.raises(KeyError):
        store.set_status("ghost", "scored")
    with pytest.raises(KeyError):
        store.set_fields("ghost", data_commit="x")


def test_node_id_that_is_unsafe_as_a_path_is_rejected(tmp_path):
    """node_id becomes a directory and a telemetry file name."""
    import pytest
    from ar_kernel.archive.db import open_db
    from ar_kernel.archive.nodes import NodeStore
    store = NodeStore(open_db(tmp_path))
    for bad in ("../escape", "a/b", "", "x" * 80, ".hidden"):
        with pytest.raises(ValueError):
            store.create(bad, None, 0)


def test_new_fields_and_eval_failed(tmp_path):
    nodes = NodeStore(open_db(tmp_path))
    nodes.create("n1", None, 0)
    nodes.set_fields("n1", edit_component="prompts", recipe_path="nodes/n1/recipe.yaml",
                     attempt_counts='{"edit_self": 2}', error="render failed")
    nodes.set_status("n1", "eval_failed")
    got = nodes.get("n1")
    assert (got["edit_component"], got["error"], got["status"]) == ("prompts", "render failed", "eval_failed")


def test_interrupted_is_a_status(tmp_path):
    nodes = NodeStore(open_db(tmp_path))
    nodes.create("n1", None, 0)
    nodes.set_status("n1", "interrupted")
    assert nodes.get("n1")["status"] == "interrupted"


def test_run_relative_paths(tmp_path):
    p = tmp_path / "nodes" / "n1" / "train" / "checkpoint-2"
    assert run_rel(tmp_path, p) == "nodes/n1/train/checkpoint-2"
