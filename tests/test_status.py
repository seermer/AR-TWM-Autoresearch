import json

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.guards import alert
from ar_kernel.status import format_status, run_status
from ar_kernel.telemetry.recorder import Recorder


def test_status_is_plain_json(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text(
        "budget: {max_usd: 10, usd_per_mtok: {input: 1, cached_input: 0.1, output: 4}}\n"
        "selection: {decay: 0.5, prior_weight: 1.0, subtree_share: 0.3, size_scale: 4, temperature: 2.0,"
        " noise_floor: 4.4e-4, epsilon: 0.2}\n")
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.78, ["m"], {"m": 0.78})
    nodes.create("n1", "root", 1), nodes.set_status("n1", "invalid_code")
    nodes.set_fields("n1", error="contract import failed")
    nodes.create("n2", "root", 1), nodes.set_status("n2", "interrupted")
    rec.event("llm.response", node="n1", usage={"prompt_tokens": 1000, "completion_tokens": 100}, mock=False)
    alert(rec, "node_failed", "n1 ended invalid_code")
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "run_args.json").write_text('{"max_nodes": 5}')
    d = run_status(tmp_path)
    json.dumps(d)                                               # serializable
    assert d["best"]["node_id"] == "root" and d["max_nodes"] == 5 and d["loop_pid"] is None
    assert {n["node_id"]: n["P"] for n in d["nodes"]} == {"root": 1.0, "n1": None, "n2": None}
    assert d["spend"]["usd"] == (1000 * 1 + 100 * 4) / 1e6
    assert [a["kind"] for a in d["alerts"]] == ["node_failed"]
    text = format_status(d)
    assert "n1" in text and "invalid_code" in text and "contract import failed" in text
    assert "nodes: 1 of 5" in text                              # interrupted n2 is not counted


def test_status_skips_a_torn_trailing_event_line(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text("selection: {decay: 0.5, prior_weight: 1.0, subtree_share: 0.3,"
                                                     " size_scale: 4, temperature: 2.0, noise_floor: 4.4e-4, epsilon: 0.2}\n")
    open_db(tmp_path).close()
    rec = Recorder(tmp_path)
    alert(rec, "node_failed", "n1 ended invalid_code")
    with open(tmp_path / "telemetry" / "events" / "run.jsonl", "a") as f:
        f.write('{"type": "alert", "kind": "dis')                # kill -9 / ENOSPC mid-write
    assert [a["kind"] for a in run_status(tmp_path)["alerts"]] == ["node_failed"]
