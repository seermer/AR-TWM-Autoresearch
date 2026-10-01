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


def test_alert_timestamp_in_format_status(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text(
        "budget: {max_usd: 10, usd_per_mtok: {input: 1, cached_input: 0.1, output: 4}}\n"
        "selection: {decay: 0.5, prior_weight: 1.0, subtree_share: 0.3, size_scale: 4, temperature: 2.0,"
        " noise_floor: 4.4e-4, epsilon: 0.2}\n")
    conn, rec = open_db(tmp_path), Recorder(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.78, ["m"], {"m": 0.78})
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "run_args.json").write_text('{"max_nodes": 5}')
    alert(rec, "root_failed", "RuntimeError: something went wrong")
    d = run_status(tmp_path)
    text = format_status(d)
    # The timestamp should appear in the alert line
    assert "alerts (latest last):" in text
    alert_lines = [line for line in text.split("\n") if "[" in line and "]" in line and "failed" in line]
    assert len(alert_lines) > 0
    # Check that the alert line contains a timestamp in the format YYYY-MM-DD HH:MM:SS
    for alert_line in alert_lines:
        assert any(c.isdigit() for c in alert_line[:19]), f"Timestamp not found in alert line: {alert_line}"


def test_status_writes_nothing_into_the_run(tmp_path):
    """Also with a non-empty -wal, as a killed loop leaves it."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "kernel.yaml").write_text(
        "selection: {decay: 0.5, prior_weight: 1.0, subtree_share: 0.3, size_scale: 4, temperature: 2.0,"
        " noise_floor: 4.4e-4, epsilon: 0.2}\n")
    conn = open_db(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.record_score("root", 0.78, ["m"], {"m": 0.78})
    killed = {p.name: p.read_bytes() for p in tmp_path.glob("archive.db*")}
    conn.close()
    for name, data in killed.items():
        (tmp_path / name).write_bytes(data)
    assert (tmp_path / "archive.db-wal").stat().st_size > 0

    def tree():
        return {str(p.relative_to(tmp_path)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    before = tree()
    assert run_status(tmp_path)["best"]["node_id"] == "root"
    assert tree() == before
    open_db(tmp_path).close()                                       # a cleanly closed archive: no -wal
    assert not (tmp_path / "archive.db-wal").exists()
    before = tree()
    assert run_status(tmp_path)["best"]["node_id"] == "root" and tree() == before
