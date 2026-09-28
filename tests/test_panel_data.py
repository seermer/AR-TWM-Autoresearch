import json
import os

import pytest

from fixtures.panel_run import make_run
from panel.runfiles import RunFiles, fmt_ts, loads


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path)


def test_runfiles_needs_a_run_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunFiles(tmp_path)


def test_paths_outside_the_run_are_refused(run_dir, tmp_path):
    files = RunFiles(run_dir)
    for bad in ("../x", "/etc/passwd", "nodes/../../x"):
        with pytest.raises(ValueError):
            files.path(bad)
    (run_dir / "out").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        files.path("out/seed")
    assert files.path("nodes/n1") == run_dir.resolve() / "nodes" / "n1"


def test_query_reads_the_archive_without_side_files(run_dir):
    files = RunFiles(run_dir)
    assert not (run_dir / "archive.db-wal").exists()
    ids = [r["node_id"] for r in files.query("SELECT node_id FROM nodes ORDER BY created_at")]
    assert ids == ["root", "n1", "n2"]
    assert not (run_dir / "archive.db-wal").exists() and not (run_dir / "archive.db-shm").exists()


def test_query_without_a_database_is_empty(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "run.json").write_text("{}")
    assert RunFiles(tmp_path).query("SELECT * FROM nodes") == []


def test_git_reads_and_misses(run_dir, tmp_path):
    files = RunFiles(run_dir)
    assert "X = 3" in files.git("show", "refs/heads/node/n1:agent/entry.py")
    assert files.git("show", "no-such-ref:agent/entry.py") is None
    (tmp_path / "bare" / "config").mkdir(parents=True)
    (tmp_path / "bare" / "config" / "run.json").write_text("{}")
    assert RunFiles(tmp_path / "bare").git("log") is None


def test_read_text_limits_and_json(run_dir):
    files = RunFiles(run_dir)
    big = run_dir / "big.log"
    big.write_bytes(b"A" * 300_000 + b"MIDDLE" + b"B" * 300_000)
    text = files.read_text("big.log", limit=100_000)
    assert text.startswith("A") and text.endswith("B") and "MIDDLE" not in text and "not shown" in text
    assert files.read_json("control/run_args.json") == {"max_nodes": 3}
    assert files.read_json("nope.json") is None and files.read_text("nope.txt") is None


def test_host_path_maps_container_paths(run_dir):
    files = RunFiles(run_dir)
    assert files.host_path("n1", "improve_recipe", 1, "/workspace/staging/work/v1.mp4") == \
        "staging/n1/improve_recipe-1/work/v1.mp4"
    assert files.host_path("n1", "improve_recipe", 1, "/workspace/tool_output/a.log") == \
        "nodes/n1/attempts/improve_recipe-1/workspace/tool_output/a.log"
    assert files.host_path("n1", "edit_self", 2, "/agent/agent/entry.py") == \
        "nodes/n1/attempts/edit_self-2/agent/agent/entry.py"
    assert files.host_path("n1", "edit_self", 2, "/etc/passwd") is None


def test_loop_pid_needs_a_matching_start_time(run_dir):
    files = RunFiles(run_dir)
    assert files.loop_pid() is None
    started = open(f"/proc/{os.getpid()}/stat").read().rsplit(")", 1)[1].split()[19]
    (run_dir / "control" / "loop.pid").write_text(f"{os.getpid()} {started}")
    assert files.loop_pid() == os.getpid()
    (run_dir / "control" / "loop.pid").write_text(f"{os.getpid()} 1")
    assert files.loop_pid() is None


def test_helpers():
    assert loads('{"a": 1}') == {"a": 1} and loads(None, []) == [] and loads("{bad", {}) == {}
    assert len(fmt_ts(0)) == 19

from panel.events import EventLog, event_row, filter_events


def test_event_log_reads_incrementally_and_waits_for_torn_lines(run_dir):
    log = EventLog(RunFiles(run_dir))
    before = len(log.events())
    path = run_dir / "telemetry" / "events" / "n1.jsonl"
    line = json.dumps({"ts_wall": 9e9, "type": "x.y", "node": "n1"})
    with path.open("a") as f:
        f.write(line[:10])                                  # a half-written line
    assert len(log.events()) == before
    with path.open("a") as f:
        f.write(line[10:] + "\n")
    events = log.events()
    assert len(events) == before + 1 and events[-1]["type"] == "x.y" and events[-1]["_file"] == "n1"


def test_event_log_finds_new_files_and_hides_gpu(run_dir):
    log = EventLog(RunFiles(run_dir))
    assert not any(e["_file"] == "gpu" for e in log.events())
    assert any(e["type"] == "gpu.sample" for e in log.events(include_gpu=True))
    (run_dir / "telemetry" / "events" / "n9.jsonl").write_text(json.dumps({"ts_wall": 1.0, "type": "a.b"}) + "\n")
    assert [e["type"] for e in log.events(files=["n9"])] == ["a.b"]


def test_payloads_load_and_cache(run_dir):
    log = EventLog(RunFiles(run_dir))
    req = next(e for e in log.events() if e["type"] == "llm.request")
    body = log.payload(req["payload"])
    assert body["body"]["messages"][0]["role"] == "system"
    assert log.payload(req["payload"]) is body                    # served from the cache


def test_missing_payload_is_none(run_dir):
    log = EventLog(RunFiles(run_dir))
    assert log.payload(None) is None and log.payload("0" * 64) is None
    req = next(e for e in log.events() if e["type"] == "llm.request")
    (run_dir / "telemetry" / "payloads" / f"{req['payload']}.json.zst").write_bytes(b"not zstd")
    assert EventLog(RunFiles(run_dir)).payload(req["payload"]) is None


def test_filter_and_rows(run_dir):
    events = EventLog(RunFiles(run_dir)).events()
    llm = filter_events(events, node="n1", types=["llm.request"])
    assert llm and all(e["type"] == "llm.request" for e in llm)
    assert filter_events(events, text="quiet 30") and not filter_events(events, text="nothing-like-this")
    assert filter_events(events, phase="improve_recipe", attempt=1, component="tools")
    row = event_row(llm[0])
    assert row["type"] == "llm.request" and "model=m" in row["summary"] and isinstance(row["seq"], int)
