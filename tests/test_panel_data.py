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


def _ev(ts, kind, node="n1"):
    return json.dumps({"ts_wall": ts, "type": kind, "node": node}) + "\n"


@pytest.mark.parametrize("how", ["replace", "truncate", "rewrite_longer", "delete"])
def test_event_log_rereads_a_file_that_was_rewritten_under_it(run_dir, how):
    """reset_run_to_root.py deletes a node's file and rewrites run.jsonl while the panel runs; the
    panel kept the old events and read the new file from the old byte offset (two edit_planner
    conversations for one n1)."""
    ev = run_dir / "telemetry" / "events"
    log = EventLog(RunFiles(run_dir))
    (ev / "zz.jsonl").write_text(_ev(1e9, "old.a", "zz") + _ev(1e9 + 1, "old.b", "zz"))
    assert [e["type"] for e in log.events(files=["zz"])] == ["old.a", "old.b"]
    if how == "replace":                                  # a new file (new inode) under the same name
        (ev / "zz.tmp").write_text(_ev(2e9, "new.a", "zz"))
        (ev / "zz.tmp").replace(ev / "zz.jsonl")
        expected = ["new.a"]
    elif how == "truncate":                               # same inode, shorter
        (ev / "zz.jsonl").write_text(_ev(2e9, "new.a", "zz"))
        expected = ["new.a"]
    elif how == "rewrite_longer":                         # same inode, grown past the old offset
        (ev / "zz.jsonl").write_text(_ev(2e9, "new.first", "zz") + _ev(2e9 + 1, "new.second", "zz")
                                     + _ev(2e9 + 2, "new.third", "zz"))
        expected = ["new.first", "new.second", "new.third"]
    else:
        (ev / "zz.jsonl").unlink()
        expected = []
    assert [e["type"] for e in log.events(files=["zz"])] == expected
    assert all(log.by_seq(e["_seq"]) is e for e in log.events(files=["zz"]))


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

from fixtures.panel_run import BUILDER, COMPACT, PLANNER, SUMMARY
from panel.chat import chains, chat_items, infer_role, segments


def _chat(run_dir):
    log = EventLog(RunFiles(run_dir))
    segs = segments(log.events(), "n1", "improve_recipe", 1)
    return log, segs, chains(segs, log.payload)


def test_segments_group_calls_by_conversation(run_dir):
    _, segs, _ = _chat(run_dir)
    assert [s.conversation_id for s in segs] == ["p1", "b1", "b2"]
    assert [len(s.calls) for s in segs] == [2, 3, 2]
    assert segs[2].calls[-1].response is None


def test_compaction_is_stitched_across_conversation_ids(run_dir):
    _, _, chs = _chat(run_dir)
    assert [[s.conversation_id for s in c] for c in chs] == [["p1"], ["b1", "b2"]]


def test_chat_shows_compaction_as_a_turn_then_continues(run_dir):
    log, _, chs = _chat(run_dir)
    items = chat_items(chs[1], log.payload, awaiting=False)
    kinds = [i["kind"] for i in items]
    req = kinds.index("compaction_request")
    assert items[req]["text"] == COMPACT and items[req]["title"].startswith("Compaction #1: ~900")
    summaries = [i for i in items if i["kind"] == "compaction_summary"]
    assert len(summaries) == 2 and "retry without tools" in summaries[1]["title"]
    assert summaries[1]["text"] == SUMMARY.strip()
    assert "(no text" in summaries[0]["text"]
    ctx = kinds.index("context")
    assert ctx > req and SUMMARY.strip() in items[ctx]["text"]
    assert kinds.count("system") == 1                                  # system prompt shown once
    assert items[-1] == {"kind": "note", "title": None, "text": "no response recorded"}
    assert chat_items(chs[1], log.payload, awaiting=True)[-1]["text"] == "awaiting response"


def test_chat_items_cover_reasoning_tools_and_usage(run_dir):
    log, _, chs = _chat(run_dir)
    items = chat_items(chs[0], log.payload, awaiting=False)
    kinds = [i["kind"] for i in items]
    assert kinds[:3] == ["system", "tools", "user"]
    assert "reasoning" in kinds and "tool_call" in kinds and "tool_output" in kinds
    call = next(i for i in items if i["kind"] == "tool_call")
    assert call["title"] == "Tool call: list_dir" and '"path": "."' in call["text"]
    out = next(i for i in items if i["kind"] == "tool_output")
    assert out["title"] == "Tool output: list_dir" and out["text"] == "a\nb"
    assert items[-2]["kind"] == "assistant" and items[-2]["text"] == "plan done"
    assert items[-1]["kind"] == "usage" and "100 in / 10 out" in items[-1]["text"]


@pytest.mark.parametrize("message,expected", [
    ({"reasoning": "main"}, "main"), ({"reasoning_content": "backup"}, "backup"),
    ({"reasoning": "", "reasoning_content": "backup"}, "backup"),
    ({"reasoning": "main", "reasoning_content": "other"}, "main"), ({}, None)])
def test_reasoning_is_shown_from_reasoning_then_reasoning_content(message, expected):
    from panel.chat import message_items
    items = message_items({"role": "assistant", "content": "hi", **message}, {})
    shown = [i["text"] for i in items if i["kind"] == "reasoning"]
    assert shown == ([expected] if expected else [])


def test_stitching_ignores_whitespace_and_unrelated_conversations():
    from panel.chat import Call, Segment
    payloads = {
        "a_req": {"body": {"messages": [{"role": "user", "content": "task A"}]}},
        "a_res": {"body": {"choices": [{"message": {"role": "assistant", "content": "\n  the summary \n"}}]}},
        "b_req": {"body": {"messages": [{"role": "user", "content": "Summary:\nthe summary\nContinue."}]}},
        "b_res": {"body": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}},
        "c_req": {"body": {"messages": [{"role": "user", "content": "unrelated"}]}},
    }
    seg = lambda cid, ts, req, res: Segment(cid, [Call(cid + "1", cid, 0, ts, req, res)])
    a, b, c = seg("A", 1.0, "a_req", "a_res"), seg("B", 2.0, "b_req", "b_res"), seg("C", 3.0, "c_req", None)
    got = chains([a, b, c], payloads.get)
    assert [[s.conversation_id for s in ch] for ch in got] == [["A", "B"], ["C"]]
    early = seg("E", 0.5, "b_req", "b_res")                             # starts before A ended
    assert [[s.conversation_id for s in ch] for ch in chains([early, a], payloads.get)] == [["E"], ["A"]]


def test_roles_are_inferred_from_the_prompt_files():
    files = {"planner": PLANNER, "data_builder": BUILDER}
    assert infer_role(BUILDER + "\n\n# Reference: x.md\n\nref", files) == "data_builder"
    assert infer_role(PLANNER, files) == "planner"
    assert infer_role("A new role I invented.\nMore.", files) == "A new role I invented."
    assert infer_role(None, files) == "(no system prompt)"
    assert infer_role("x", {}) == "x"

from panel import views


@pytest.fixture
def run(run_dir):
    return views.Run(run_dir)


def test_overview(run):
    ov = views.overview(run)
    assert ov["loop"]["alive"] is False and ov["loop"]["state_label"] == "last recorded"
    assert ov["loop"]["max_nodes"] == 3
    rows = {r["node"]: r for r in ov["nodes"]}
    assert rows["n1"]["vs parent"] == pytest.approx(0.05) and rows["n1"]["duration_min"] == 10.0
    assert rows["root"]["parent"] == "" and rows["n2"]["status"] == "running"
    assert ov["spend"]["calls"] == 6 and ov["spend"]["usd"] == pytest.approx(0.006)
    assert ov["alerts"][0]["kind"] == "stall" and [s["node"] for s in ov["scores"]] == ["root", "n1"]
    assert len(ov["recent"]) <= 50 and ov["disk_free_gb"] > 0


def test_gpu_series(run):
    series = views.gpu_series(run, hours=1e6)
    assert {s["gpu"] for s in series} == {"0", "1"} and series[0]["memory_gib"] == 2.0


def test_trace_filters_pages_and_detail(run):
    choices = views.trace_choices(run)
    assert "llm.request" in choices["type"] and "n1" in choices["node"]
    got = views.trace(run, node="n1", phase=None, attempt=None, types=["llm.request"], component=None,
                      text=None, include_gpu=False, page=0, page_size=2)
    assert got["total"] == 7 and got["pages"] == 4 and len(got["rows"]) == 2
    detail = views.trace_detail(run, got["rows"][0]["seq"])
    assert detail["event"]["type"] == "llm.request" and detail["chat"][0]["kind"] == "system"
    resp = next(e for e in run.log.events() if e["type"] == "llm.response")
    assert views.trace_detail(run, resp["_seq"])["chat"]
    assert views.trace_detail(run, 10 ** 9) is None
    all_gpu = views.trace(run, node=None, phase=None, attempt=None, types=None, component=None,
                          text=None, include_gpu=True, page=0)
    assert any(r["type"] == "gpu.sample" for r in all_gpu["rows"])


def test_node_detail(run):
    d = views.node_detail(run, "n1")
    assert d["lineage"] == "root → n1" and d["edit"]["summary"] == "tighter planner"
    assert "+optimizer.lr" in d["recipe_diff"] and d["data_commit"]["datasets"] == {"ds1": 1}
    m1 = next(r for r in d["metrics"] if r["metric"] == "m1")
    assert (m1["node"], m1["parent"], m1["root"]) == (0.7, 0.6, 0.6)
    assert [(a["phase"], a["attempt"], a["outcome"]) for a in d["attempts"]][0] == ("edit_self", 1, "contract_failed")
    assert views.node_detail(run, "nope") is None


def test_root_is_a_baseline_everywhere(run):
    assert views.node_detail(run, "root")["baseline"] is True
    assert views.code_edits(run, "root")["baseline"] is True
    assert views.training_attempts(run, "root") == []
    assert views.training(run, "root", 1)["reached_training"] is False


def test_conversations_and_logs(run):
    assert ("improve_recipe", 1) in views.agent_attempts(run, "n1")
    convs = views.conversations(run, "n1", "improve_recipe", 1)
    assert [(c["role"], c["calls"], c["compactions"]) for c in convs] == [("planner", 2, 0), ("data_builder", 5, 1)]
    chat = views.conversation_chat(run, "n1", "improve_recipe", 1, 1)
    assert any(i["kind"] == "compaction_request" for i in chat)
    assert views.tool_logs(run, "n1", "improve_recipe", 1) == [
        "nodes/n1/attempts/improve_recipe-1/workspace/tool_output/run_command-20260927-120000-0001.log"]


def test_code_edits(run):
    d = views.code_edits(run, "n1")
    assert "-X = 1" in d["node_diff"] and "+X = 3" in d["node_diff"]
    a1, a2 = d["attempts"]
    assert a1["outcome"] == "contract_failed" and "+X = 2" in a1["diff"] and a1["contract"]["ok"] is False
    assert a2["outcome"] == "passed" and "-X = 2" in a2["diff"] and "+X = 3" in a2["diff"]
    [n2] = views.code_edits(run, "n2")["attempts"]
    assert n2["commit"] is None and n2["diff"] is None and n2["contract"] is None and "check out" in n2["detail"]
    assert "agent/entry.py" in views.agent_tree(run, d["agent_commit"])
    assert views.agent_file(run, d["agent_commit"], "agent/entry.py") == "X = 3\n"


def test_training(run):
    assert views.training_attempts(run, "n1") == [1]
    t = views.training(run, "n1", 1)
    assert [p["step"] for p in t["loss"]] == [1, 2] and t["loss"][0]["loss"] == pytest.approx(0.347656)
    assert [g["result"] for g in t["gates"]] == ["failed", "passed"]
    assert t["reached_training"] and t["config"].startswith("optimizer")
    assert {g["gpu"] for g in t["gpu"]} == {"0", "1"}


def test_selection_and_cost(run):
    [sel] = views.selection(run)
    assert sel["child"] == "n1" and sel["chosen"] == "root" and sel["candidates"][0]["P"] == 1.0
    c = views.cost(run)
    assert {r["role (inferred)"] for r in c["by_role"]} == {"planner", "data_builder"}
    assert sum(r["calls"] for r in c["by_phase"]) == 6 and c["errors"] == []


def test_views_on_an_empty_run(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "run.json").write_text("{}")
    r = views.Run(tmp_path)
    assert views.overview(r)["nodes"] == [] and views.gpu_series(r) == []
    assert views.trace(r, node=None, phase=None, attempt=None, types=None, component=None, text=None,
                       include_gpu=False, page=0)["rows"] == []
    assert views.node_detail(r, "root") is None and views.selection(r) == []
    assert views.cost(r) == {"by_phase": [], "by_role": [], "errors": []}
    assert views.code_edits(r, "root")["attempts"] == []

from panel import media


def test_training_data_and_clips(run):
    d = media.training_data(run, "n1")
    assert d["datasets"] == [{"dataset": "ds1", "clips": 1}] and d["commit"]["message"] == "ds1: one clip"
    got = media.clips(run, "n1", "ds1")
    [clip] = got["rows"]
    assert clip["video"] == "store/blobs/video/vd1.mp4" and clip["caption"]["caption"] == "a hallway"
    assert clip["leakage"] == {"matches": [], "near": [{"case_id": "7"}]}
    assert media.training_data(run, "root")["datasets"] == []
    assert media.clip_detail(run, "nope") is None


def test_ingest_calls_and_staging(run):
    rows = media.ingest_calls(run, "n1")
    assert [(r["outcome"], r["clip_id"]) for r in rows] == [("accepted", "clip1"), ("rejected", None)]
    assert rows[1]["reasons"] == "aspect 4:3" and rows[1]["video"] == "/workspace/staging/work/v2.mp4"
    assert rows[1]["host_video"] == "staging/n1/improve_recipe-1/work/v2.mp4"
    assert media.staging_files(run, "n1") == [{"path": "staging/n1/improve_recipe-1/work/v2.mp4", "bytes": 9}]


def test_camera_path(run):
    path = media.camera_path(run, "store/blobs/pose/pd1.npz")
    assert path["x"] == [0, 1, 2, 3, 4] and path["z"] == [0, 2, 4, 6, 8]


def test_eval_view(run):
    ev = media.eval_view(run, "n1")
    [case] = ev["cases"]
    assert case["case"] == "7" and case["scores"] == {"m1": 0.7, "m2": 0.8}
    assert case["video"].endswith("videos/case_7_combined.mp4") and case["prompt_schedule"][0]["prompt"] == "a corridor"
    assert ev["aggregates"]["metrics"]["m1"] == 0.7


def test_eval_view_without_eval(run):
    assert media.eval_view(run, "n2") == {"cases": [], "aggregates": None}


def test_files_list_and_preview(run, run_dir):
    names = [e["name"] for e in media.list_dir(run, "")]
    assert names[:3] == ["agents.git", "config", "control"] and "archive.db" in names
    assert media.preview(run, "config/run.json")["kind"] == "text"
    assert media.preview(run, "store/blobs/video/vd1.mp4")["kind"] == "video"
    npz = media.preview(run, "store/blobs/pose/pd1.npz")
    assert npz["kind"] == "npz" and npz["arrays"] == [{"name": "cam_c2w", "shape": [5, 4, 4], "dtype": "float64"}]
    assert media.preview(run, "archive.db")["kind"] == "binary"
    assert media.preview(run, "nodes")["kind"] == "dir"
    (run_dir / "big.txt").write_bytes(b"x" * 3_000_000)
    assert "not shown" in media.preview(run, "big.txt")["text"]


def test_preview_non_utf8_text(run, run_dir):
    (run_dir / "latin.log").write_bytes(b"caf\xe9 ok\n")
    got = media.preview(run, "latin.log")
    assert got["kind"] == "text" and "caf" in got["text"]


def test_preview_refuses_symlink_out(run, run_dir, tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    (run_dir / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ValueError):
        media.preview(run, "link.txt")
    with pytest.raises(ValueError):
        media.list_dir(run, "..")

import hashlib


def _snapshot(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def test_every_view_leaves_the_run_untouched(run_dir):
    before = _snapshot(run_dir)
    r = views.Run(run_dir)
    views.overview(r), views.gpu_series(r, hours=1e6), views.trace_choices(r)
    got = views.trace(r, node=None, phase=None, attempt=None, types=None, component=None, text="n1",
                      include_gpu=True, page=0)
    for row in got["rows"]:
        views.trace_detail(r, row["seq"])
    for node in ("root", "n1", "n2"):
        views.node_detail(r, node), views.code_edits(r, node), media.training_data(r, node), media.eval_view(r, node)
        for phase, attempt in views.agent_attempts(r, node):
            for c in views.conversations(r, node, phase, attempt):
                views.conversation_chat(r, node, phase, attempt, c["index"])
            views.tool_logs(r, node, phase, attempt)
        for attempt in views.training_attempts(r, node):
            views.training(r, node, attempt)
    media.clips(r, "n1", "ds1"), media.camera_path(r, "store/blobs/pose/pd1.npz")
    views.selection(r), views.cost(r)
    from panel import problems as problem_view
    problem_view.problems(r), problem_view.problem_counts(r)
    commit = views.code_edits(r, "n1")["agent_commit"]
    for path in views.agent_tree(r, commit):
        views.agent_file(r, commit, path)
    for entry in media.list_dir(r, ""):
        media.preview(r, entry["name"])
    assert _snapshot(run_dir) == before


# ---- final review fixes ----

def test_git_arguments_cannot_become_options(run, tmp_path):
    target = tmp_path / "pwned"
    assert views.agent_file(run, f"--output={target}", "x") is None
    assert views.agent_tree(run, f"--output={target}") == []
    assert not list(tmp_path.glob("pwned*"))


def test_an_empty_wal_left_by_a_stopped_run_is_not_touched(run_dir):
    (run_dir / "archive.db-wal").write_bytes(b"")
    before = _snapshot(run_dir)
    assert [r["node_id"] for r in RunFiles(run_dir).query("SELECT node_id FROM nodes ORDER BY created_at")] == ["root", "n1", "n2"]
    assert _snapshot(run_dir) == before and not (run_dir / "archive.db-shm").exists()


def test_rows_committed_in_a_leftover_wal_are_read_without_touching_the_run(run_dir):
    import sqlite3
    conn = sqlite3.connect(run_dir / "archive.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("UPDATE nodes SET status='eval_failed' WHERE node_id='n2'")
    conn.commit()
    crashed = {p.name: p.read_bytes() for p in run_dir.glob("archive.db*")}   # as a killed writer leaves them
    conn.close()
    for name, data in crashed.items():
        (run_dir / name).write_bytes(data)
    assert (run_dir / "archive.db-wal").stat().st_size > 0
    before = _snapshot(run_dir)
    rows = RunFiles(run_dir).query("SELECT status FROM nodes WHERE node_id = 'n2'")
    assert rows == [{"status": "eval_failed"}] and _snapshot(run_dir) == before


def test_a_live_writer_is_read_with_plain_read_only(run_dir, monkeypatch):
    import sqlite3
    started = open(f"/proc/{os.getpid()}/stat").read().rsplit(")", 1)[1].split()[19]
    (run_dir / "control" / "loop.pid").write_text(f"{os.getpid()} {started}")
    (run_dir / "archive.db-wal").write_bytes(b"")
    uris, real = [], sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda target, **kw: uris.append(target) or real(target, **kw))
    assert RunFiles(run_dir).query("SELECT count(*) AS n FROM nodes") == [{"n": 3}]
    assert uris[0].endswith("?mode=ro")


def test_a_missing_request_payload_does_not_break_the_chat(run_dir):
    log, segs, chs = _chat(run_dir)
    digest = next(c for c in chs[1][0].calls[::-1]).request
    (run_dir / "telemetry" / "payloads" / f"{digest}.json.zst").unlink()
    log = EventLog(RunFiles(run_dir))
    chs = chains(segments(log.events(), "n1", "improve_recipe", 1), log.payload)
    items = [i for ch in chs for i in chat_items(ch, log.payload, awaiting=False)]
    assert any(i["kind"] == "note" and "payload missing" in i["text"] for i in items)


def test_a_blank_system_prompt_has_no_role():
    assert infer_role("  \n ", {"planner": PLANNER}) == "(no system prompt)"


# ---- owner's first look (2026-09-28) ----

def test_gpu_series_covers_the_runs_last_hours_not_the_clock(run_dir):
    path = run_dir / "telemetry" / "events" / "gpu.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for e in events:
        e["ts_wall"] -= 2 * 86400                             # a run that stopped two days ago
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert views.gpu_series(views.Run(run_dir), hours=6)


def test_trace_offers_attempts_and_any(run):
    assert {"0", "1"} <= set(views.trace_choices(run)["attempt"])


# ---- problems view (2026-09-28) ----

from collections import Counter  # noqa: E402

from panel import problems  # noqa: E402


def _add_problems(run_dir):
    """One of every problem kind on n2 (n1 already has a contract failure, a gate failure, an
    ingest rejection and an unanswered LLM call; the run has a stall alert)."""
    from ar_kernel.telemetry.recorder import Recorder
    from fixtures.panel_run import Llm, tool_call
    rec = Recorder(run_dir)
    base = dict(node="n2", phase="edit_self", attempt=1)
    rec.event("phase.start", payload={"code_commit": None}, **base)
    calls = [tool_call("t1", "run_command", {"command": "false"}), tool_call("t2", "nope", {}),
             tool_call("t3", "data_query", {"filter": {}}), tool_call("t4", "read_file", {"path": "a"})]
    history = [{"role": "system", "content": "You code."}, {"role": "user", "content": "go"}]
    llm = Llm(rec, **base)
    llm.call("c1", 0, history, {"content": "", "tool_calls": calls})
    rec.event("tool.call", tool="data_query", component="tools", payload={"tool": "data_query", "args": {}}, **base)
    rec.event("tool.error", tool="data_query", component="tools",
              payload={"tool": "data_query", "error": "bad filter"}, **base)
    llm.call("c1", 1, history + [
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "t1", "content": "exit 2\nboom"},
        {"role": "tool", "tool_call_id": "t2", "content": "Error: nope is not a valid tool, try one of [a]."},
        {"role": "tool", "tool_call_id": "t3", "content": "Error: ToolException('bad filter')\n Please fix your mistakes."},
        {"role": "tool", "tool_call_id": "t4", "content": "fine"}], {"content": "done"})
    rec.event("llm.request", payload={"body": {"messages": [{"role": "user", "content": "x"}]}}, call_id="r1",
              conversation_id="c9", turn_index=0, model="m", component="gateway", **base)
    rec.event("llm.response", payload={"body": {"error": {"message": "rate limited"}}}, call_id="r1",
              conversation_id="c9", status=429, usage={}, cost_usd=None, latency_s=0.1, component="gateway", **base)
    rec.event("job.finished", node="n2", job_id="j1", component="tools", payload={"error": None, "result": {
        "clips": {"/workspace/a.mp4": {"error": "HTTP 400: too many tokens"}, "/workspace/b.mp4": {"caption": "ok"}}}})
    rec.event("job.finished", node="n2", job_id="j2", component="tools", payload={"error": "worker died", "result": None})
    rec.event("subproc.end", node="n2", phase="render", returncode=1, payload={"stdout": "", "stderr": "Traceback: x"})
    rec.event("subproc.end", node="n2", phase="render", returncode=0, payload={"stdout": "", "stderr": ""})
    rec.event("subproc.cancelled", node="n2", phase="caption_videos", returncode=-15, payload={})
    rec.event("sandbox.end", exit_code=1, timed_out=False, component="sandbox",
              payload={"stdout": "", "stderr": "crash", "reason": None}, **base)
    rec.event("render.error", node="n2", phase="render", payload={"error": "RuntimeError('x')", "traceback": "tb"})
    rec.event("run.warning", payload={"message": "disk low"}, message="disk low")
    rec.event("phase.end", **base)


def test_problems_lists_every_failure_once(run_dir):
    _add_problems(run_dir)
    got = problems.problems(views.Run(run_dir))
    assert Counter((p["source"], p["kind"]) for p in got) == Counter({
        ("recorded", "alert"): 1, ("recorded", "contract_failed"): 1, ("recorded", "gate_failed"): 1,
        ("recorded", "ingest_rejected"): 1, ("recorded", "llm_call"): 2, ("recorded", "tool_error"): 1,
        ("recorded", "job_item"): 1, ("recorded", "job_failed"): 1, ("recorded", "subprocess"): 1,
        ("recorded", "sandbox"): 1, ("recorded", "error_event"): 1, ("recorded", "warning"): 1,
        ("inferred", "agent_tool"): 2})
    assert [p["ts"] for p in got] == sorted((p["ts"] for p in got), reverse=True)        # newest first
    by = {(p["kind"], p["node"]): p for p in got}
    assert "429" in by[("llm_call", "n2")]["summary"] and "rate limited" in by[("llm_call", "n2")]["summary"]
    assert "no response" in by[("llm_call", "n1")]["summary"]
    assert "a.mp4" in by[("job_item", "n2")]["summary"] and "too many tokens" in by[("job_item", "n2")]["summary"]
    assert by[("tool_error", "n2")]["seq"] is not None and "data_query" in by[("tool_error", "n2")]["summary"]
    inferred = sorted(p["summary"] for p in got if p["source"] == "inferred")
    assert inferred[0].startswith("nope:") and inferred[1].startswith("run_command: exit 2")
    assert all(p["where"] == "edit_self-1 · conversation 0" and p["seq"] is None for p in got
               if p["source"] == "inferred")


def test_an_unanswered_call_in_a_live_attempt_is_not_a_problem(run_dir):
    started = open(f"/proc/{os.getpid()}/stat").read().rsplit(")", 1)[1].split()[19]
    (run_dir / "control" / "loop.pid").write_text(f"{os.getpid()} {started}")
    path = run_dir / "telemetry" / "events" / "n1.jsonl"
    path.write_text("".join(line + "\n" for line in path.read_text().splitlines()
                            if '"phase.end"' not in line))                  # the attempt is still running
    assert not [p for p in problems.problems(views.Run(run_dir)) if p["kind"] == "llm_call"]


def test_problem_counts_by_kind(run_dir):
    _add_problems(run_dir)
    r = views.Run(run_dir)
    counts = problems.problem_counts(r, hours=1)
    assert sum(c["total"] for c in counts) == len(problems.problems(r))
    assert all(c["last hour"] == c["total"] for c in counts)          # everything happened in the last hour


def test_error_tool_outputs_are_marked_in_chats(run_dir):
    _add_problems(run_dir)
    r = views.Run(run_dir)
    items = views.conversation_chat(r, "n2", "edit_self", 1, 0)
    marked = [i["title"] for i in items if i["kind"] == "tool_error"]
    assert marked == ["⚠ Tool error: run_command", "⚠ Tool error: nope", "⚠ Tool error: data_query"]
    assert any(i["kind"] == "tool_output" and i["text"] == "fine" for i in items)
