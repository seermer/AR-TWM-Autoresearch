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





from panel import views


@pytest.fixture
def run(run_dir):
    return views.Run(run_dir)


def test_overview(run):
    ov = views.overview(run)
    rows = {r["node"]: r for r in ov["nodes"]}
    assert rows["n1"]["vs parent"] == pytest.approx(0.05) and rows["n1"]["duration_min"] == 10.0
    assert rows["root"]["parent"] == "" and rows["n2"]["status"] == "running"
    assert ov["alerts"][0]["kind"] == "monitor.stall" and len(ov["recent"]) <= 50
    assert ov["recent"][0]["type"] == "end ar.run" and ov["recent"][0]["seq"] == len(run.records.all()) - 1
    assert [(o["name"], o["node"], o["facts"]) for o in ov["open_spans"]] == [
        ("llm.request", "n1", "no end recorded; " + ov["open_spans"][0]["facts"].split("; ", 1)[1])]


def test_gpu_full_keeps_every_sample_of_a_short_run(run):
    g = views.gpu_full(run)
    assert set(g["gpus"]) == {"0", "1"} and g["buckets"] == g["samples"] > 0
    assert g["gpus"]["0"]["memory_gib"][0][1] == 2.0


def test_gpu_full_buckets_a_long_run_and_keeps_its_peaks(run, monkeypatch):
    samples = [{"kind": "sample", "ts": float(t), "gpus": {"0": {"util": 100 if t == 777 else 10, "memory_mib": 1024}}}
               for t in range(3000)]
    monkeypatch.setattr(run.records, "all", lambda: samples)
    g = views.gpu_full(run, buckets=100)
    util = g["gpus"]["0"]["util"]
    assert g["samples"] == 3000 and g["buckets"] == 100 and len(util) <= 200
    assert [777.0, 100] in [list(p) for p in util] and [p[0] for p in util] == sorted(p[0] for p in util)


def test_trace_filters_pages_and_detail(run):
    choices = views.trace_choices(run)
    assert "start llm.request" in choices["type"] and "n1" in choices["node"] and {"1", "2"} <= set(choices["attempt"])
    got = views.trace(run, node="n1", types=["start llm.request"], limit=2)
    assert got["total"] == 7 and len(got["rows"]) == 2
    rest = views.trace(run, node="n1", types=["start llm.request"], offset=2, limit=100)
    assert len(rest["rows"]) == 5 and rest["rows"][0]["seq"] not in {r["seq"] for r in got["rows"]}
    detail = views.trace_detail(run, got["rows"][0]["seq"])
    assert detail["event"]["type"] == "start llm.request" and detail["chat"][0]["kind"] == "system"
    assert detail["payload"]["output"]["status"] == 200 and detail["event"]["node"] == "n1"
    end = next(i for i, r in enumerate(run.records.all()) if r["kind"] == "end" and r["span"] == detail["event"]["span"])
    assert views.trace_detail(run, end)["chat"][-1]["kind"] == "tool_call"
    assert views.trace_detail(run, 10 ** 9) is None
    assert views.trace(run, text="quiet 30")["total"] == 1 and views.trace(run, text="nothing-like-this")["total"] == 0
    assert any(r["type"] == "sample" for r in views.trace(run, include_gpu=True, limit=10 ** 6)["rows"])
    assert not any(r["type"] == "sample" for r in views.trace(run, limit=10 ** 6)["rows"])


def test_trace_keeps_one_spans_subtree(run):
    tool = next(n for n in run.records.spans() if n.start["name"] == "tool.data_ingest")
    rows = views.trace(run, span=tool.id, limit=10 ** 6)["rows"]
    assert rows[0]["type"] == "start tool.data_ingest" and rows[-1]["type"] == "end tool.data_ingest"
    assert {r["type"] for r in rows} >= {"start ingest.candidate", "decision ingest.check", "start file.kept"}
    assert not any(r["type"].endswith("llm.request") for r in rows)


def test_records_are_read_as_they_are_appended_and_a_torn_line_waits(run_dir):
    from panel.records import Records
    from panel.runfiles import RunFiles
    records = Records(RunFiles(run_dir))
    before = len(records.all())
    [path] = sorted((run_dir / "telemetry" / "records").glob("*.jsonl"))
    line = json.dumps({"v": 1, "writer": "w2", "seq": 1, "ts": 9e9, "mono": 0, "kind": "log", "span": None,
                       "stream": "stdout", "text": "late"})
    with path.open("a") as f:
        f.write(line[:10])                                  # a half-written line
    assert len(records.all()) == before
    with path.open("a") as f:
        f.write(line[10:] + "\n")
    assert len(records.all()) == before + 1 and records.all()[-1]["text"] == "late"


def test_payloads_load_cache_and_miss(run_dir):
    from panel.records import Records
    from panel.runfiles import RunFiles
    records = Records(RunFiles(run_dir))
    req = next(r for r in records.all() if r["kind"] == "start" and r["name"] == "llm.request")
    body = records.payload(req["inputs"])
    assert body["endpoint"] == "/chat/completions" and records.payload(req["inputs"]) is body    # from the cache
    assert records.payload(None) is None and records.payload("0" * 64) is None
    (run_dir / "telemetry" / "payloads" / f"{req['inputs']}.json.zst").write_bytes(b"not zstd")
    assert Records(RunFiles(run_dir)).payload(req["inputs"]) is None


def test_each_record_is_placed_in_its_phase_attempt(run):
    places = {(r["kind"], r.get("name")): run.records.place(r) for r in run.records.all()}
    assert places[("decision", "contract.step")][:2] == ("n1", "edit_self")
    assert places[("decision", "llm.link")] == ("n1", "improve_recipe", 1)
    assert places[("decision", "node.status")] == ("n1", None, None)
    assert places[("alert", "monitor.stall")] == (None, None, None)

from fixtures.panel_run import COMPACT, SUMMARY  # noqa: E402
from panel.chat import chains, chat_items, segments  # noqa: E402


def _chat(run):
    segs = segments(run.records.calls("n1", "improve_recipe", 1))
    return segs, chains(segs, run.records.payload)


def test_segments_group_calls_by_conversation(run):
    segs, _ = _chat(run)
    assert [s.conversation_id for s in segs] == ["p1", "b1", "b2"]
    assert [len(s.calls) for s in segs] == [2, 3, 2]
    assert segs[2].calls[-1].response is None and segs[0].calls[0].latency_s == 1.0


def test_compaction_is_stitched_across_conversation_ids(run):
    _, chs = _chat(run)
    assert [[s.conversation_id for s in c] for c in chs] == [["p1"], ["b1", "b2"]]


def test_chat_shows_compaction_as_a_turn_then_continues(run):
    _, chs = _chat(run)
    items = chat_items(chs[1], run.records.payload, awaiting=False)
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
    assert chat_items(chs[1], run.records.payload, awaiting=True)[-1]["text"] == "awaiting response"


def test_chat_items_cover_reasoning_tools_and_usage(run):
    _, chs = _chat(run)
    items = chat_items(chs[0], run.records.payload, awaiting=False)
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


def test_a_conversation_without_a_role_label_is_unlabeled(run):
    calls = run.records.calls("n1", "improve_recipe", 1)
    for c in calls:
        c.role = None
    run.records.calls = lambda *a: calls
    assert {c["role"] for c in views.conversations(run, "n1", "improve_recipe", 1)} == {"unlabeled"}


def test_a_missing_request_payload_does_not_break_the_chat(run_dir):
    run = views.Run(run_dir)
    _, chs = _chat(run)
    span = run.records.span(chs[1][0].calls[-1].span)
    (run_dir / "telemetry" / "payloads" / f"{span.start['inputs']}.json.zst").unlink()    # the raw request
    run = views.Run(run_dir)
    _, chs = _chat(run)
    items = [i for ch in chs for i in chat_items(ch, run.records.payload, awaiting=False)]
    assert any(i["kind"] == "note" and "payload missing" in i["text"] for i in items)


def test_node_detail(run):
    d = views.node_detail(run, "n1")
    assert d["lineage"] == "root → n1" and d["edit"]["summary"] == "tighter planner"
    assert d["summary"] == "Tested pool clips.\n"
    assert "+optimizer.lr" in d["recipe_diff"] and d["data_commit"]["datasets"] == {"ds1": 1}
    m1 = next(r for r in d["metrics"] if r["metric"] == "m1")
    assert (m1["node"], m1["parent"], m1["root"]) == (0.7, 0.6, 0.6)
    assert [(a["phase"], a["attempt"], a["outcome"]) for a in d["attempts"]][0] == ("edit_self", 1, "contract_failed")
    assert views.node_detail(run, "nope") is None


def test_a_continued_part_is_listed_with_its_outcome(run_dir):
    from ar_kernel.archive.db import open_db
    from ar_kernel.archive.nodes import NodeStore
    NodeStore(open_db(run_dir)).add_attempt("n2", "improve_recipe", 1, "continued", {"snapshot_id": "s", "summary": "x"})
    run = views.Run(run_dir)
    assert ("improve_recipe", 1, "continued") in [(a["phase"], a["attempt"], a["outcome"])
                                                  for a in views.node_detail(run, "n2")["attempts"]]
    assert views.training(run, "n2", 1)["outcome"]["outcome"] == "continued"


def test_root_is_a_baseline_everywhere(run):
    assert views.node_detail(run, "root")["baseline"] is True
    assert views.code_edits(run, "root")["baseline"] is True
    assert views.training_attempts(run, "root") == []
    assert views.training(run, "root", 1)["reached_training"] is False


def test_code_edits(run):
    d = views.code_edits(run, "n1")
    assert "-X = 1" in d["node_diff"] and "+X = 3" in d["node_diff"]
    a1, a2 = d["attempts"]
    assert a1["outcome"] == "contract_failed" and "+X = 2" in a1["diff"] and a1["contract"]["ok"] is False
    assert a1["contract"]["steps"] == [{"step": "import", "result": "fail", "detail": "import failed"}]
    assert a2["outcome"] == "passed" and "-X = 2" in a2["diff"] and "+X = 3" in a2["diff"] and a2["contract"]["ok"]
    [n2] = views.code_edits(run, "n2")["attempts"]
    assert n2["commit"] is None and n2["diff"] is None and n2["contract"] is None and "check out" in n2["detail"]
    assert "agent/entry.py" in views.agent_tree(run, d["agent_commit"])
    assert views.agent_file(run, d["agent_commit"], "agent/entry.py") == "X = 3\n"


def test_training(run):
    assert views.training_attempts(run, "n1") == [1]
    t = views.training(run, "n1", 1)
    assert [p["step"] for p in t["loss"]] == [1, 2] and t["loss"][0]["loss"] == pytest.approx(0.347656)
    assert t["config"].startswith("optimizer") and t["outcome"]["outcome"] == "passed"
    assert [(g["result"], g["failures"]) for g in t["gates"]] == [("failed", "resolution 1x1 is not allowed"),
                                                                   ("passed", None)]
    assert t["reached_training"] and t["duration_min"] == 1.0 and {g["gpu"] for g in t["gpu"]} == {"0", "1"}


def test_selection_and_cost(run):
    [sel] = views.selection(run)
    assert sel["child"] == "n1" and sel["chosen"] == "root" and sel["candidates"][0]["P"] == 1.0
    c = views.cost(run)
    assert {r["role"] for r in c["by_role"]} == {"planner", "data_builder"}
    assert sum(r["calls"] for r in c["by_phase"]) == 6 and c["errors"] == []
    calls = views.llm_calls(run)
    assert len(calls) == 6 and calls[0]["role"] == "planner" and calls[0]["total_tokens"] == 110
    assert views.trace_detail(run, calls[0]["seq"])["event"]["type"] == "start llm.request"


def test_conversations_and_logs(run):
    assert views.agent_attempts(run, "n1") == [("edit_self", 1), ("edit_self", 2), ("improve_recipe", 1)]
    convs = views.conversations(run, "n1", "improve_recipe", 1)
    assert [(c["role"], c["calls"], c["compactions"], c["tool_errors"]) for c in convs] == [
        ("planner", 2, 0, 0), ("data_builder", 5, 1, 0)]
    chat = views.conversation_chat(run, "n1", "improve_recipe", 1, 1)
    assert any(i["kind"] == "compaction_request" for i in chat)
    assert views.tool_logs(run, "n1", "improve_recipe", 1) == [
        "nodes/n1/attempts/improve_recipe-1/workspace/tool_output/data_engineer-data_query-20260927-120001-2.txt",
        "nodes/n1/attempts/improve_recipe-1/workspace/tool_output/run_command-20260927-120000-0001.log"]


def test_views_on_an_empty_run(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "run.json").write_text("{}")
    r = views.Run(tmp_path)
    assert views.overview(r)["nodes"] == [] and views.gpu_full(r)["samples"] == 0
    assert views.trace(r)["rows"] == []
    assert views.node_detail(r, "root") is None and views.selection(r) == []
    assert views.cost(r) == {"by_phase": [], "by_role": [], "errors": []}
    assert views.code_edits(r, "root")["attempts"] == []

from panel import media


def test_training_data_and_clips(run):
    d = media.training_data(run, "n1")
    assert d["datasets"] == [{"dataset": "ds1", "clips": 1}] and d["commit"]["message"] == "ds1: one clip"
    [clip] = media.clips(run, "n1", "ds1")
    assert clip["video"] == "store/blobs/video/vd1.mp4" and clip["caption"]["caption"] == "a hallway"
    assert clip["annotation"]["turns"] == [{"start_s": 0.0, "perspective": "first_person"}] and clip["derived_from"] == {"kind": "dataset", "name": "org/ds"}
    assert clip["leakage"] == {"matches": [], "near": [{"case_id": "7"}]}
    assert media.training_data(run, "root")["datasets"] == []
    assert media.clip_detail(run, "nope") is None


def test_ingest_calls_and_staging(run):
    rows = media.ingest_calls(run, "n1")
    assert [(r["outcome"], r["clip_id"]) for r in rows] == [("accepted", "clip1"), ("rejected", None)]
    assert rows[1]["reasons"] == "aspect 4:3" and rows[1]["video"] == "/workspace/staging/work/v2.mp4"
    assert rows[0]["annotation"] == "/workspace/staging/work/v1.json" and rows[0]["derived_from"] == "dataset: org/ds"
    assert rows[1]["host_video"] == "staging/n1/improve_recipe-1/work/v2.mp4"
    assert media.training_data(run, "n1")["ingests"] == rows
    assert media.staging_files(run, "n1") == [{"path": "staging/n1/improve_recipe-1/work/v2.mp4", "bytes": 9}]


def test_camera_path(run):
    path = media.camera_path(run, "store/blobs/pose/pd1.npz")
    assert path["x"] == [0, 1, 2, 3, 4] and path["z"] == [0, 2, 4, 6, 8]


def test_eval_view(run):
    ev = media.eval_view(run, "n1")
    [case] = ev["cases"]
    assert list(case["scores"].items()) == [("event_edit_adherence", 1.0), ("m1", 0.7), ("m2", 0.8)]   # judged first
    assert case["video"].endswith("videos/case_7_combined.mp4") and case["prompt_schedule"][0]["prompt"] == "a corridor"
    assert ev["aggregates"]["metrics"]["m1"] == 0.7


def test_eval_view_without_eval(run):
    assert media.eval_view(run, "n2") == {"cases": [], "aggregates": None}
    assert media.judge_view(run, "n2", "7") == []


def test_judge_view_gives_each_answer_and_whether_it_is_the_expected_one(run, run_dir):
    assert media.judge_view(run, "root", "7") == [
        {"turn": "2: Rain begins.", "question": "Scene unchanged?", "answer": "no", "passed": True},
        {"turn": "2: Rain begins.", "question": "Event visible?", "answer": "no", "passed": False}]
    evaluation = run_dir / "nodes" / "n1" / "eval" / "work_dirs" / "ar_r1_n1" / "evaluation"
    for metric, data in {
            "perspective_switch_adherence": {"turn_details": [{"turn": 1, "action": "Switch to first person.",
                                                               "q_results": {"Q1": True, "Q3": False, "Q4": True, "Q_cam": True}}]},
            "causal_fidelity": {"track1": {"score": 1, "max": 3, "reason": "r"},
                                "track2": {"details": [{"dim": "gravity", "dim_name": "Gravity", "score": 2, "max": 3}]}}}.items():
        (evaluation / metric).mkdir()
        (evaluation / metric / "case_7.json").write_text(json.dumps(data))
    rows = media.judge_view(run, "n1", "7")
    assert [(r["question"], r["answer"], r["passed"]) for r in rows[2:]] == [
        ("Viewpoint changes?", "yes", True), ("Ends in the asked view?", "no", False), ("End view valid?", "yes", True),
        ("Changed by the camera? (not scored)", "yes", True),
        ("Causal fidelity: overall", "1/3", None), ("Causal fidelity: Gravity", "2/3", None)]


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
    views.overview(r)
    for node in ("root", "n1", "n2"):
        views.node_detail(r, node), views.code_edits(r, node), media.training_data(r, node), media.eval_view(r, node)
        for attempt in views.training_attempts(r, node):
            views.training(r, node, attempt)
    media.clips(r, "n1", "ds1"), media.camera_path(r, "store/blobs/pose/pd1.npz")
    views.selection(r)
    commit = views.code_edits(r, "n1")["agent_commit"]
    for path in views.agent_tree(r, commit):
        views.agent_file(r, commit, path)
    for entry in media.list_dir(r, ""):
        media.preview(r, entry["name"])
    assert _snapshot(run_dir) == before


# ---- reading a run's files safely ----

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









# ---- problems view ----

from collections import Counter  # noqa: E402

from panel import problems  # noqa: E402


def _add_problems(run_dir):
    """One of every problem kind on n2 (n1 already has a contract failure, a gate failure, an ingest rejection
    and an unanswered LLM call; the run has a stall alert)."""
    from ar_kernel import telemetry
    from fixtures.panel_run import Scripted, tool_call
    rec = Scripted(run_dir)
    calls = [tool_call("t1", "run_command", {"command": "false"}), tool_call("t2", "nope", {}),
             tool_call("t3", "data_query", {"filter": {}}), tool_call("t4", "read_file", {"path": "a"})]
    history = [{"role": "system", "content": "You code."}, {"role": "user", "content": "go"}]
    try:
        with telemetry.span("phase.attempt", {"node": "n2", "phase": "edit_self", "attempt": 1}) as attempt:
            rec.llm(attempt, "c1", 0, history, {"content": "", "tool_calls": calls})
            try:
                with telemetry.span("tool.request", {"tool": "data_query"}), telemetry.span("tool.data_query", {}):
                    raise ValueError("bad filter")
            except ValueError:
                pass
            rec.llm(attempt, "c1", 1, history + [
                {"role": "assistant", "content": "", "tool_calls": calls},
                {"role": "tool", "tool_call_id": "t1", "content": "exit 2\nboom"},
                {"role": "tool", "tool_call_id": "t2", "content": "Error: nope is not a valid tool, try one of [a]."},
                {"role": "tool", "tool_call_id": "t3", "content": "Error: ToolException('bad filter')\n Please fix your mistakes."},
                {"role": "tool", "tool_call_id": "t4", "content": "fine"}], {"content": "done"})
            sp = rec.llm_start(attempt, "c9", 0, {"messages": [{"role": "user", "content": "x"}]})
            rec.llm_end(sp, {"error": {"message": "rate limited"}}, status=429, cost=None)
            with telemetry.span("llm.request", {"endpoint": "/chat/completions", "role": None, "body": "{}"}) as refused:
                telemetry.decision("llm.check", {"check": "model"}, ["pass", "refuse"], "refuse", "not allowed")
                refused.status, refused.output = "refused", {"status": 403, "body": {"error": {"message": "not allowed"}}}
            telemetry.error(OSError("disk full"), "the path keeps its mode")
            telemetry.decision("phase.result", {}, ["accept", "fail"], "fail", "no result.json")
            telemetry.decision("phase.result", {}, ["accept", "fail"], "accept", "fine")    # not a problem
            with telemetry.span("process", {"command": ["python", "render.py"]}) as proc:
                proc.status, proc.output = "cancelled", {"exit_code": -15}                  # not a problem
            with telemetry.span("process", {"command": ["git", "status"]}) as proc:
                proc.status, proc.output = "error", {"exit_code": 128, "stderr": "fatal: not a git repository\n"}
            with telemetry.span("process", {"command": ["python", "train.py"]}) as proc:
                telemetry.log("stdout", "step 1")
                telemetry.log("stderr", "CUDA out of memory")
                proc.status, proc.output = "error", {"exit_code": 1}
            attempt.output = {}
        telemetry.alert("monitor.disk_low", "warning", "disk low")
    finally:
        rec.close()


def test_problems_lists_every_failure_once(run_dir):
    _add_problems(run_dir)
    got = problems.problems(views.Run(run_dir))
    assert Counter((p["source"], p["kind"]) for p in got) == Counter({
        ("recorded", "alert"): 2, ("recorded", "contract.step"): 1, ("recorded", "recipe.gate"): 1,
        ("recorded", "ingest_rejected"): 1, ("recorded", "no end recorded"): 1, ("recorded", "llm.request"): 1,
        ("recorded", "refused"): 1, ("recorded", "tool.data_query"): 1, ("recorded", "handled error"): 1,
        ("recorded", "phase.result"): 1, ("recorded", "process"): 2, ("inferred", "agent_tool"): 2})
    assert [p["ts"] for p in got] == sorted((p["ts"] for p in got), reverse=True)        # newest first
    by = {(p["kind"], p["node"]): p for p in got}
    assert "429" in by[("llm.request", "n2")]["summary"] and "rate limited" in by[("llm.request", "n2")]["summary"]
    assert "403" in by[("refused", "n2")]["summary"] and by[("no end recorded", "n1")]["summary"].startswith("llm.request")
    assert "ValueError: bad filter" in by[("tool.data_query", "n2")]["summary"] and by[("tool.data_query", "n2")]["seq"] is not None
    assert "disk full" in by[("handled error", "n2")]["summary"]
    assert sorted(p["summary"] for p in got if p["kind"] == "process") == [
        "process: exit code 128: fatal: not a git repository", "process: exit code 1: CUDA out of memory"]
    inferred = sorted(p["summary"] for p in got if p["source"] == "inferred")
    assert inferred[0].startswith("nope:") and inferred[1].startswith("run_command: exit 2")
    assert all(p["where"] == "edit_self-1 · conversation 0" and p["seq"] is None for p in got
               if p["source"] == "inferred")


def test_open_spans_run_only_while_their_writer_runs(run_dir):
    """A resume: the killed invocation's open spans have no end recorded; the live invocation's are running."""
    import time

    from ar_kernel import telemetry
    from fixtures.panel_run import Scripted
    from panel import timeline
    time.sleep(1.05 - time.time() % 1)            # a writer's id is its start second and pid: a new one here
    rec = Scripted(run_dir, t=round(time.time()), alive=True)
    telemetry.start("loop")
    rec.close(end=False)
    r = views.Run(run_dir)
    facts = sorted((o["name"], o["facts"].split(";")[0]) for o in views.open_spans(r))
    assert facts == [("ar.run", "running"), ("llm.request", "no end recorded"), ("loop", "running")]
    assert [p["summary"].split(":")[0] for p in problems.problems(r) if p["kind"] == "no end recorded"] == \
        ["llm.request"]
    spans = {s["label"]: s["status"] for s in timeline.timeline(r)["spans"] if s["status"] != "ok"}
    assert "no end recorded" in spans.values() and "running" not in spans.values()   # loop/ar.run have no lane


def test_problem_counts_by_kind(run_dir):
    _add_problems(run_dir)
    r = views.Run(run_dir)
    counts = problems.problem_counts(r, hours=2)
    assert sum(c["total"] for c in counts) == len(problems.problems(r))
    assert all(c["last hour"] == c["total"] for c in counts)          # everything happened in the window


def test_error_tool_outputs_are_marked_in_chats(run_dir):
    _add_problems(run_dir)
    r = views.Run(run_dir)
    items = views.conversation_chat(r, "n2", "edit_self", 1, 0)
    marked = [i["title"] for i in items if i["kind"] == "tool_error"]
    assert marked == ["⚠ Tool error: run_command", "⚠ Tool error: nope", "⚠ Tool error: data_query"]
    assert any(i["kind"] == "tool_output" and i["text"] == "fine" for i in items)


def test_tools_offered_show_each_tools_parameters():
    from panel.chat import tool_lines
    tools = [{"type": "function", "function": {"name": "data_query", "description": "Search the pool.", "parameters": {
        "type": "object", "required": ["limit"], "properties": {
            "limit": {"type": "integer"},
            "camera_motion": {"anyOf": [{"enum": ["moving", "static"], "type": "string"}, {"type": "null"}]},
            "clip_ids": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "function", "function": {"name": "bare"}}]
    assert tool_lines(tools) == ("- data_query(limit: integer, camera_motion?: moving | static, clip_ids?: [string]): "
                                 "Search the pool.\n- bare(): ")
    odd = [{"type": "function", "function": {"name": "t", "parameters": {"properties": {"x": True, "y": {"items": True, "type": "array"}}}}}]
    assert tool_lines(odd) == "- t(x?: any, y?: [any]): "
