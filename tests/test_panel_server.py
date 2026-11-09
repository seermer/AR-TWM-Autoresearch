import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from ar_kernel import telemetry
from fixtures.panel_run import Scripted, make_run
from panel import server, timeline, views


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path)


@pytest.fixture
def client(run_dir, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><div id=root></div>")
    (dist / "app.js").write_text("1")
    monkeypatch.setattr(server, "DIST", dist)
    return TestClient(server.create_app(run_dir))


def test_every_endpoint_answers_and_the_run_stays_untouched(client, run_dir):
    before = _snapshot(run_dir)
    ok = lambda url: (r := client.get(url), r.status_code == 200 or pytest.fail(f"{url}: {r.status_code} {r.text[:300]}"))[0].json()
    status = ok("/api/status")
    assert status["name"] == "r1" and status["alive"] is False and status["max_nodes"] == 3
    ok("/api/overview"), ok("/api/gpu"), ok("/api/timeline"), ok("/api/event-choices"), ok("/api/problems")
    ok("/api/training-compare"), ok("/api/eval-matrix"), ok("/api/selection"), ok("/api/cost"), ok("/api/files")
    for row in ok("/api/events?limit=5000")["rows"]:
        ok(f"/api/event/{row['seq']}")
    for node in ("root", "n1", "n2"):
        d = ok(f"/api/node/{node}")
        ok(f"/api/timeline?node={node}"), ok(f"/api/code/{node}"), ok(f"/api/data/{node}"), ok(f"/api/training/{node}")
        ok(f"/api/eval/{node}"), ok(f"/api/eval/{node}/7?other=root")
        assert d["training_attempts"] == ([1] if node == "n1" else [])
        for a in d["agent_attempts"]:
            agents = ok(f"/api/agents/{node}/{a['phase']}/{a['attempt']}")
            for c in agents["conversations"]:
                ok(f"/api/chat/{node}/{a['phase']}/{a['attempt']}/{c['index']}")
            for t in agents["graph"]["tasks"]:
                ok(f"/api/task/{node}/{a['phase']}/{a['attempt']}/{t['index']}")
            for log in agents["tool_logs"]:
                ok(f"/api/preview?path={log}"), ok(f"/api/tail?path={log}&offset=-10")
    tool = next(n for n in views.Run(run_dir).records.spans() if n.start["name"] == "tool.data_ingest")
    assert ok(f"/api/events?span={tool.id}")["total"] > 2
    ok("/api/clips/n1/ds1"), ok("/api/clip/clip1")
    commit = ok("/api/code/n1")["agent_commit"]
    for path in ok(f"/api/agent/tree?commit={commit}"):
        ok(f"/api/agent/file?commit={commit}&path={path}")
    assert client.get("/raw/config/run.json").status_code == 200
    assert _snapshot(run_dir) == before


def test_node_agents_code_and_cost_carry_the_new_fields(client):
    n1 = client.get("/api/node/n1").json()
    assert n1["parent_recipe"] is None and n1["recipe"] and n1["children"] == ["n2"]
    code = client.get("/api/code/n1").json()
    assert code["commits"][0]["label"] == "n1 · this node" and code["parent_commit"]
    assert any(f["status"] for f in code["node_files"]) and all("base" in a for a in code["attempts"])
    convs = client.get("/api/agents/n1/improve_recipe/1").json()
    tasks = convs["graph"]["tasks"]
    assert tasks and all(t["calls"] and t["role"] for t in tasks) and all(
        {"source", "target", "tools"} <= set(e) for e in convs["graph"]["edges"])
    chat = client.get(f"/api/chat/n1/improve_recipe/1/{convs['conversations'][0]['index']}").json()
    usage = [i for i in chat if i["kind"] == "usage"]
    assert usage and usage[0]["turn"] == 1 and usage[0]["prompt_tokens"] == 100 and all("at" in i for i in chat)
    calls = [i for i in chat if i["kind"] == "tool_call"]
    outputs = [i for i in chat if i["kind"] in ("tool_output", "tool_error")]
    assert calls and {c["ref"] for c in calls} >= {o["ref"] for o in outputs}
    assert client.get("/api/cost").json()["calls"][0]["prompt_tokens"] is not None
    status = client.get("/api/status").json()
    assert (status["calls"], status["tokens"], status["usd"]) == (1, 110, 0.001)        # the archive's llm_calls row
    assert status["events"] == status["last_seq"] + 1


def test_errors_are_json_and_paths_stay_in_the_run(client):
    assert client.get("/api/node/nope").status_code == 404
    assert client.get("/api/event/999999").status_code == 404
    assert client.get("/api/tail?path=missing.log").status_code == 404
    bad = client.get("/api/preview?path=../../etc/passwd")
    assert bad.status_code == 400 and "outside the run folder" in bad.json()["error"]
    assert client.get("/raw/%2E%2E/%2E%2E/etc/passwd").status_code in (400, 404)
    assert client.get("/api/agent/tree?commit=--output=x").json() == []
    assert client.get("/api/nope").status_code == 404
    assert client.get("/raw/config/run.json").headers["content-security-policy"] == "sandbox"


def test_events_page_past_any_size_and_problem_ids_are_stable(client, run_dir):
    total = client.get("/api/events?limit=1").json()["total"]
    first, rest = client.get("/api/events?limit=3").json(), client.get(f"/api/events?offset=3&limit={10 ** 6}").json()
    assert len(first["rows"]) + len(rest["rows"]) == total and rest["offset"] == 3
    ids = [p["id"] for p in client.get("/api/problems").json()]
    assert ids and len(set(ids)) == len(ids)
    rec = Scripted(run_dir)
    telemetry.alert("monitor.disk_low", "warning", "3 GB free")
    rec.close()
    again = [p["id"] for p in client.get("/api/problems").json()]
    assert len(again) == len(ids) + 1 and set(ids) <= set(again)


def test_nan_is_sent_as_null():
    body = json.loads(server.reply({"loss": [1.0, float("nan"), float("inf")]}).body)
    assert body == {"loss": [1.0, None, None]}


def test_raw_files_serve_byte_ranges(client):
    r = client.get("/raw/config/run.json", headers={"Range": "bytes=0-4"})
    assert r.status_code == 206 and len(r.content) == 5


def test_the_web_app_answers_every_other_path(client):
    assert "root" in client.get("/node/n1/agents?pa=x").text
    assert client.get("/app.js").text == "1"
    assert "root" in client.get("/../../etc/passwd").text


def test_launcher_refuses_an_unknown_run_or_an_unbuilt_app(tmp_path, monkeypatch, capsys):
    from panel import __main__ as launcher
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    assert launcher.main(["--run-id", "nope"]) == 2 and "no run 'nope'" in capsys.readouterr().err
    make_run(tmp_path)
    monkeypatch.setattr(server, "DIST", tmp_path / "missing")
    assert launcher.main(["--run-id", "r1"]) == 2 and "setup_envs.sh panel" in capsys.readouterr().err


def test_changed_files_keep_unusual_names(run_dir):
    import subprocess
    run = views.Run(run_dir)

    def git(*args, stdin=None):
        return subprocess.run(["git", "--git-dir", str(run_dir / "agents.git"), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              input=stdin or "", capture_output=True, text=True, check=True).stdout.strip()
    base = views.code_edits(run, "n1")["agent_commit"]
    blob = git("rev-parse", f"{base}:agent/entry.py")
    tree = git("mktree", stdin="".join(f"100644 blob {blob}\t{name}\n" for name in ("naïve.py", "a b.py")))
    head = git("commit-tree", tree, "-p", base, "-m", "x")
    got = {f["path"]: f for f in views.changed_files(run, base, head)}
    assert {"naïve.py", "a b.py"} <= set(got)                      # unquoted, spaces kept
    renamed = [f for f in got.values() if f["status"].startswith("R")]
    assert renamed and all(f["old_path"] != f["path"] and f["old_path"].startswith("agent/") for f in renamed)


def test_host_paths_are_found_only_where_a_container_path_starts():
    assert views.HOST_PATH.findall("see /home/u/workspace/x.txt, https://h/agent/y and /workspace/ok.log") == ["/workspace/ok.log"]


# ---- role graph ----

def _req(*messages):
    return {"messages": list(messages)}


def _res(content="", tools=()):
    message = {"role": "assistant", "content": content}
    if tools:
        message["tool_calls"] = [{"id": f"t-{t}", "type": "function", "function": {"name": t, "arguments": "{}"}} for t in tools]
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _fake(run_dir, calls) -> views.Run:
    """calls: (conversation, role, sent, answered, request, response), in phase attempt n p 1 of a new writer."""
    rec = Scripted(run_dir, t=0.0)
    try:
        attempt = telemetry.start("phase.attempt", {"node": "n", "phase": "p", "attempt": 1})
        spans = {}
        for t, answered, k in sorted([(c[2], False, k) for k, c in enumerate(calls)]
                                     + [(c[3], True, k) for k, c in enumerate(calls)]):
            conv, role, _, _, request, response = calls[k]
            rec.at(t)
            if answered:
                rec.llm_end(spans[k], response)
            else:
                spans[k] = rec.llm_start(attempt, conv, 0, request, role=role)
    finally:
        rec.close()
    return views.Run(run_dir)


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def _tool_turn(name, content=""):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": f"t-{name}", "type": "function", "function": {"name": name, "arguments": "{}"}}]}


def test_each_task_is_a_node_and_follows_the_task_that_ended_last(run_dir):
    s, go = ("system", "s"), ("user", "plan")
    first = _req(*_msgs(s, go))
    resumed = _req(*_msgs(s, go), _tool_turn("worker"), {"role": "tool", "tool_call_id": "t-worker", "content": "built"})
    run = _fake(run_dir, [
        ("a", "planner", 0.0, 1.0, first, _res(tools=["worker"])),
        ("b", "worker", 2.0, 3.0, _req(*_msgs(s, ("user", "do it"))), _res(tools=["run_command"])),
        ("b", "worker", 4.0, 5.0, _req(*_msgs(s, ("user", "do it"))), _res("worker done")),
        ("c", "reviewer", 6.0, 7.0, _req(*_msgs(s, ("user", "review"))), _res("ok")),
        ("a", "planner", 10.0, 11.0, resumed, _res(tools=["read"])),
        ("a", "planner", 12.0, 13.0, _req(*_msgs(s, go), _tool_turn("worker"),
                                          {"role": "tool", "tool_call_id": "t-worker", "content": "built"},
                                          _tool_turn("read"), {"role": "tool", "tool_call_id": "t-read", "content": "x"}),
         _res("final")),
    ])
    g = views.tasks(run, "n", "p", 1)
    assert [(t["role"], t["of_role"], t["role_tasks"], t["calls"], t["turns"]) for t in g["tasks"]] == [
        ("planner", 1, 2, 1, [1, 1]), ("worker", 1, 1, 2, [1, 2]), ("reviewer", 1, 1, 1, [1, 1]),
        ("planner", 2, 2, 2, [2, 3])]
    assert [(e["source"], e["target"], e["tools"]) for e in g["edges"]] == [
        (0, 1, ["worker"]), (1, 2, []), (2, 3, [])]
    items = views.task_chat(run, "n", "p", 1, 3)
    prior = [(i["kind"], i["text"]) for i in items if i.get("prior")]
    assert prior == [("system", "s"), ("user", "plan"), ("tool_call", "{}")]
    new = [(i["kind"], i.get("turn")) for i in items if not i.get("prior") and i["kind"] != "tools"]
    assert new == [("tool_output", None), ("tool_call", None), ("usage", 2), ("tool_output", None),
                   ("assistant", None), ("usage", 3)]
    assert all("at" not in i for i in items if i.get("prior"))


def test_tasks_running_in_parallel_are_listed_not_taken_as_callers(run_dir):
    sys_, user = {"role": "system", "content": "s"}, {"role": "user", "content": "go"}
    run = _fake(run_dir, [
        ("a", "planner", 0.0, 1.0, _req(sys_, user), _res(tools=["x", "y"])),
        ("b", "x", 2.0, 10.0, _req(sys_, user), _res(tools=["run"])),
        ("c", "y", 3.0, 4.0, _req(sys_, user), _res("y done")),
        ("b", "x", 11.0, 12.0, _req(sys_, user), _res("x done")),
        ("a", "planner", 13.0, 14.0, _req(sys_, user), _res("end")),
    ])
    g = views.tasks(run, "n", "p", 1)
    assert [(t["role"], t["calls"], t["alongside"]) for t in g["tasks"]] == [
        ("planner", 1, []), ("x", 2, []), ("y", 1, [1]), ("planner", 1, [])]
    assert [(e["source"], e["target"]) for e in g["edges"]] == [(0, 1), (0, 2), (1, 3)]


def test_compaction_stitches_only_within_one_role():
    from panel.chat import Call, Segment, chains
    payloads = {"a_req": {"body": _req({"role": "user", "content": "task"})}, "a_res": {"body": _res("the summary")},
                "b_req": {"body": _req({"role": "user", "content": "Summary: the summary"})}, "b_res": {"body": _res("ok")}}
    seg = lambda cid, ts, req, res, role: Segment(cid, [Call(cid + "1", cid, 0, ts, req, res, role=role)])
    same = chains([seg("A", 1.0, "a_req", "a_res", "x"), seg("B", 2.0, "b_req", "b_res", "x")], payloads.get)
    other = chains([seg("A", 1.0, "a_req", "a_res", "x"), seg("B", 2.0, "b_req", "b_res", "y")], payloads.get)
    assert [[s.conversation_id for s in c] for c in same] == [["A", "B"]]
    assert [[s.conversation_id for s in c] for c in other] == [["A"], ["B"]]


# ---- timeline ----

def test_timeline_has_a_lane_per_kind_of_span_and_marks(run_dir):
    rec = Scripted(run_dir, t=0.0)
    try:
        with telemetry.span("phase.attempt", {"node": "n", "phase": "x", "attempt": 1}) as attempt:
            rec.at(1)
            a = telemetry.start("process", {"command": ["python", "-m", "scripts.a"]})
            rec.at(2)
            b = telemetry.start("process", {"command": ["python", "b.py"]})
            rec.at(3)
            a.end("ok", {"returncode": 0})
            rec.at(4)
            b.end("error", {"returncode": 1})
            rec.at(5)
            with telemetry.span("train") as train:
                rec.at(9)
                train.output = {}
            rec.at(9.5)
            telemetry.start("eval.render")                     # never ends
            rec.at(9.6)
            telemetry.alert("monitor.stall", "warning", "stall")
            rec.at(10)
            attempt.output = {}
    finally:
        rec.close()
    got = timeline.timeline(views.Run(run_dir), "n")
    by_lane = {}
    for s in got["spans"]:
        by_lane.setdefault(s["lane"], []).append(s)
    assert [(s["label"], s["status"]) for s in by_lane["phase"]] == [("x-1", "ok")]
    assert [(s["label"], s["status"]) for s in by_lane["subprocess"]] == [("scripts.a", "ok"), ("b.py", "error")]
    assert by_lane["train"][0]["end"] == 9.0 and by_lane["render"][0]["status"] == "no end recorded"
    assert [(m["type"], m["level"]) for m in got["marks"]] == [("alert monitor.stall", "warning")]
