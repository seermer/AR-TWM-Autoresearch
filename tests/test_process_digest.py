import json

from ar_kernel.process_digest import process_digest
from ar_kernel.telemetry.recorder import Recorder


def _rec(tmp_path):
    return Recorder(tmp_path)


def test_missing_events_is_empty_not_an_error(tmp_path):
    assert process_digest(tmp_path, "n9") == {}


def test_a_torn_line_is_skipped_and_the_rest_is_still_digested(tmp_path):
    """A killed writer leaves one unfinished line; it used to empty the whole digest."""
    rec = _rec(tmp_path)
    rec.event("tool.error", node="n1", phase="improve_recipe", tool="hf_download", component="tools",
              payload={"error": "403 denied"})
    with rec.events_path("n1").open("a") as f:
        f.write("{not json\n")
    assert [row["tool"] for row in process_digest(tmp_path, "n1")["tool_errors"]] == ["hf_download"]


def test_tool_errors_are_grouped_and_normalised(tmp_path):
    rec = _rec(tmp_path)
    for i in range(2):
        rec.event("tool.error", node="n1", phase="improve_recipe", tool="hf_download", component="tools",
                  payload={"error": f"403 (Request ID: Root={i}) job {'a' * 32} at /mnt/x/y/z.mp4 denied"})
    d = process_digest(tmp_path, "n1")
    [row] = d["tool_errors"]
    assert row["tool"] == "hf_download" and row["count"] == 2
    assert "<id>" in row["example"] and "<path>" in row["example"] and "Request ID" not in row["example"]
    assert len(row["example"]) <= 1000


def _reply(text):
    return {"body": {"choices": [{"message": {"role": "assistant", "content": text}}]}}


def test_conversations_turns_and_compactions_per_phase(tmp_path):
    rec = _rec(tmp_path)
    # c1 is compacted into c2; c3 is another role's conversation, not a compaction.
    firsts = {"c1": "build data", "c2": "Continued. <summary>\nSUMMARY-TEXT\n</summary>", "c3": "write a recipe"}
    for conv, turns in (("c1", 3), ("c2", 2), ("c3", 1)):
        for t in range(turns):
            rec.event("llm.request", node="n1", phase="improve_recipe", conversation_id=conv,
                      payload={"body": {"messages": [{"role": "user", "content": firsts[conv]}]}})
        rec.event("llm.response", node="n1", phase="improve_recipe", conversation_id=conv,
                  payload=_reply("SUMMARY-TEXT" if conv == "c1" else f"done {conv}"))
    assert process_digest(tmp_path, "n1")["roles"] == [
        {"phase": "improve_recipe", "role": "conversation", "conversations": 3, "turns": 6, "compactions": 1}]


def test_gates_and_local_errors_and_no_runtimes(tmp_path):
    rec = _rec(tmp_path)
    rec.event("phase.start", node="n1", phase="edit_self", attempt=1)
    rec.event("phase.end", node="n1", phase="edit_self", attempt=1)
    rec.event("phase.start", node="n1", phase="improve_recipe", attempt=1)      # killed: no end
    rec.event("gate.failed", node="n1", phase="gate")
    rec.event("gate.passed", node="n1", phase="gate")
    rec.event("train.end", node="n1", phase="train", duration_s=12.4)
    messages = [{"role": "tool", "content": "Error: ToolException('x')"}, {"role": "tool", "content": "exit 2\nboom"},
                {"role": "tool", "content": "exit 0\nfine"}]
    rec.event("llm.request", node="n1", phase="edit_self", conversation_id="c", payload={"body": {"messages": messages}})
    d = process_digest(tmp_path, "n1")
    assert "phases" not in d and "train_s" not in json.dumps(d)     # runtimes are not the edit planner's concern
    assert d["gates"] == {"failed": 1, "passed": 1}
    assert d["local_errors"] == {"tool_error_messages": 1, "run_command_nonzero": 1}


def test_the_digest_stays_under_its_size_cap(tmp_path):
    from ar_kernel.process_digest import MAX_BYTES
    rec = _rec(tmp_path)
    for i in range(40):
        rec.event("tool.error", node="n1", phase="p", tool=f"tool{i % 3}", component="tools",
                  payload={"error": f"failure kind {i}: " + "x y " * 600})
    d = process_digest(tmp_path, "n1")
    assert len(json.dumps(d)) <= MAX_BYTES and 0 < len(d["tool_errors"]) <= 8


def test_ask_calls_are_not_conversations_turns_or_compactions(tmp_path):
    """An ask is recorded like a gateway call under the same phase. A short answer ("42") also occurs
    in a role's first message, which would read as a compacted conversation continuing."""
    from ar_kernel.gateway.store import CallStore
    from ar_kernel.tools.context import TokenRegistry
    from ar_kernel.transcripts import write_transcripts
    rec = Recorder(tmp_path)
    store = CallStore(rec)
    caller = TokenRegistry(rec).issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                                      staging_host=tmp_path)

    def call(messages, answer, tool=None):
        meta = store.begin(caller, "/v1/chat/completions", {"model": "m", "messages": messages}, tool=tool)
        store.end(meta, caller, status=200, latency_s=1, attempts=1,
                  body={"id": "x", "choices": [{"message": {"role": "assistant", "content": answer}}]})

    call([{"role": "system", "content": "planner"}, {"role": "user", "content": "42 clips in the pool"}], "ok")
    call([{"role": "user", "content": [{"type": "text", "text": "how many cars?"}]}], "42", tool="ask")
    assert process_digest(tmp_path, "n1")["roles"] == [
        {"phase": "improve_recipe", "role": "conversation", "conversations": 1, "turns": 1, "compactions": 0}]
    write_transcripts(tmp_path, "n1")
    assert len(list((tmp_path / "nodes" / "n1" / "transcripts").rglob("*.jsonl"))) == 1


def test_roles_tools_jobs_ingest_and_rounds(tmp_path):
    rec = _rec(tmp_path)
    base = dict(node="n1", phase="improve_recipe", attempt=1)

    def turn(conv, tool):
        body = {"messages": [{"role": "user", "content": conv}],
                "tools": [{"type": "function", "function": {"name": tool}}]}
        rec.event("llm.request", conversation_id=conv, payload={"body": body}, **base)
        rec.event("llm.response", conversation_id=conv, latency_s=30.0, payload=_reply("ok"), **base)
    turn("c1", "submit_plan"), turn("c1", "submit_plan"), turn("c2", "submit_data_and_recipe")
    for tool, failed in (("data_ingest", False), ("data_ingest", True), ("hf_search", False)):
        rec.event("tool.call", tool=tool, component="tools", payload={"tool": tool, "args": {}}, **base)
        if failed:
            rec.event("tool.error", tool=tool, component="tools", payload={"error": "bad path"}, **base)
    rec.event("job.finished", node="n1", job_id="j1", state="done", gpu_seconds=120.0, payload={})
    rec.event("job.finished", node="n1", job_id="j2", state="failed", gpu_seconds=5.0, payload={})
    rec.event("ingest.accepted", node="n1", phase="ingest", payload={"clip_id": "x"})
    for _ in range(2):
        rec.event("ingest.rejected", node="n1", phase="ingest", payload={"reasons": ["moving clips need poses/<id>.npz"]})
    ws = tmp_path / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    ws.mkdir(parents=True)
    (ws / "plans.json").write_text(json.dumps([{"plan": {}, "report": "r"}, {"plan": {}}]))
    d = process_digest(tmp_path, "n1")
    assert d["roles"] == [
        {"phase": "improve_recipe", "role": "plan", "conversations": 1, "turns": 2, "compactions": 0},
        {"phase": "improve_recipe", "role": "data_and_recipe", "conversations": 1, "turns": 1, "compactions": 0}]
    assert d["tools"] == {"data_ingest": {"calls": 2, "errors": 1}, "hf_search": {"calls": 1, "errors": 0}}
    assert d["gpu_jobs"] == {"run": 2, "failed": 1}
    assert d["ingest"] == {"accepted": 1, "rejected": 2,
                           "reasons": [{"reason": "moving clips need poses/<id>.npz", "count": 2}]}
    assert d["rounds"] == {"improve_recipe": {"plans": 2, "reports": 1}}


def test_a_plans_file_of_another_shape_does_not_break_the_digest(tmp_path):
    rec = _rec(tmp_path)
    rec.event("tool.call", node="n1", phase="edit_self", tool="ask", component="tools", payload={})
    for attempt, text in ((1, "{not json"), (2, json.dumps({"rounds": "an edited agent's own shape"})),
                          (3, json.dumps(["a string", {"plan": 1}]))):
        ws = tmp_path / "nodes" / "n1" / "attempts" / f"edit_self-{attempt}" / "workspace"
        ws.mkdir(parents=True)
        (ws / "plans.json").write_text(text)
    d = process_digest(tmp_path, "n1")
    assert d["tools"] == {"ask": {"calls": 1, "errors": 0}}
    assert d["rounds"] == {"edit_self": {"plans": 1, "reports": 0}}


def test_the_kernels_own_contract_run_is_not_a_role(tmp_path):
    rec = _rec(tmp_path)
    for phase in ("contract", "edit_self"):
        rec.event("llm.request", node="n1", phase=phase, conversation_id=phase,
                  payload={"body": {"messages": [{"role": "user", "content": "ping"}]}})
    assert [r["phase"] for r in process_digest(tmp_path, "n1")["roles"]] == ["edit_self"]


def test_a_plans_json_that_is_not_a_regular_file_is_skipped(tmp_path):
    import os
    rec = _rec(tmp_path)
    rec.event("tool.call", node="n1", phase="edit_self", tool="ask", component="tools", payload={})
    ws = tmp_path / "nodes" / "n1" / "attempts" / "edit_self-1" / "workspace"
    ws.mkdir(parents=True)
    os.mkfifo(ws / "plans.json")                    # reading it would block forever
    ws2 = tmp_path / "nodes" / "n1" / "attempts" / "edit_self-2" / "workspace"
    ws2.mkdir(parents=True)
    (tmp_path / "real.json").write_text(json.dumps([{"plan": {}}]))
    (ws2 / "plans.json").symlink_to(tmp_path / "real.json")
    assert process_digest(tmp_path, "n1")["rounds"] == {}
