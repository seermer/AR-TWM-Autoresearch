import json

from ar_kernel.process_digest import process_digest
from ar_kernel.telemetry.recorder import Recorder


def _rec(tmp_path):
    return Recorder(tmp_path)


def test_missing_events_is_empty_not_an_error(tmp_path):
    assert process_digest(tmp_path, "n9") == {}


def test_a_torn_line_is_skipped_and_the_rest_is_still_digested(tmp_path):
    """A killed writer leaves one unfinished line; it must not empty the whole digest."""
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


def test_each_phase_attempt_lists_its_conversations_snapshots_and_accepted_snapshot(tmp_path):
    rec = _rec(tmp_path)
    for attempt, accepted in ((1, None), (2, "s3")):
        base = dict(node="n1", phase="edit_self", attempt=attempt)
        rec.event("phase.start", **base)
        for conv, role, turns in (("p", "edit_planner", 2), ("c", "coder", 1), ("r", None, 1)):
            for _ in range(turns):
                rec.event("llm.request", conversation_id=f"{conv}{attempt}", **({"role": role} if role else {}),
                          payload={"body": {"messages": []}}, **base)
            if conv == "c":
                span = rec.event("tool.call", tool="snapshot", component="tools", payload={"args": {}}, **base)
                rec.event("tool.result", tool="snapshot", parent_span_id=span, component="tools",
                          payload={"tool": "snapshot", "result": {"snapshot_id": f"s{attempt + 1}"}}, **base)
        rec.event("phase.end", payload={"result": {"snapshot_id": accepted, "summary": "x"} if accepted else None,
                                        "error": None if accepted else "no result"}, **base)
    rec.event("llm.request", node="n1", phase="contract", attempt=1, conversation_id="smoke",
              payload={"body": {"messages": []}})                                  # the kernel's own smoke run
    conversations = lambda attempt: [
        {"role": "edit_planner", "turns": 2, "transcript": f"transcripts/edit_self-{attempt}/01-edit_planner.jsonl"},
        {"role": "coder", "turns": 1, "transcript": f"transcripts/edit_self-{attempt}/02-coder.jsonl"},
        {"role": "unlabeled", "turns": 1, "transcript": f"transcripts/edit_self-{attempt}/03-unlabeled.jsonl"}]
    assert process_digest(tmp_path, "n1")["phases"] == [
        {"phase": "edit_self", "attempt": 1, "conversations": conversations(1), "snapshots": ["s2"], "accepted": None},
        {"phase": "edit_self", "attempt": 2, "conversations": conversations(2), "snapshots": ["s3"], "accepted": "s3"}]


def test_the_digest_holds_no_seed_conventions_and_no_runtimes(tmp_path):
    rec = _rec(tmp_path)
    rec.event("gate.failed", node="n1", phase="gate")
    rec.event("gate.passed", node="n1", phase="gate")
    rec.event("train.end", node="n1", phase="train", duration_s=12.4)
    messages = [{"role": "tool", "content": "Error: ToolException('x')"}, {"role": "tool", "content": "exit 2\nboom"}]
    rec.event("llm.request", node="n1", phase="edit_self", attempt=1, conversation_id="c",
              payload={"body": {"messages": messages}})
    d = process_digest(tmp_path, "n1")
    assert "train_s" not in json.dumps(d) and "12.4" not in json.dumps(d)
    assert d["gates"] == {"failed": 1, "passed": 1}
    assert "local_errors" not in d and "compactions" not in json.dumps(d) and "rounds" not in d


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
    [phase] = process_digest(tmp_path, "n1")["phases"]
    assert phase["conversations"] == [{"role": "unlabeled", "turns": 1,
                                       "transcript": "transcripts/improve_recipe-1/01-unlabeled.jsonl"}]
    write_transcripts(tmp_path, "n1")
    assert len(list((tmp_path / "nodes" / "n1" / "transcripts").rglob("*.jsonl"))) == 1


def test_tools_jobs_and_ingest(tmp_path):
    rec = _rec(tmp_path)
    base = dict(node="n1", phase="improve_recipe", attempt=1)

    for tool, failed in (("data_ingest", False), ("data_ingest", True), ("hf_search", False)):
        rec.event("tool.call", tool=tool, component="tools", payload={"tool": tool, "args": {}}, **base)
        if failed:
            rec.event("tool.error", tool=tool, component="tools", payload={"error": "bad path"}, **base)
    rec.event("job.finished", node="n1", job_id="j1", state="done", gpu_seconds=120.0, payload={})
    rec.event("job.finished", node="n1", job_id="j2", state="failed", gpu_seconds=5.0, payload={})
    rec.event("ingest.accepted", node="n1", phase="ingest", payload={"clip_id": "x"})
    for _ in range(2):
        rec.event("ingest.rejected", node="n1", phase="ingest", payload={"reasons": ["moving clips need poses/<id>.npz"]})
    d = process_digest(tmp_path, "n1")
    assert d["tools"] == {"data_ingest": {"calls": 2, "errors": 1}, "hf_search": {"calls": 1, "errors": 0}}
    assert d["gpu_jobs"] == {"run": 2, "failed": 1}
    assert d["ingest"] == {"accepted": 1, "rejected": 2,
                           "reasons": [{"reason": "moving clips need poses/<id>.npz", "count": 2}]}
