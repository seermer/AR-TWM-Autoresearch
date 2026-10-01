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
    assert process_digest(tmp_path, "n1")["llm"] == {
        "improve_recipe": {"conversations": 3, "turns": 6, "compactions": 1}}


def test_phase_times_gates_and_local_errors(tmp_path):
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
    assert set(d["phases"]) == {"edit_self", "train_s"} and d["phases"]["train_s"] == 12
    assert d["gates"] == {"failed": 1, "passed": 1}
    assert d["local_errors"] == {"tool_error_messages": 1, "run_command_nonzero": 1}


def test_the_digest_stays_under_two_kilobytes(tmp_path):
    rec = _rec(tmp_path)
    for i in range(40):
        rec.event("tool.error", node="n1", phase="p", tool=f"tool{i}", component="tools",
                  payload={"error": "e" * 400 + str(i)})
    assert len(json.dumps(process_digest(tmp_path, "n1"))) <= 2000


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
    assert process_digest(tmp_path, "n1")["llm"] == {
        "improve_recipe": {"conversations": 1, "turns": 1, "compactions": 0}}
    write_transcripts(tmp_path, "n1")
    assert len(list((tmp_path / "nodes" / "n1" / "transcripts").rglob("*.md"))) == 1
