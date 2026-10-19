import json

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.transcripts import write_transcripts


def test_one_file_per_conversation_named_by_its_role_label_with_a_message_per_line(tmp_path):
    rec = Recorder(tmp_path)
    tools = [{"function": {"name": "read_file"}}, {"function": {"name": "submit_plan"}}]
    messages = [{"role": "system", "content": "# Role\nplan"}, {"role": "user", "content": "<context>x</context>"},
                {"role": "assistant", "content": "", "reasoning": "think",
                 "tool_calls": [{"id": "t1", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "file text"}]
    rec.event("llm.request", node="n1", phase="improve_recipe", attempt=1, conversation_id="c1", role="plan",
              payload={"body": {"messages": messages, "tools": tools}})
    rec.event("llm.response", node="n1", phase="improve_recipe", attempt=1, conversation_id="c1",
              payload={"body": {"choices": [{"message": {"role": "assistant", "content": "all done"}}]}})
    rec.event("llm.request", node="n1", phase="contract", attempt=1, conversation_id="c2",
              payload={"body": {"messages": [{"role": "user", "content": "ping"}]}})
    write_transcripts(tmp_path, "n1", rec)
    [path] = [p for p in (tmp_path / "nodes" / "n1" / "transcripts").rglob("*") if p.is_file()]      # one format only
    assert path.relative_to(tmp_path / "nodes" / "n1" / "transcripts").as_posix() == "improve_recipe-1/01-plan.jsonl"
    turns = [json.loads(line) for line in path.read_text().splitlines()]
    assert [m["role"] for m in turns] == ["system", "user", "assistant", "tool", "assistant"]
    assert turns[2]["reasoning"] == "think" and turns[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert turns[3] == {"role": "tool", "tool_call_id": "t1", "content": "file text"} and turns[4]["content"] == "all done"


def test_missing_telemetry_writes_nothing_and_does_not_raise(tmp_path):
    rec = Recorder(tmp_path)
    write_transcripts(tmp_path, "n9", rec)
    assert not (tmp_path / "nodes").exists() and not [e for e in rec.read_events() if e["type"] == "alert"]


def test_a_failure_is_a_warning_alert_not_an_exception(tmp_path):
    rec = Recorder(tmp_path)
    rec.event("llm.request", node="n1", phase="edit_self", attempt=1, conversation_id="c1",
              payload={"body": {"messages": [{"role": "user", "content": "x"}]}})
    (tmp_path / "nodes" / "n1").mkdir(parents=True)
    (tmp_path / "nodes" / "n1" / "transcripts").write_text("a file where the folder goes")
    write_transcripts(tmp_path, "n1", rec)
    [alert] = [e for e in rec.read_events() if e["type"] == "alert"]
    assert alert["kind"] == "transcripts_failed" and alert["level"] == "warning"


def test_a_conversation_without_a_label_is_unlabeled(tmp_path):
    rec = Recorder(tmp_path)
    rec.event("llm.request", node="n1", phase="edit_self", attempt=2, conversation_id="c1",
              payload={"body": {"messages": [{"role": "user", "content": "x"}]}})
    write_transcripts(tmp_path, "n1", rec)
    assert (tmp_path / "nodes" / "n1" / "transcripts" / "edit_self-2" / "01-unlabeled.jsonl").is_file()
