from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.transcripts import write_transcripts


def test_one_readable_file_per_conversation_named_by_its_submit_tool(tmp_path):
    rec = Recorder(tmp_path)
    tools = [{"function": {"name": "read_file"}}, {"function": {"name": "submit_plan"}}]
    messages = [{"role": "system", "content": "# Role\nplan"}, {"role": "user", "content": "<context>x</context>"},
                {"role": "assistant", "content": "", "reasoning": "think",
                 "tool_calls": [{"id": "t1", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "file text"}]
    rec.event("llm.request", node="n1", phase="improve_recipe", attempt=1, conversation_id="c1",
              payload={"body": {"messages": messages, "tools": tools}})
    rec.event("llm.response", node="n1", phase="improve_recipe", attempt=1, conversation_id="c1",
              payload={"body": {"choices": [{"message": {"role": "assistant", "content": "all done"}}]}})
    rec.event("llm.request", node="n1", phase="contract", attempt=1, conversation_id="c2",
              payload={"body": {"messages": [{"role": "user", "content": "ping"}]}})
    write_transcripts(tmp_path, "n1")
    [path] = (tmp_path / "nodes" / "n1" / "transcripts").rglob("*.md")
    assert path.relative_to(tmp_path / "nodes" / "n1" / "transcripts").as_posix() == "improve_recipe-1/01-plan.md"
    text = path.read_text()
    for part in ("# Role\nplan", "### reasoning\n\nthink", "### tool call: read_file", "## tool result: read_file\n\nfile text",
                 "all done"):
        assert part in text


def test_missing_telemetry_writes_nothing_and_does_not_raise(tmp_path):
    write_transcripts(tmp_path, "n9")
    assert not (tmp_path / "nodes").exists()
