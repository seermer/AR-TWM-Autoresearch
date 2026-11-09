"""Transcripts, written by the gateway's store as each agent call ends."""
import json

from fastapi.testclient import TestClient

from ar_kernel.archive.activity import calls
from ar_kernel.archive.db import open_db
from ar_kernel.gateway.app import create_gateway_app
from ar_kernel.gateway.mock import MockBook
from ar_kernel.gateway.store import CallStore
from ar_kernel.tools.context import TokenRegistry

CHAT = "/v1/chat/completions"


def _setup(run, phase="improve_recipe", attempt=1):
    reg = TokenRegistry()
    caller = reg.issue(node="n1", phase=phase, attempt=attempt, workspace_host=run, staging_host=run)
    return reg, CallStore(run), caller


def _call(store, caller, messages, answer, role=None, tool=None):
    meta = store.begin(caller, CHAT, {"model": "m", "messages": messages}, tool=tool, role=role)
    store.end(meta, caller, status=200, body={"id": "x", "choices": [{"message": {"role": "assistant", "content": answer}}]})
    return [{"role": "assistant", "content": answer}]


def _files(run, folder="improve_recipe-1"):
    path = run / "nodes" / "n1" / "transcripts" / folder
    return {p.name: [json.loads(line) for line in p.read_text().splitlines()] for p in sorted(path.iterdir())}


def test_one_file_per_conversation_named_by_its_role_label_with_a_message_per_line(tmp_path):
    _, store, caller = _setup(tmp_path)
    messages = [{"role": "system", "content": "# Role\nplan"}, {"role": "user", "content": "<context>x</context>"},
                {"role": "assistant", "content": "", "reasoning": "think",
                 "tool_calls": [{"id": "t1", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "file text"}]
    meta = store.begin(caller, CHAT, {"model": "m", "messages": messages[:2]}, role="plan")
    store.end(meta, caller, status=200, body={"choices": [{"message": messages[2]}]})
    _call(store, caller, messages, "all done", role="plan")             # the same conversation, one turn on
    [(name, turns)] = _files(tmp_path).items()
    assert name == "01-plan.jsonl"
    assert [m["role"] for m in turns] == ["system", "user", "assistant", "tool", "assistant"]
    assert turns[2]["reasoning"] == "think" and turns[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert turns[3] == {"role": "tool", "tool_call_id": "t1", "content": "file text"} and turns[4]["content"] == "all done"
    assert [(c["turn"], c["transcript"]) for c in calls(open_db(tmp_path), "n1")] == [
        (0, "transcripts/improve_recipe-1/01-plan.jsonl"), (1, "transcripts/improve_recipe-1/01-plan.jsonl")]
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_compacted_conversation_continues_in_the_next_file_and_an_unlabeled_one_says_so(tmp_path):
    _, store, caller = _setup(tmp_path)
    first = [{"role": "user", "content": "task"}]
    first += _call(store, caller, first, "step 1")
    _call(store, caller, [{"role": "user", "content": "summary of the work so far"}], "step 2")     # compacted
    _call(store, caller, [{"role": "user", "content": "a helper"}], "h", role="coder")
    assert list(_files(tmp_path)) == ["01-unlabeled.jsonl", "02-unlabeled.jsonl", "03-coder.jsonl"]


def test_ask_calls_and_the_kernels_smoke_runs_write_no_transcript(tmp_path):
    _, store, caller = _setup(tmp_path)
    _call(store, caller, [{"role": "user", "content": "how many cars?"}], "42", tool="ask")
    _, smoke_store, smoke = _setup(tmp_path, phase="contract")
    _call(smoke_store, smoke, [{"role": "user", "content": "ping"}], "ok")
    assert not (tmp_path / "nodes").exists()
    assert [c["transcript"] for c in calls(open_db(tmp_path), "n1")] == [None, None]


def test_a_resumed_attempt_adds_files_and_overwrites_none(tmp_path):
    _, store, caller = _setup(tmp_path)
    _call(store, caller, [{"role": "user", "content": "before the kill"}], "a", role="plan")
    _, store, caller = _setup(tmp_path)                                  # a new process, the same attempt
    _call(store, caller, [{"role": "user", "content": "after the resume"}], "b", role="plan")
    files = _files(tmp_path)
    assert list(files) == ["01-plan.jsonl", "02-plan.jsonl"] and files["01-plan.jsonl"][0]["content"] == "before the kill"


def test_two_first_turns_at_once_get_different_numbers(tmp_path):
    _, store, caller = _setup(tmp_path)
    metas = [store.begin(caller, CHAT, {"model": "m", "messages": [{"role": "user", "content": t}]}, role=r)
             for t, r in (("one", "a"), ("two", "b"))]
    for meta in reversed(metas):
        store.end(meta, caller, status=200, body={"choices": [{"message": {"content": "x"}}]})
    assert list(_files(tmp_path)) == ["01-a.jsonl", "02-b.jsonl"]


def test_lines_keep_the_old_key_order_and_secrets_are_redacted(tmp_path):
    reg = TokenRegistry()
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    _call(CallStore(tmp_path, ["sk-SECRET123", ""]), caller,
          [{"role": "tool", "tool_call_id": "t", "content": f"key sk-SECRET123, token {caller.token}"}], "I saw sk-SECRET123")
    path = tmp_path / "nodes" / "n1" / "transcripts" / "improve_recipe-1" / "01-unlabeled.jsonl"
    assert path.read_text() == ('{"content": "key [REDACTED], token [REDACTED]", "role": "tool", "tool_call_id": "t"}\n'
                                '{"content": "I saw [REDACTED]", "role": "assistant"}\n')


def test_a_failed_call_keeps_its_request(tmp_path):
    _, store, caller = _setup(tmp_path)
    meta = store.begin(caller, CHAT, {"model": "m", "messages": [{"role": "user", "content": "q"}]})
    store.end(meta, caller, status=502, body={"error": {"message": "down"}})
    assert _files(tmp_path) == {"01-unlabeled.jsonl": [{"role": "user", "content": "q"}]}


def test_a_blocked_name_in_a_tool_result_never_reaches_a_transcript(tmp_path):
    reg, store, caller = _setup(tmp_path)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path,
                       mock_script="smoke")
    app = create_gateway_app(registry=reg, store=store, allowed_models={"m"}, upstream=None,
                             mocks=MockBook.default(), censor=["wbench"])
    body = {"model": "m", "messages": [{"role": "user", "content": "go"},
                                       {"role": "tool", "tool_call_id": "t", "content": "WBench ranks it. Clip 3 moves."}]}
    reply = TestClient(app).post(CHAT, json=body, headers={"Authorization": f"Bearer {caller.token}"})
    assert reply.status_code == 200
    [turns] = _files(tmp_path).values()
    assert "wbench" not in json.dumps(turns).lower() and turns[1]["content"] == "Clip 3 moves."


def test_a_transcript_that_cannot_be_written_stops_the_phase(tmp_path):
    reg, store, caller = _setup(tmp_path)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path,
                       mock_script="smoke")
    (tmp_path / "nodes" / "n1").mkdir(parents=True)
    (tmp_path / "nodes" / "n1" / "transcripts").write_text("a file where the folder goes")
    app = create_gateway_app(registry=reg, store=store, allowed_models={"m"}, upstream=None, mocks=MockBook.default())
    reply = TestClient(app).post(CHAT, json={"model": "m", "messages": [{"role": "user", "content": "go"}]},
                                 headers={"Authorization": f"Bearer {caller.token}"})
    assert reply.status_code == 500 and caller.fault.event.is_set()                 # the reply is withheld
    assert caller.fault.reason.startswith("gateway: the transcript transcripts/improve_recipe-1/01-unlabeled.jsonl")
    [row] = calls(open_db(tmp_path), "n1")                              # the call itself is still counted
    assert row["status"] == 200


def test_retries_of_one_turn_ending_at_once_all_write_the_transcript(tmp_path):
    import threading
    _, store, caller = _setup(tmp_path)
    first = [{"role": "user", "content": "task"}]
    history = first + _call(store, caller, first, "step 1") + [{"role": "user", "content": "go on"}]
    metas = [store.begin(caller, CHAT, {"model": "m", "messages": history}) for _ in range(20)]  # a client's retries
    errors = []

    def end(meta):
        try:
            store.end(meta, caller, status=200, body={"choices": [{"message": {"role": "assistant", "content": "step 2"}}]})
        except Exception as exc:                                    # noqa: BLE001 -- collected for the assert
            errors.append(exc)
    threads = [threading.Thread(target=end, args=(m,)) for m in metas]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and not list(tmp_path.rglob("*.tmp"))
    assert [m["content"] for m in _files(tmp_path)["01-unlabeled.jsonl"]] == ["task", "step 1", "go on", "step 2"]


def test_a_transcript_write_is_recorded_by_its_hash_not_its_text(tmp_path):
    import hashlib
    from conftest import telemetry_payload, telemetry_records
    from ar_kernel import telemetry
    telemetry.open(tmp_path)
    telemetry.start("ar.run")
    _, store, caller = _setup(tmp_path)
    first = [{"role": "user", "content": "task"}]
    first += _call(store, caller, first, "step 1")
    _call(store, caller, first + [{"role": "user", "content": "go on"}], "step 2")
    [path] = (tmp_path / "nodes" / "n1" / "transcripts" / "improve_recipe-1").iterdir()
    writes = [telemetry_payload(tmp_path, r["inputs"]) for r in telemetry_records(tmp_path)
              if r["kind"] == "start" and r["name"] == "fs.write"]
    writes = [w for w in writes if w["path"] == str(path)]
    assert len(writes) == 2 and all("text" not in w for w in writes)
    assert writes[-1]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()       # the version on disk
