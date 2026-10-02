import json

from ar_kernel.config import KernelConfig
from ar_kernel.isolation import blocked, copies_held_out, scrub

CFG = KernelConfig.load()
CASE = json.loads((CFG.wbench / "data" / "cases" / "case_196.json").read_text())
SENTENCE = "Torch sconces on both walls cast flickering orange light."      # nine words of case 196's scene text


def test_blocked_names_match_any_case_and_position():
    assert blocked(CFG, "meituan-longcat/WBench") and blocked(CFG, "someone/my-wbench-mirror")
    assert not blocked(CFG, "org/walks") and not blocked(CFG, "") and not blocked(CFG, None)


def test_a_copied_run_of_eight_words_is_caught_with_a_prefix_a_suffix_or_a_split():
    assert SENTENCE in CASE["environment_prompt"]
    assert copies_held_out(CFG, f"A clip. {SENTENCE} Then more.")
    assert copies_held_out(CFG, "torch sconces, on both walls: cast flickering orange light")     # case, punctuation
    assert copies_held_out(CFG, "Torch sconces on both walls. Cast flickering orange light.")    # split in two
    assert copies_held_out(CFG, "unrelated", CASE["interactions"][0]["prompt"])


def test_short_phrases_and_paraphrases_pass():
    assert not copies_held_out(CFG, "First-person viewer.")
    assert not copies_held_out(CFG, "Wall torches throw a wavering orange glow along a stone castle corridor.")
    assert not copies_held_out(CFG, None, "", 3)


def test_scrub_replaces_names_in_nested_values():
    out = scrub({"log": ["[run_WBench] 4 items", 3], "ok": True}, ["wbench"])
    assert out == {"log": ["[run_render] 4 items", 3], "ok": True}


def _reply(rec, message, **where):
    rec.event("llm.response", conversation_id="c", payload={"body": {"choices": [{"message": message}]}}, **where)


def _tool_call(name, arguments):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}]}


def test_the_audit_reads_everything_the_model_wrote_in_one_attempt(tmp_path):
    from ar_kernel.isolation import audit
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    here = dict(node="n2", phase="improve_recipe", attempt=1)
    _reply(rec, _tool_call("run_command", json.dumps({"command": "ls /workspace"})), **here)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []
    _reply(rec, _tool_call("run_command", json.dumps(
        {"command": "curl -L https://huggingface.co/datasets/meituan-longcat/WBench/resolve/main/x.json"})), **here)
    _reply(rec, {"role": "assistant", "content": "done", "reasoning": "arXiv 2605.25874 describes the cases"}, **here)
    _reply(rec, {"role": "assistant", "content": "The eval is WBench."}, node="n2", phase="edit_self", attempt=1)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == [
        "reasoning names 2605.25874", "run_command names meituan-longcat", "run_command names wbench"]
    assert audit(CFG, rec, "n2", "edit_self", 1) == ["reply names wbench"]
    assert audit(CFG, rec, "n2", "improve_recipe", 2) == []


def test_tool_results_are_censored_not_audited(tmp_path):
    from ar_kernel.isolation import audit, censor_tool_results
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.event("llm.request", node="n2", phase="improve_recipe", attempt=1, conversation_id="c", payload={"body": {
        "messages": [{"role": "tool", "content": "a paper abstract that compares models on WBench"}]}})
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []
    names = ["wbench", "2605.25874"]
    chat = {"model": "m", "messages": [
        {"role": "user", "content": "plan"},
        {"role": "tool", "tool_call_id": "c1", "content": "WBench (arXiv:2605.25874) ranks models"},
        {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "see run_wbench.py"}]}]}
    out = censor_tool_results(chat, names)
    assert out["messages"][0] == chat["messages"][0]
    assert out["messages"][1]["content"] == "render (arXiv:render) ranks models"
    assert out["messages"][2]["content"] == [{"type": "text", "text": "see run_render.py"}]
    assert chat["messages"][1]["content"].startswith("WBench")              # the caller's body is not changed
    responses = {"input": [{"type": "function_call_output", "call_id": "c", "output": "about WBench"},
                           {"type": "message", "role": "user", "content": "x"}]}
    assert censor_tool_results(responses, names)["input"][0]["output"] == "about render"
    assert censor_tool_results({"input": "a plain string"}, names) == {"input": "a plain string"}
