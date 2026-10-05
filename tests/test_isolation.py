import json

import pytest

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


def test_scrub_removes_this_machines_folders_wherever_the_project_sits(tmp_path, monkeypatch):
    """ffprobe's command line in a rejection, a log tail in a gate failure: they name kernel files by host
    path. The folders come from the config and the environment, never from a fixed path."""
    from ar_kernel.isolation import host_roots
    raw = {**CFG.raw, "paths": {**CFG.raw["paths"], "worldmodel": str(tmp_path / "elsewhere" / "WM"),
                                "runs_dir": "runs"}}
    moved = KernelConfig(raw=raw, repo_root=tmp_path / "proj" / "AR")
    monkeypatch.setenv("HOME", str(tmp_path / "home" / "someone"))
    roots = host_roots(moved)
    text = (f"ffprobe {tmp_path}/proj/AR/runs/r1/quarantine/ab/video.mp4; cwd {tmp_path}/elsewhere/WM/tools; "
            f"cache {tmp_path}/home/someone/.cache/hf; code {tmp_path}/proj/AR/kernel/x.py")
    assert scrub({"tail": [text]}, ["wbench"], roots) == {"tail": [
        "ffprobe <host>/runs/r1/quarantine/ab/video.mp4; cwd <host>/WM/tools; cache <host>/home/.cache/hf; "
        "code <host>/AR/kernel/x.py"]}
    assert str(tmp_path) not in json.dumps(scrub(text, [], roots))
    assert scrub(text, []) == text                                      # no folders given, nothing replaced



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
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == ["2605.25874", "meituan-longcat", "wbench"]
    assert audit(CFG, rec, "n2", "edit_self", 1) == ["wbench"]
    assert audit(CFG, rec, "n2") == ["2605.25874", "meituan-longcat", "wbench"]          # the whole node
    assert audit(CFG, rec, "n2", "improve_recipe", 2) == []


def test_tool_results_are_not_audited(tmp_path):
    from ar_kernel.isolation import audit
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.event("llm.request", node="n2", phase="improve_recipe", attempt=1, conversation_id="c", payload={"body": {
        "messages": [{"role": "tool", "content": "a paper abstract that compares models on WBench"}]}})
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []


NAMES = ["wbench", "meituan-longcat", "2605.25874"]


@pytest.mark.parametrize("text,left", [
    ("We evaluate on WBench (arXiv:2605.25874). Our model also improves FVD.", "Our model also improves FVD."),
    ("First line.\nWBench has 289 cases\nLast line.", "First line.\n\nLast line."),
    ('{"title": "WBench: A World Model Benchmark", "published": "2026-05-01"}', '{"title": "", "published": "2026-05-01"}'),
    ("see https://huggingface.co/datasets/meituan-longcat/WBench", "see https://huggingface."),
    ("DrawBench has prompts. Nothing to drop here.", "DrawBench has prompts. Nothing to drop here."),
])
def test_a_sentence_that_names_the_evaluation_is_dropped_whole(text, left):
    from ar_kernel.isolation import drop_sentences
    assert drop_sentences(text, NAMES) == left
    assert drop_sentences({"a": [text, 3]}, NAMES) == {"a": [left, 3]}


def test_the_gateway_drops_such_sentences_from_everything_the_model_did_not_write():
    from ar_kernel.isolation import censor_request
    chat = {"model": "m", "messages": [
        {"role": "user", "content": "Plan. The WBench paper is public."},
        {"role": "tool", "tool_call_id": "c1", "content": "WBench ranks models. Clip 3 is static."},
        {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "ok. see WBench"}]},
        {"role": "function", "name": "f", "content": "WBench list"},
        {"role": "assistant", "content": "I will look at WBench"}]}
    out = censor_request(chat, NAMES)
    assert [m["content"] for m in out["messages"]] == [
        "Plan.", "Clip 3 is static.", [{"type": "text", "text": "ok."}], "", "I will look at WBench"]
    assert chat["messages"][1]["content"].startswith("WBench")              # the caller's body is not changed
    responses = {"input": [{"type": "function_call_output", "call_id": "c", "output": "about WBench"},
                           {"type": "custom_tool_call_output", "output": "WBench"},
                           {"type": "function_call", "name": "f", "arguments": "WBench"},
                           {"type": "reasoning", "summary": [{"text": "WBench"}]},
                           {"type": "message", "role": "assistant", "content": "WBench"}]}
    out = censor_request(responses, NAMES)
    assert out["input"][0]["output"] == "" and out["input"][1]["output"] == ""
    assert out["input"][2:] == responses["input"][2:]                        # the model's own words stay
    assert censor_request({"input": "a plain string"}, NAMES) == {"input": "a plain string"}


def test_a_name_that_only_ends_like_a_blocked_one_is_left_alone():
    assert not blocked(CFG, "shunk031/DrawBench") and blocked(CFG, "x/run_wbench") and blocked(CFG, "WBench-mirror")
    assert scrub("DrawBench prompts; run_wbench.py; WBench", ["wbench"]) == "DrawBench prompts; run_render.py; render"


def test_the_audit_reads_any_response_shape_and_ignores_lookalike_names(tmp_path):
    from ar_kernel.isolation import audit
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    here = dict(node="n2", phase="improve_recipe", attempt=1)
    rec.event("llm.response", conversation_id="c", payload={"body": {"output": [
        {"type": "message", "content": [{"type": "output_text", "text": "DrawBench has prompts"}]}]}}, **here)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []
    rec.event("llm.response", conversation_id="c", payload={"body": {"output": [
        {"type": "function_call", "name": "run_command", "arguments": '{"command": "git clone x/WBench"}'}]}}, **here)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == ["wbench"]


import asyncio
import re
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"wbench|benchmark|meituan|2605\.25874|\b(test|eval\w*|proxy) cases?\b", re.IGNORECASE)


def _agent_visible_texts(tmp_path):
    """Everything an agent can read that the kernel or the seed ships: the seed agent, the contract
    package, the skills, and every kernel tool's description and schema."""
    from ar_kernel.contract.verify import _MockData, _MockHf
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.tools.ask import MockAsk, register_ask_tool
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.tools.context import TokenRegistry
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.gpu_jobs import build_gpu_backends, register_gpu_tools
    from ar_kernel.tools.hf_tools import register_hf_tools
    from ar_kernel.tools.jobs import JobQueue, register_job_tools
    from ar_kernel.tools.server import ToolKit, new_mcp
    from ar_kernel.tools.skills import register_skill_tool
    texts = {}
    for root in (REPO / "seed_agent", REPO / "contract", REPO / "kernel" / "ar_kernel" / "skills"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md", ".txt"):
                texts[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8")
    rec = Recorder(tmp_path)
    reg = TokenRegistry(rec)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    try:
        for backend in build_gpu_backends(CFG, tmp_path, [0, 1, 2, 3], reg, rec):
            queue.register(backend)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, queue)
        register_gpu_tools(mcp, kit, queue)
        register_caption_tool(mcp, kit, queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_skill_tool(mcp, kit)
        for tool in asyncio.run(mcp.list_tools()):
            texts[f"tool {tool.name}"] = f"{tool.description}\n{json.dumps(tool.input_schema)}"
    finally:
        queue.shutdown()
    return texts


def test_nothing_an_agent_sees_names_the_evaluation(tmp_path):
    texts = _agent_visible_texts(tmp_path)
    assert {"tool rollout_alayaworld", "tool read_skill", "tool data_ingest"} <= set(texts)
    found = {name: sorted(set(m.group(0) for m in FORBIDDEN.finditer(text))) for name, text in texts.items()}
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_agent_facing_messages_are_neutral():
    from ar_kernel.context_bundle import KERNEL_FAILURES
    from ar_kernel.eval.score import AGENT_METRICS
    from ar_kernel.isolation import EXCLUDED_CLIP, EXCLUDED_PROMPT
    words = " ".join([EXCLUDED_CLIP, EXCLUDED_PROMPT, *KERNEL_FAILURES.values(),
                      *[f"{alias} {text}" for alias, _, text in AGENT_METRICS.values()]])
    assert not FORBIDDEN.search(words) and "leak" not in words.lower()


def test_boilerplate_shared_by_several_evaluation_prompts_is_not_a_copy():
    assert not copies_held_out(CFG, "First-person view at eye level from the sidewalk.")
    assert not copies_held_out(CFG, "Third-person view from behind and slightly above the hiker.")
    assert copies_held_out(CFG, CASE["environment_prompt"])               # a whole prompt still is


def test_text_without_a_blocked_name_or_a_sentence_end_is_passed_through_at_once():
    """A 100,000-character line (base64, one row of numbers) took 20 s per model request."""
    import time
    from ar_kernel.isolation import drop_sentences
    text, started = "ab12" * 50_000, time.monotonic()
    assert drop_sentences(text, ["wbench"]) == text
    assert time.monotonic() - started < 2


def test_a_finished_nodes_training_config_and_logs_are_scrubbed_in_place(tmp_path):
    from ar_kernel.isolation import host_roots, scrub_train_files
    cfg = KernelConfig(raw=CFG.raw, repo_root=tmp_path / "proj" / "AR")
    attempt = tmp_path / "proj" / "AR" / "runs" / "r" / "nodes" / "n1" / "attempts" / "improve_recipe-2"
    (attempt / "train" / "logs" / "train_config").mkdir(parents=True)
    files = [attempt / "train_config.yaml", attempt / "train" / "train.log",
             attempt / "train" / "logs" / "train_config" / "train_node0.log"]
    for f in files:
        f.write_text(f"config: {attempt}/train_config.yaml\n[Train] step=1 loss=0.5\n")
    (attempt / "train" / "untouched.bin").write_bytes(b"\xff" + str(attempt).encode())
    scrub_train_files(attempt.parents[1], ["wbench"], host_roots(cfg))
    for f in files:
        assert f.read_text() == ("config: <host>/runs/r/nodes/n1/attempts/improve_recipe-2/train_config.yaml\n"
                                 "[Train] step=1 loss=0.5\n")
    assert (attempt / "train" / "untouched.bin").read_bytes().startswith(b"\xff")
    assert not list(attempt.rglob("*.tmp"))
