"""The seed agent: pure helpers in-process, then both entry points run the way a
container runs them (python -m ar_contract.run in a subprocess) against the kernel's
real gateway (scripted mock mode) and the mock tool server, over Unix sockets."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import typing
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "seed_agent"
sys.path.insert(0, str(SEED))


def test_snap_segments_puts_boundaries_on_rounds():
    from agent.tools import snap_segments
    segs = [{"time_range_s": [0.0, 3.1], "prompt": "walk"}, {"time_range_s": [3.1, 8.0], "prompt": "turn"}]
    out = snap_segments(segs, duration=8.0)
    boundary = out[0]["time_range_s"][1]
    k = (boundary - 25 / 24) / (32 / 24)
    assert abs(k - round(k)) < 1e-9                         # on 25/24 + k*32/24
    assert out[0]["time_range_s"][0] == 0.0 and out[-1]["time_range_s"][1] == 8.0
    assert out[1]["time_range_s"][0] == boundary            # contiguous


def test_snap_segments_drops_segments_that_collapse():
    from agent.tools import snap_segments
    segs = [{"time_range_s": [0.0, 1.0], "prompt": "a"}, {"time_range_s": [1.0, 1.1], "prompt": "b"},
            {"time_range_s": [1.1, 6.0], "prompt": "c"}]
    assert [s["prompt"] for s in snap_segments(segs, duration=6.0)] == ["a", "c"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "sub/../../x"])
def test_file_tools_stay_inside_their_root(tmp_path, bad):
    from agent.tools import resolve_inside
    with pytest.raises(ValueError):
        resolve_inside(str(tmp_path), bad)


def test_file_tools_accept_paths_inside(tmp_path):
    from agent.tools import resolve_inside
    assert resolve_inside(str(tmp_path), "a/b.txt") == tmp_path / "a" / "b.txt"


def test_edit_file_needs_exactly_one_match():
    from agent.tools import replace_once
    assert replace_once("a b a", "b", "c") == "a c a"
    with pytest.raises(ValueError, match="found 2"):
        replace_once("a b a", "a", "c")
    with pytest.raises(ValueError, match="found 0"):
        replace_once("a", "z", "c")


def test_edit_file_refuses_non_utf8_files_and_leaves_them_untouched(tmp_path):
    from agent.tools import make_file_tools
    raw = b"caption: caf\xe9\n"                              # Latin-1, not UTF-8
    (tmp_path / "c.txt").write_bytes(raw)
    tools = {t.name: t for t in make_file_tools(str(tmp_path))}
    assert "caf" in tools["read_file"].invoke({"path": "c.txt"})     # reading replaces bad bytes
    with pytest.raises(ValueError, match="not UTF-8"):
        tools["edit_file"].invoke({"path": "c.txt", "old": "caption", "new": "title"})
    assert (tmp_path / "c.txt").read_bytes() == raw


def test_run_command_keeps_the_full_output_in_the_workspace(tmp_path, monkeypatch):
    from agent.tools import OUTPUT_SHOWN, make_file_tools
    agent, workspace = tmp_path / "agent", tmp_path / "ws"
    agent.mkdir(), workspace.mkdir()
    monkeypatch.setenv("AR_WORKSPACE", str(workspace))
    run = {t.name: t for t in make_file_tools(str(agent))}["run_command"]      # an edit_self tool root
    out = run.invoke({"command": "echo first; seq 1 20000"})
    logs = list((workspace / "tool_output").glob("*.log"))
    assert len(logs) == 1 and not list(agent.rglob("*.log"))                   # never inside /agent
    full = logs[0].read_text()
    assert full.startswith("$ echo first; seq 1 20000\nexit 0\n") and "first\n1\n2\n" in full
    assert full.endswith("20000\n")
    assert out.endswith("20000\n") and str(logs[0]) in out and len(out) < OUTPUT_SHOWN + 500
    assert "Long output" in out
    short = run.invoke({"command": "echo hi"})
    assert short == "exit 0\nhi\n" and len(list((workspace / "tool_output").glob("*.log"))) == 2


def test_read_tools_take_absolute_paths_anywhere(tmp_path):
    from agent.tools import make_file_tools
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "k.md").write_text("fact")
    tools = {t.name: t for t in make_file_tools(str(tmp_path / "root"))}
    assert tools["read_file"].invoke({"path": str(tmp_path / "elsewhere" / "k.md")}) == "fact"
    assert tools["list_dir"].invoke({"path": str(tmp_path / "elsewhere")}) == ["k.md"]


def test_read_file_shows_a_small_head_of_a_large_file_and_reads_line_ranges(tmp_path):
    from agent.tools import READ_LIMIT, READ_SHOWN, make_file_tools
    read = {t.name: t for t in make_file_tools(str(tmp_path))}["read_file"]
    (tmp_path / "big.txt").write_text("".join(f"line {i}\n" for i in range(READ_LIMIT // 5)))
    out = read.invoke({"path": "big.txt"})
    assert out.startswith("[Large file: 20000 lines") and len(out) < READ_SHOWN + 300
    assert read.invoke({"path": "big.txt", "offset": 10, "limit": 2}) == "[lines 11-12 of 20000]\nline 10\nline 11\n"


def test_discarded_changes_restores_the_tree_but_keeps_logs(tmp_path):
    from agent.tools import KEEP_BYTES, discarded_changes
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.txt").write_text("old")
    (tmp_path / "gone.txt").write_text("deleted, then back")
    (tmp_path / "link").symlink_to("keep/a.txt")
    (tmp_path / "big.bin").write_bytes(b"0" * (KEEP_BYTES + 1))
    with discarded_changes(str(tmp_path), keep=(str(tmp_path / "tool_output"),)):
        (tmp_path / "keep" / "a.txt").write_text("new")
        (tmp_path / "gone.txt").unlink()
        (tmp_path / "link").unlink()
        (tmp_path / "new" / "deep").mkdir(parents=True)
        (tmp_path / "new" / "deep" / "x.txt").write_text("x")
        (tmp_path / "keep" / "y.txt").write_text("y")
        (tmp_path / "big.bin").write_bytes(b"1")
        (tmp_path / "tool_output").mkdir()
        (tmp_path / "tool_output" / "cmd.log").write_text("log")
    assert (tmp_path / "keep" / "a.txt").read_text() == "old"
    assert (tmp_path / "gone.txt").read_text() == "deleted, then back"
    assert (tmp_path / "link").is_symlink() and (tmp_path / "link").read_text() == "old"
    assert not (tmp_path / "new").exists() and not (tmp_path / "keep" / "y.txt").exists()
    assert (tmp_path / "big.bin").read_bytes() == b"1"          # too large to keep a copy of
    assert (tmp_path / "tool_output" / "cmd.log").read_text() == "log"


def test_run_command_timeout_kills_what_the_shell_started(tmp_path, monkeypatch):
    import time
    from agent.tools import make_file_tools
    monkeypatch.setenv("AR_WORKSPACE", str(tmp_path))
    run = {t.name: t for t in make_file_tools(str(tmp_path))}["run_command"]
    out = run.invoke({"command": "(sleep 3; touch late.txt) & wait", "timeout_s": 1})
    assert out.startswith("timed out after 1s")
    time.sleep(3)
    assert not (tmp_path / "late.txt").exists()


def test_snap_timed_prompts_refuses_overlaps_and_names_dropped_segments():
    from agent.tools import snap_segments, snap_timed_prompts
    with pytest.raises(ValueError, match="overlap"):
        snap_segments([{"time_range_s": [0, 3], "prompt": "a"}, {"time_range_s": [2.9, 8], "prompt": "c"}], 8)
    out = snap_timed_prompts.invoke({"segments_json": json.dumps(
        [{"time_range_s": [0.0, 1.0], "prompt": "a"}, {"time_range_s": [1.0, 1.1], "prompt": "b"},
         {"time_range_s": [1.1, 6.0], "prompt": "c"}]), "duration": 6.0})
    assert out.endswith('Dropped, shorter than a round after snapping: ["b"]')


def test_long_tool_results_go_to_a_file(tmp_path, monkeypatch):
    from agent.harness import TOOL_RESULT_LIMIT, TOOL_RESULT_SHOWN, cap_result
    monkeypatch.setenv("AR_WORKSPACE", str(tmp_path))
    assert cap_result("t", "short") == "short"
    text = "".join(f"{i}\n" for i in range(TOOL_RESULT_LIMIT))
    out = cap_result("data_query", text)
    [saved] = (tmp_path / "tool_output").glob("data_query-*.txt")
    assert saved.read_text() == text and str(saved) in out and len(out) < TOOL_RESULT_SHOWN + 500
    assert out.split("\n", 2)[1] == "0" and out.endswith(text[-100:])          # head and tail


def test_check_roles_builds_every_role_and_the_selftest_uses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")
    from agent.orchestration import check_roles, selftest
    check_roles()
    copy = tmp_path / "copy"
    shutil.copytree(SEED, copy)
    (copy / "agent" / "prompts" / "coder.md").unlink()
    assert any("coder.md" in e for e in selftest(str(copy)))


def _lineage(depth):
    node = {"status": "scored", "score": 0.7, "component": "prompts", "edit": {"summary": "[prompts] x\n\ncoder text"},
            "data": {"d": {"format": "video_caption_camera", "clips": 8, "weight": 1.0}}, "recipe": {"optimizer.lr": 1e-4},
            "aggregates": {"dimensions": {"quality": 0.7}, "strata": {"category": {"Urban": {"quality": 0.7}}, "perspective": {"first_person": {"quality": 0.7}}},
                           "metrics": {"aesthetic_quality": 0.6}}, "process": {"phases": {"train_s": 600}}}
    return [{"node_id": "root", "score": 0.6, "status": "scored", "aggregates": node["aggregates"]},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


def test_the_digest_describes_the_recent_lineage_and_keeps_the_root_as_baseline():
    from ar_contract.models import RecipeContext
    from agent.briefing import LINEAGE_SHOWN, recipe_context
    ctx = RecipeContext(nodes_remaining=1, attempt=2, max_attempts=3, lineage=_lineage(100), n_gpus=4,
                        tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
                        recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
                        resolution_allowlist=[[416, 736]], lora_allowlist=[[64, 64]], format_rules="rules text",
                        retry={"kind": "train", "log_tail": "oom\nline", "data_commit": "c1"})
    text = recipe_context(ctx, [{"plan": {"hypothesis": "h"}, "report": "r"}])
    assert text.index("## Retry") < text.index("## Recipe") < text.index("## Lineage")      # small sections first
    assert "```\noom\nline\n```" in text and '"hypothesis": "h"' in text
    assert text.count("\n### n") == LINEAGE_SHOWN and "### n99:" in text and "### n89:" not in text
    assert "|  | root | n90 | n91 |" in text                    # score tables: the root, then the last 10
    assert "### Dimensions by group of test cases" in text and "| perspective: first_person / quality | 0.7 |" in text
    assert len(text) < 40_000                                   # the same size at any depth


def test_brief_marks_a_cut_and_points_to_the_full_context(monkeypatch):
    from agent import orchestration
    monkeypatch.setattr(orchestration, "BRIEF_CHARS", 20)
    out = orchestration.brief("x" * 100)
    assert out.startswith("<context>") and out.endswith("/context/context.json]\n</context>")


def test_replans_stop_at_the_round_limit():
    from langchain_core.tools import ToolException
    from agent.entry import MAX_ROUNDS
    from agent.orchestration import replan_tool
    rounds = [{}] * (MAX_ROUNDS - 1)
    tool, box = replan_tool(rounds)
    asyncio.run(tool.ainvoke({"report": "r"}))
    assert box.value.report == "r"
    rounds.append({})
    box.value = None
    with pytest.raises(ToolException, match="No replans left"):
        asyncio.run(tool.ainvoke({"report": "r"}))
    assert box.value is None


def test_submit_tool_validates_then_captures():
    from pydantic import ValidationError
    from agent.orchestration import EditPlan
    from agent.tools import submit_tool
    tool, box = submit_tool("submit_edit_plan", "d", EditPlan)
    plan = {"component": "everything", "change": "x", "files": [], "rationale": "r", "expected_effect": "e"}
    with pytest.raises(ValidationError):
        asyncio.run(tool.ainvoke(plan))
    assert box.value is None
    asyncio.run(tool.ainvoke({**plan, "component": "tools"}))
    assert box.value.component == "tools"


def test_kernel_results_prefer_structured_content():
    """MCP splits a list result into one text block per item (an empty list gives no text),
    so structured content is returned JSON-encoded; errors raise ToolException."""
    from types import SimpleNamespace as NS
    from langchain_core.tools import ToolException
    from agent.tools import result_text
    text = [NS(type="text", text="a"), NS(type="text", text="b")]
    assert result_text(NS(is_error=False, structured_content={"result": []}, content=[])) == '{"result": []}'
    assert result_text(NS(is_error=False, structured_content=None, content=text)) == "a\nb"
    with pytest.raises(ToolException, match="a\nb"):
        result_text(NS(is_error=True, structured_content=None, content=text))


def test_edit_components_match_the_contract():
    from ar_contract.models import EDIT_COMPONENTS
    from agent.orchestration import COMPONENTS, EditPlan
    assert tuple(COMPONENTS) == EDIT_COMPONENTS
    assert set(typing.get_args(EditPlan.model_fields["component"].annotation)) == set(EDIT_COMPONENTS)
    for component, where in COMPONENTS.items():
        assert (SEED / where.split(" ")[0].split("*")[0]).exists(), f"{component} -> {where}"


def test_selftest_passes_on_the_seed_and_catches_a_broken_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")                  # set in every container; building roles needs it
    from agent.orchestration import selftest
    copy = tmp_path / "copy"
    shutil.copytree(SEED, copy)
    assert selftest(str(copy)) == []
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, extra):\n    pass\n")
    errors = selftest(str(copy))
    assert any("edit_self" in e for e in errors) and any("improve_recipe" in e for e in errors)
    # keyword-only parameters are rejected too, as in the contract's static check
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, *, extra=1):\n    pass\n"
                                             "def improve_recipe(ctx):\n    pass\n")
    assert [e for e in selftest(str(copy)) if "exactly one parameter" in e] == [
        "agent/entry.py needs top-level edit_self(ctx) with exactly one parameter"]


# ---- both entry points against the kernel services -------------------------------------------

C0 = "0" * 64                                        # the mock data_commit id


def _call(name, args):
    from ar_kernel.gateway.mock import function_call
    return function_call(name, args, f"call_{uuid.uuid4().hex[:12]}")


def _scripts():
    """An accepted submit ends the role's run, so no reply follows one."""
    recipe = [
        [_call("submit_plan", {"hypothesis": "more walking clips"})],               # invalid: no actions
        [_call("submit_plan", {"hypothesis": "more walking clips", "actions": ["download walking clips"]})],
        [_call("data_query", {"filter": {"format": "video_caption_camera"}}), _call("no_such_tool", {}),
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("request_replan", {"report": "downloads are disabled; the pool has clips"})],
        [_call("submit_plan", {"hypothesis": "pool clips suffice", "actions": ["commit the pool clips"]})],
        [_call("submit_data_and_recipe", {"data_commit": C0, "notes": "pool clips", "rationale": "fits the data",
                                          "recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5}})],
    ]
    edit = [
        [_call("read_file", {"path": "agent/prompts/planner.md"}),       # planning may try things out ...
         _call("run_command", {"command": "echo junk >> agent/prompts/coder.md && touch agent/stray.py"})],
        [_call("submit_edit_plan", {"component": "everything", "change": "x", "files": [],
                                    "rationale": "r", "expected_effect": "e"})],   # invalid component
        [_call("submit_edit_plan", {"component": "prompts", "change": "ask for posed clips first",
                                    "files": ["agent/prompts/planner.md"], "rationale": "static-only clips",
                                    "expected_effect": "more moving clips"})],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext)",
                             "new": "def edit_self(ctx: EditContext, extra)"})],       # breaks the self-test
        [_call("submit_edit", {"summary": "too early"})],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext, extra)",
                             "new": "def edit_self(ctx: EditContext)"})],
        [_call("edit_file", {"path": "agent/prompts/planner.md", "old": "Finish by calling submit_plan.",
                             "new": "Prefer clips with poses. Finish by calling submit_plan."})],
        [_call("submit_edit", {"summary": "Changed the planner prompt to prefer clips with poses."})],
    ]
    return {"recipe": recipe, "edit": edit}


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    from ar_kernel.contract.verify import _MockCaption, _MockData, _MockHf
    from ar_kernel.gateway.app import create_gateway_app
    from ar_kernel.gateway.mock import MockBook
    from ar_kernel.gateway.store import CallStore
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.tools.context import TokenRegistry
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.hf_tools import register_hf_tools
    from ar_kernel.tools.jobs import JobQueue, register_job_tools
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
    run = tmp_path_factory.mktemp("run")
    rec = Recorder(run)
    registry, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    kit, mcp = ToolKit(registry, rec), new_mcp()
    register_data_tools(mcp, kit, _MockData())
    register_hf_tools(mcp, kit, _MockHf())
    register_job_tools(mcp, kit, queue)
    queue.register(_MockCaption())
    register_caption_tool(mcp, kit, queue)
    book = MockBook.default()
    for name, script in _scripts().items():
        book.add(name, script)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=registry, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=book), build_tool_app(mcp))
    yield rec, registry, services
    services.stop()
    queue.shutdown()


def _run(kernel, tmp_path, kind, script, node, ctx):
    rec, registry, services = kernel
    agent = tmp_path / "agent_copy"
    shutil.copytree(SEED, agent)
    ws, ctx_dir = tmp_path / "ws", tmp_path / "ctx"
    ws.mkdir(), ctx_dir.mkdir()
    (ctx_dir / "context.json").write_text(json.dumps(ctx))
    caller = registry.issue(node=node, phase=kind, attempt=1, workspace_host=ws, staging_host=tmp_path,
                            mock_script=script)
    env = {**os.environ, "AR_SOCKET_DIR": str(services.socket_dir), "AR_TOKEN": caller.token,
           "AR_DEFAULT_MODEL": "mock-model", "AR_WORKSPACE": str(ws), "AR_CONTEXT_DIR": str(ctx_dir),
           "AR_AGENT_DIR": str(agent)}
    proc = subprocess.run([sys.executable, "-m", "ar_contract.run", kind], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    registry.revoke(caller.token)
    return proc, json.loads((ws / "result.json").read_text()), agent


def _tool_outputs(rec, node) -> list[str]:
    """Every tool result the agent sent back to the model (Chat Completions `tool` messages),
    from gateway telemetry."""
    out = []
    for event in rec.read_events(node):
        if event["type"] == "llm.request":
            body = rec.load_payload(event["payload"])["body"]
            assert "messages" in body                            # the Chat Completions path
            out += [m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                    for m in body["messages"] if m.get("role") == "tool"]
    return out


def _panel_chats(rec, node, phase) -> list[tuple[str, list[dict]]]:
    """(role, chat items) per conversation, as the run panel builds them."""
    from panel.chat import chains, chat_items, infer_role, request_messages, segments, text_of
    prompts = {p.stem: p.read_text() for p in (SEED / "agent" / "prompts").glob("*.md")}
    out = []
    for chain in chains(segments(rec.read_events(node), node, phase, 1), rec.load_payload):
        system = next(text_of(m["content"]) for m in request_messages(rec.load_payload(chain[0].calls[0].request))
                      if m.get("role") == "system")
        out.append((infer_role(system, prompts), chat_items(chain, rec.load_payload, awaiting=False)))
    return out


def _tools_of(rec, node, system_start) -> set[str]:
    """The tools offered in the requests of the role whose system prompt starts with `system_start`."""
    names = set()
    for event in rec.read_events(node):
        if event["type"] == "llm.request":
            body = rec.load_payload(event["payload"])["body"]
            if str(body["messages"][0].get("content", "")).startswith(system_start):
                names |= {t["function"]["name"] for t in body.get("tools") or []}
    assert names, f"no requests for {system_start!r}"
    return names


def test_chat_model_works_sync_and_async_over_the_socket(kernel, tmp_path, monkeypatch):
    """The container has no network: both the sync and the async client must use the socket."""
    _, registry, services = kernel
    caller = registry.issue(node="n-chat", phase="edit_self", attempt=1, workspace_host=tmp_path,
                            staging_host=tmp_path, mock_script="smoke")
    monkeypatch.setenv("AR_SOCKET_DIR", str(services.socket_dir))
    monkeypatch.setenv("AR_TOKEN", caller.token)
    monkeypatch.setenv("AR_DEFAULT_MODEL", "mock-model")
    from langchain_core.messages import HumanMessage
    from ar_contract.client import chat_model
    model = chat_model()
    assert model.invoke([HumanMessage("ping")]).text == "ok"
    assert asyncio.run(model.ainvoke([HumanMessage("ping")])).text == "ok"
    registry.revoke(caller.token)


BASE = {"nodes_remaining": 3, "attempt": 1, "max_attempts": 3}


@pytest.mark.parametrize("kind", ["edit_self", "improve_recipe"])
def test_dry_run_against_the_kernel_services(kernel, tmp_path, kind):
    proc, body, _ = _run(kernel, tmp_path, kind, "smoke", f"dry-{kind}", {**BASE, "dry_run": True})
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])


def test_improve_recipe_full_flow(kernel, tmp_path):
    rec = kernel[0]
    ctx = {**BASE, "tunable_rules": {"optimizer.max_steps": {"type": "int"}, "optimizer.lr": {"type": "float"}},
           "resolution_allowlist": [[352, 640]]}
    proc, body, _ = _run(kernel, tmp_path, "improve_recipe", "recipe", "n-recipe", ctx)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["data_commit"] == C0
    assert body["result"]["recipe"] == {"optimizer.max_steps": 200, "optimizer.lr": 1e-5}   # int coerced
    outputs = _tool_outputs(rec, "n-recipe")
    assert any("Error invoking tool 'submit_plan'" in o and "actions" in o for o in outputs)   # bad args
    assert any("no_such_tool is not a valid tool" in o for o in outputs)                       # unknown tool
    assert any(o.startswith("Error: ToolException(") and "downloads are disabled" in o
               for o in outputs)                                                 # kernel tool error reported
    kinds = [e["type"] for e in rec.read_events("n-recipe")]
    assert "tool.call" in kinds and "tool.error" in kinds                        # kernel-side records
    turns = [e["turn_index"] for e in rec.read_events("n-recipe") if e["type"] == "llm.request"]
    assert max(turns) >= 2                     # the harness's resent chat history links
    tasks = {str(m["content"]) for e in rec.read_events("n-recipe") if e["type"] == "llm.request"
             for m in rec.load_payload(e["payload"])["body"]["messages"] if m.get("role") == "user"}
    first = [t for t in tasks if t.startswith("<plan>")]
    assert first and all("Resolutions (height x width): 352x640." in t for t in first)
    assert any(t.startswith("<engineer_report>\ndownloads are disabled") for t in tasks)      # back to the planner
    assert any(t.startswith("The planner revised the plan.") and "pool clips suffice" in t for t in tasks)
    rounds = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert [r["plan"]["hypothesis"] for r in rounds] == ["more walking clips", "pool clips suffice"]
    assert rounds[0]["report"].startswith("downloads are disabled") and "report" not in rounds[1]
    assert "pool clips suffice" in body["result"]["rationale"]                  # the final plan
    planner_tools = _tools_of(rec, "n-recipe", "# Role\nYou plan the training-data work")
    assert {"hf_search", "data_query", "read_file", "run_command", "arxiv_search"} <= planner_tools
    assert not {"hf_download", "data_ingest", "data_commit", "caption_videos"} & planner_tools
    # The panel shows one conversation per role, each holding all of its rounds.
    chats = _panel_chats(rec, "n-recipe", "improve_recipe")
    assert [role for role, _ in chats] == ["planner", "data_engineer"]
    planner_users = [i["text"] for i in chats[0][1] if i["kind"] == "user"]
    assert planner_users[0].startswith("<context>") and planner_users[-1].startswith("<engineer_report>")
    engineer_users = [i["text"] for i in chats[1][1] if i["kind"] == "user"]
    assert engineer_users[0].startswith("<plan>") and engineer_users[-1].startswith("The planner revised the plan.")


def test_edit_self_plans_exactly_one_component(kernel, tmp_path):
    rec = kernel[0]
    proc, body, agent = _run(kernel, tmp_path, "edit_self", "edit", "n-edit", BASE)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["component"] == "prompts"
    assert body["result"]["summary"].startswith("[prompts] ask for posed clips first")
    assert "prefer clips with poses" in body["result"]["summary"]
    assert "Prefer clips with poses." in (agent / "agent" / "prompts" / "planner.md").read_text()
    # ... but what the planner changed is undone before the coder starts; its command log stays
    assert (agent / "agent" / "prompts" / "coder.md").read_text() == (SEED / "agent" / "prompts" / "coder.md").read_text()
    assert not (agent / "agent" / "stray.py").exists()
    assert list((tmp_path / "ws" / "tool_output").glob("run_command-*.log"))
    [round_] = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert round_["plan"]["component"] == "prompts" and "report" not in round_
    outputs = _tool_outputs(rec, "n-edit")
    assert any("Error invoking tool 'submit_edit_plan'" in o and "component" in o for o in outputs)
    assert any("The self-test failed" in o and "exactly one parameter" in o for o in outputs)   # submit refused
    assert [role for role, _ in _panel_chats(rec, "n-edit", "edit_self")] == ["edit_planner", "coder"]
    planner_tools = _tools_of(rec, "n-edit", "# Role\nYou plan one improvement")
    assert {"read_file", "list_dir", "run_command", "arxiv_search", "arxiv_read"} <= planner_tools


@pytest.mark.docker
def test_seed_agent_passes_contract_verification(tmp_path):
    """On a context with a lineage and siblings, as a real node gets: the dry run builds the first message."""
    from ar_contract.models import EditContext, RecipeContext
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import ContractHarness, verify_contract
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.vcs.agents_repo import AgentsRepo
    harness = ContractHarness(KernelConfig.load(), tmp_path / "run", Recorder(tmp_path / "run"))
    harness.start()
    try:
        repo = AgentsRepo(tmp_path / "agents.git")
        commit = repo.init(SEED)
        common = dict(nodes_remaining=1, attempt=1, max_attempts=1, dry_run=True, lineage=_lineage(3),
                      siblings=_lineage(3)[1:])
        recipe = RecipeContext(**common, n_gpus=4, clip_pool_size=7, format_rules="rules text",
                               tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
                               recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
                               resolution_allowlist=[[416, 736]], lora_allowlist=[[64, 64]])
        report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                                 commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                                 node="root", attempt=1,
                                 contexts={"edit_self": EditContext(**common).model_dump(mode="json"),
                                           "improve_recipe": recipe.model_dump(mode="json")})
    finally:
        harness.stop()
    assert report.ok, report.to_retry()


def test_edit_file_parallel_calls_do_not_lose_edits(tmp_path):
    import threading
    from agent.tools import make_file_tools
    tools = {t.name: t for t in make_file_tools(str(tmp_path))}
    for _ in range(200):
        (tmp_path / "p.md").write_text("alpha beta")
        threads = [threading.Thread(target=tools["edit_file"].invoke, args=({"path": "p.md", "old": o, "new": n},))
                   for o, n in (("alpha", "A"), ("beta", "B"))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert (tmp_path / "p.md").read_text() == "A B"


def test_system_prompt_is_the_role_prompt_then_the_knowledge_index():
    from agent.orchestration import system_prompt
    text = system_prompt("planner")
    assert text.startswith("# Role") and "<memory>" not in text
    index = text.split("<knowledge>\n", 1)[1]
    for path in (SEED / "agent" / "knowledge").glob("*.md"):
        assert f"{path.resolve()}: Use when" in index
    assert "\n\n\n" not in text and not text.endswith("\n")


def test_knowledge_files_start_with_their_name_and_when_to_use_them():
    for path in (SEED / "agent" / "knowledge").glob("*.md"):
        head = path.read_text().split("\n")
        assert head[0] == "---" and head[1] == f"name: {path.stem}" and head[2].startswith("description: Use when")
        assert head[3] == "---"


def test_every_role_prompt_points_to_the_knowledge_index():
    for name in ("planner", "data_engineer", "edit_planner", "coder"):
        assert "Read a knowledge file only at the moment you are about to do what its description names" in (SEED / "agent" / "prompts" / f"{name}.md").read_text()


def test_prompts_and_knowledge_have_no_wrapped_commands_or_dated_notes():
    agent = SEED / "agent"
    for path in [*(agent / "prompts").glob("*.md"), *(agent / "knowledge").glob("*.md")]:
        text = path.read_text()
        assert not any(line.count("`") % 2 for line in text.splitlines()), f"{path.name}: inline code wraps"
        assert "*(20" not in text, f"{path.name}: dated note"


def test_the_digest_describes_the_parents_finished_children_before_the_lineage():
    from ar_contract.models import EditContext
    from agent.briefing import SIBLINGS_SHOWN, edit_context
    siblings = [{**n, "node_id": f"s{i}"} for i, n in enumerate(_lineage(15)[1:])]
    ctx = EditContext(nodes_remaining=1, attempt=1, max_attempts=3, lineage=_lineage(2), siblings=siblings)
    text = edit_context(ctx, {"prompts": "p.md"}, None)
    assert text.index("## Siblings") < text.index("## Lineage")
    assert text.count("\n### s") == SIBLINGS_SHOWN and "### s13:" in text and "### s3:" not in text
    assert "### s13:" in text and "| s3 | scored | 0.7 | prompts |" in text         # every sibling in one line
    assert "| root | s13 |" not in text                            # but not a column of the lineage tables
    assert "## Siblings" not in edit_context(EditContext(nodes_remaining=1, attempt=1, max_attempts=3), {}, None)


def test_the_eval_knowledge_states_the_score_weights_of_the_kernel_config():
    """The knowledge file is fixed text; it must not drift from configs/kernel.yaml."""
    import re
    from ar_kernel.config import KernelConfig
    from ar_kernel.eval.score import DIMENSION_METRICS
    weights = KernelConfig.load().get("eval.score_weights")
    text = (SEED / "agent" / "knowledge" / "eval_prompts_and_turns.md").read_text()
    [line] = [l for l in text.splitlines() if l.startswith("- Score: ")]
    names, value = re.search(r"means: (.+) weigh ([\d.]+) each, every other metric weighs 1\.", line).groups()
    assert set(re.findall(r"`(\w+)`", names)) == set(weights)
    assert set(weights.values()) == {float(value)}
    assert f"of the {len(DIMENSION_METRICS)} metric means" in line
    half = sum(weights.values()) == len(DIMENSION_METRICS) - len(weights)
    assert half and "Those four are half the score." in line


def test_an_earlier_node_is_described_by_what_the_phase_acts_on():
    from agent.briefing import data_node, edit_node
    node = {"node_id": "n1", "status": "scored", "score": 0.7, "component": "prompts",
            "edit": {"summary": "[prompts] ask for posed clips\n\ncoder text"},
            "code_diff_stats": [{"path": "agent/prompts/planner.md", "added": 2, "removed": 1}],
            "data": {"d": {"format": "video_caption_camera", "clips": 8, "weight": 1.0, "sources": {"hf:org/set": 8}}},
            "recipe": {"optimizer.lr": 1e-4},
            "rationale": 'why\n\nPlan: {"hypothesis": "more turning clips", "actions": ["a"]}\nData: notes',
            "process": {"phases": {"train_s": 600}, "tool_errors": [{"tool": "data_ingest", "count": 3, "example": "long message"}]}}
    data, edit = data_node(node), edit_node(node)
    assert "- Hypothesis: more turning clips" in data and "- Recipe: optimizer.lr 0.0001" in data
    assert "from hf:org/set x8" in data and "from hf:org/set x8" in edit
    assert not any(word in data for word in ("Edit:", "Code changed", "Process", "min", "edited"))   # no code, no timings
    assert "- Edit: [prompts] ask for posed clips" in edit and "agent/prompts/planner.md (+2/-1)" in edit
    assert "tool errors: data_ingest x3" in edit and "long message" not in edit and "Recipe" not in edit


def test_the_eval_knowledge_lists_each_dimensions_metrics_as_wbench_groups_them():
    import ast
    from ar_kernel.config import KernelConfig
    source = (KernelConfig.load().wbench / "main.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "DIMENSION_MAP" for t in n.targets))
    text = (SEED / "agent" / "knowledge" / "eval_prompts_and_turns.md").read_text()
    for dimension, metrics in ast.literal_eval(node.value).items():
        [line] = [l for l in text.splitlines() if l.startswith(f"  - `{dimension}`: ")]
        assert line.split(": ", 1)[1].rstrip(".").split(", ") == metrics


def test_the_engineer_gets_what_it_acts_on_and_the_planner_also_the_history():
    from ar_contract.models import RecipeContext
    from agent.briefing import engineer_context, recipe_context
    ctx = RecipeContext(nodes_remaining=1, attempt=2, max_attempts=3, lineage=_lineage(3), siblings=_lineage(3)[1:],
                        n_gpus=4, format_rules="rules text", retry={"kind": "gate", "failures": ["too few clips"]},
                        tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}})
    engineer, planner = engineer_context(ctx, None), recipe_context(ctx, None)
    for part in ("## This node", "## Retry", "too few clips", "## Recipe", "## Data formats"):
        assert part in engineer and part in planner
    assert planner.startswith(engineer)
    assert not any(part in engineer for part in ("## Lineage", "## Siblings", "## Archive"))
    assert "category: Urban" not in planner            # scene-category groups are left to context.json


def test_a_rationale_line_that_only_looks_like_the_plan_is_ignored():
    from agent.briefing import _hypothesis
    assert _hypothesis('why\n\nPlan: {"hypothesis": "more turns", "actions": ["a"]}\nData: x') == "more turns"
    assert _hypothesis("why\n\nPlan: {not json}\nData: x") is None
