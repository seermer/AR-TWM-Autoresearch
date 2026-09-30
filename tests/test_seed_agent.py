"""The seed agent: pure helpers in-process, then both entry points run the way a
container runs them (python -m ar_contract.run in a subprocess) against the kernel's
real gateway (scripted mock mode) and the Task 14 mock tool server, over Unix sockets."""
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


def test_planners_file_tools_are_read_only(tmp_path):
    from agent.tools import make_file_tools
    assert {t.name for t in make_file_tools(str(tmp_path), writable=False)} == {"read_file", "list_dir"}


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


def test_selftest_passes_on_the_seed_and_catches_a_broken_entry(tmp_path):
    from agent.orchestration import selftest
    copy = tmp_path / "copy"
    shutil.copytree(SEED, copy)
    assert selftest(str(copy)) == []
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, extra):\n    pass\n")
    errors = selftest(str(copy))
    assert any("edit_self" in e for e in errors) and any("improve_recipe" in e for e in errors)
    # keyword-only parameters are rejected too, as in the contract's static check (Task 14)
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, *, extra=1):\n    pass\n"
                                             "def improve_recipe(ctx):\n    pass\n")
    assert [e for e in selftest(str(copy)) if "exactly one parameter" in e] == [
        "agent/entry.py needs top-level edit_self(ctx) with exactly one parameter"]


# ---- both entry points against the kernel services -------------------------------------------

C0 = "0" * 64                                        # the Task 14 mock data_commit id


def _call(name, args):
    from ar_kernel.gateway.mock import function_call
    return function_call(name, args, f"call_{uuid.uuid4().hex[:12]}")


def _scripts():
    from ar_kernel.gateway.mock import message
    recipe = [
        [_call("submit_plan", {"hypotheses": ["more walking clips"]})],            # invalid: no actions
        [_call("submit_plan", {"hypotheses": ["more walking clips"], "actions": ["download walking clips"]})],
        [message("planned")],
        [_call("data_query", {"filter": {"format": "video_caption_camera"}}), _call("no_such_tool", {}),
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("request_replan", {"report": "downloads are disabled; the pool has clips"})],
        [message("asked for a new plan")],
        [_call("submit_plan", {"hypotheses": ["pool clips suffice"], "actions": ["commit the pool clips"]})],
        [message("replanned")],
        [_call("submit_data_and_recipe", {"data_commit": C0, "notes": "pool clips", "rationale": "fits the data",
                                          "recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5}})],
        [message("done")],
    ]
    edit = [
        [_call("read_file", {"path": "agent/prompts/planner.md"})],
        [_call("submit_edit_plan", {"component": "everything", "change": "x", "files": [],
                                    "rationale": "r", "expected_effect": "e"})],   # invalid component
        [_call("submit_edit_plan", {"component": "prompts", "change": "ask for posed clips first",
                                    "files": ["agent/prompts/planner.md"], "rationale": "static-only clips",
                                    "expected_effect": "more moving clips"})],
        [message("planned")],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext)",
                             "new": "def edit_self(ctx: EditContext, extra)"})],       # breaks the self-test
        [_call("submit_edit", {"summary": "too early"})],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext, extra)",
                             "new": "def edit_self(ctx: EditContext)"})],
        [_call("edit_file", {"path": "agent/prompts/planner.md", "old": "Finish by calling submit_plan.",
                             "new": "Prefer clips with poses. Finish by calling submit_plan."})],
        [_call("submit_edit", {"summary": "Changed the planner prompt to prefer clips with poses."})],
        [message("done")],
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
    assert max(turns) >= 2                     # the harness's resent chat history links (Task 19)
    tasks = {str(m["content"]) for e in rec.read_events("n-recipe") if e["type"] == "llm.request"
             for m in rec.load_payload(e["payload"])["body"]["messages"] if m.get("role") == "user"}
    first = [t for t in tasks if t.startswith("<plan>")]
    assert first and all('"resolution_allowlist": [[352, 640]]' in t for t in first)
    assert any(t.startswith("<engineer_report>\ndownloads are disabled") for t in tasks)      # back to the planner
    assert any(t.startswith("The planner revised the plan.") and "pool clips suffice" in t for t in tasks)
    rounds = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert [r["plan"]["hypotheses"] for r in rounds] == [["more walking clips"], ["pool clips suffice"]]
    assert rounds[0]["report"].startswith("downloads are disabled") and "report" not in rounds[1]
    assert "more walking clips" in body["result"]["rationale"] and "pool clips suffice" in body["result"]["rationale"]
    planner_tools = _tools_of(rec, "n-recipe", "# Role\nYou plan the training-data work")
    assert {"hf_search", "data_query", "read_file", "arxiv_search"} <= planner_tools
    assert not {"write_file", "run_command", "hf_download", "data_ingest", "data_commit"} & planner_tools


def test_edit_self_plans_exactly_one_component(kernel, tmp_path):
    rec = kernel[0]
    proc, body, agent = _run(kernel, tmp_path, "edit_self", "edit", "n-edit", BASE)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["component"] == "prompts"
    assert body["result"]["summary"].startswith("[prompts] ask for posed clips first")
    assert "prefer clips with poses" in body["result"]["summary"]
    assert "Prefer clips with poses." in (agent / "agent" / "prompts" / "planner.md").read_text()
    [round_] = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert round_["plan"]["component"] == "prompts" and "report" not in round_
    outputs = _tool_outputs(rec, "n-edit")
    assert any("Error invoking tool 'submit_edit_plan'" in o and "component" in o for o in outputs)
    assert any("The self-test failed" in o and "exactly one parameter" in o for o in outputs)   # submit refused
    planner_tools = _tools_of(rec, "n-edit", "# Role\nYou plan one improvement")
    assert {"read_file", "list_dir", "arxiv_search", "arxiv_read"} <= planner_tools
    assert not {"write_file", "edit_file", "run_command"} & planner_tools


@pytest.mark.docker
def test_seed_agent_passes_contract_verification(tmp_path):
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import ContractHarness, verify_contract
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.vcs.agents_repo import AgentsRepo
    harness = ContractHarness(KernelConfig.load(), tmp_path / "run", Recorder(tmp_path / "run"))
    harness.start()
    try:
        repo = AgentsRepo(tmp_path / "agents.git")
        commit = repo.init(SEED)
        report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                                 commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                                 node="root", attempt=1)
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
        assert "read the knowledge files whose descriptions match" in (SEED / "agent" / "prompts" / f"{name}.md").read_text()


def test_brief_marks_a_cut_and_closes_its_block(monkeypatch):
    from agent import orchestration
    monkeypatch.setattr(orchestration, "BRIEF_CHARS", 20)
    out = orchestration.brief({"lineage": "x" * 100})
    assert out.startswith("<context>") and out.endswith("</context>") and "[truncated]" in out


def test_prompts_and_knowledge_have_no_wrapped_commands_or_dated_notes():
    agent = SEED / "agent"
    for path in [*(agent / "prompts").glob("*.md"), *(agent / "knowledge").glob("*.md")]:
        text = path.read_text()
        assert not any(line.count("`") % 2 for line in text.splitlines()), f"{path.name}: inline code wraps"
        assert "*(20" not in text, f"{path.name}: dated note"
