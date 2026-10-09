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


def test_file_tools_accept_the_workspace_from_another_root(tmp_path, monkeypatch):
    from agent.tools import resolve_inside
    monkeypatch.setenv("AR_WORKSPACE", str(tmp_path / "ws"))
    target = str(tmp_path / "ws" / "scratch" / "x.py")
    assert resolve_inside(str(tmp_path / "agent"), target) == Path(target)


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
        tools["edit_file"].invoke({"path": "c.txt", "old_string": "caption", "new_string": "title"})
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
    same = snap_timed_prompts.invoke({"segments_json": json.dumps(     # the dropped one shares its text with a kept one
        [{"time_range_s": [0.0, 1.0], "prompt": "a"}, {"time_range_s": [1.0, 1.1], "prompt": "c"},
         {"time_range_s": [1.1, 6.0], "prompt": "c"}]), "duration": 6.0})
    assert same.endswith('Dropped, shorter than a round after snapping: ["c"]')


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






def _data_lineage(depth):
    node = {"status": "scored", "score": 0.7, "error": None, "recipe": {"optimizer.lr": 1e-4},
            "data": {"d": {"format": "video_caption_camera", "prompt_mode": None, "clips": 8, "weight": 1.0,
                           "sources": {"hf:org/walks": 8}}},
            "rationale": 'why\n\nPlan: {"hypothesis": "more turning clips"}\nData: notes',
            "metrics": {"frame_aesthetics": 0.6},
            "aggregates": {"dimensions": {"quality": 0.7}, "metrics": {"frame_aesthetics": 0.6},
                           "groups": {"category": {"Urban": {"quality": 0.7}},
                                      "viewpoint": {"first_person": {"quality": 0.7}}}}}
    return [{**node, "node_id": "root", "score": 0.6, "data": {}, "recipe": None, "rationale": None},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


def _edit_lineage(depth):
    process = {"roles": [{"phase": "improve_recipe", "role": "plan", "conversations": 1, "turns": 12,
                          "compactions": 0}],
               "tools": {"data_ingest": {"calls": 9, "errors": 4}}, "tool_errors": [],
               "local_errors": {"run_command_nonzero": 3}, "gpu_jobs": {"run": 2, "failed": 1},
               "ingest": {"accepted": 5, "rejected": 4, "reasons": [{"reason": "moving clips need poses", "count": 4}]},
               "rounds": {"improve_recipe": {"plans": 1, "reports": 0}}, "gates": {},
               "attempts": [{"phase": "edit_self", "attempt": 1, "outcome": "contract_failed"},
                            {"phase": "edit_self", "attempt": 2, "outcome": "passed"}]}
    node = {"status": "scored", "error": None, "edit": {"summary": "Added a pose check tool."},
            "code_diff_stats": [{"path": "agent/tools.py", "added": 30, "removed": 2}], "process": process}
    return [{"node_id": "root", "status": "scored", "error": None, "edit": None, "code_diff_stats": [],
             "process": {"attempts": []}},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


GUIDE = {"cause_and_effect": {"dimension": "physics", "weight": 4.5, "measures": "physics holds"}}


def _recipe_ctx(**over):
    from ar_contract.models import RecipeContext
    return RecipeContext(**{**dict(
        nodes_remaining=1, attempt=1, max_attempts=3, n_gpus=4, metric_guide=GUIDE,
        tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
        recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
        resolution_allowlist=[[416, 736]], lora_allowlist=[[64, 64]]), **over})


def test_the_data_digest_has_scores_the_metric_guide_and_the_recent_lineage():
    from agent.briefing import LINEAGE_SHOWN, recipe_context
    ctx = _recipe_ctx(attempt=2, lineage=_data_lineage(100), siblings=_data_lineage(3)[1:],
                      retry={"kind": "train", "log_tail": "oom\nline", "data_commit": "c1"})
    text = recipe_context(ctx, [{"plan": {"hypothesis": "h"}, "report": "r"}])
    assert text.index("## Retry") < text.index("## Recipe") < text.index("## Metrics") < text.index("## Siblings") \
        < text.index("## Lineage")
    assert "```\noom\nline\n```" in text and '"hypothesis": "h"' in text
    assert "| cause_and_effect | physics | 4.5 | physics holds |" in text
    assert "|  | root | n90 | n91 |" in text                    # score tables: the root, then the last 10
    assert "| viewpoint: first_person / quality | 0.7 |" in text and "Urban" not in text
    assert text.count("\n### n") == LINEAGE_SHOWN + 2 and "### n99: scored, score 0.7" in text and "### n89:" not in text
    assert "- Hypothesis: more turning clips" in text and "hf:org/walks x8" in text
    assert "Data formats" not in text and len(text) < 40_000


def test_a_node_that_failed_before_scoring_still_renders_in_the_data_digest():
    from agent.briefing import recipe_context
    failed = {"node_id": "n1", "status": "train_failed", "score": None, "error": "loss became nan",
              "metrics": {}, "aggregates": None, "data": {}, "recipe": None, "rationale": None}
    text = recipe_context(_recipe_ctx(lineage=[_data_lineage(1)[0], failed]), None)
    assert "### n1: train_failed, score -" in text and "- Error: loss became nan" in text


def test_the_edit_digest_shows_how_runs_went_and_no_score():
    from ar_contract.models import EditContext
    from agent.briefing import edit_context
    ctx = EditContext(nodes_remaining=4, attempt=1, max_attempts=3, lineage=_edit_lineage(3),
                      siblings=_edit_lineage(2)[1:],
                      archive={"nodes": [{"node_id": "root", "status": "scored"}, {"node_id": "n1", "status": "scored"},
                                         {"node_id": "n2", "status": "train_failed"}]})
    text = edit_context(ctx, {"tools": "agent/tools.py -- the agent's tools"}, None)
    assert ", score " not in text and "| score |" not in text and "0.7" not in text      # a status may say "scored"
    assert "### Scores" not in text and "### Metrics" not in text and " min" not in text
    assert "3 nodes: scored 2, train_failed 1" in text
    assert "  - tools: agent/tools.py -- the agent's tools" in text
    for line in ("### n2: scored", "- Edit: Added a pose check tool.", "- Code changed: agent/tools.py (+30/-2)",
                 "- improve_recipe / plan: 12 model turns, 0 compactions",
                 "- Kernel tool calls (errors): data_ingest 9 (4)", "- Local tool failures: run_command_nonzero 3",
                 "- GPU jobs: 2 run, 1 failed",
                 "- Ingest: 5 accepted, 4 rejected; 4x moving clips need poses",
                 "- improve_recipe rounds: 1 plans, 0 reports back",
                 "- Failed attempts: edit_self 1 contract_failed"):
        assert line in text, line


def test_engineers_get_what_they_act_on_and_not_the_history():
    from agent.briefing import coder_context, engineer_context
    text = engineer_context(_recipe_ctx(lineage=_data_lineage(5), retry={"kind": "gate", "failures": ["too few clips"]}),
                            [{"plan": {"hypothesis": "h"}}])
    assert "## Recipe" in text and "## Metrics" in text and "too few clips" in text
    assert "## Lineage" not in text and "## Archive" not in text
    from ar_contract.models import EditContext
    ctx = EditContext(nodes_remaining=1, attempt=1, max_attempts=3, folders={"/agent": "the agent code you change"})
    coder = coder_context(ctx, {"harness": "agent/harness.py"})
    assert "  - harness: agent/harness.py" in coder and "## Folders\n\n- `/agent`: the agent code you change\n- `/workspace/scratch`: throwaway" in coder
    assert "## Folders\n\n- `/workspace/staging`: a separate mount" in engineer_context(
        _recipe_ctx(folders={"/workspace/staging": "a separate mount"}), None)
    assert "- Node id: n7\n" in engineer_context(_recipe_ctx(node_id="n7"), None)      # data_query's ingested_by asks for it
    retried = EditContext(nodes_remaining=1, attempt=2, max_attempts=3, retry={"kind": "contract", "error": "import fails"})
    assert "## Retry" in coder_context(retried, {}) and "import fails" in coder_context(retried, {})
    assert "## Retry" not in coder


def test_a_trained_node_with_an_empty_recipe_says_so():
    from agent.briefing import data_node
    node = {"node_id": "n1", "status": "scored", "score": 0.5, "rationale": None, "recipe": {},
            "data": {"d": {"format": "video_caption_camera", "weight": 1.0, "clips": 4, "sources": {"hf:o/s": 4}}},
            "data_commit": "c1"}
    assert data_node(node).endswith("- Recipe: all values kept default")
    assert "Recipe" not in data_node({**node, "data": {}})                    # the root trained nothing


def test_file_tools_name_their_folder():
    from agent.tools import make_file_tools
    tools = {t.name: t for t in make_file_tools("/agent")}
    assert "relative to /agent" in tools["read_file"].description and "tool root" not in str(
        [t.description for t in tools.values()])
    assert tools["write_file"].description.endswith("absolute under /agent or /workspace).")
    assert make_file_tools("/workspace")[2].description.endswith("absolute under /workspace).")
    assert "in /agent" in tools["run_command"].description and "`timeout_s` seconds" in tools["run_command"].description


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


def test_a_role_that_never_submits_is_reminded_three_times(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from agent import orchestration as orch
    monkeypatch.setattr(orch, "SCRATCH", tmp_path / "scratch")
    sent = []

    class Agent:
        async def ainvoke(self, state):
            sent.append(state["messages"][-1].content)
            return {"messages": state["messages"]}
    role = orch.Role.__new__(orch.Role)
    role.agent, role.messages, role.submissions = Agent(), [], [NS(value=None, name="submit_plan")]
    with pytest.raises(orch.NoSubmission):
        asyncio.run(role.run("the task"))
    assert sent == ["the task", *[orch.REMIND.format(tools="submit_plan")] * 3]


def test_a_review_that_never_answers_keeps_the_submitted_result(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from agent import orchestration as orch
    monkeypatch.setattr(orch, "RECORD", tmp_path / "plans.json")
    plan, accept, done, replan = (NS(value=None, name=n) for n in ("plan", "accept", "done", "replan"))
    calls = []

    class Planner:
        async def run(self, task):
            calls.append(task)
            if len(calls) == 2:                      # the review turn: no tool call, even when reminded
                raise orch.NoSubmission("the role finished without calling submit_plan or accept_result")
            plan.value = NS(model_dump=lambda: {"hypothesis": "h"})
            return plan

    class Engineer:
        messages = []

        async def run(self, work):
            done.value = NS(model_dump=lambda: {"data_commit": "c"}, model_dump_json=lambda indent: "{}")
            return done
    team = orch.Team(Planner(), plan, accept, Engineer(), done, replan, [])
    asyncio.run(orch.plan_and_engineer(team, task="t", show=lambda p: "work", engineer_context="ctx"))
    assert len(calls) == 2 and team.rounds == [{"plan": {"hypothesis": "h"}, "result": {"data_commit": "c"}}]
    done.value = None                                # with no result, a silent planner still fails the phase
    calls.clear(), calls.append("first")
    with pytest.raises(orch.NoSubmission, match="finished without"):
        asyncio.run(orch.plan_and_engineer(team, task="t", show=lambda p: "work", engineer_context="ctx"))


def test_submit_tool_validates_then_captures():
    from pydantic import ValidationError
    from agent.orchestration import EditPlan
    from agent.tools import submit_tool
    tool, box = submit_tool("submit_edit_plan", "d", EditPlan)
    plan = {"problem": "", "evidence": "e", "mechanism": "m", "check": "c"}
    with pytest.raises(ValidationError):
        asyncio.run(tool.ainvoke(plan))
    assert box.value is None
    asyncio.run(tool.ainvoke({**plan, "problem": "x"}))
    assert box.value.problem == "x"


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


def test_every_listed_component_exists_in_the_seed():
    from agent.orchestration import COMPONENTS
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


PLAN = {"hypothesis": "more walking clips", "expected_change": "camera path accuracy up",
        "data": "forward-walking clips with poses from the pool"}
EDIT_PLAN = {"problem": "the planner sees too many siblings", "evidence": "the first message is long",
             "mechanism": "show fewer siblings in the briefing", "check": "fewer planner compactions"}


def _scripts():
    """An accepted submit ends the role's run, so no reply follows one. Replies are consumed in order
    by whichever role calls the model next."""
    recipe = [
        [_call("submit_plan", {**PLAN, "data": "x" * 4001})],                       # planner: over the cap
        [_call("submit_plan", PLAN)],
        [_call("data_query", {"format": "video_caption_camera"}), _call("no_such_tool", {}),     # engineer
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("request_replan", {"report": "downloads are disabled; the pool has clips"})],
        [_call("submit_plan", {**PLAN, "hypothesis": "pool clips suffice"})],                    # planner
        [_call("submit_data_and_recipe", {"data_commit": C0, "notes": "pool clips", "rationale": "fits the data",
                                          "recipe": {"optimizer.max_steps": 100, "optimizer.lr": 1e-5}})],
        [_call("submit_plan", {**PLAN, "hypothesis": "pool clips suffice, trained for longer"})],   # planner: reviews
        [_call("submit_data_and_recipe", {"data_commit": C0, "notes": "pool clips, 200 steps", "rationale": "fits the data",
                                          "recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5}})],
        [_call("accept_result", {"reason": "the commit holds the pool clips and trains for 200 steps"})],
    ]
    edit = [
        [_call("read_file", {"path": "agent/prompts/planner.md"}),       # planning may try things out ...
         _call("run_command", {"command": "ls $AR_WORKSPACE/scratch && touch $AR_WORKSPACE/scratch/notes.txt"})],
        [_call("accept_result", {"reason": "nothing to do"})],          # refused: no result yet
        [_call("submit_edit_plan", {**EDIT_PLAN, "problem": ""})],       # invalid: no problem
        [_call("submit_edit_plan", EDIT_PLAN)],
        [_call("edit_file", {"path": "agent/entry.py", "old_string": "def edit_self(ctx: EditContext)",
                             "new_string": "def edit_self(ctx: EditContext, extra)"})],       # coder: breaks the self-test
        [_call("submit_edit", {"summary": "too early"})],
        [_call("edit_file", {"path": "agent/entry.py", "old_string": "def edit_self(ctx: EditContext, extra)",
                             "new_string": "def edit_self(ctx: EditContext)"})],
        [_call("edit_file", {"path": "agent/briefing.py", "old_string": "SIBLINGS_SHOWN = 10",
                             "new_string": "SIBLINGS_SHOWN = 8"})],
        [_call("submit_edit", {"summary": "The briefing now shows eight siblings."})],
        [_call("accept_result", {"reason": "briefing.py shows eight siblings and the self-test passes"})],
    ]
    return {"recipe": recipe, "edit": edit}


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    from ar_kernel.contract.verify import _MockCaption, _MockData, _MockHf
    from ar_kernel.tools.ask import MockAsk, register_ask_tool
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
    register_ask_tool(mcp, kit, MockAsk())
    from ar_kernel.tools.skills import register_skill_tool
    register_skill_tool(mcp, kit)
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
    assert any("Error invoking tool 'submit_plan'" in o and "at most 4000 characters" in o for o in outputs)
    assert any("no_such_tool is not a valid tool" in o for o in outputs)                       # unknown tool
    assert any(o.startswith("Error: ") and "ToolException" not in o and "downloads are disabled" in o
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
    assert [r["plan"]["hypothesis"] for r in rounds] == ["more walking clips", "pool clips suffice",
                                                         "pool clips suffice, trained for longer"]
    assert rounds[0]["report"].startswith("downloads are disabled") and "report" not in rounds[1]
    # the planner is shown each result: it had the second plan redone, then accepted the third
    assert [r.get("result", {}).get("notes") for r in rounds] == [None, "pool clips", "pool clips, 200 steps"]
    assert any(t.startswith("<engineer_result>") and '"notes": "pool clips"' in t and "accept_result" in t for t in tasks)
    assert "pool clips suffice" in body["result"]["rationale"]                  # the final plan
    planner_tools = _tools_of(rec, "n-recipe", "# Mission\nYou choose the one data idea")
    assert {"hf_search", "data_query", "data_fetch", "ask", "read_skill", "read_file", "run_command",
            "arxiv_search"} <= planner_tools
    engineer_tools = _tools_of(rec, "n-recipe", "# Mission\nYou build the training data")
    assert {"request_replan", "read_skill", "data_ingest", "arxiv_search"} <= engineer_tools
    assert "ask_planner" not in engineer_tools
    assert not {"hf_download", "data_ingest", "data_commit", "caption_videos"} & planner_tools
    # The panel shows one conversation per role, each holding all of its rounds.
    chats = _panel_chats(rec, "n-recipe", "improve_recipe")
    assert [role for role, _ in chats] == ["planner", "data_engineer"]
    planner_users = [i["text"] for i in chats[0][1] if i["kind"] == "user"]
    assert planner_users[0].startswith("<context>") and planner_users[1].startswith("<engineer_report>")
    assert planner_users[-1].startswith("<engineer_result>")
    engineer_users = [i["text"] for i in chats[1][1] if i["kind"] == "user"]
    assert engineer_users[0].startswith("<plan>") and engineer_users[-1].startswith("The planner revised the plan.")


def test_edit_self_plans_then_edits(kernel, tmp_path):
    rec = kernel[0]
    proc, body, agent = _run(kernel, tmp_path, "edit_self", "edit", "n-edit", BASE)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["summary"] == "The briefing now shows eight siblings."      # the coder's own words
    assert "SIBLINGS_SHOWN = 8" in (agent / "agent" / "briefing.py").read_text()
    # every role has a scratch folder, emptied when its turn ends; its command log stays
    assert not (tmp_path / "ws" / "scratch").exists()
    assert list((tmp_path / "ws" / "tool_output").glob("run_command-*.log"))
    [round_] = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert round_["plan"] == EDIT_PLAN and "report" not in round_
    assert round_["result"] == {"summary": "The briefing now shows eight siblings."}
    assert any("There is no result to accept yet" in o for o in _tool_outputs(rec, "n-edit"))
    outputs = _tool_outputs(rec, "n-edit")
    assert any("Error invoking tool 'submit_edit_plan'" in o and "problem" in o for o in outputs)
    assert any("The self-test failed" in o and "exactly one parameter" in o for o in outputs)   # submit refused
    assert [role for role, _ in _panel_chats(rec, "n-edit", "edit_self")] == ["edit_planner", "coder"]
    planner_tools = _tools_of(rec, "n-edit", "# Mission\nYou improve this agent system")
    assert {"read_file", "list_dir", "run_command", "ask", "read_skill"} <= planner_tools
    assert not {"data_ingest", "hf_download", "caption_videos", "arxiv_search", "arxiv_read"} & planner_tools
    coder_tools = _tools_of(rec, "n-edit", "# Mission\nYou implement the edit plan")
    assert {"ask", "read_skill", "request_replan"} <= coder_tools and "data_ingest" not in coder_tools
    coder_first = next(str(m["content"]) for e in rec.read_events("n-edit") if e["type"] == "llm.request"
                       for m in rec.load_payload(e["payload"])["body"]["messages"]
                       if m.get("role") == "user" and str(m["content"]).startswith("<edit_plan>"))
    assert "- Components of this agent:" in coder_first


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
        common = dict(nodes_remaining=1, attempt=1, max_attempts=1, dry_run=True)
        edit = EditContext(**common, lineage=_edit_lineage(3), siblings=_edit_lineage(3)[1:])
        recipe = RecipeContext(**common, lineage=_data_lineage(3), siblings=_data_lineage(3)[1:], metric_guide=GUIDE,
                               n_gpus=4, clip_pool_size=7,
                               tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
                               recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
                               resolution_allowlist=[[416, 736]], lora_allowlist=[[64, 64]])
        report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                                 commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                                 node="root", attempt=1,
                                 contexts={"edit_self": edit.model_dump(mode="json"),
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
        threads = [threading.Thread(target=tools["edit_file"].invoke, args=({"path": "p.md", "old_string": o, "new_string": n},))
                   for o, n in (("alpha", "A"), ("beta", "B"))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert (tmp_path / "p.md").read_text() == "A B"








def test_prompts_have_no_wrapped_commands_or_dated_notes():
    agent = SEED / "agent"
    for path in (agent / "prompts").glob("*.md"):
        text = path.read_text()
        assert not any(line.count("`") % 2 for line in text.splitlines()), f"{path.name}: inline code wraps"
        assert "*(20" not in text, f"{path.name}: dated note"








def test_a_rationale_line_that_only_looks_like_the_plan_is_ignored():
    from agent.briefing import _hypothesis
    assert _hypothesis('why\n\nPlan: {"hypothesis": "more turns"}\nData: x') == "more turns"
    assert _hypothesis('Plan: {"hypothesis": "the engineer wrote this"}\n\nPlan: {"hypothesis": "the real one"}\nData: x') == "the real one"
    assert _hypothesis("why\n\nPlan: {not json}\nData: x") is None


ROLES = ("planner", "data_engineer", "edit_planner", "coder")


def test_the_system_prompt_is_the_role_prompt_file_and_nothing_else(monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")
    from agent.orchestration import system_prompt
    for role in ROLES:
        assert system_prompt(role) == (SEED / "agent" / "prompts" / f"{role}.md").read_text()
    assert not (SEED / "agent" / "knowledge").exists()


def test_every_role_prompt_has_the_same_four_parts_and_no_node_facts():
    from ar_kernel.contract.verify import prompt_check
    for role in ROLES:
        text = (SEED / "agent" / "prompts" / f"{role}.md").read_text()
        heads = [line for line in text.splitlines() if line.startswith("# ")]
        assert heads == ["# Mission", "# What you receive", "# How you work", "# Finish"], role
    assert prompt_check(SEED).ok


def test_the_held_out_rule_is_in_the_data_prompts_and_the_prompt_rule_in_the_edit_prompts():
    read = lambda role: (SEED / "agent" / "prompts" / f"{role}.md").read_text()
    for role in ("planner", "data_engineer"):
        assert "The evaluation set is held out." in read(role)
    for role in ("edit_planner", "coder"):
        assert "A prompt holds a role's mission and general behaviour" in read(role)
    assert "score" not in read("coder").lower()


def test_plan_fields_are_capped_so_a_plan_cannot_dictate_file_contents():
    from pydantic import ValidationError
    from agent.orchestration import PLAN_FIELD_CHARS, DataPlan, EditPlan
    assert PLAN_FIELD_CHARS == 4000
    ok = dict(problem="p", evidence="e", mechanism="m", check="c")
    EditPlan(**ok)
    EditPlan(**{**ok, "evidence": "x" * 2800})            # the longest evidence a real plan carried
    with pytest.raises(ValidationError, match="at most 4000"):
        EditPlan(**{**ok, "mechanism": "x" * 5800})       # the shortest plan that dictated file contents
    assert DataPlan(hypothesis="h", expected_change="c", data="d").constraints == ""
