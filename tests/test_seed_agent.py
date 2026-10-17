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
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "seed_agent"
sys.path.insert(0, str(SEED))


def _data_lineage(depth):
    node = {"status": "scored", "score": 0.7, "error": None, "recipe": {"optimizer.lr": 1e-4},
            "data": {"d": {"clips": 8, "weight": 1.0,
                           "sources": {"hf:org/walks": 8}}},
            "rationale": "why", "summary": "more turning clips",
            "metrics": {"frame_aesthetics": 0.6},
            "aggregates": {"dimensions": {"quality": 0.7}, "metrics": {"frame_aesthetics": 0.6},
                           "groups": {"category": {"Urban": {"quality": 0.7}},
                                      "viewpoint": {"first_person": {"quality": 0.7}}}}}
    return [{**node, "node_id": "root", "score": 0.6, "data": {}, "recipe": None, "rationale": None, "summary": None},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


def _edit_lineage(depth):
    process = {"phases": [{"phase": "improve_recipe", "attempt": 1, "snapshots": ["s1", "s2"], "accepted": "s2",
                           "conversations": [{"role": "recipe_planner", "turns": 12,
                                              "transcript": "transcripts/improve_recipe-1/01-recipe_planner.jsonl"}]}],
               "tools": {"data_ingest": {"calls": 9, "errors": 4}}, "tool_errors": [],
               "gpu_jobs": {"run": 2, "failed": 1},
               "ingest": {"accepted": 5, "rejected": 4, "reasons": [{"reason": "moving clips need poses", "count": 4}]},
               "gates": {},
               "attempts": [{"phase": "edit_self", "attempt": 1, "outcome": "contract_failed"},
                            {"phase": "edit_self", "attempt": 2, "outcome": "passed"}]}
    node = {"status": "scored", "error": None, "edit": {"snapshot_id": "a" * 40, "summary": "Added a pose check tool."},
            "code_diff_stats": [{"path": "agent/tools.py", "added": 30, "removed": 2}], "process": process}
    return [{"node_id": "root", "status": "scored", "error": None, "edit": None, "code_diff_stats": [],
             "process": {"attempts": []}},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


GUIDE = {"cause_and_effect": {"dimension": "physics", "weight": 4.5, "measures": "physics holds"}}


def _recipe_ctx(**over):
    from ar_contract.models import RecipeContext
    return RecipeContext(**{**dict(
        n_gpus=4, metric_guide=GUIDE,
        tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
        recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
        lora_allowlist=[[64, 64]]), **over})


# ---- contexts ----

def test_the_data_digest_has_scores_the_metric_guide_and_the_recent_lineage():
    from agent.briefing import LINEAGE_SHOWN, recipe_planner_context
    ctx = _recipe_ctx(lineage=_data_lineage(100), siblings=_data_lineage(3)[1:], tools=["data_query"])
    text = recipe_planner_context(ctx, "/workspace/scratch/recipe_planner-1")
    assert text.index("## Recipe") < text.index("## Metrics") < text.index("## Siblings") < text.index("## Lineage")
    assert "## Retry" not in text and "Kernel tools" not in text and "data_query" not in text
    assert "| cause_and_effect | physics | 4.5 | physics holds |" in text
    assert "|  | root | n90 | n91 |" in text                    # score tables: the root, then the last 10
    assert "| viewpoint: first_person / quality | 0.7 |" in text and "Urban" not in text
    assert text.count("\n### n") == LINEAGE_SHOWN + 2 and "### n99: scored, score 0.7" in text and "### n89:" not in text
    assert "- Summary: more turning clips" in text and "hf:org/walks x8" in text and "Hypothesis" not in text
    assert "Data formats" not in text and len(text) < 40_000


def test_a_node_that_failed_before_scoring_still_renders_in_the_data_digest():
    from agent.briefing import recipe_planner_context
    failed = {"node_id": "n1", "status": "train_failed", "score": None, "error": "loss became nan",
              "metrics": {}, "aggregates": None, "data": {}, "recipe": None, "rationale": None, "summary": None}
    text = recipe_planner_context(_recipe_ctx(lineage=[_data_lineage(1)[0], failed]), "/s")
    assert "### n1: train_failed, score -" in text and "- Error: loss became nan" in text


def test_the_edit_digest_shows_how_runs_went_and_no_score():
    from ar_contract.models import EditContext
    from agent.briefing import edit_planner_context
    ctx = EditContext(lineage=_edit_lineage(3),
                      siblings=_edit_lineage(2)[1:],
                      archive={"nodes": [{"node_id": "root", "status": "scored"}, {"node_id": "n1", "status": "scored"},
                                         {"node_id": "n2", "status": "train_failed"}]})
    text = edit_planner_context(ctx, {"tools": "agent/tools.py -- the agent's tools"}, "/workspace/scratch/edit_planner-1")
    assert ", score " not in text and "| score |" not in text and "0.7" not in text      # a status may say "scored"
    assert "### Scores" not in text and "### Metrics" not in text and " min" not in text
    assert "3 nodes: scored 2, train_failed 1" in text
    assert "  - tools: agent/tools.py -- the agent's tools" in text
    assert "- `/workspace/scratch/edit_planner-1`: your scratch folder" in text
    assert "- `/workspace/tool_output`: every tool call's full output" in text
    for line in ("### n2: scored", "- Summary: Added a pose check tool.", "- Code changed: agent/tools.py (+30/-2)",
                 "- improve_recipe attempt 1: recipe_planner 12 turns "
                 "(/nodes/n2/transcripts/improve_recipe-1/01-recipe_planner.jsonl); snapshots: s1, s2; result: s2",
                 "- Kernel tool calls (errors): data_ingest 9 (4)", "- GPU jobs: 2 run, 1 failed",
                 "- Ingest: 5 accepted, 4 rejected; 4x moving clips need poses",
                 "- Failed attempts: edit_self 1 contract_failed"):
        assert line in text, line
    assert "rounds" not in text and "compactions" not in text and "Local tool" not in text


def test_builders_and_reviewers_get_what_they_act_on_and_not_the_history():
    from ar_contract.models import EditContext
    from agent.briefing import coder_context, data_engineer_context, edit_reviewer_context, recipe_reviewer_context
    ctx = _recipe_ctx(lineage=_data_lineage(5),
                      target_skill={"name": "rain starting", "sentence": "rain begins to fall", "description": "d"})
    for build in (data_engineer_context, recipe_reviewer_context):
        text = build(ctx, "/workspace/scratch/x-3")
        assert "## Recipe" in text and "## Metrics" in text
        assert "## Lineage" not in text and "## Archive" not in text
        assert "## Target skill" in text and "- name: rain starting" in text and "- sentence: rain begins to fall" in text
        assert "`/workspace/scratch/x-3`: your scratch folder" in text
    assert "## Target skill" not in data_engineer_context(_recipe_ctx(), "/s")
    assert "- Node id: n7\n" in data_engineer_context(_recipe_ctx(node_id="n7"), "/s")      # data_query's ingested_by
    edit = EditContext(folders={"/agent": "the agent code you change"})
    for build in (coder_context, edit_reviewer_context):
        text = build(edit, {"harness": "agent/harness.py"}, "/workspace/scratch/coder-2")
        assert "  - harness: agent/harness.py" in text and "## Retry" not in text and "Nodes left" not in text
        assert "## Folders\n\n- `/agent`: the agent code you change\n- `/workspace/tool_output`: " in text


def test_a_trained_node_with_an_empty_recipe_says_so():
    from agent.briefing import data_node
    node = {"node_id": "n1", "status": "scored", "score": 0.5, "rationale": None, "recipe": {},
            "data": {"d": {"weight": 1.0, "clips": 4, "sources": {"dataset:o/s": 4}}},
            "data_commit": "c1"}
    assert data_node(node).endswith("- Recipe: all values kept at their base values")
    assert "Recipe" not in data_node({**node, "data": {}})                    # the root trained nothing
    baseline = "- The root is the baseline: the model with no further fine-tuning."
    assert baseline in data_node({**node, "node_id": "root", "data": {}}) and baseline not in data_node(node)


def test_brief_marks_a_cut_and_points_to_the_full_context(monkeypatch):
    from agent import orchestration
    monkeypatch.setattr(orchestration, "BRIEF_CHARS", 20)
    out = orchestration.brief("x" * 100)
    assert out.startswith("<context>") and out.endswith("/context/context.json]\n</context>")


# ---- roles and the delegate tool, in-process ----

class FakeModel:
    """chat_model's stand-in: replies from `replies[role]` in order, and records each request."""
    def __init__(self, replies: dict, seen: list, role: str) -> None:
        self.replies, self.seen, self.role = replies, seen, role

    def bind_tools(self, tools, **kw):
        return self

    def bind(self, **kw):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage
        self.seen.append((self.role, list(messages)))
        if isinstance(self.replies[self.role][0], Exception):
            raise self.replies[self.role].pop(0)
        text, calls = self.replies[self.role].pop(0)
        return AIMessage(content=text, tool_calls=[{"name": n, "args": a, "id": f"c{uuid.uuid4().hex[:6]}",
                                                    "type": "tool_call"} for n, a in calls])


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")
    monkeypatch.setenv("AR_WORKSPACE", str(tmp_path / "ws"))
    from agent import orchestration
    monkeypatch.setattr(orchestration, "SCRATCH", tmp_path / "ws" / "scratch")
    replies, seen = {}, []
    monkeypatch.setattr(orchestration, "chat_model", lambda model, role: FakeModel(replies, seen, role))
    return orchestration, replies, seen


class Schema:
    from pydantic import BaseModel

    class Report(BaseModel):
        """Hand back."""
        report: str


def test_a_role_keeps_its_conversation_and_scratch_until_closed(orch):
    o, replies, seen = orch
    replies["coder"] = [("", [("submit_coder", {"report": "one"})]), ("", [("submit_coder", {"report": "two"})])]
    role = o.Role("coder", lambda scratch: f"CONTEXT {scratch}", [], "submit_coder", Schema.Report)
    assert asyncio.run(role.run("first")) == {"report": "one"}
    assert role.scratch.is_dir() and role.scratch.name.startswith("coder-")
    assert asyncio.run(role.run("second")) == {"report": "two"}
    first, second = (messages for _, messages in seen)
    assert first[0].content == role.system and first[1].content == f"<context>\nCONTEXT {role.scratch}\n</context>\n\nfirst"
    assert [m.content for m in second[1:] if m.type == "human"][-1] == "second" and len(second) == 5
    role.close()
    assert not role.scratch.exists()


def test_a_role_whose_call_raises_keeps_the_work_before_it(orch):
    o, replies, seen = orch
    replies["coder"] = [("", [("write_file", {"path": "a.txt", "content": "x"})]), RuntimeError("gateway gave up"),
                        ("", [("submit_coder", {"report": "two"})])]
    role = o.Role("coder", lambda scratch: "C", [], "submit_coder", Schema.Report)
    with pytest.raises(RuntimeError, match="gateway gave up"):
        asyncio.run(role.run("first"))
    asyncio.run(role.run("second"))
    assert [m.type for m in seen[-1][1]] == ["system", "human", "ai", "tool", "human"]
    assert seen[-1][1][3].content.startswith("wrote 1 chars to ") and seen[-1][1][4].content == "second"


def test_a_role_that_never_submits_is_reminded_four_times_then_raises(orch):
    o, replies, seen = orch
    replies["coder"] = [("I am done", [])] * 5
    role = o.Role("coder", lambda scratch: "C", [], "submit_coder", Schema.Report)
    with pytest.raises(o.NoSubmission, match="coder ended its turn without calling submit_coder, after 4 reminders"):
        asyncio.run(role.run("task"))
    assert [m.content for m in seen[-1][1] if m.type == "human"][1:] == [o.REMIND.format(tool="submit_coder")] * 4


def test_a_check_refuses_a_submission_or_adds_to_it(orch):
    from langchain_core.tools import ToolException
    o, replies, seen = orch

    async def check(value):
        if value["report"] == "bad":
            raise ToolException("not yet")
        return {"snapshot_id": "s1"}
    replies["coder"] = [("", [("submit_coder", {"report": "bad"})]), ("", [("submit_coder", {"report": "good"})])]
    role = o.Role("coder", lambda scratch: "C", [], "submit_coder", Schema.Report, check)
    assert asyncio.run(role.run("task")) == {"report": "good", "snapshot_id": "s1"}
    assert seen[-1][1][-1].content == "Error: not yet"


class FakeRole:
    def __init__(self, results, log, name="coder"):
        self.results, self.log, self.name, self.closed = results, log, name, False
        self.tools, self.submit = [], f"submit_{name}"

    async def run(self, briefing):
        self.log.append((self.name, briefing))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def test_the_delegate_tool_runs_the_builder_then_a_fresh_reviewer_and_hands_back(orch):
    o = orch[0]
    from agent.entry import MAX_ROUNDS
    from langchain_core.tools import ToolException
    log, reviewers, handbacks = [], [], []
    builder = FakeRole([{"report": f"r{i}", "snapshot_id": f"s{i}"} for i in range(MAX_ROUNDS)], log)

    def reviewer():
        reviewers.append(FakeRole([{"verdict": "falls_short", "findings": "f"}], log, "edit_reviewer"))
        return reviewers[-1]
    tool = o.delegate(builder, reviewer, o.EditPlan, handbacks, lambda briefing, b: f"REVIEW {b['snapshot_id']}")
    plan = dict(problem="p", evidence="e", mechanism="m", check="c")
    out = json.loads(asyncio.run(tool.ainvoke(plan)))
    assert out == {"snapshot_id": "s0", "report": "r0", "review": {"verdict": "falls_short", "findings": "f"}}
    assert log[0][1].startswith("<plan>\n") and log[1] == ("edit_reviewer", "REVIEW s0") and reviewers[0].closed
    asyncio.run(tool.ainvoke(plan))
    assert log[2][1].startswith("The planner revised the plan.\n\n<plan>") and len(reviewers) == 2
    assert tool.metadata == {"ends_batch": True} and tool.name == "coder"
    for _ in range(MAX_ROUNDS - 2):
        asyncio.run(tool.ainvoke(plan))
    with pytest.raises(ToolException, match=f"No calls left: coder was called {MAX_ROUNDS} times"):
        asyncio.run(tool.ainvoke(plan))
    assert [h["snapshot_id"] for h in handbacks] == [f"s{i}" for i in range(MAX_ROUNDS)]


def test_a_builder_that_fails_reaches_the_planner_as_an_error_and_uses_a_call(orch):
    o = orch[0]
    from agent.harness import run_tool
    builder = FakeRole([o.NoSubmission("coder ended its turn without calling submit_coder, after 4 reminders"),
                        {"report": "r", "snapshot_id": "s"}], [])
    reviewer = lambda: FakeRole([{"verdict": "carries_out_plan", "findings": "f"}], [])
    handbacks = []
    tool = o.delegate(builder, reviewer, o.EditPlan, handbacks, lambda b, s: "x")
    call = {"name": "coder", "args": dict(problem="p", evidence="e", mechanism="m", check="c"), "id": "c1"}
    message = asyncio.run(run_tool({"coder": tool}, call, "edit_planner", None))
    assert message.status == "error" and message.content == (
        "Error: NoSubmission: coder ended its turn without calling submit_coder, after 4 reminders")
    assert handbacks == []
    asyncio.run(run_tool({"coder": tool}, call, "edit_planner", None))
    assert [h["snapshot_id"] for h in handbacks] == ["s"]


def test_submit_result_accepts_only_a_handed_back_snapshot_and_lists_the_valid_ones(orch):
    o = orch[0]
    from langchain_core.tools import ToolException
    handbacks = [{"snapshot_id": "a", "review": {"verdict": "falls_short"}, "submission": {"data_commit": None}},
                 {"snapshot_id": "b", "review": {"verdict": "carries_out_plan"}, "submission": {"data_commit": "c"}}]
    asyncio.run(o.result_check(handbacks, needs_result=False)({"snapshot_id": "a"}))
    with pytest.raises(ToolException, match=r"^z is not a snapshot handed back in this phase\. The ones you can "
                                            r"accept: a \(falls_short\); b \(carries_out_plan\)\.$"):
        asyncio.run(o.result_check(handbacks, needs_result=False)({"snapshot_id": "z"}))
    with pytest.raises(ToolException, match=r"with a data commit and a recipe\. The ones you can accept: b "):
        asyncio.run(o.result_check(handbacks, needs_result=True)({"snapshot_id": "a"}))
    with pytest.raises(ToolException, match="none yet"):
        asyncio.run(o.result_check([], needs_result=True)({"snapshot_id": "a"}))
    from ar_contract.models import SUMMARY_CHARS
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        o.RecipeSubmit(snapshot_id="b", summary="x" * (SUMMARY_CHARS + 1))   # the contract's cap, checked here


class FakeSession:
    def __init__(self, recipe_ok=True):
        self.calls, self.recipe_ok = [], recipe_ok

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        result = {"snapshot": {"snapshot_id": f"s{len(self.calls)}"},
                  "recipe_check": {"ok": self.recipe_ok, "failures": [] if self.recipe_ok else ["too few clips"]},
                  "data_commit": {"commit_id": "c" * 64, "datasets": {}}}[name]
        return NS(is_error=False, structured_content=result, content=[])


def _builder(o, role_name, ctx, session):
    roles = (o.recipe_roles(ctx, session, []) if role_name == "data_engineer" else o.edit_roles(ctx, session, [], None))
    return roles[1]


def test_the_data_engineers_hand_back_is_checked_then_snapshotted(orch):
    o, replies, _ = orch
    ctx = _recipe_ctx(tunable_rules={"optimizer.max_steps": {"type": "int"}})
    session = FakeSession()
    engineer = _builder(o, "data_engineer", ctx, session)
    replies["data_engineer"] = [
        ("", [("submit_data_engineer", {"report": "r", "data_commit": "c1"})]),
        ("", [("submit_data_engineer", {"report": "r", "data_commit": "c1", "recipe": {"optimizer.max_steps": 9.6},
                                        "rationale": "why"})])]
    value = asyncio.run(engineer.run("plan"))
    assert value["recipe"] == {"optimizer.max_steps": 10} and value["snapshot_id"] == "s2"
    assert session.calls == [("recipe_check", {"recipe": {"optimizer.max_steps": 10}, "data_commit": "c1"}),
                             ("snapshot", {"data_commit": "c1", "recipe": {"optimizer.max_steps": 10}, "rationale": "why"})]
    replies["data_engineer"] = [("", [("submit_data_engineer", {"report": "cannot be built"})])]
    assert asyncio.run(engineer.run("plan"))["snapshot_id"] == "s3" and session.calls[-1] == ("snapshot", {})
    refused = FakeSession(recipe_ok=False)
    engineer = _builder(o, "data_engineer", ctx, refused)
    replies["data_engineer"] = [("", [("submit_data_engineer", {"report": "r", "data_commit": "c1", "recipe": {},
                                                                "rationale": "why"})]),
                                ("", [("submit_data_engineer", {"report": "r"})])]
    asyncio.run(engineer.run("plan"))
    assert [name for name, _ in refused.calls] == ["recipe_check", "snapshot"]


def test_the_coders_hand_back_needs_a_passing_self_test(orch, monkeypatch):
    o, replies, seen = orch
    results = [["agent/entry.py defines no top-level improve_recipe"], []]
    monkeypatch.setattr(o, "selftest", lambda root: results.pop(0))
    session = FakeSession()
    from ar_contract.models import EditContext
    coder = _builder(o, "coder", EditContext(), session)
    replies["coder"] = [("", [("submit_coder", {"report": "r"})])] * 2
    assert asyncio.run(coder.run("plan")) == {"report": "r", "snapshot_id": "s1"}
    refusal = [m.content for m in seen[-1][1] if m.type == "tool"][0]
    assert refusal.startswith("Error: The self-test failed. Fix these first:\nagent/entry.py defines no top-level")
    assert session.calls == [("snapshot", {})]


def test_check_roles_builds_all_six_roles(orch, monkeypatch):
    o = orch[0]
    built = []
    real = o.Role.__init__

    def record(self, name, *a, **k):
        built.append(name)
        real(self, name, *a, **k)
    monkeypatch.setattr(o.Role, "__init__", record)
    o.check_roles()
    assert sorted(built) == sorted(["data_engineer", "recipe_planner", "recipe_reviewer", "coder", "edit_planner",
                                    "edit_reviewer"])


def test_the_planner_tools_follow_the_roles_missions(orch):
    o = orch[0]
    from langchain_core.tools import StructuredTool

    async def noop() -> str:
        return ""
    ktools = [StructuredTool.from_function(coroutine=noop, name=n, description=n)
              for n in ("data_query", "data_ingest", "hf_download", "ask", "read_skill", "caption_videos")]
    names = lambda role: {t.name for t in role.tools}
    planner, engineer, reviewer = o.recipe_roles(None, None, ktools)
    assert names(planner) == {"data_engineer", "data_query", "ask", "read_skill", "write_file", "edit_file",
                              "run_command", "arxiv_search", "arxiv_read", "submit_result"}
    assert {"data_ingest", "hf_download", "caption_videos", "submit_data_engineer"} <= names(engineer)
    assert names(reviewer()) == {"data_query", "ask", "read_skill", "write_file", "edit_file", "run_command",
                                 "submit_review"}
    planner, coder, reviewer = o.edit_roles(None, None, ktools, None)
    assert names(planner) == {"coder", "ask", "read_skill", "write_file", "edit_file", "run_command", "arxiv_search",
                              "arxiv_read", "submit_result"}
    assert names(coder) == {"ask", "read_skill", "self_test", "write_file", "edit_file", "run_command", "submit_coder"}
    assert names(reviewer()) == {"ask", "read_skill", "self_test", "write_file", "edit_file", "run_command",
                                 "submit_review"}
    run_command = lambda role: next(t for t in role.tools if t.name == "run_command").description
    assert f"in {o.AGENT_ROOT} " in run_command(coder) and f"in {planner.scratch} " in run_command(planner)


def test_a_planners_delegate_tool_names_its_builders_tools(orch):
    o = orch[0]
    for planner, builder, _ in (o.recipe_roles(None, None, []), o.edit_roles(None, None, [], None)):
        tool = next(t for t in planner.tools if t.name == builder.name)
        listed = tool.description.split(" Its tools: ")[1].split(", ")
        assert listed == [t.name for t in builder.tools if t.name != builder.submit] and builder.submit not in listed
    assert tool.description == "Brief the coder. Its tools: self_test, write_file, edit_file, run_command"


# ---- pure helpers ----

def test_kernel_results_prefer_structured_content():
    """MCP splits a list result into one text block per item (an empty list gives no text),
    so structured content is returned JSON-encoded; errors raise ToolException."""
    from langchain_core.tools import ToolException
    from agent.tools import result_text
    text = [NS(type="text", text="a"), NS(type="text", text="b")]
    assert result_text(NS(is_error=False, structured_content={"result": []}, content=[])) == '{"result": []}'
    assert result_text(NS(is_error=False, structured_content=None, content=text)) == "a\nb"
    with pytest.raises(ToolException, match="a\nb"):
        result_text(NS(is_error=True, structured_content=None, content=text))


def test_arxiv_search_refuses_fewer_than_one_result_without_a_request(monkeypatch):
    import arxiv
    from langchain_core.tools import ToolException
    from agent.tools import arxiv_search
    monkeypatch.setattr(arxiv, "Client", None)                  # any request would fail differently
    for n in (0, -1):
        with pytest.raises(ToolException, match="^max_results must be at least 1$"):
            arxiv_search.invoke({"query": "q", "max_results": n})


def test_every_listed_component_exists_in_the_seed():
    from agent.orchestration import COMPONENTS
    for component, where in COMPONENTS.items():
        assert (SEED / where.split(" ")[0].split("*")[0]).exists(), f"{component} -> {where}"


def test_selftest_passes_on_the_seed_and_catches_a_broken_entry_or_role(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")                  # set in every container; building roles needs it
    from agent.orchestration import selftest
    copy = tmp_path / "copy"
    shutil.copytree(SEED, copy)
    assert selftest(str(copy)) == []
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, *, extra=1):\n    pass\n"
                                             "def improve_recipe(ctx):\n    pass\n")
    assert selftest(str(copy))[0] == "edit_self must take exactly one positional parameter (ctx)"
    shutil.copy(SEED / "agent" / "entry.py", copy / "agent" / "entry.py")
    (copy / "agent" / "prompts" / "coder.md").unlink()
    assert any("coder.md" in e for e in selftest(str(copy)))
    shutil.copy(SEED / "agent" / "prompts" / "coder.md", copy / "agent" / "prompts" / "coder.md")
    (copy / "agent" / "notes.md").write_text("Remember: node0004 lost on follows_subject_action.\n")
    assert selftest(str(copy)) == ["agent/notes.md:1 mentions follows_subject_action", "agent/notes.md:1 mentions node0004"]
    (copy / "agent" / "notes.md").unlink()
    (copy / "agent" / "__pycache__").mkdir(exist_ok=True)
    (copy / "agent" / "__pycache__" / "notes.py").write_text("node0004\n")     # ignored: the kernel never commits it
    assert selftest(str(copy)) == []


def test_the_self_test_tool_reports_the_result(monkeypatch):
    from agent import orchestration
    monkeypatch.setattr(orchestration, "selftest", lambda root: [])
    assert asyncio.run(orchestration.SELF_TEST.ainvoke({})) == "self-test passed"
    monkeypatch.setattr(orchestration, "selftest", lambda root: ["e1", "e2"])
    assert asyncio.run(orchestration.SELF_TEST.ainvoke({})) == "self-test failed:\ne1\ne2"


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
    assert EditPlan(**ok).constraints == ""


# ---- prompts ----

ROLES = ("edit_planner", "coder", "edit_reviewer", "recipe_planner", "data_engineer", "recipe_reviewer")


def test_each_role_has_its_prompt_file_used_as_is(orch):
    o = orch[0]
    assert sorted(p.stem for p in (SEED / "agent" / "prompts").glob("*.md")) == sorted([*ROLES, "compact"])
    for role in ROLES:
        text = (SEED / "agent" / "prompts" / f"{role}.md").read_text()
        assert o.Role(role, lambda s: "", [], "submit_x", Schema.Report).system == text
        assert "{" not in text and text.startswith("You are the ")


def test_prompts_have_no_node_facts_wrapped_lines_or_dated_notes():
    from ar_contract.static import mentions
    assert mentions(SEED) == []
    for role in ROLES:
        text = (SEED / "agent" / "prompts" / f"{role}.md").read_text()
        assert not any(line.count("`") % 2 for line in text.splitlines()), f"{role}: inline code wraps"
        assert "*(20" not in text, f"{role}: dated note"


def test_the_held_out_rule_is_in_the_data_prompts_and_the_node_rule_in_the_edit_prompts():
    read = lambda role: (SEED / "agent" / "prompts" / f"{role}.md").read_text()
    for role in ("recipe_planner", "data_engineer", "recipe_reviewer"):
        assert "The evaluation set is held out." in read(role)
    for role in ("edit_planner", "coder"):
        assert "write no specific node, nor its scores or metrics, into the change" in read(role)


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
    """Replies in call order; SNAPk becomes the id of the script's k-th snapshot."""
    recipe = [
        [_call("data_engineer", {**PLAN, "data": "x" * 4001})],                        # planner: over the cap
        [_call("data_engineer", PLAN), _call("data_query", {})],                       # the query is not run
        [_call("data_query", {}), _call("no_such_tool", {}),                           # engineer
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("submit_data_engineer", {"report": "r", "data_commit": C0})],           # refused: not all three
        [_call("submit_data_engineer", {"report": "downloads are disabled; the pool has clips"})],      # SNAP1
        [_call("submit_review", {"verdict": "falls_short", "findings": "no data commit"})],
        [_call("submit_result", {"snapshot_id": "SNAP1", "summary": "s"})],             # planner: refused
        [_call("data_engineer", {**PLAN, "hypothesis": "pool clips suffice"})],
        [_call("submit_data_engineer", {"report": "pool clips, 200 steps", "data_commit": C0,      # SNAP2
                                        "recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5},
                                        "rationale": "fits the data"})],
        [_call("submit_review", {"verdict": "carries_out_plan", "findings": "checked 3 clips"})],
        [_call("submit_result", {"snapshot_id": "SNAP2", "summary": "Pool clips suffice; trained for 200 steps."})],
    ]
    edit = [
        [_call("run_command", {"command": "pwd && touch notes.txt"}),                   # planner: in its scratch
         _call("run_command", {"command": "touch $AR_AGENT_DIR/agent/stray.py"})],      # reverted: /agent is the coder's
        [_call("submit_result", {"snapshot_id": "nope", "summary": "nothing to do"})],  # refused: nothing handed back
        [_call("coder", {**EDIT_PLAN, "problem": ""})],                                 # invalid: no problem
        [_call("coder", EDIT_PLAN)],
        [_call("edit_file", {"path": "agent/entry.py", "old_string": "def edit_self(ctx: EditContext)",
                             "new_string": "def edit_self(ctx: EditContext, extra)"})],       # coder: breaks the self-test
        [_call("submit_coder", {"report": "too early"})],
        [_call("edit_file", {"path": "agent/entry.py", "old_string": "def edit_self(ctx: EditContext, extra)",
                             "new_string": "def edit_self(ctx: EditContext)"})],
        [_call("edit_file", {"path": "agent/briefing.py", "old_string": "SIBLINGS_SHOWN = 10",
                             "new_string": "SIBLINGS_SHOWN = 8"})],
        [_call("submit_coder", {"report": "The briefing now shows eight siblings."})],                 # SNAP1
        [_call("self_test", {}),                                                                        # reviewer
         _call("run_command", {"command": "grep -c 'SIBLINGS_SHOWN = 8' $AR_AGENT_DIR/agent/briefing.py"})],
        [_call("submit_review", {"verdict": "carries_out_plan", "findings": "briefing.py shows eight"})],
        [_call("submit_result", {"snapshot_id": "SNAP1", "summary": "Fewer siblings in the planner's context."})],
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
    from ar_kernel.tools.skills import register_skill_tool
    from ar_kernel.tools.snapshot import SnapshotStore, register_snapshot_tool
    from ar_kernel.vcs.agents_repo import AgentsRepo
    run = tmp_path_factory.mktemp("run")
    rec = Recorder(run)
    repo = AgentsRepo(run / "agents.git")
    repo.init(SEED)
    book = MockBook.default()
    for name, script in _scripts().items():
        book.add(name, script)

    class NamingStore(SnapshotStore):
        """Puts each snapshot's real id where its script says SNAPk."""
        taken: dict = {}

        def take(self, caller, *args):
            out = super().take(caller, *args)
            k = self.taken[caller.mock_script] = self.taken.get(caller.mock_script, 0) + 1
            for reply in book._scripts.get(caller.mock_script, []):
                for item in reply:
                    if "arguments" in item:
                        item["arguments"] = item["arguments"].replace(f"SNAP{k}", out["snapshot_id"])
            return out

    registry, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    kit, mcp = ToolKit(registry, rec), new_mcp()
    register_data_tools(mcp, kit, _MockData())
    register_snapshot_tool(mcp, kit, NamingStore(run, repo, _MockData().commit_exists))
    register_hf_tools(mcp, kit, _MockHf())
    register_job_tools(mcp, kit, queue)
    queue.register(_MockCaption())
    register_caption_tool(mcp, kit, queue)
    register_ask_tool(mcp, kit, MockAsk())
    register_skill_tool(mcp, kit)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=registry, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=book), build_tool_app(mcp))
    yield rec, registry, services, repo, run
    services.stop()
    queue.shutdown()


def _run(kernel, tmp_path, kind, script, node, ctx):
    rec, registry, services, repo, _ = kernel
    agent = tmp_path / "agent_copy"
    shutil.copytree(SEED, agent, ignore=shutil.ignore_patterns("__pycache__"))
    ws, ctx_dir = tmp_path / "ws", tmp_path / "ctx"
    ws.mkdir(), ctx_dir.mkdir()
    (ctx_dir / "context.json").write_text(json.dumps(ctx))
    edits = kind == "edit_self"
    caller = registry.issue(node=node, phase=kind, attempt=1, workspace_host=ws, staging_host=tmp_path,
                            mock_script=script, agent_host=agent if edits else None,
                            base_commit=repo.commit_tree(agent, None, "copy") if edits else None)
    env = {**os.environ, "AR_SOCKET_DIR": str(services.socket_dir), "AR_TOKEN": caller.token,
           "AR_DEFAULT_MODEL": "mock-model", "AR_WORKSPACE": str(ws), "AR_CONTEXT_DIR": str(ctx_dir),
           "AR_AGENT_DIR": str(agent)}
    proc = subprocess.run([sys.executable, "-m", "ar_contract.run", kind], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    registry.revoke(caller.token)
    return proc, json.loads((ws / "result.json").read_text()), agent


def _requests(rec, node) -> list[dict]:
    return [e for e in rec.read_events(node) if e["type"] == "llm.request"]


def _tool_outputs(rec, node) -> list[str]:
    """Every tool result the agent sent back to the model (Chat Completions `tool` messages)."""
    out = []
    for event in _requests(rec, node):
        body = rec.load_payload(event["payload"])["body"]
        out += [m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                for m in body["messages"] if m.get("role") == "tool"]
    return out


def _by_role(rec, node) -> dict[str, list[dict]]:
    """Request bodies by the role label the gateway recorded."""
    out: dict[str, list[dict]] = {}
    for event in _requests(rec, node):
        out.setdefault(event.get("role"), []).append(rec.load_payload(event["payload"])["body"])
    return out


def _tools_of(bodies) -> set[str]:
    return {t["function"]["name"] for body in bodies for t in body.get("tools") or []}


def _users(bodies) -> list[str]:
    return [str(m["content"]) for m in bodies[-1]["messages"] if m.get("role") == "user"]


def test_chat_model_works_sync_and_async_over_the_socket(kernel, tmp_path, monkeypatch):
    """The container has no network: both the sync and the async client must use the socket."""
    _, registry, services, _, _ = kernel
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




@pytest.mark.parametrize("kind", ["edit_self", "improve_recipe"])
def test_dry_run_against_the_kernel_services_returns_a_real_snapshot(kernel, tmp_path, kind):
    from ar_kernel.tools import snapshot
    rec, _, _, repo, run = kernel
    proc, body, _ = _run(kernel, tmp_path, kind, "smoke", f"dry-{kind}", {"dry_run": True})
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    sid = body["result"]["snapshot_id"]
    if kind == "edit_self":
        assert snapshot.edit_snapshot(repo, "dry-edit_self", "edit_self", 1, sid) == sid
    else:
        assert snapshot.recipe_snapshot(run, "dry-improve_recipe", "improve_recipe", 1, sid)["data_commit"] == C0
    assert set(_by_role(rec, f"dry-{kind}")) == {"ping"}


def test_improve_recipe_full_flow(kernel, tmp_path):
    from ar_kernel.tools import snapshot
    rec, _, _, _, run = kernel
    ctx = {"tunable_rules": {"optimizer.max_steps": {"type": "int"}, "optimizer.lr": {"type": "float"}},
           "lora_allowlist": [[48, 48]], "target_skill": {"name": "rain starting", "sentence": "rain begins",
                                                          "description": "d"}}
    proc, body, _ = _run(kernel, tmp_path, "improve_recipe", "recipe", "n-recipe", ctx)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    result = body["result"]
    assert result["summary"] == "Pool clips suffice; trained for 200 steps."
    assert snapshot.recipe_snapshot(run, "n-recipe", "improve_recipe", 1, result["snapshot_id"]) == {
        "has_result": True, "data_commit": C0, "recipe": {"optimizer.max_steps": 200, "optimizer.lr": 1e-5},
        "rationale": "fits the data"}                                                   # int coerced
    outputs = _tool_outputs(rec, "n-recipe")
    for fragment in ("Error: invalid arguments for data_engineer:\ndata: ", "Not run: data_engineer ran in this turn",
                     "Error: no_such_tool is not a valid tool", "Give data_commit, recipe and rationale together",
                     "is not a snapshot handed back in this phase with a data commit and a "
                     "recipe. The ones you can accept: none yet."):
        assert any(fragment in o for o in outputs), fragment
    assert any(o.startswith("Error: ") and "ToolException" not in o and "downloads are disabled" in o
               for o in outputs)                                                 # kernel tool error reported
    roles = _by_role(rec, "n-recipe")
    assert set(roles) == {"recipe_planner", "data_engineer", "recipe_reviewer"}
    planner_tools = _tools_of(roles["recipe_planner"])
    assert {"data_engineer", "hf_search", "data_query", "data_fetch", "ask", "read_skill", "run_command",
            "arxiv_search", "submit_result"} <= planner_tools
    assert not {"hf_download", "data_ingest", "data_commit", "caption_videos", "snapshot"} & planner_tools
    assert {"submit_data_engineer", "read_skill", "data_ingest", "arxiv_search"} <= _tools_of(roles["data_engineer"])
    assert "snapshot" not in _tools_of(roles["data_engineer"])
    assert {"submit_review", "data_query", "ask"} <= _tools_of(roles["recipe_reviewer"])
    engineer = _users(roles["data_engineer"])
    assert engineer[0].startswith("<context>") and "<plan>" in engineer[0] and "- name: rain starting" in engineer[0]
    assert "LoRA (rank/alpha): 48/48." in engineer[0]
    assert engineer[-1].startswith("The planner revised the plan.") and "pool clips suffice" in engineer[-1]
    reviewer_first = _users(roles["recipe_reviewer"])[0]
    assert "<report>\npool clips, 200 steps\n</report>" in reviewer_first and '"rationale": "fits the data"' in reviewer_first
    handbacks = [json.loads(m["content"]) for m in roles["recipe_planner"][-1]["messages"]
                 if m.get("role") == "tool" and m["content"].startswith('{\n "snapshot_id"')]
    assert [h["review"]["verdict"] for h in handbacks] == ["falls_short", "carries_out_plan"]
    assert {p.name.split("-")[0] for p in (tmp_path / "ws" / "tool_output").iterdir()} == {
        "recipe_planner", "data_engineer", "recipe_reviewer"}
    assert not list((tmp_path / "ws" / "scratch").iterdir()) and not (tmp_path / "ws" / "staging" / ".guard").exists()


def test_edit_self_plans_builds_reviews_and_accepts_a_snapshot(kernel, tmp_path):
    rec, _, _, repo, _ = kernel
    from ar_kernel.tools import snapshot
    proc, body, agent = _run(kernel, tmp_path, "edit_self", "edit", "n-edit", {})
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    sid = body["result"]["snapshot_id"]
    assert body["result"]["summary"] == "Fewer siblings in the planner's context."
    assert snapshot.edit_snapshot(repo, "n-edit", "edit_self", 1, sid) == sid
    assert "SIBLINGS_SHOWN = 8" in repo.read_file(sid, "agent/briefing.py")
    assert not (agent / "agent" / "stray.py").exists() and "SIBLINGS_SHOWN = 8" in (agent / "agent" / "briefing.py").read_text()
    assert not any("belongs to coder: agent/briefing.py" in o for o in _tool_outputs(rec, "n-edit"))
    outputs = _tool_outputs(rec, "n-edit")
    for fragment in (f"Error: run_command changed files in {agent}, which belongs to coder: agent/stray.py (reverted)",
                     "nope is not a snapshot handed back in this phase. The ones you can accept: none yet.",
                     "Error: invalid arguments for coder:\nproblem: ",
                     "Error: The self-test failed. Fix these first:\nedit_self must take exactly one positional",
                     "self-test passed", "exit 0\n1\n"):
        assert any(fragment in o for o in outputs), fragment
    planner_scratch = next(o for o in outputs if o.startswith("exit 0\n") and "/scratch/edit_planner-" in o)
    assert planner_scratch.split("\n")[1].startswith(str(tmp_path / "ws" / "scratch" / "edit_planner-"))
    roles = _by_role(rec, "n-edit")
    assert set(roles) == {"edit_planner", "coder", "edit_reviewer"}
    assert {"coder", "run_command", "ask", "read_skill", "arxiv_search", "submit_result"} <= _tools_of(roles["edit_planner"])
    assert not {"data_ingest", "hf_download", "caption_videos", "snapshot", "self_test"} & _tools_of(roles["edit_planner"])
    assert {"self_test", "ask", "read_skill", "submit_coder"} <= _tools_of(roles["coder"])
    assert not {"arxiv_search", "data_ingest"} & _tools_of(roles["coder"])
    assert {"self_test", "submit_review"} <= _tools_of(roles["edit_reviewer"])
    coder_first = _users(roles["coder"])[0]
    assert coder_first.startswith("<context>") and "- Components of this agent:" in coder_first and "<plan>" in coder_first


@pytest.mark.docker
def test_seed_agent_passes_contract_verification(tmp_path):
    """On a context with a lineage and siblings, as a real node gets: the dry run builds every context."""
    from ar_contract.models import EditContext, RecipeContext
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import ContractHarness, verify_contract
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.vcs.agents_repo import AgentsRepo
    repo = AgentsRepo(tmp_path / "agents.git")
    commit = repo.init(SEED)
    harness = ContractHarness(KernelConfig.load(), tmp_path / "run", Recorder(tmp_path / "run"), repo)
    harness.start()
    try:
        common = dict(dry_run=True)
        edit = EditContext(**common, lineage=_edit_lineage(3), siblings=_edit_lineage(3)[1:])
        recipe = RecipeContext(**common, lineage=_data_lineage(3), siblings=_data_lineage(3)[1:], metric_guide=GUIDE,
                               n_gpus=4, clip_pool_size=7,
                               tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
                               recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
                               lora_allowlist=[[64, 64]])
        report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                                 commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                                 node="root", attempt=1,
                                 contexts={"edit_self": edit.model_dump(mode="json"),
                                           "improve_recipe": recipe.model_dump(mode="json")})
    finally:
        harness.stop()
    assert report.ok, report.to_record()
