"""The roles, their tools and the phases."""
from __future__ import annotations

import asyncio
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult
from ar_contract.static import mentions, static_check
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from .briefing import (coder_context, data_engineer_context, edit_planner_context, edit_reviewer_context,
                       recipe_planner_context, recipe_reviewer_context)
from .entry import AGENT_ROOT, BRIEF_CHARS, COMPACT_AT, CONTEXT_WINDOW, MAX_ROUNDS, MODEL, WORKSPACE
from .harness import Guard, bind, compact, remove_tree, run_loop
from .tools import kernel_tools, local_tools, result_text

AGENT_PKG = Path(__file__).resolve().parent
SCRATCH = Path(WORKSPACE) / "scratch"
GUARD_DIR = str(Path(WORKSPACE) / "staging" / ".guard")
PLAN_FIELD_CHARS = 4000                       # chars per plan field
REMIND = ("Your last message made no tool call, which ends your turn. Keep working with the tools; when the "
          "work is done, call {tool}. Nothing is recorded unless you call {tool}.")
REMINDERS = 4
READ_ONLY_KERNEL_TOOLS = {"hf_search", "hf_list_files", "data_query", "data_fetch", "ask", "read_skill"}
EDIT_KERNEL_TOOLS = {"ask", "read_skill"}

COMPONENTS = {
    "tools": "agent/tools.py -- the agent's own tools and the adapter for kernel tools",
    "orchestration": "agent/orchestration.py -- the roles and how work passes between them",
    "briefing": "agent/briefing.py -- each role's context",
    "harness": "agent/harness.py -- the single-agent loop logic",
    "prompts": "agent/prompts/*.md -- the role prompts and the compaction prompt",
    "entry": "agent/entry.py -- the two entry points, and the settings",
    "requirements": "agent/requirements.txt -- the packages installed before every phase",
}


class NoSubmission(RuntimeError):
    """A role ended its turn without submitting."""


def brief(text: str) -> str:
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS] + " ...[truncated: the full context is in /context/context.json]"
    return f"<context>\n{text.strip()}\n</context>"


class Role:
    """One role: prompt, context, tools, a submit tool, its conversation and scratch folder."""
    _ids = itertools.count(1)

    def __init__(self, name: str, context, tools: list, submit: str, schema: type[BaseModel], check=None, *,
                 owns: str | None = None, papers: bool = False, guard: Guard | None = None,
                 compact_on_delegation: bool = False) -> None:
        """`compact_on_delegation`: compact when the work passes to another role (stateful roles only)."""
        self.name, self.context_fn, self.submit, self.guard = name, context, submit, guard
        self.compact_on_delegation = compact_on_delegation
        self.scratch = SCRATCH / f"{name}-{next(Role._ids)}"
        self.system = (AGENT_PKG / "prompts" / f"{name}.md").read_text()
        self.messages: list = []
        self.context: str | None = None
        self.pinned = ""
        self.value: dict | None = None

        async def accept(**kwargs) -> str:
            value = schema.model_validate(kwargs).model_dump()
            value.update((await check(value) if check is not None else None) or {})
            self.value = value
            return json.dumps({"submitted": value}, ensure_ascii=False)

        self.tools = [*tools, *local_tools(owns or str(self.scratch), papers),
                      StructuredTool(name=submit, description=schema.__doc__ or submit, args_schema=schema,
                                     coroutine=accept, return_direct=True)]
        self.model = chat_model(MODEL, role=name)
        self.model.bind_tools(self.tools)          # validates the tool schemas

    async def run(self, briefing: str) -> dict:
        """Works on `briefing` until the submit is accepted; returns the submission."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        if self.context is None:
            self.context = brief(self.context_fn(str(self.scratch)))
        self.pinned = f"{self.context}\n\n{briefing}"
        self.value = None
        self.messages.append(HumanMessage(content=briefing if len(self.messages) else self.pinned))
        for k in range(REMINDERS + 1):
            if k:
                self.messages.append(HumanMessage(content=REMIND.format(tool=self.submit)))
            self.messages = await run_loop(self.model, self.tools, self.system, self.messages, role=self.name,
                                           pinned=self.pinned, compact_at=COMPACT_AT, context_window=CONTEXT_WINDOW,
                                           guard=self.guard, compact_on_delegation=self.compact_on_delegation)
            if self.value is not None:
                return self.value
        raise NoSubmission(f"{self.name} ended its turn without calling {self.submit}, after {REMINDERS} reminders")

    async def compact(self) -> None:
        await compact(self.model, bind(self.model, self.tools), self.tools, self.system, self.messages, self.pinned)

    def close(self) -> None:
        remove_tree(self.scratch)


async def ping() -> str:
    """One model call, for the dry run."""
    messages = await run_loop(chat_model(MODEL, role="ping"), [], "Reply with the single word ok.",
                              [HumanMessage(content="ping")], role="ping", pinned="ping", compact_at=COMPACT_AT,
                              context_window=CONTEXT_WINDOW)
    return messages[-1].text


def plan_text(length: str, description: str, **kwargs):
    return Field(max_length=PLAN_FIELD_CHARS, description=f"{description} ({length})", **kwargs)


class Review(BaseModel):
    """Submit your review of the build."""
    verdict: Literal["carries_out_plan", "falls_short"]
    findings: str = Field(min_length=1, description="each problem with where it shows, most important first; "
                                                    "which ones decide the verdict; what you checked")


def delegate(builder: Role, reviewer, schema: type[BaseModel], handbacks: list, describe,
             described: list = ()) -> StructuredTool:
    """The planner's tool for `builder`: build, review, hand back. `described`: builder tools listed with their
    descriptions."""
    calls = 0

    async def call(**kwargs) -> str:
        nonlocal calls
        if calls >= MAX_ROUNDS:
            raise ToolException(f"No calls left: {builder.name} was called {MAX_ROUNDS} times in this phase. "
                                "Finish with submit_result.")
        calls += 1
        plan = schema.model_validate(kwargs)
        briefing = f"<plan>\n{plan.model_dump_json(indent=1)}\n</plan>"
        built = await builder.run(briefing if calls == 1 else f"The planner revised the plan.\n\n{briefing}")
        if builder.compact_on_delegation:
            await builder.compact()
        review_role = reviewer()
        try:
            review = await review_role.run(describe(briefing, built))
        finally:
            review_role.close()
        handback = {"snapshot_id": built["snapshot_id"], "report": built["report"], "review": review}
        handbacks.append({**handback, "submission": built})
        return json.dumps(handback, indent=1, ensure_ascii=False)

    tools = ", ".join(t.name for t in builder.tools if t.name != builder.submit)
    description = f"{schema.__doc__} Its tools: {tools}"
    if described:
        description += (f"\n\nThe {builder.name} tools you cannot call yourself:\n"
                        + "\n".join(f"- {t.name}: {t.description}" for t in described))
    return StructuredTool(name=builder.name, description=description, args_schema=schema,
                          coroutine=call, metadata={"ends_batch": True, "delegates": True})


def result_check(handbacks: list, needs_result: bool):
    async def check(value: dict) -> None:
        valid = [h for h in handbacks if not needs_result or h["submission"].get("data_commit")]
        if not any(h["snapshot_id"] == value["snapshot_id"] for h in valid):
            ids = "; ".join(f"{h['snapshot_id']} ({h['review']['verdict']})" for h in valid) or "none yet"
            raise ToolException(f"{value['snapshot_id']} is not a snapshot handed back in this phase"
                                f"{' with a data commit and a recipe' if needs_result else ''}. The ones you can "
                                f"accept: {ids}.")
    return check


class EditSubmit(EditResult):
    """End the phase with the snapshot you accept and the node's summary."""


class RecipeSubmit(RecipeResult):
    """End the phase with the snapshot you accept, one with a data commit and a recipe, and the node's summary."""


async def snapshot(session, args: dict) -> str:
    return json.loads(result_text(await session.call_tool("snapshot", args)))["snapshot_id"]


# ---- improve_recipe ----

class DataPlan(BaseModel):
    """Brief the data engineer."""
    hypothesis: str = plan_text("one or two sentences", "the one data idea this node tests, and what in "
                                "earlier nodes suggests it", min_length=1)
    expected_change: str = plan_text("a sentence", "which metrics or groups should move, and in which direction",
                                     min_length=1)
    data: str = plan_text("a short paragraph", "what the training set should contain and where it comes from "
                          "(sources you checked exist); not the steps to build it", min_length=1)
    constraints: str = plan_text("optional", "what the engineer must keep or must not do, if anything", default="")


class DataHandback(BaseModel):
    """Hand back to the planner: your report, and the data commit and recipe when the data is ready."""
    report: str = Field(min_length=1, description="what you built, how you checked it and what the checks showed, "
                                                  "where it departs from the plan, what the planner should know")
    data_commit: str | None = Field(None, min_length=1, description="the data_commit id to train on")
    recipe: dict[str, float] | None = Field(None, description="tunable key -> value")
    rationale: str | None = Field(None, min_length=1, description="the reason for each changed key's value")


def typed(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


def recipe_roles(ctx: RecipeContext | None, session, ktools: list) -> tuple:
    """(planner, engineer, reviewer factory)."""
    async def built(value: dict) -> dict:
        given = [value[k] is not None for k in ("data_commit", "recipe", "rationale")]
        if any(given) and not all(given):
            raise ToolException("Give data_commit, recipe and rationale together, or none of them.")
        if not any(given):
            return {"snapshot_id": await snapshot(session, {})}
        recipe = typed(value["recipe"], ctx.tunable_rules)
        check = json.loads(result_text(await session.call_tool(
            "recipe_check", {"recipe": recipe, "data_commit": value["data_commit"]})))
        if not check.get("ok"):
            raise ToolException(f"recipe_check failed: {check.get('failures')}")
        return {"recipe": recipe, "snapshot_id": await snapshot(session, {
            "data_commit": value["data_commit"], "recipe": recipe, "rationale": value["rationale"]})}

    def describe(briefing: str, b: dict) -> str:
        shown = {k: b[k] for k in ("snapshot_id", "data_commit", "recipe", "rationale")}
        return (f"{briefing}\n\n<report>\n{b['report']}\n</report>\n\n"
                f"<snapshot>\n{json.dumps(shown, indent=1, ensure_ascii=False)}\n</snapshot>")

    read_only = [t for t in ktools if t.name in READ_ONLY_KERNEL_TOOLS]
    others = [t for t in ktools if t.name not in READ_ONLY_KERNEL_TOOLS]
    handbacks: list = []
    engineer = Role("data_engineer", lambda s: data_engineer_context(ctx, s), ktools, "submit_data_engineer",
                    DataHandback, built, owns=WORKSPACE, papers=True, compact_on_delegation=True)
    reviewer = lambda: Role("recipe_reviewer", lambda s: recipe_reviewer_context(ctx, s), read_only, "submit_review",
                            Review)
    planner = Role("recipe_planner", lambda s: recipe_planner_context(ctx, s),
                   [delegate(engineer, reviewer, DataPlan, handbacks, describe, others), *read_only], "submit_result",
                   RecipeSubmit, result_check(handbacks, needs_result=True), papers=True)
    return planner, engineer, reviewer


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        planner, engineer, reviewer = recipe_roles(ctx, session, ktools)
        try:
            if ctx.dry_run:
                for role in (planner, engineer, reviewer()):
                    role.context_fn(str(role.scratch))
                commit = json.loads(result_text(await session.call_tool(
                    "data_commit", {"parent": None, "datasets": {}, "message": "dry run"})))["commit_id"]
                sid = await snapshot(session, {"data_commit": commit, "recipe": {}, "rationale": "dry run"})
                return RecipeResult(snapshot_id=sid, summary=f"dry run: roles and contexts built, {len(ktools)} "
                                                             f"kernel tools, model said {await ping()!r}")
            return RecipeResult(**await planner.run("Plan this node's data, have it built, and submit the result."))
        finally:
            planner.close()
            engineer.close()


# ---- edit_self ----

class EditPlan(BaseModel):
    """Brief the coder."""
    problem: str = plan_text("one or two sentences", "where the agent system made its roles waste effort, lack a "
                             "capability or lose information", min_length=1)
    evidence: str = plan_text("a short paragraph", "where it shows: nodes, transcript files, process lines",
                              min_length=1)
    mechanism: str = plan_text("a short paragraph", "what should change in the agent system and in which "
                               "component, as intent; not the code or the text to write", min_length=1)
    check: str = plan_text("a sentence", "what a later reader of the process lines and transcripts would see if "
                           "it worked", min_length=1)
    constraints: str = plan_text("optional", "what the coder must keep or must not do, if anything", default="")


class CodeHandback(BaseModel):
    """Hand back to the planner. Accepted only if the self-test passes on /agent."""
    report: str = Field(min_length=1, description="what you changed and in which files, how you checked it, where "
                                                  "you departed from the plan and why, what is still open")


def committed_files(root: str) -> list[str]:
    """The files the kernel commits from `root`: every file its ignore rules keep."""
    with tempfile.TemporaryDirectory() as tmp:
        git = ["git", "--git-dir", f"{tmp}/.git", "--work-tree", root]
        subprocess.run(["git", "init", "-q", tmp], check=True, capture_output=True, text=True, timeout=60)
        out = subprocess.run([*git, "ls-files", "-z", "--others", "--exclude-standard"], check=True,
                             capture_output=True, text=True, timeout=60).stdout
    return sorted(f for f in out.split("\0") if f)


def selftest(root: str) -> list[str]:
    """The contract's checks plus building every role; returns the errors."""
    try:
        problem = static_check((Path(root) / "agent" / "entry.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        problem = f"cannot read agent/entry.py: {exc}"
    errors = [problem] if problem else []
    try:
        errors += mentions(root, committed_files(root))
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"cannot list the files of {root}: {getattr(exc, 'stderr', '') or exc}")
    r = subprocess.run([sys.executable, "-c", "import agent.entry, agent.orchestration as o; o.check_roles()"],
                       cwd=root, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        errors.append(r.stderr[-6000:])
    return errors


async def _self_test() -> str:
    errors = await asyncio.to_thread(selftest, AGENT_ROOT)
    return "self-test failed:\n" + "\n".join(errors) if errors else "self-test passed"


SELF_TEST = StructuredTool.from_function(coroutine=_self_test, name="self_test",
                                         description="Run the self-test on /agent: the kernel's entry-point "
                                                     "check, the refusal of code that mentions a specific node or "
                                                     "a metric, the import, and building every role.")


def edit_roles(ctx: EditContext | None, session, ktools: list, guard: Guard | None) -> tuple:
    """(planner, coder, reviewer factory)."""
    async def built(value: dict) -> dict:
        errors = await asyncio.to_thread(selftest, AGENT_ROOT)
        if errors:
            raise ToolException("The self-test failed. Fix these first:\n" + "\n".join(errors))
        return {"snapshot_id": await snapshot(session, {})}

    def describe(briefing: str, b: dict) -> str:
        return (f"{briefing}\n\n<report>\n{b['report']}\n</report>\n\n"
                f"<snapshot>\n{b['snapshot_id']}: /agent as the coder handed it back\n</snapshot>")

    kernel = [t for t in ktools if t.name in EDIT_KERNEL_TOOLS]
    handbacks: list = []
    coder = Role("coder", lambda s: coder_context(ctx, COMPONENTS, s), [*kernel, SELF_TEST], "submit_coder",
                 CodeHandback, built, owns=AGENT_ROOT, guard=guard, compact_on_delegation=True)
    reviewer = lambda: Role("edit_reviewer", lambda s: edit_reviewer_context(ctx, COMPONENTS, s),
                            [*kernel, SELF_TEST], "submit_review", Review, guard=guard)
    planner = Role("edit_planner", lambda s: edit_planner_context(ctx, COMPONENTS, s),
                   [delegate(coder, reviewer, EditPlan, handbacks, describe), *kernel], "submit_result",
                   EditSubmit, result_check(handbacks, needs_result=False), papers=True, guard=guard)
    return planner, coder, reviewer


async def run_meta(ctx: EditContext) -> EditResult:
    async with mcp_session() as session:
        guard = Guard(AGENT_ROOT, "coder", GUARD_DIR)
        planner, coder, reviewer = edit_roles(ctx, session, await kernel_tools(session), guard)
        try:
            if ctx.dry_run:
                for role in (planner, coder, reviewer()):
                    role.context_fn(str(role.scratch))
                return EditResult(snapshot_id=await snapshot(session, {}),
                                  summary=f"dry run: roles and contexts built, model said {await ping()!r}")
            return EditResult(**await planner.run("Find the problem, have the coder fix it, and submit the result."))
        finally:
            planner.close()
            coder.close()
            guard.remove()


def check_roles() -> None:
    """Builds every role without a model call."""
    for _, _, reviewer in (recipe_roles(None, None, []), edit_roles(None, None, [], None)):
        if reviewer().compact_on_delegation:
            raise ValueError("compact_on_delegation needs a role that keeps its conversation; a reviewer is new "
                             "for every review")
