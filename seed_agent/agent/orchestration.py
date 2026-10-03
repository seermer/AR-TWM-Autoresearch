"""Which roles run, in what order, with which tools and what they are told first.

A role = a system prompt and a tool list, run on the harness. It returns its result by calling a
submit tool; one that stops without submitting gets one reminder.
Both phases are a planner and an engineer. The planner submits a plan that states intent; the engineer
carries it out, then either submits its result or reports back with request_replan for a new plan. The
planner is shown a submitted result and accepts it or revises the plan, up to MAX_ROUNDS plans per phase.
Each role keeps its conversation throughout. Every plan, report and result is recorded in plans.json. A planner has the engineer's file and shell tools, but what it
changes on disk while planning is undone, and its kernel tools only read.
- improve_recipe: planner -> data engineer, who builds the data commit and writes the recipe
  (recipe_check must pass before the submission is accepted).
- edit_self: edit planner -> coder (the self-test must pass before the submission is accepted).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from .entry import AGENT_ROOT, BRIEF_CHARS, COMPACT_AT, CONTEXT_WINDOW, MAX_ROUNDS, MODEL, WORKSPACE
from .briefing import coder_context, edit_context, engineer_context, recipe_context
from .harness import build_react_agent
from .tools import discarded_changes, kernel_tools, local_tools, result_text, snap_timed_prompts, submit_tool

AGENT_PKG = Path(__file__).resolve().parent
RECORD = Path(WORKSPACE) / "plans.json"       # every plan and report; carried to a retry with the workspace
PLAN_FIELD_CHARS = 4000                       # a safety cap: plans that state intent stay well under it, ones that
                                              # dictate file contents (5,800 characters and up) do not fit
REMIND = ("You stopped without calling {tools}. Finish the task, then call {tools} with the result. "
          "The work is only recorded through {tools}.")
REPLAN = "Revise the plan. The engineer carries out the plan you submit next."
REVIEW = ("The engineer finished. Check the result against your plan by looking at what was built, not only at "
          "this summary. If it tests the plan, call accept_result. If it falls short in a way the engineer can "
          "fix, submit a revised plan that says what to change: the work so far stays.")
PLANNING_KERNEL_TOOLS = {"hf_search", "hf_list_files", "data_query", "ask", "read_skill"}    # read-only
EDIT_KERNEL_TOOLS = {"ask", "read_skill"}

# Where each component lives in this agent, in the order an edit should consider them.
COMPONENTS = {
    "tools": "agent/tools.py -- the agent's own tools and the adapter for kernel tools (the kernel's tools and "
             "skills themselves are fixed)",
    "orchestration": "agent/orchestration.py -- the roles, their tools, the plan and replan loop, the plan "
                     "and result schemas",
    "briefing": "agent/briefing.py -- what each role is told first, built from the kernel's context",
    "harness": "agent/harness.py -- the single-agent inner loop: ReAct graph, tool execution, auto-compaction",
    "prompts": "agent/prompts/*.md -- each role's mission and general behaviour",
    "settings": "agent/entry.py -- limits: plans per phase, the context digest cap",
}


# ---- roles ----

def block(tag: str, text: str, **attrs) -> str:
    attributes = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"<{tag}{attributes}>\n{text.strip()}\n</{tag}>"


def system_prompt(name: str) -> str:
    return (AGENT_PKG / "prompts" / f"{name}.md").read_text()


def brief(text: str) -> str:
    """The context digest (briefing.py) in a <context> block. BRIEF_CHARS is only a safety cap: the digest
    ends with the lineage, so a cut would only shorten that."""
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS] + " ...[truncated: the full context is in /context/context.json]"
    return block("context", text)


class Role:
    """One role on the harness. Its conversation continues across calls to run()."""

    def __init__(self, name: str, tools: list, submissions: list, discard: tuple[str, ...] = ()) -> None:
        self.submissions = submissions            # the boxes of its submit tools
        self.discard = discard                    # dirs whose changes are undone after each run (planners)
        self.messages: list = []
        model = chat_model(MODEL)
        model.bind_tools(tools)                   # a broken tool schema fails here, not at the first call
        self.agent = build_react_agent(model, tools, system_prompt(name),
                                       context_window=CONTEXT_WINDOW, compact_at=COMPACT_AT)

    def _submitted(self):
        return next((box for box in self.submissions if box.value is not None), None)

    async def run(self, task: str):
        """Work on `task` until a submit tool is called; returns that tool's box. One reminder if it stops early."""
        for box in self.submissions:
            box.value = None
        names = " or ".join(box.name for box in self.submissions)
        with discarded_changes(*self.discard, keep=(str(Path(WORKSPACE) / "tool_output"),)):
            for message in (task, REMIND.format(tools=names)):
                state = await self.agent.ainvoke({"messages": [*self.messages, HumanMessage(content=message)]})
                self.messages = state["messages"]
                if self._submitted() is not None:
                    return self._submitted()
        raise RuntimeError(f"the role finished without calling {names}")


async def ping() -> str:
    """Contract smoke run: after building every role, one model call proves the wiring."""
    agent = build_react_agent(chat_model(MODEL), [], "Reply with the single word ok.",
                              context_window=CONTEXT_WINDOW, compact_at=COMPACT_AT)
    return (await agent.ainvoke({"messages": [HumanMessage(content="ping")]}))["messages"][-1].text


class Replan(BaseModel):
    report: str = Field(min_length=1, description="what you did and found, and why the plan should change")


def save(rounds: list) -> None:
    RECORD.write_text(json.dumps(rounds, indent=1))


def replan_tool(rounds: list) -> tuple:
    """The engineer's way back to the planner."""
    async def check(_) -> None:
        if len(rounds) >= MAX_ROUNDS:
            raise ToolException(f"No replans left: all {MAX_ROUNDS} plans are used. Finish with the current plan.")
    return submit_tool("request_replan", "Stop and send a report back to the planner, who then revises the plan. "
                       "The normal step when what you found changes what should be done.", Replan, check)


class Accept(BaseModel):
    reason: str = Field(min_length=1, description="in a sentence, what in the result shows that it tests the plan")


def accept_tool(done) -> tuple:
    """The planner's way to end the phase, once the engineer has submitted a result."""
    async def check(_) -> None:
        if done.value is None:
            raise ToolException("There is no result to accept yet. Submit a plan.")
    return submit_tool("accept_result", "Accept the engineer's result as the outcome of this phase.", Accept, check)


@dataclass
class Team:
    planner: Role
    plan: object                  # the submit boxes
    accept: object
    engineer: Role
    done: object
    replan: object
    rounds: list                  # every plan, with the engineer's report or result


async def plan_and_engineer(team: Team, *, task: str, show, engineer_context: str) -> None:
    """Plan and carry out. The planner plans again when the engineer asks, or when it is shown the engineer's
    result and wants it changed; the phase ends when the planner accepts a result or the plans are used up.
    Every round is appended to team.rounds and saved."""
    while True:
        if await team.planner.run(task) is team.accept:
            return
        team.rounds.append({"plan": team.plan.value.model_dump()})
        save(team.rounds)
        work = show(team.plan.value)
        work = f"The planner revised the plan.\n\n{work}" if team.engineer.messages else f"{work}\n\n{engineer_context}"
        if await team.engineer.run(work) is team.done:
            team.rounds[-1]["result"] = team.done.value.model_dump()
            save(team.rounds)
            if len(team.rounds) >= MAX_ROUNDS:
                return
            task = f"{block('engineer_result', team.done.value.model_dump_json(indent=2))}\n\n{REVIEW}"
        else:
            team.rounds[-1]["report"] = team.replan.value.report
            save(team.rounds)
            task = f"{block('engineer_report', team.replan.value.report)}\n\n{REPLAN}"


def check_roles() -> None:
    """Build every role of both phases (prompts, tool schemas) without calling a model."""
    recipe_team(None, None, [])
    edit_team([])


def previous_plans(ctx) -> list | None:
    """The failed attempt's plans, on a retry: read before this attempt overwrites the record."""
    return json.loads(RECORD.read_text()) if ctx.retry and RECORD.exists() else None


def plan_text(length: str, description: str, **kwargs):
    return Field(max_length=PLAN_FIELD_CHARS, description=f"{description} ({length})", **kwargs)


# ---- improve_recipe ----

class DataPlan(BaseModel):
    hypothesis: str = plan_text("one or two sentences", "the one data idea this node tests, and what in "
                                "earlier nodes suggests it", min_length=1)
    expected_change: str = plan_text("a sentence", "which metrics or groups should move, and in which direction",
                                     min_length=1)
    data: str = plan_text("a short paragraph", "what the training set should contain and where it comes from "
                          "(sources you checked exist); not the steps to build it", min_length=1)
    constraints: str = plan_text("optional", "what the engineer must keep or must not do, if anything", default="")


class DataAndRecipe(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains, and where it departs from the plan")
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1, description="why each changed key has its value")


def typed(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


def recipe_team(ctx: RecipeContext | None, session, ktools: list) -> Team:
    async def recipe_passes(result: DataAndRecipe) -> None:
        check = json.loads(result_text(await session.call_tool("recipe_check", {
            "recipe": typed(result.recipe, ctx.tunable_rules), "data_commit": result.data_commit})))
        if not check.get("ok"):
            raise ToolException(f"recipe_check failed: {check.get('failures')}")

    rounds: list[dict] = []
    plan_tool, plan = submit_tool("submit_plan", "Submit the data plan for this node.", DataPlan)
    done_tool, done = submit_tool("submit_data_and_recipe", "Submit the data commit and the recipe to train on. "
                                  "It is accepted only if recipe_check passes.", DataAndRecipe, recipe_passes)
    accept_t, accept = accept_tool(done)
    planner = Role("planner", [plan_tool, accept_t, *[t for t in ktools if t.name in PLANNING_KERNEL_TOOLS],
                               *local_tools(WORKSPACE, papers=True)], [plan, accept], discard=(WORKSPACE,))
    replan_t, replan = replan_tool(rounds)
    engineer = Role("data_engineer", [*ktools, *local_tools(WORKSPACE, papers=True), snap_timed_prompts,
                                      done_tool, replan_t], [done, replan])
    return Team(planner, plan, accept, engineer, done, replan, rounds)


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        team = recipe_team(ctx, session, ktools)
        previous = previous_plans(ctx)
        # Both built in a dry run too: it gets the real context.
        context, for_engineer = brief(recipe_context(ctx, previous)), brief(engineer_context(ctx, previous))
        if ctx.dry_run:
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: roles and the first message built, {len(ktools)} kernel "
                                          f"tools, model said {await ping()!r}")
        await plan_and_engineer(team, task=context, show=lambda p: block("plan", p.model_dump_json(indent=2)),
                                engineer_context=for_engineer)
    r = team.done.value
    return RecipeResult(data_commit=r.data_commit, recipe=typed(r.recipe, ctx.tunable_rules),
                        rationale=f"{r.rationale}\n\nPlan: {json.dumps(team.rounds[-1]['plan'])}\nData: {r.notes}")


# ---- edit_self ----

class EditPlan(BaseModel):
    problem: str = plan_text("one or two sentences", "where the agent system made its roles waste effort, lack a "
                             "capability or lose information", min_length=1)
    evidence: str = plan_text("a short paragraph", "where it shows: nodes, transcript files, process lines",
                              min_length=1)
    mechanism: str = plan_text("a short paragraph", "what should change in the agent system and in which "
                               "component, as intent; not the code or the text to write", min_length=1)
    check: str = plan_text("a sentence", "what a later reader of the process lines and transcripts would see if "
                           "it worked", min_length=1)


class EditSummary(BaseModel):
    summary: str = Field(min_length=1, description="one paragraph: what you changed and how you checked it")


def selftest(root: str) -> list[str]:
    """The contract's static and import checks, plus building every role, run locally so the agent can fix
    itself. agent.orchestration is imported too: agent.entry imports it only when an entry point runs."""
    errors = []
    try:
        tree = ast.parse((Path(root) / "agent" / "entry.py").read_text())
        top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("edit_self", "improve_recipe"):
            fn = top.get(name)
            if (fn is None or len(fn.args.posonlyargs + fn.args.args) != 1 or fn.args.vararg
                    or fn.args.kwarg or fn.args.kwonlyargs):
                errors.append(f"agent/entry.py needs top-level {name}(ctx) with exactly one parameter")
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        errors.append(f"agent/entry.py: {exc}")
    r = subprocess.run([sys.executable, "-c", "import agent.entry, agent.orchestration as o; o.check_roles()"],
                       cwd=root, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        errors.append(r.stderr[-6000:])
    return errors


def edit_team(ktools: list) -> Team:
    async def selftest_passes(_) -> None:
        errors = selftest(AGENT_ROOT)
        if errors:
            raise ToolException("The self-test failed. Fix these first:\n" + "\n".join(errors))

    rounds: list[dict] = []
    kernel = [t for t in ktools if t.name in EDIT_KERNEL_TOOLS]
    plan_tool, plan = submit_tool("submit_edit_plan", "Submit the edit plan.", EditPlan)
    done_tool, done = submit_tool("submit_edit", "Submit the summary of the change you made. It is accepted "
                                  "only if the self-test passes.", EditSummary, selftest_passes)
    accept_t, accept = accept_tool(done)
    planner = Role("edit_planner", [plan_tool, accept_t, *kernel, *local_tools(AGENT_ROOT, papers=False)],
                   [plan, accept], discard=(AGENT_ROOT, WORKSPACE))
    replan_t, replan = replan_tool(rounds)
    engineer = Role("coder", [*kernel, *local_tools(AGENT_ROOT, papers=False), done_tool, replan_t], [done, replan])
    return Team(planner, plan, accept, engineer, done, replan, rounds)


async def run_meta(ctx: EditContext) -> EditResult:
    async with mcp_session() as session:
        team = edit_team(await kernel_tools(session))
        task = brief(edit_context(ctx, COMPONENTS, previous_plans(ctx)))    # built in a dry run too: it gets the real context
        if ctx.dry_run:
            return EditResult(summary=f"dry run: roles and the first message built, model said {await ping()!r}")
        await plan_and_engineer(team, task=task, show=lambda p: block("edit_plan", p.model_dump_json(indent=2)),
                                engineer_context=brief(coder_context(ctx, COMPONENTS)))
    return EditResult(summary=team.done.value.summary)
