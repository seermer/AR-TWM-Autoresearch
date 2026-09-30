"""Which roles run, in what order, with which tools and context.

A role = a system prompt (+ the knowledge index) and a tool list, run on the harness. It returns its result
by calling a submit tool; one that stops without submitting gets one reminder.
Both phases are a planner/engineer loop: the planner submits a plan, the engineer carries it out, then either
finishes the phase or reports back for a new plan, up to MAX_ROUNDS plans. Each role keeps its conversation
across rounds. The last plan is the final one; every plan is recorded. A planner has the same local tools as
an engineer, but what it changes on disk is undone after each of its turns, and its kernel tools are read-only.
- improve_recipe: planner -> data engineer, who builds the data commit and writes the recipe
  (recipe_check must pass before the submission is accepted).
- edit_self: edit planner (ONE component) -> coder (the self-test must pass before the submission is accepted).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditComponent, EditContext, EditResult, RecipeContext, RecipeResult
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from .entry import AGENT_ROOT, BRIEF_CHARS, COMPACT_AT, CONTEXT_WINDOW, MAX_ROUNDS, MODEL, WORKSPACE
from .harness import build_react_agent
from .tools import (ARXIV_TOOLS, discarded_changes, kernel_tools, make_file_tools, result_text, snap_timed_prompts,
                    submit_tool)

AGENT_PKG = Path(__file__).resolve().parent
REMIND = ("You stopped without calling {tools}. Finish the task, then call {tools} with the result. "
          "The work is only recorded through {tools}.")
REPLAN = "Revise the plan. The engineer carries out the plan you submit next."

# Where each component lives in this agent (spec 9.1.1). An edit plan names one of them.
COMPONENTS = {
    "prompts": "agent/prompts/*.md -- the system prompt of each role",
    "tools": "agent/tools.py -- agent-local tools and the kernel-tool adapter (not the kernel tools themselves)",
    "harness": "agent/harness.py -- the single-agent inner loop: ReAct graph, tool execution, auto-compaction",
    "orchestration": "agent/orchestration.py (+ agent/entry.py settings) -- which roles run, in what order, "
                     "with which tools and context",
    "knowledge": "agent/knowledge/*.md -- reference material that roles read",
}


# ---- roles ----

def block(tag: str, text: str, **attrs) -> str:
    attributes = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"<{tag}{attributes}>\n{text.strip()}\n</{tag}>"


def description(text: str) -> str:
    """The `description:` of a knowledge file's front matter."""
    head = text.split("\n---", 1)[0] if text.startswith("---\n") else ""
    return next((line.split(":", 1)[1].strip() for line in head.splitlines() if line.startswith("description:")), "")


def system_prompt(name: str) -> str:
    """The role's prompt, then the index of knowledge files: each path with when it is needed."""
    prompt = (AGENT_PKG / "prompts" / f"{name}.md").read_text()
    index = "\n".join(f"- {path}: {description(path.read_text())}"
                      for path in sorted((AGENT_PKG / "knowledge").glob("*.md")))
    return f"{prompt.strip()}\n\n{block('knowledge', index)}"


def local_tools(root: str) -> list:
    return [*make_file_tools(root), *ARXIV_TOOLS]


def brief(payload: dict) -> str:
    """The JSON context for a role, capped at BRIEF_CHARS with a visible cut: callers put the small keys
    first and the lineage last, so a cut only shortens the lineage."""
    text = json.dumps(payload, default=str)
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
    report: str = Field(min_length=1, description="what you did and found, and why the plan must change")


def replan_tool(rounds: list) -> tuple:
    async def check(_) -> None:
        if len(rounds) >= MAX_ROUNDS:
            raise ToolException(f"No replans left: all {MAX_ROUNDS} plans are used. Finish with the current plan.")
    return submit_tool("request_replan", "Stop and send a report back to the planner, who then revises the plan.",
                       Replan, check)


@dataclass
class Team:
    planner: Role
    plan: object                  # the submit boxes
    engineer: Role
    done: object
    replan: object
    rounds: list                  # every plan, with the engineer's report when it asked for a new one


async def plan_and_engineer(team: Team, *, task: str, show, engineer_context: str | None, record: Path) -> None:
    """Plan, carry out, and replan when the engineer asks, until the engineer finishes. Every round is
    appended to team.rounds and saved to `record`."""
    while True:
        await team.planner.run(task)
        team.rounds.append({"plan": team.plan.value.model_dump()})
        record.write_text(json.dumps(team.rounds, indent=1))
        work = show(team.plan.value)
        if team.engineer.messages:
            work = f"The planner revised the plan.\n\n{work}"
        elif engineer_context:
            work = f"{work}\n\n{engineer_context}"
        if await team.engineer.run(work) is team.done:
            return
        team.rounds[-1]["report"] = team.replan.value.report
        record.write_text(json.dumps(team.rounds, indent=1))
        task = f"{block('engineer_report', team.replan.value.report)}\n\n{REPLAN}"


def check_roles() -> None:
    """Build every role of both phases (prompts, knowledge index, tool schemas) without calling a model."""
    recipe_team(None, None, [])
    edit_team()


# ---- improve_recipe ----

class DataPlan(BaseModel):
    hypothesis: str = Field(min_length=1, description="the one testable data hypothesis of this node")
    actions: list[str] = Field(min_length=1, description="concrete steps that test it")


class DataAndRecipe(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains and why")
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1, description="why each changed key has its value")


PLANNING_KERNEL_TOOLS = {"hf_search", "hf_list_files", "data_query"}       # read-only: planning changes nothing


def typed(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


def recipe_team(ctx: RecipeContext | None, session, ktools: list) -> Team:
    async def recipe_passes(result: DataAndRecipe) -> None:
        check = json.loads(result_text(await session.call_tool("recipe_check", {
            "recipe": typed(result.recipe, ctx.tunable_rules), "data_commit": result.data_commit})))
        if not check.get("ok"):
            raise ToolException(f"recipe_check failed: {check.get('failures')}")

    plan_tool, plan = submit_tool("submit_plan", "Submit the data plan for this node.", DataPlan)
    done_tool, done = submit_tool("submit_data_and_recipe", "Submit the data commit and the recipe to train on. "
                                  "It is accepted only if recipe_check passes.", DataAndRecipe, recipe_passes)
    rounds: list[dict] = []
    replan_t, replan = replan_tool(rounds)
    planner = Role("planner", [plan_tool, *[t for t in ktools if t.name in PLANNING_KERNEL_TOOLS],
                               *local_tools(WORKSPACE)], [plan], discard=(WORKSPACE,))
    engineer = Role("data_engineer", [*ktools, *local_tools(WORKSPACE), snap_timed_prompts, done_tool, replan_t],
                    [done, replan])
    return Team(planner, plan, engineer, done, replan, rounds)


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        team = recipe_team(ctx, session, ktools)
        if ctx.dry_run:
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: roles built, {len(ktools)} kernel tools, "
                                          f"model said {await ping()!r}")
        record = Path(WORKSPACE) / "plans.json"             # carried to a retry with the workspace
        previous = json.loads(record.read_text()) if ctx.retry and record.exists() else None
        context = brief({"rules": ctx.tunable_rules, "resolution_allowlist": ctx.resolution_allowlist,
                         "lora_allowlist": ctx.lora_allowlist, "n_gpus": ctx.n_gpus,
                         "parent_data_commit": ctx.parent_data_commit, "parent_recipe": ctx.parent_recipe,
                         "tools": ctx.tools, "retry": ctx.retry, "previous_attempt_plans": previous,
                         "format_rules": ctx.format_rules, "recipe_guide": ctx.recipe_guide,
                         "base_recipe": ctx.base_recipe, "clip_pool_size": len(ctx.clip_pool),
                         "archive": ctx.archive, "lineage": ctx.lineage})
        await plan_and_engineer(team, task=context, show=lambda p: block("plan", p.model_dump_json(indent=2)),
                                engineer_context=context, record=record)
    r = team.done.value
    return RecipeResult(data_commit=r.data_commit, recipe=typed(r.recipe, ctx.tunable_rules),
                        rationale=f"{r.rationale}\n\nPlan: {json.dumps(team.rounds[-1]['plan'])}\nData: {r.notes}")


# ---- edit_self ----

class EditSummary(BaseModel):
    summary: str = Field(min_length=1, description="one paragraph: what you changed and why")


class EditPlan(BaseModel):
    component: EditComponent = Field(description="the ONE component this edit changes")
    change: str = Field(min_length=1, description="the concrete change")
    files: list[str] = Field(description="files you expect to change, relative to /agent")
    rationale: str = Field(min_length=1, description="evidence from the lineage for this change")
    expected_effect: str = Field(min_length=1, description="what should improve, and how you will know")


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
        errors.append(r.stderr[-3000:])
    return errors


def edit_team() -> Team:
    async def selftest_passes(_) -> None:
        errors = selftest(AGENT_ROOT)
        if errors:
            raise ToolException("The self-test failed. Fix these first:\n" + "\n".join(errors))

    plan_tool, plan = submit_tool("submit_edit_plan",
                                  "Submit the edit plan: exactly one component and one focused change.", EditPlan)
    done_tool, done = submit_tool("submit_edit", "Submit the summary of the change you made. It is accepted "
                                  "only if the self-test passes.", EditSummary, selftest_passes)
    rounds: list[dict] = []
    replan_t, replan = replan_tool(rounds)
    planner = Role("edit_planner", [plan_tool, *local_tools(AGENT_ROOT)], [plan], discard=(AGENT_ROOT, WORKSPACE))
    engineer = Role("coder", [*local_tools(AGENT_ROOT), done_tool, replan_t], [done, replan])
    return Team(planner, plan, engineer, done, replan, rounds)


async def run_meta(ctx: EditContext) -> EditResult:
    team = edit_team()
    if ctx.dry_run:
        return EditResult(summary=f"dry run: roles built, model said {await ping()!r}")
    record = Path(WORKSPACE) / "plans.json"                 # carried to a retry with the workspace
    previous = json.loads(record.read_text()) if ctx.retry and record.exists() else None
    await plan_and_engineer(
        team, record=record, engineer_context=None,
        task=brief({"components": COMPONENTS, "nodes_remaining": ctx.nodes_remaining, "retry": ctx.retry,
                    "previous_attempt_plans": previous, "archive": ctx.archive, "lineage": ctx.lineage}),
        show=lambda p: block("edit_plan", p.model_dump_json(indent=2), component=p.component))
    p = team.plan.value
    return EditResult(summary=f"[{p.component}] {p.change}\n\n{team.done.value.summary}", component=p.component)
