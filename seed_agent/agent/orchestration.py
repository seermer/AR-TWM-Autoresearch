"""Which roles run, in what order, with which tools and context.

A role = a system prompt (+ the knowledge index) and a tool list, run on the harness. It returns its result
by calling a submit tool; one that stops without submitting gets one reminder.
Both phases are a planner/engineer loop: the planner (read-only tools) submits a plan, the engineer carries
it out, then either finishes the phase or reports back for a new plan, up to MAX_ROUNDS plans. Each role
keeps its conversation across rounds. The last plan is the final one; every plan is recorded.
- improve_recipe: planner -> data engineer, who builds the data commit and writes the recipe
  (recipe_check must pass before the submission is accepted).
- edit_self: edit planner (ONE component) -> coder (the self-test must pass before the submission is accepted).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditComponent, EditContext, EditResult, RecipeContext, RecipeResult
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from .entry import AGENT_ROOT, BRIEF_CHARS, COMPACT_AT, CONTEXT_WINDOW, MAX_ROUNDS, MODEL, WORKSPACE
from .harness import build_react_agent
from .tools import ARXIV_TOOLS, kernel_tools, make_file_tools, result_text, snap_timed_prompts, submit_tool

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


def local_tools(root: str, *, writable: bool = True) -> list:
    return [*make_file_tools(root, writable=writable), *ARXIV_TOOLS]


def brief(payload: dict) -> str:
    """The JSON context for a role, capped at BRIEF_CHARS with a visible cut."""
    text = json.dumps(payload, default=str)
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS] + " ...[truncated]"
    return block("context", text)


class Role:
    """One role on the harness. Its conversation continues across calls to run()."""

    def __init__(self, name: str, tools: list, submissions: list) -> None:
        self.submissions = submissions            # the boxes of its submit tools
        self.messages: list = []
        self.agent = build_react_agent(chat_model(MODEL), tools, system_prompt(name),
                                       context_window=CONTEXT_WINDOW, compact_at=COMPACT_AT)

    def _submitted(self):
        return next((box for box in self.submissions if box.value is not None), None)

    async def run(self, task: str):
        """Work on `task` until a submit tool is called; returns that tool's box. One reminder if it stops early."""
        for box in self.submissions:
            box.value = None
        names = " or ".join(box.name for box in self.submissions)
        for message in (task, REMIND.format(tools=names)):
            state = await self.agent.ainvoke({"messages": [*self.messages, HumanMessage(content=message)]})
            self.messages = state["messages"]
            if self._submitted() is not None:
                return self._submitted()
        raise RuntimeError(f"the role finished without calling {names}")


async def ping() -> str:
    """Contract smoke run: prove the wiring, do no work."""
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


async def plan_and_engineer(planner: Role, plan, engineer: Role, done, replan, *, rounds: list, task: str,
                            show, engineer_context: str | None, record: Path) -> None:
    """Plan, carry out, and replan when the engineer asks, until the engineer finishes. Every plan (with
    the engineer's report when it asked for a new one) is appended to `rounds` and saved to `record`."""
    while True:
        await planner.run(task)
        rounds.append({"plan": plan.value.model_dump()})
        record.write_text(json.dumps(rounds, indent=1))
        work = show(plan.value)
        if engineer.messages:
            work = f"The planner revised the plan.\n\n{work}"
        elif engineer_context:
            work = f"{work}\n\n{engineer_context}"
        if await engineer.run(work) is done:
            return
        rounds[-1]["report"] = replan.value.report
        record.write_text(json.dumps(rounds, indent=1))
        task = f"{block('engineer_report', replan.value.report)}\n\n{REPLAN}"


# ---- improve_recipe ----

class DataPlan(BaseModel):
    hypotheses: list[str] = Field(min_length=1, description="1-3 testable data hypotheses")
    actions: list[str] = Field(min_length=1, description="concrete steps that test them")


class DataAndRecipe(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains and why")
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1, description="why each changed key has its value")


READ_ONLY_KERNEL_TOOLS = {"hf_search", "hf_list_files", "data_query"}


def typed(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        if ctx.dry_run:
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: {len(ktools)} kernel tools, model said {await ping()!r}")
        context = brief({"rules": ctx.tunable_rules, "resolution_allowlist": ctx.resolution_allowlist,
                         "lora_allowlist": ctx.lora_allowlist,        # first, so the length cap never cuts them
                         "lineage": ctx.lineage, "archive": ctx.archive, "n_gpus": ctx.n_gpus,
                         "parent_data_commit": ctx.parent_data_commit, "parent_recipe": ctx.parent_recipe,
                         "base_recipe": ctx.base_recipe, "recipe_guide": ctx.recipe_guide,
                         "clip_pool_size": len(ctx.clip_pool), "tools": ctx.tools, "retry": ctx.retry,
                         "format_rules": ctx.format_rules})

        async def recipe_passes(result: DataAndRecipe) -> None:
            check = json.loads(result_text(await session.call_tool("recipe_check", {
                "recipe": typed(result.recipe, ctx.tunable_rules), "data_commit": result.data_commit})))
            if not check.get("ok"):
                raise ToolException(f"recipe_check failed: {check.get('failures')}")

        plan_tool, plan = submit_tool("submit_plan", "Submit the data plan for this node.", DataPlan)
        done_tool, done = submit_tool("submit_data_and_recipe", "Submit the data commit and the recipe to train "
                                      "on. It is accepted only if recipe_check passes.", DataAndRecipe, recipe_passes)
        rounds: list[dict] = []
        replan_t, replan = replan_tool(rounds)
        planner = Role("planner", [plan_tool, *[t for t in ktools if t.name in READ_ONLY_KERNEL_TOOLS],
                                   *local_tools(WORKSPACE, writable=False)], [plan])
        engineer = Role("data_engineer", [*ktools, *local_tools(WORKSPACE), snap_timed_prompts, done_tool, replan_t],
                        [done, replan])
        await plan_and_engineer(planner, plan, engineer, done, replan, rounds=rounds, task=context,
                                         show=lambda p: block("plan", p.model_dump_json(indent=2)),
                                         engineer_context=context, record=Path(WORKSPACE) / "plans.json")
    r = done.value
    return RecipeResult(data_commit=r.data_commit, recipe=typed(r.recipe, ctx.tunable_rules),
                        rationale=f"{r.rationale}\n\nPlans: {json.dumps(rounds)}\nData: {r.notes}")


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
    """The contract's static and import checks, run locally so the agent can fix itself.
    agent.orchestration is imported too: agent.entry imports it only when an entry point runs."""
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
    r = subprocess.run([sys.executable, "-c", "import agent.entry, agent.orchestration"], cwd=root,
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        errors.append(r.stderr[-3000:])
    return errors


async def run_meta(ctx: EditContext) -> EditResult:
    if ctx.dry_run:
        return EditResult(summary=f"dry run: model said {await ping()!r}")
    record = Path(WORKSPACE) / "plans.json"                 # carried to a retry with the workspace
    previous = json.loads(record.read_text()) if ctx.retry and record.exists() else None

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
    planner = Role("edit_planner", [plan_tool, *local_tools(AGENT_ROOT, writable=False)], [plan])
    engineer = Role("coder", [*local_tools(AGENT_ROOT), done_tool, replan_t], [done, replan])
    await plan_and_engineer(
        planner, plan, engineer, done, replan, rounds=rounds, record=record, engineer_context=None,
        task=brief({"components": COMPONENTS, "lineage": ctx.lineage, "archive": ctx.archive,
                    "nodes_remaining": ctx.nodes_remaining, "retry": ctx.retry,
                    "previous_attempt_plans": previous}),
        show=lambda p: block("edit_plan", p.model_dump_json(indent=2), component=p.component))
    p = plan.value
    return EditResult(summary=f"[{p.component}] {p.change}\n\n{done.value.summary}", component=p.component)
