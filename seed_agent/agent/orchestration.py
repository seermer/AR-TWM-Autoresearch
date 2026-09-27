"""Which roles run, in what order, with which tools and context.

A role = a system prompt (+ reference knowledge) and a tool list, run on the harness. It returns
its result by calling a submit_<x> tool; one that stops without submitting gets one reminder.
- improve_recipe: plan -> (build data -> write recipe -> recipe_check) x up to CHECK_ROUNDS.
- edit_self: plan ONE edit to ONE component -> implement -> self-test -> (fix | finish).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditComponent, EditContext, EditResult, RecipeContext, RecipeResult
from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, Field

from .entry import (AGENT_ROOT, BRIEF_CHARS, CHECK_ROUNDS, COMPACT_AT, CONTEXT_WINDOW, MODEL,
                    SELFTEST_ROUNDS, WORKSPACE)
from .harness import build_react_agent
from .tools import kernel_tools, make_file_tools, result_text, snap_timed_prompts, submit_tool

AGENT_PKG = Path(__file__).resolve().parent
REMIND = ("You stopped without calling {tool}. Finish the task, then call {tool} with the result. "
          "The work is only recorded through {tool}.")

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

def system_prompt(name: str, knowledge: tuple[str, ...] = ()) -> str:
    text = (AGENT_PKG / "prompts" / f"{name}.md").read_text()
    for doc in knowledge:
        text += f"\n\n# Reference: {doc}\n\n" + (AGENT_PKG / "knowledge" / doc).read_text()
    return text


async def run_role(system: str, tools: list, task: str, submission=None) -> list[AnyMessage]:
    """Run one role to completion. A role that must submit (`submission`: the box from submit_tool)
    gets one reminder if it stops early."""
    agent = build_react_agent(chat_model(MODEL), tools, system, context_window=CONTEXT_WINDOW,
                              compact_at=COMPACT_AT)
    state = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    if submission is not None and submission.value is None:
        state = await agent.ainvoke({"messages": [*state["messages"],
                                                  HumanMessage(content=REMIND.format(tool=submission.name))]})
    if submission is not None and submission.value is None:
        raise RuntimeError(f"the role finished without calling {submission.name}")
    return state["messages"]


# ---- improve_recipe ----

class DataPlan(BaseModel):
    hypotheses: list[str] = Field(min_length=1, description="1-3 testable data hypotheses")
    actions: list[str] = Field(min_length=1, description="concrete steps that test them")


class BuildOutcome(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains and why")


class RecipeDraft(BaseModel):
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1)


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        if ctx.dry_run:                  # contract smoke run: prove the wiring, do no work
            messages = await run_role("Reply with the single word ok.", [], "ping")
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: {len(ktools)} kernel tools, model said "
                                          f"{messages[-1].text!r}")
        recipe_rules = {"rules": ctx.tunable_rules, "resolution_allowlist": ctx.resolution_allowlist,
                        "lora_allowlist": ctx.lora_allowlist}      # first, so the length cap never cuts them
        context = json.dumps({**recipe_rules,
                              "lineage": ctx.lineage, "archive": ctx.archive, "n_gpus": ctx.n_gpus,
                              "parent_data_commit": ctx.parent_data_commit, "parent_recipe": ctx.parent_recipe,
                              "clip_pool_size": len(ctx.clip_pool), "tools": ctx.tools, "retry": ctx.retry,
                              "format_rules": ctx.format_rules}, default=str)[:BRIEF_CHARS]
        plan_tool, plan = submit_tool("submit_plan", "Submit the data plan for this node.", DataPlan)
        await run_role(system_prompt("planner"), [plan_tool], context, plan)
        failures: list = []
        for _ in range(CHECK_ROUNDS):
            build_tool, built = submit_tool("submit_data_commit", "Submit the data commit to train on.",
                                            BuildOutcome)
            task = f"PLAN:\n{plan.value.model_dump_json(indent=2)}\n\nCONTEXT:\n{context}"
            if failures:
                task += ("\n\nTHE LAST RECIPE CHECK FAILED. Fix the data if the failures are about data:\n"
                         + json.dumps(failures))
            await run_role(system_prompt("data_builder", knowledge=("data_building.md",)),
                           [*ktools, *make_file_tools(WORKSPACE), snap_timed_prompts,
                            build_tool], task, built)
            recipe_tool, draft = submit_tool("submit_recipe", "Submit the training recipe.", RecipeDraft)
            await run_role(system_prompt("recipe_writer"), [recipe_tool], json.dumps(
                {**recipe_rules, "n_gpus": ctx.n_gpus, "data_notes": built.value.notes,
                 "parent_recipe": ctx.parent_recipe, "previous_failures": failures}), draft)
            recipe = {key: int(round(value)) if ctx.tunable_rules.get(key, {}).get("type") == "int"
                      else float(value) for key, value in draft.value.recipe.items()}
            try:
                check = json.loads(result_text(await session.call_tool(
                    "recipe_check", {"recipe": recipe, "data_commit": built.value.data_commit})))
                failures = [] if check.get("ok") else check.get("failures", [])
            except Exception as exc:  # noqa: BLE001 -- a failed check is a failure to fix
                failures = [str(exc)]
            if not failures:
                break
    rationale = draft.value.rationale
    if failures:
        rationale += f"\n\nUnresolved recipe_check failures: {failures}"
    return RecipeResult(data_commit=built.value.data_commit, recipe=recipe,
                        rationale=f"{rationale}\n\nPlan: {plan.value.model_dump_json()}\nData: {built.value.notes}")


# ---- edit_self ----

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
        messages = await run_role("Reply with the single word ok.", [], "ping")
        return EditResult(summary=f"dry run: model said {messages[-1].text!r}")
    plan_file = Path(WORKSPACE) / "edit_plan.json"          # carried to a retry with the workspace
    previous = json.loads(plan_file.read_text()) if ctx.retry and plan_file.exists() else None
    plan_tool, plan = submit_tool("submit_edit_plan",
                                  "Submit the edit plan: exactly one component and one focused change.",
                                  EditPlan)
    brief = json.dumps({"components": COMPONENTS, "lineage": ctx.lineage, "archive": ctx.archive,
                        "nodes_remaining": ctx.nodes_remaining, "retry": ctx.retry,
                        "previous_attempt_plan": previous}, default=str)[:BRIEF_CHARS]
    await run_role(system_prompt("edit_planner"), [*make_file_tools(AGENT_ROOT, writable=False), plan_tool],
                   brief, plan)
    p = plan.value
    plan_file.write_text(p.model_dump_json(indent=2))
    errors: list[str] = []
    summary = ""
    for _ in range(SELFTEST_ROUNDS):
        task = f"EDIT PLAN (component: {p.component}):\n{p.model_dump_json(indent=2)}"
        if errors:
            task += "\n\nTHE SELF-TEST FAILED; fix these first:\n" + "\n".join(errors)
        messages = await run_role(system_prompt("coder"), make_file_tools(AGENT_ROOT), task)
        summary = messages[-1].text
        errors = selftest(AGENT_ROOT)
        if not errors:
            break
    note = f"\n\n(self-test still failing: {errors})" if errors else ""
    return EditResult(summary=f"[{p.component}] {p.change}\n\n{summary or 'no summary'}{note}",
                      component=p.component)
