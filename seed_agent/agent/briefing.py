"""What a role is told first: the kernel's context as a Markdown digest. The exact data is in
/context/context.json and each finished node's files are under /nodes/<node>/; the digest points there
instead of repeating them. The two phases are told different things: the data roles see how earlier
nodes scored, on what data and with which data idea; the edit roles see how the agent code changed and
how each run went, and no score. Small sections come first and the lineage last, and only the most
recent LINEAGE_SHOWN ancestors are described, so the digest stays about the same size at any depth."""
from __future__ import annotations

import json
import re
from collections import Counter

LINEAGE_SHOWN = 10          # most recent ancestors described; the root stays in the score tables as the baseline
SIBLINGS_SHOWN = 10         # most recent finished children of the parent described
TEXT_CHARS = 4000           # safety cap on a free text quoted per node
TOP_NODES = 5               # best nodes listed from the whole archive
GROUP_AXES = ("instruction_kind", "viewpoint")     # groups in the group table; scene categories are in context.json
POINTER = "Exact data: /context/context.json. What each finished node left behind: /nodes/<node>/."


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows])


def clip(text: str, limit: int = TEXT_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _fields(value: dict) -> str:
    """A small dict as Markdown bullets: multi-line text in a code block, the rest inline."""
    out = []
    for key, item in value.items():
        if isinstance(item, str) and "\n" in item:
            out.append(f"- {key}:\n```\n{item.strip()}\n```")
        else:
            out.append(f"- {key}: {item if isinstance(item, str) else json.dumps(item)}")
    return "\n".join(out)


def retry_section(retry: dict | None, previous_plans: list | None) -> str:
    if not retry:
        return ""
    parts = ["## Retry", "The previous attempt of this phase failed:", _fields(retry)]
    for i, rnd in enumerate(previous_plans or [], 1):
        parts.append(f"Its plan {i}:\n```json\n{json.dumps(rnd, indent=1)}\n```")
    return "\n\n".join(parts)


def components_section(components: dict) -> str:
    return "- Components of this agent:\n" + "\n".join(f"  - {k}: {v}" for k, v in components.items())


# ---- improve_recipe: scores, data, data ideas ----

def metric_section(guide: dict) -> str:
    if not guide:
        return ""
    return ("## Metrics\n\nThe score is the weighted mean of these metrics. A dimension is the plain mean of its "
            "metrics and is not part of the score.\n\n"
            + table(["metric", "dimension", "weight", "measures"],
                    [[name, g.get("dimension"), g.get("weight"), g.get("measures")] for name, g in guide.items()]))


def best_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    best = sorted((n for n in nodes if n.get("score") is not None), key=lambda n: -n["score"])[:TOP_NODES]
    return "\n\n".join([
        "## Archive",
        f"{len(nodes)} nodes, {archive.get('n_scored', len(best))} scored. The best {len(best)}:",
        table(["node", "parent", "depth", "score", "status"],
              [[n["node_id"], n.get("parent_id"), n.get("depth"), n.get("score"), n.get("status")] for n in best])])


def _score_rows(nodes: list[dict], pick) -> list[list]:
    keys: list[str] = []
    for n in nodes:
        keys += [k for k in pick(n.get("aggregates") or {}) if k not in keys]
    return [[k, *[pick(n.get("aggregates") or {}).get(k) for n in nodes]] for k in keys]


def _groups(aggregates: dict) -> dict:
    """One row per group and dimension: the dimension's score over that group of evaluation items alone."""
    return {f"{axis}: {name} / {dimension}": value
            for axis, groups in (aggregates.get("groups") or {}).items() if axis in GROUP_AXES
            for name, dimensions in groups.items() for dimension, value in dimensions.items()}


def _data(n: dict) -> str:
    return "- Data: " + "; ".join(
        f"{name} ({d.get('format')}{'/' + d['prompt_mode'] if d.get('prompt_mode') else ''}): "
        f"{d.get('clips')} clips, weight {d.get('weight')}"
        + (", from " + ", ".join(f"{s} x{k}" for s, k in d["sources"].items()) if d.get("sources") else "")
        for name, d in n["data"].items())


def _hypothesis(rationale: str | None) -> str | None:
    """The data idea a node tested: run_task records its final plan in the rationale as a `Plan: {json}` line."""
    plans = re.findall(r"^Plan: (\{.*\})$", rationale or "", re.MULTILINE)
    try:
        return json.loads(plans[-1]).get("hypothesis") if plans else None      # the last one: run_task appends it
    except ValueError:                                      # a line of the rationale that only looks like one
        return None


def data_node(n: dict) -> str:
    """An earlier node as the data planner needs it: what data idea it tested, on what data, with which recipe."""
    lines = [f"### {n['node_id']}: {n.get('status')}, score {_cell(n.get('score'))}"]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if _hypothesis(n.get("rationale")):
        lines.append(f"- Hypothesis: {clip(_hypothesis(n['rationale']))}")
    if n.get("data"):
        lines.append(_data(n))
    if n.get("recipe"):
        lines.append("- Recipe: " + ", ".join(f"{k} {v}" for k, v in n["recipe"].items()))
    return "\n".join(lines)


# ---- edit_self: code edits and how each run went ----

def status_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    counts = Counter(n.get("status") for n in nodes)
    return f"## Archive\n\n{len(nodes)} nodes: " + ", ".join(f"{status} {n}" for status, n in counts.most_common())


def _process(p: dict) -> list[str]:
    """How a node's run went, one line per kind of fact. The tool error messages are in context.json."""
    lines = [f"- {r['phase']} / {r['role']}: {r['turns']} model turns, {r['compactions']} compactions"
             for r in p.get("roles") or []]
    if p.get("tools"):
        lines.append("- Kernel tool calls (errors): "
                     + ", ".join(f"{tool} {v['calls']} ({v['errors']})" for tool, v in p["tools"].items()))
    if p.get("local_errors"):
        lines.append("- Local tool failures: " + ", ".join(f"{k} {v}" for k, v in p["local_errors"].items()))
    jobs = p.get("gpu_jobs") or {}
    if jobs.get("run"):
        lines.append(f"- GPU jobs: {jobs['run']} run, {jobs['failed']} failed")
    ingest = p.get("ingest") or {}
    if ingest.get("accepted") or ingest.get("rejected"):
        reasons = "".join(f"; {r['count']}x {clip(r['reason'], 200)}" for r in ingest.get("reasons") or [])
        lines.append(f"- Ingest: {ingest['accepted']} accepted, {ingest['rejected']} rejected{reasons}")
    lines += [f"- {phase} rounds: {r['plans']} plans, {r['reports']} reports back"
              for phase, r in (p.get("rounds") or {}).items()]
    failed = [a for a in p.get("attempts") or [] if a["outcome"] != "passed"]
    if failed:
        lines.append("- Failed attempts: " + ", ".join(f"{a['phase']} {a['attempt']} {a['outcome']}" for a in failed))
    if p.get("gates"):
        lines.append("- Pre-training checks: " + ", ".join(f"{k} {v}" for k, v in p["gates"].items()))
    return lines


def edit_node(n: dict) -> str:
    """An earlier node as the edit planner needs it: how the agent code changed and how the run went."""
    lines = [f"### {n['node_id']}: {n.get('status')}"]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if n.get("edit"):
        lines.append(f"- Edit: {clip(n['edit'].get('summary', ''))}")
    if n.get("code_diff_stats"):
        lines.append("- Code changed: " + ", ".join(f"{d['path']} (+{d['added']}/-{d['removed']})"
                                                    for d in n["code_diff_stats"]))
    return "\n".join(lines + _process(n.get("process") or {}))


# ---- sections shared by both phases ----

def lineage_section(lineage: list[dict], describe, scores: bool) -> str:
    if not lineage:
        return ""
    shown = lineage[-LINEAGE_SHOWN:]
    older = "" if len(shown) == len(lineage) else (
        f" Only the last {len(shown)} are described here; all of them are in /context/context.json and /nodes/.")
    parts = ["## Lineage", f"From the root to the parent: {len(lineage)} nodes, oldest first.{older}"]
    if scores:
        columns = shown if shown[0] is lineage[0] else [lineage[0], *shown]       # the root: the baseline
        ids = [n["node_id"] for n in columns]
        parts += [
            "### Scores\n\n" + table(["", *ids], [["score", *[n.get("score") for n in columns]],
                                                  *_score_rows(columns, lambda a: a.get("dimensions") or {})]),
            "### Dimensions by group\n\n" + table(["", *ids], _score_rows(columns, _groups)),
            "### Metrics\n\n" + table(["", *ids], _score_rows(columns, lambda a: a.get("metrics") or {}))]
    return "\n\n".join([*parts, *[describe(n) for n in shown]])


def siblings_section(siblings: list[dict], describe, scores: bool) -> str:
    if not siblings:
        return ""
    shown = siblings[-SIBLINGS_SHOWN:]
    older = "" if len(shown) == len(siblings) else (
        f" Only the last {len(shown)} are described below; all of them are in /context/context.json and /nodes/.")
    header = ["node", "status", "score"] if scores else ["node", "status"]
    return "\n\n".join([
        "## Siblings",
        f"The parent's other children that have finished: {len(siblings)}, oldest first.{older}",
        table(header, [[n["node_id"], n.get("status"), n.get("score")][:len(header)] for n in siblings]),
        *[describe(n) for n in shown]])


def _node_and_recipe(ctx, previous_plans: list | None) -> list[str]:
    """What the data phase needs to act: this node's facts, the retry report, the recipe keys and the metrics."""
    guide = ctx.recipe_guide or {}
    keys = [[k, r.get("type"), r.get("min"), r.get("max"), (guide.get(k) or {}).get("base"),
             ctx.parent_recipe.get(k), (guide.get(k) or {}).get("meaning")]
            for k, r in (ctx.tunable_rules or {}).items()]
    pairs = lambda allowed, sep: ", ".join(f"{a}{sep}{b}" for a, b in allowed)
    return [
        POINTER,
        "## This node\n\n" + "\n".join([
            f"- Training GPUs: {ctx.n_gpus}",
            f"- Parent data commit: {ctx.parent_data_commit or 'none (the parent is the root)'}",
            f"- Clip pool: {ctx.clip_pool_size} clips in the archive",
            f"- Kernel tools: {', '.join(ctx.tools)}"]),
        retry_section(ctx.retry, previous_plans),
        "## Recipe\n\nTunable keys, with the base recipe's and the parent's values:\n\n"
        + table(["key", "type", "min", "max", "base", "parent", "meaning"], keys)
        + f"\n\nResolutions (height x width): {pairs(ctx.resolution_allowlist, 'x')}. "
          f"LoRA (rank/alpha): {pairs(ctx.lora_allowlist, '/')}. The full base recipe is `base_recipe` "
          f"in /context/context.json.",
        metric_section(ctx.metric_guide)]


def recipe_context(ctx, previous_plans: list | None) -> str:
    """The data planner's first message."""
    return "\n\n".join(s for s in [
        *_node_and_recipe(ctx, previous_plans),
        best_section(ctx.archive),
        siblings_section(ctx.siblings, data_node, scores=True),
        lineage_section(ctx.lineage, data_node, scores=True)] if s)


def engineer_context(ctx, previous_plans: list | None) -> str:
    """What the data engineer gets with the plan: the plan already carries what the planner drew from history."""
    return "\n\n".join(s for s in _node_and_recipe(ctx, previous_plans) if s)


def edit_context(ctx, components: dict, previous_plans: list | None) -> str:
    """The edit planner's first message."""
    return "\n\n".join(s for s in [
        POINTER,
        f"## This node\n\n- Nodes left in the run after this one: {ctx.nodes_remaining}\n" + components_section(components),
        retry_section(ctx.retry, previous_plans),
        status_section(ctx.archive),
        siblings_section(ctx.siblings, edit_node, scores=False),
        lineage_section(ctx.lineage, edit_node, scores=False)] if s)


def coder_context(components: dict) -> str:
    """What the coder gets with the plan."""
    return f"{POINTER}\n\n## This agent\n\n{components_section(components)}"
