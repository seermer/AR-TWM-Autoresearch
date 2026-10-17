"""Each role's context as Markdown."""
from __future__ import annotations

from collections import Counter

LINEAGE_SHOWN = 10          # ancestors shown
SIBLINGS_SHOWN = 10         # siblings shown
TEXT_CHARS = 4000           # chars per quoted text
TOP_NODES = 5               # best nodes listed
GROUP_AXES = ("viewpoint",)     # groups in the group table
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


TOOL_OUTPUT = ("every tool call's full output, one file per call named <role>-<tool>-<time>-<n>.txt; an attempt "
               "workspace under /nodes/<node>/attempts/ keeps its own tool_output folder")


def folders_section(folders: dict, scratch: str) -> str:
    """What each folder is for."""
    folders = {**folders, "/workspace/tool_output": TOOL_OUTPUT,
               scratch: "your scratch folder, for throwaway files; removed when your conversation ends"}
    return "## Folders\n\n" + "\n".join(f"- `{path}`: {what}" for path, what in folders.items())


def skill_section(skill: dict) -> str:
    if not skill:
        return ""
    return "## Target skill\n\nThe skill this run aims to improve.\n\n" + "\n".join(
        f"- {key}: {skill[key]}" for key in ("name", "sentence", "description") if skill.get(key))


def _pool(ctx) -> str:
    """The clip pool in one line."""
    parts = [f"{name}: " + ", ".join(f"{key} {n}" for key, n in (ctx.clip_pool_stats.get(by) or {}).items())
             for name, by in (("ingested by", "by_node"), ("sources", "by_source")) if ctx.clip_pool_stats.get(by)]
    return f"- Clip pool: {ctx.clip_pool_size} clips in the archive" + (f" ({'; '.join(parts)})" if parts else "")


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
        table(["node", "parent", "score", "status", "data commit"],
              [[n["node_id"], n.get("parent_id"), n.get("score"), n.get("status"), n.get("data_commit")]
               for n in best])])


def _score_rows(nodes: list[dict], pick) -> list[list]:
    keys: list[str] = []
    for n in nodes:
        keys += [k for k in pick(n.get("aggregates") or {}) if k not in keys]
    return [[k, *[pick(n.get("aggregates") or {}).get(k) for n in nodes]] for k in keys]


def _groups(aggregates: dict) -> dict:
    """One row per group and dimension."""
    return {f"{axis}: {name} / {dimension}": value
            for axis, groups in (aggregates.get("groups") or {}).items() if axis in GROUP_AXES
            for name, dimensions in groups.items() for dimension, value in dimensions.items()}


def _data(n: dict) -> str:
    return "- Data: " + "; ".join(
        f"{name}: {d.get('clips')} clips, weight {d.get('weight')}"
        + (", from " + ", ".join(f"{s} x{k}" for s, k in d["sources"].items()) if d.get("sources") else "")
        for name, d in n["data"].items())


def data_node(n: dict) -> str:
    """An earlier node for the data roles."""
    lines = [f"### {n['node_id']}: {n.get('status')}, score {_cell(n.get('score'))}"]
    if n["node_id"] == "root":
        lines.append("- The root is the baseline: the model with no further fine-tuning.")
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if n.get("summary"):
        lines.append(f"- Summary: {clip(n['summary'])}")
    if n.get("data"):
        lines.append(_data(n))
        lines.append(f"- Data commit: {n.get('data_commit')}")
    if n.get("recipe"):
        lines.append("- Recipe: " + ", ".join(f"{k} {v}" for k, v in n["recipe"].items()))
    elif n.get("data"):
        lines.append("- Recipe: all values kept at their base values")
    return "\n".join(lines)


# ---- edit_self: code edits and how each run went ----

def status_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    counts = Counter(n.get("status") for n in nodes)
    return f"## Archive\n\n{len(nodes)} nodes: " + ", ".join(f"{status} {n}" for status, n in counts.most_common())


def phase_line(p: dict, node: str) -> str:
    """One phase attempt."""
    convs = ", ".join(f"{c['role']} {c['turns']} turns (/nodes/{node}/{c['transcript']})"
                      for c in p.get("conversations") or []) or "no conversation"
    return (f"- {p['phase']} attempt {p['attempt']}: {convs}; snapshots: {', '.join(p.get('snapshots') or []) or 'none'}; "
            f"result: {p.get('accepted') or 'none'}")


def _process(p: dict, node: str) -> list[str]:
    """How a node's run went."""
    lines = [phase_line(row, node) for row in p.get("phases") or []]
    if p.get("tools"):
        lines.append("- Kernel tool calls (errors): "
                     + ", ".join(f"{tool} {v['calls']} ({v['errors']})" for tool, v in p["tools"].items()))
    jobs = p.get("gpu_jobs") or {}
    if jobs.get("run"):
        lines.append(f"- GPU jobs: {jobs['run']} run, {jobs['failed']} failed")
    ingest = p.get("ingest") or {}
    if ingest.get("accepted") or ingest.get("rejected"):
        reasons = "".join(f"; {r['count']}x {clip(r['reason'], 200)}" for r in ingest.get("reasons") or [])
        lines.append(f"- Ingest: {ingest['accepted']} accepted, {ingest['rejected']} rejected{reasons}")
    failed = [a for a in p.get("attempts") or [] if a["outcome"] != "passed"]
    if failed:
        lines.append("- Failed attempts: " + ", ".join(f"{a['phase']} {a['attempt']} {a['outcome']}" for a in failed))
    if p.get("gates"):
        lines.append("- Pre-training checks: " + ", ".join(f"{k} {v}" for k, v in p["gates"].items()))
    return lines


def edit_node(n: dict) -> str:
    """An earlier node for the edit roles."""
    lines = [f"### {n['node_id']}: {n.get('status')}"]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if n.get("edit"):
        lines.append(f"- Summary: {clip(n['edit'].get('summary', ''))}")
    if n.get("code_diff_stats"):
        lines.append("- Code changed: " + ", ".join(f"{d['path']} (+{d['added']}/-{d['removed']})"
                                                    for d in n["code_diff_stats"]))
    return "\n".join(lines + _process(n.get("process") or {}, n["node_id"]))


# ---- sections shared by both phases ----

def lineage_section(lineage: list[dict], describe, scores: bool) -> str:
    if not lineage:
        return ""
    shown = lineage[-LINEAGE_SHOWN:]
    older = "" if len(shown) == len(lineage) else (
        f" Only the last {len(shown)} are described here; all of them are in /context/context.json and /nodes/.")
    parts = ["## Lineage", f"From the root to the parent: {len(lineage)} nodes, oldest first.{older}"]
    if scores:
        columns = shown if shown[0] is lineage[0] else [lineage[0], *shown]
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


def _node_and_recipe(ctx, scratch: str) -> list[str]:
    """This node's facts for the data roles."""
    guide = ctx.recipe_guide or {}
    keys = [[k, r.get("type"), r.get("min"), r.get("max"), (guide.get(k) or {}).get("base"),
             ctx.parent_recipe.get(k), (guide.get(k) or {}).get("meaning")]
            for k, r in (ctx.tunable_rules or {}).items()]
    pairs = lambda allowed, sep: ", ".join(f"{a}{sep}{b}" for a, b in allowed)
    return [
        POINTER,
        "## This node\n\n" + "\n".join([
            f"- Node id: {ctx.node_id}",
            f"- Training GPUs: {ctx.n_gpus}",
            f"- Parent data commit: {ctx.parent_data_commit or 'none (the parent is the root)'}",
            _pool(ctx)]),
        skill_section(ctx.target_skill),
        folders_section(ctx.folders, scratch),
        "## Recipe\n\nTunable keys, with the base recipe's and the parent's values:\n\n"
        + table(["key", "type", "min", "max", "base", "parent", "meaning"], keys)
        + f"\n\nLoRA (rank/alpha): {pairs(ctx.lora_allowlist, '/')}. The full base recipe is `base_recipe` "
          f"in /context/context.json.",
        metric_section(ctx.metric_guide)]


def _join(sections) -> str:
    return "\n\n".join(s for s in sections if s)


def recipe_planner_context(ctx, scratch: str) -> str:
    return _join([*_node_and_recipe(ctx, scratch), best_section(ctx.archive),
                  siblings_section(ctx.siblings, data_node, scores=True),
                  lineage_section(ctx.lineage, data_node, scores=True)])


def data_engineer_context(ctx, scratch: str) -> str:
    return _join(_node_and_recipe(ctx, scratch))


def recipe_reviewer_context(ctx, scratch: str) -> str:
    return _join(_node_and_recipe(ctx, scratch))


def _this_agent(ctx, components: dict, scratch: str) -> list[str]:
    return [POINTER, f"## This agent\n\n{components_section(components)}", folders_section(ctx.folders, scratch)]


def edit_planner_context(ctx, components: dict, scratch: str) -> str:
    return _join([*_this_agent(ctx, components, scratch), status_section(ctx.archive),
                  siblings_section(ctx.siblings, edit_node, scores=False),
                  lineage_section(ctx.lineage, edit_node, scores=False)])


def coder_context(ctx, components: dict, scratch: str) -> str:
    return _join(_this_agent(ctx, components, scratch))


def edit_reviewer_context(ctx, components: dict, scratch: str) -> str:
    return _join(_this_agent(ctx, components, scratch))
