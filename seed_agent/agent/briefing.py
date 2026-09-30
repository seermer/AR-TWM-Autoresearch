"""What a role is told first: the kernel's context as a Markdown digest. The exact data is in
/context/context.json and each ancestor's files are under /lineage/<node>/; this digest points there
instead of repeating them. Small sections come first and the lineage last, and only the most recent
LINEAGE_SHOWN ancestors are described, so the digest stays about the same size at any depth."""
from __future__ import annotations

import json

LINEAGE_SHOWN = 10          # most recent ancestors described; the root stays in the score tables as the baseline
TEXT_CHARS = 1200           # longest free text quoted per ancestor
TOP_NODES = 5               # best nodes listed from the whole archive


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


def archive_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    best = sorted((n for n in nodes if n.get("score") is not None), key=lambda n: -n["score"])[:TOP_NODES]
    return "\n\n".join([
        "## Archive",
        f"{len(nodes)} nodes, {archive.get('n_scored', len(best))} scored. The best {len(best)}:",
        table(["node", "parent", "depth", "score", "edited component", "status"],
              [[n["node_id"], n.get("parent_id"), n.get("depth"), n.get("score"), n.get("component"), n.get("status")]
               for n in best])])


def _score_rows(nodes: list[dict], pick) -> list[list]:
    keys: list[str] = []
    for n in nodes:
        keys += [k for k in pick(n.get("aggregates") or {}) if k not in keys]
    return [[k, *[pick(n.get("aggregates") or {}).get(k) for n in nodes]] for k in keys]


def _strata(aggregates: dict) -> dict:
    return {f"{group}: {name}": value for group, values in (aggregates.get("strata") or {}).items()
            for name, value in values.items()}


def _process(p: dict) -> str:
    bits = [", ".join(f"{k.removesuffix('_s')} {round(v / 60)} min" for k, v in (p.get("phases") or {}).items())]
    bits += [f"{phase}: {v.get('turns')} LLM turns, {v.get('compactions')} compactions"
             for phase, v in (p.get("llm") or {}).items()]
    errors = p.get("tool_errors") or []
    if errors:
        bits.append("tool errors: " + "; ".join(f"{e['tool']} x{e['count']} ({e['example']})" for e in errors[:3]))
    if p.get("gates"):
        bits.append("gates: " + ", ".join(f"{k} {v}" for k, v in p["gates"].items()))
    return "; ".join(b for b in bits if b)


def _node(n: dict) -> str:
    head = f"### {n['node_id']}: {n.get('status')}, score {_cell(n.get('score'))}"
    if n.get("component"):
        head += f", edited {n['component']}"
    lines = [head]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if n.get("edit"):
        change = str(n["edit"].get("summary", "")).split("\n\n")[0]       # "[component] change"; the rest is the coder's
        lines.append(f"- Edit: {clip(change)}")
    if n.get("code_diff_stats"):
        lines.append("- Code changed: " + ", ".join(f"{d['path']} (+{d['added']}/-{d['removed']})"
                                                    for d in n["code_diff_stats"]))
    if n.get("data"):
        lines.append("- Data: " + "; ".join(
            f"{name} ({d.get('format')}{'/' + d['prompt_mode'] if d.get('prompt_mode') else ''}): "
            f"{d.get('clips')} clips, weight {d.get('weight')}" for name, d in n["data"].items()))
    if n.get("recipe"):
        lines.append("- Recipe: " + ", ".join(f"{k} {v}" for k, v in n["recipe"].items()))
    if n.get("process"):
        lines.append(f"- Process: {_process(n['process'])}")
    if n.get("recipe") or n.get("edit"):                    # the root left no files, only its score
        lines.append(f"- Files: /lineage/{n['node_id']}/ (rationale.md with the plan and data notes, edit.json, "
                     f"transcripts/, attempts/)")
    return "\n".join(lines)


def lineage_section(lineage: list[dict]) -> str:
    if not lineage:
        return ""
    shown = lineage[-LINEAGE_SHOWN:]
    columns = shown if shown[0] is lineage[0] else [lineage[0], *shown]       # the root: the baseline
    ids = [n["node_id"] for n in columns]
    scores = [["score", *[n.get("score") for n in columns]],
              *_score_rows(columns, lambda a: a.get("dimensions") or {})]
    older = "" if len(shown) == len(lineage) else (
        f" Only the last {len(shown)} are described here (and the root, in the tables); all of them are in "
        f"/context/context.json and /lineage/.")
    return "\n\n".join([
        "## Lineage",
        f"From the root to the parent: {len(lineage)} nodes, oldest first.{older}",
        "### Scores\n\n" + table(["", *ids], scores),
        "### Strata\n\n" + table(["", *ids], _score_rows(columns, _strata)),
        "### Metrics\n\n" + table(["", *ids], _score_rows(columns, lambda a: a.get("metrics") or {})),
        *[_node(n) for n in shown]])


def recipe_context(ctx, previous_plans: list | None) -> str:
    guide = ctx.recipe_guide or {}
    keys = [[k, r.get("type"), r.get("min"), r.get("max"), (guide.get(k) or {}).get("base"),
             ctx.parent_recipe.get(k), (guide.get(k) or {}).get("meaning")]
            for k, r in (ctx.tunable_rules or {}).items()]
    pairs = lambda allowed, sep: ", ".join(f"{a}{sep}{b}" for a, b in allowed)
    return "\n\n".join(s for s in [
        "Exact data: /context/context.json. What each ancestor left behind: /lineage/<node>/.",
        "## This node\n\n" + "\n".join([
            f"- Training GPUs: {ctx.n_gpus}",
            f"- Parent data commit: {ctx.parent_data_commit or 'none (the parent is the root)'}",
            f"- Clip pool: {len(ctx.clip_pool)} clips in the archive (data_query lists them)",
            f"- Kernel tools: {', '.join(ctx.tools)}"]),
        retry_section(ctx.retry, previous_plans),
        "## Recipe\n\nTunable keys, with the base recipe's and the parent's values:\n\n"
        + table(["key", "type", "min", "max", "base", "parent", "meaning"], keys)
        + f"\n\nResolutions (height x width): {pairs(ctx.resolution_allowlist, 'x')}. "
          f"LoRA (rank/alpha): {pairs(ctx.lora_allowlist, '/')}. The full base recipe is `base_recipe` "
          f"in /context/context.json.",
        f"## Data formats\n\n{ctx.format_rules.strip()}" if ctx.format_rules else "",
        archive_section(ctx.archive),
        lineage_section(ctx.lineage)] if s)


def edit_context(ctx, components: dict, previous_plans: list | None) -> str:
    return "\n\n".join(s for s in [
        "Exact data: /context/context.json. What each ancestor left behind: /lineage/<node>/.",
        "## This node\n\n" + f"- Nodes left in the run after this one: {ctx.nodes_remaining}\n"
        + "- Components of this agent:\n" + "\n".join(f"  - {k}: {v}" for k, v in components.items()),
        retry_section(ctx.retry, previous_plans),
        archive_section(ctx.archive),
        lineage_section(ctx.lineage)] if s)
