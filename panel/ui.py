"""Gradio UI for the run panel (spec section 5). Every handler takes the Run first and returns
its outputs plus the tab's error text last, so one broken view never takes the panel down."""
from __future__ import annotations

import functools
import html
import json
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from . import media, problems, views

FOLDED = {"system", "tools", "reasoning", "tool_call", "tool_output", "context"}
TOOL_ERROR_MARK = "⚠ Tool error"
# Passed to launch(css=...): the tool-error mark stays red even if the chat's sanitizer drops `style`.
CSS = ".panel-tool-error { color: #cf222e; font-weight: 600; }"


def guarded(n: int):
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args):
            try:
                out = fn(*args)
            except Exception as exc:  # noqa: BLE001 -- shown in the tab; the panel stays up
                return (*[gr.update()] * n, f"**Error:** `{type(exc).__name__}: {exc}`")
            return (*(out if isinstance(out, tuple) else (out,)), "")
        return inner
    return wrap


# Every table has named columns, also when empty (Gradio 6 shows "1 2 3" for a table without any).
COLUMNS = {
    "nodes": ["node", "parent", "status", "score", "vs parent", "component", "attempts", "duration_min", "error"],
    "alerts": ["time", "level", "kind", "message"],
    "events": ["seq", "time", "node", "phase", "attempt", "type", "component", "summary"],
    "metrics": ["metric", "node", "parent", "root"],
    "attempts": ["phase", "attempt", "outcome", "time", "detail"],
    "datasets": ["dataset", "clips"],
    "clips": ["clip_id", "camera_motion", "formats", "frames", "license", "leakage"],
    "ingests": ["time", "phase", "attempt", "video", "caption", "pose", "camera_motion", "outcome", "clip_id",
                "reasons", "host_video"],
    "staging": ["path", "bytes"],
    "gates": ["time", "result", "failures"],
    "gpu": ["gpu", "mean_util", "peak_memory_gib"],
    "cases": ["case", "perspective"],
    "case_scores": ["metric", "this node", "compared node"],
    "selection": ["time", "child", "chosen", "seed", "candidates"],
    "candidates": ["node_id", "score", "subtree_mean", "value", "size", "penalty", "w", "P"],
    "by_phase": ["node", "phase", "calls", "tokens", "usd", "latency_s", "errors"],
    "by_role": ["role (inferred)", "conversations", "calls", "tokens", "usd"],
    "errors": ["time", "node", "phase", "status"],
    "files": ["name", "type", "bytes", "modified"],
    "problems": ["id", "time", "source", "kind", "node", "phase", "attempt", "where", "summary"],
    "problem_counts": ["kind", "source", "last hour", "total"],
}
PROBLEM_KINDS = ["llm_call", "tool_error", "agent_tool", "job_item", "job_failed", "gate_failed", "ingest_rejected",
                 "contract_failed", "subprocess", "sandbox", "error_event", "warning", "alert"]
DYNAMIC = {"cases", "candidates"}          # their columns come from the data (metrics, selection fields)


def df(rows: list[dict] | None, key: str) -> pd.DataFrame:
    columns = None if rows and key in DYNAMIC else COLUMNS[key]
    frame = pd.DataFrame(rows or [], columns=columns)
    for col in frame.columns:
        if frame[col].map(lambda v: isinstance(v, (dict, list))).any():
            frame[col] = frame[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    return frame


def fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text or "")), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def to_messages(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        kind, title, text = it["kind"], it.get("title"), it.get("text") or ""
        if kind == "user":
            out.append({"role": "user", "content": text})
        elif kind == "assistant":
            out.append({"role": "assistant", "content": text})
        elif kind in ("usage", "note"):
            out.append({"role": "assistant", "content": f"*{text}*"})
        elif kind == "compaction_request":
            out.append({"role": "user", "content": f"**{title}**\n\n{text}"})
        elif kind == "compaction_summary":
            out.append({"role": "assistant", "content": f"**{title}**\n\n{text}"})
        elif kind == "tool_error":             # no status: Gradio 6 shows the section open
            rest = (title or "").removeprefix(TOOL_ERROR_MARK)
            red = f'<span class="panel-tool-error" style="color:#cf222e">{TOOL_ERROR_MARK}</span>{html.escape(rest)}'
            out.append({"role": "assistant", "content": fence(text), "metadata": {"title": red}})
        else:
            body = text if kind in ("system", "context") else fence(text)
            # status "done": Gradio 6 starts the section closed (open when status is absent)
            out.append({"role": "assistant", "content": body, "metadata": {"title": title or kind, "status": "done"}})
    return out


def diff_html(text: str | None) -> str:
    if not text:
        return "<p><i>no changes</i></p>"
    lines = []
    for line in text.splitlines():
        colour = ("#1a7f37" if line.startswith("+") and not line.startswith("+++") else
                  "#cf222e" if line.startswith("-") and not line.startswith("---") else
                  "#8250df" if line.startswith("@@") else None)
        escaped = html.escape(line)
        lines.append(f'<span style="color:{colour}">{escaped}</span>' if colour else escaped)
    return '<pre style="white-space:pre-wrap;font-size:12px">' + "\n".join(lines) + "</pre>"


def _abs(run: views.Run, rel: str | None) -> str | None:
    return str(run.files.path(rel)) if rel else None


# ---- handlers ----

@guarded(9)
def h_overview(run):
    ov = views.overview(run)
    loop, spend = ov["loop"], ov["spend"]
    state = loop["state"] or {}
    head = (f"**Loop:** running (pid {loop['pid']})" if loop["alive"] else "**Loop:** stopped") + \
        f" — {loop['state_label']} state: {state.get('node')} · {state.get('phase')} · attempt {state.get('attempt')}"
    usd = "n/a" if spend["usd"] is None else f"${spend['usd']:.4f}"
    head += (f"\n\n**Spend:** {usd} · {spend['tokens']:,} tokens · {spend['calls']} calls · "
             f"**Nodes:** {max(len(ov['nodes']) - 1, 0)} of {loop['max_nodes']} · **Disk free:** {ov['disk_free_gb']} GB"
             + (f" · **Judge:** {ov['judge']}" if ov.get("judge") else ""))
    scores = go.Figure(go.Scatter(x=[s["node"] for s in ov["scores"]], y=[s["score"] for s in ov["scores"]],
                                  mode="lines+markers"))
    scores.update_layout(title="Score by node", height=300, margin=dict(t=40, b=30))
    gpu = go.Figure()
    series = views.gpu_series(run)
    for g in sorted({s["gpu"] for s in series}):
        pts = [s for s in series if s["gpu"] == g]
        gpu.add_trace(go.Scatter(x=[p["time"] for p in pts], y=[p["memory_gib"] for p in pts], name=f"GPU {g}"))
    gpu.update_layout(title="GPU memory (GiB), the run's last 6 h of samples", height=300, margin=dict(t=40, b=30))
    nodes = [n["node"] for n in ov["nodes"]]
    return (head, df(ov["nodes"], "nodes"), scores, gpu, df(ov["alerts"], "alerts"), df(ov["recent"], "events"),
            df(problems.problem_counts(run), "problem_counts"),
            gr.update(choices=nodes), gr.update(choices=nodes))


@guarded(3)
def h_trace(run, node, phase, attempt, types, component, text, include_gpu, page):
    got = views.trace(run, node=node or None, phase=phase or None,
                      attempt=None if attempt in (None, "", "any") else int(attempt), types=types or None,
                      component=component or None, text=text, include_gpu=bool(include_gpu),
                      page=max(0, int(page or 1) - 1))
    return df(got["rows"], "events"), \
        f"{got['total']} events · page {min(int(page or 1), got['pages'])} of {got['pages']}", \
        gr.update(maximum=got["pages"])


@guarded(3)
def h_trace_detail(run, seq):
    d = views.trace_detail(run, int(seq))
    if d is None:
        return None, None, []
    return d["event"], d["payload"], to_messages(d["chat"])


@guarded(9)
def h_node(run, node):
    d = views.node_detail(run, node)
    if d is None:
        return (f"No node {node!r}.", None, "", "", None, df([], "metrics"), df([], "attempts"), None, None)
    n = d["node"]
    head = (f"### {node} · {n['status']} · score {n['score']}\n\n**Lineage:** {d['lineage']}"
            + ("\n\n*Baseline: no parent, edit, recipe, rationale or training.*" if d["baseline"] else "")
            + (f"\n\n**Error:** {n['error']}" if n["error"] else ""))
    return (head, d["edit"], d["rationale"] or "", diff_html(d["recipe_diff"]) if d["recipe_diff"] else "",
            d["data_commit"], df(d["metrics"], "metrics"), df(d["attempts"], "attempts"), d["timings"], dict(n))


@guarded(1)
def h_attempts(run, node):
    keys = [f"{phase}-{attempt}" for phase, attempt in views.agent_attempts(run, node)]
    return gr.update(choices=keys, value=keys[-1] if keys else None)


def _split(key: str) -> tuple[str, int]:
    phase, _, attempt = key.rpartition("-")
    return phase, int(attempt)


@guarded(2)
def h_conversations(run, node, attempt_key):
    if not attempt_key:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    phase, attempt = _split(attempt_key)
    convs = views.conversations(run, node, phase, attempt)
    labels = [f"{c['index']}: {c['role']} (inferred) · {c['calls']} calls · {c['compactions']} compactions · "
              f"{c['tokens']:,} tokens" for c in convs]
    logs = views.tool_logs(run, node, phase, attempt)
    return (gr.update(choices=labels, value=labels[0] if labels else None),
            gr.update(choices=logs, value=None))


@guarded(1)
def h_chat(run, node, attempt_key, conv_label):
    if not attempt_key or not conv_label:
        return []
    phase, attempt = _split(attempt_key)
    return to_messages(views.conversation_chat(run, node, phase, attempt, int(conv_label.split(":", 1)[0])))


@guarded(1)
def h_tool_log(run, rel):
    return run.files.read_text(rel) if rel else ""


def _commit_choices(run, node, d) -> list[tuple[str, str]]:
    """Every agent version worth browsing for this node: its own, its parent's, each edit attempt's."""
    n = run.node(node)
    parent = run.node(n["parent_id"]) if n and n["parent_id"] else None
    out = []
    if d["agent_commit"]:
        out.append((f"{node} · this node" if parent else f"{node} · seed agent", d["agent_commit"]))
    if parent and parent["agent_commit"]:
        out.append((f"{parent['node_id']} · parent", parent["agent_commit"]))
    out += [(f"edit_self attempt {a['attempt']} · {a['outcome']} · {a['commit'][:8]}", a["commit"])
            for a in d["attempts"] if a["commit"]]
    return out


@guarded(5)
def h_code(run, node):
    d = views.code_edits(run, node)
    commits = gr.update(choices=_commit_choices(run, node, d), value=d["agent_commit"])
    tree = gr.update(choices=views.agent_tree(run, d["agent_commit"]) if d["agent_commit"] else [], value=None)
    if d["baseline"]:
        return "*Baseline: the root runs the seed agent unedited.*", "", gr.update(choices=[], value=None), \
            commits, tree
    choices = [a["attempt"] for a in d["attempts"]]
    return ("**Node diff** (parent's agent code → this node's)", diff_html(d["node_diff"]),
            gr.update(choices=choices, value=choices[-1] if choices else None), commits, tree)


@guarded(2)
def h_commit_tree(run, commit):
    return gr.update(choices=views.agent_tree(run, commit) if commit else [], value=None), ""


@guarded(3)
def h_code_attempt(run, node, attempt):
    a = next((a for a in views.code_edits(run, node)["attempts"] if a["attempt"] == attempt), None)
    if a is None:
        return "", "", None
    head = f"**Attempt {attempt}:** {a['outcome']}" + (f" · commit `{a['commit'][:12]}`" if a["commit"] else
                                                       " · no commit (checkout or commit failed)")
    if a["detail"]:
        head += f"\n\n**Detail:** `{a['detail']}`"
    contract = a["contract"] if a["contract"] is not None else {
        "note": "no contract check: the attempt failed before it"}
    return head, diff_html(a["diff"]) if a["commit"] else "", contract


@guarded(1)
def h_agent_file(run, commit, path):
    return (views.agent_file(run, commit, path) or "") if commit and path else ""


@guarded(9)
def h_training_data(run, node):
    d = media.training_data(run, node)
    names = [x["dataset"] for x in d["datasets"]]
    head = (f"**Data commit** `{d['commit']['commit_id'][:16]}`: {d['commit']['message']}" if d["commit"]
            else "*No data commit (baseline, or not built yet).*")
    return (head, df(d["datasets"], "datasets"), gr.update(choices=names, value=names[0] if names else None),
            df(d["ingests"], "ingests"), df(d["staging"], "staging"),
            None, None, None, None)                  # the previously opened clip: video, caption, plot, record


@guarded(3)
def h_clips(run, node, dataset, page):
    if not dataset:
        return df([], "clips"), "", gr.update()
    got = media.clips(run, node, dataset, page=max(0, int(page or 1) - 1))
    rows = [{"clip_id": r["clip_id"], "camera_motion": r["camera_motion"], "formats": r["formats"],
             "frames": r["metadata"].get("frames"), "license": r["license"],
             "leakage": "none" if not r["leakage"] else ("inferred " if r["leakage"]["inferred"] else "") +
             f"{len(r['leakage']['matches'] or [])} matches / {len(r['leakage']['near'] or [])} near"}
            for r in got["rows"]]
    return df(rows, "clips"), f"{got['total']} clips · page {min(int(page or 1), got['pages'])} of {got['pages']}", \
        gr.update(maximum=got["pages"])


@guarded(4)
def h_clip(run, clip_id):
    c = media.clip_detail(run, clip_id)
    if c is None:
        return None, None, None, None
    fig = None
    if c["pose"]:
        path = media.camera_path(run, c["pose"])
        fig = go.Figure(go.Scatter(x=path["x"], y=path["z"], mode="lines+markers"))
        fig.update_layout(title="Camera path, top-down (x vs z)", height=350, yaxis_scaleanchor="x")
    info = {k: c[k] for k in ("clip_id", "camera_motion", "metadata", "formats", "warnings", "provenance",
                              "license", "derived_from", "ingested_by", "leakage")}
    return _abs(run, c["video"]), c["caption"], fig, info


@guarded(1)
def h_training_attempts(run, node):
    choices = views.training_attempts(run, node)
    return gr.update(choices=choices, value=choices[-1] if choices else None)


@guarded(5)
def h_training(run, node, attempt):
    if attempt is None:
        return "*No improve_recipe attempt (baseline, or not started).*", None, "", df([], "gates"), df([], "gpu")
    t = views.training(run, node, int(attempt))
    head = (f"Training started {t['started']}" + (f", {t['duration_min']} min" if t["duration_min"] else "")
            if t["reached_training"] else "*This attempt never reached training.*")
    if t["outcome"]:
        head += f"\n\n**Attempt outcome:** {t['outcome']['outcome']} · `{t['outcome']['detail']}`"
    fig = None
    if t["loss"]:
        fig = go.Figure(go.Scatter(x=[p["step"] for p in t["loss"]], y=[p["loss"] for p in t["loss"]],
                                   mode="lines", name="loss"))
        fig.update_layout(title="Training loss", height=350)
    return head, fig, t["config"] or "", df(t["gates"], "gates"), df(t["gpu"], "gpu")


@guarded(3)
def h_eval(run, node, other):
    ev = media.eval_view(run, node)
    rows = [{"case": c["case"], "perspective": c["perspective"], **c["scores"]} for c in ev["cases"]]
    cases = [c["case"] for c in ev["cases"]]
    return df(rows, "cases"), gr.update(choices=cases, value=cases[0] if cases else None), ev["aggregates"]


@guarded(5)
def h_eval_case(run, node, other, case):
    mine = next((c for c in media.eval_view(run, node)["cases"] if c["case"] == case), None)
    theirs = next((c for c in media.eval_view(run, other)["cases"] if c["case"] == case), None) if other else None
    if mine is None:
        return None, None, df([], "case_scores"), None, None
    metrics = sorted(set(mine["scores"]) | set((theirs or {}).get("scores", {})))
    table = [{"metric": m, "this node": mine["scores"].get(m),
              "compared node": (theirs or {}).get("scores", {}).get(m)} for m in metrics]
    return (_abs(run, mine["video"]), _abs(run, theirs["video"]) if theirs else None, df(table, "case_scores"),
            {"perspective": mine["perspective"], "actions": mine["actions"]}, mine["prompt_schedule"])


@guarded(2)
def h_selection(run):
    rows = views.selection(run)
    return df([{k: v for k, v in r.items() if k != "candidates"} | {"candidates": len(r["candidates"])}
               for r in rows], "selection"), rows


@guarded(1)
def h_selection_row(run, rows, index):
    return df(rows[index]["candidates"] if rows and 0 <= index < len(rows) else [], "candidates")


@guarded(3)
def h_cost(run):
    c = views.cost(run)
    return df(c["by_phase"], "by_phase"), df(c["by_role"], "by_role"), df(c["errors"], "errors")


@guarded(2)
def h_files(run, rel):
    return df(media.list_dir(run, rel or ""), "files"), rel or ""


@guarded(5)
def h_preview(run, rel):
    p = media.preview(run, rel)
    meta = {k: v for k, v in p.items() if k not in ("text",)}
    text = p.get("text") if p["kind"] == "text" else ""
    video = _abs(run, rel) if p["kind"] == "video" else None
    image = _abs(run, rel) if p["kind"] == "image" else None
    return meta, text, video, image, p["kind"]


@guarded(3)
def h_problems(run, kinds, source):
    rows = [p for p in problems.problems(run)
            if (not kinds or p["kind"] in kinds) and source in (None, "", "any", p["source"])]
    rows = [dict(p, id=i) for i, p in enumerate(rows)]
    return df(rows, "problems"), rows, f"{len(rows)} problems · newest first · inferred = read from a chat's tool output"


@guarded(3)
def h_problem_detail(run, rows, index):
    if not rows or not 0 <= index < len(rows):
        return None, None, ""
    p = rows[index]
    if p["seq"] is None:
        return None, None, p.get("detail") or p["summary"]
    d = views.trace_detail(run, p["seq"])
    return (d["event"], d["payload"], p.get("detail") or "") if d else (None, None, p["summary"])


# ---- row selection: by the selected row's own values (a sorted table's index is not the frame's) ----

def select_trace(run, evt):
    return h_trace_detail(run, int(evt.row_value[0]))


def select_problem(run, rows, evt):
    return h_problem_detail(run, rows, int(evt.row_value[0]))


def select_clip(run, evt):
    return h_clip(run, evt.row_value[0])


def select_selection(run, rows, evt):
    time_, child = evt.row_value[0], evt.row_value[1]
    index = next((i for i, r in enumerate(rows or []) if (r["time"], r["child"]) == (time_, child)), -1)
    return h_selection_row(run, rows, index)


def select_file(run, folder, evt):
    name, kind = evt.row_value[0], evt.row_value[1]
    rel = f"{folder}/{name}" if folder else name
    if kind == "dir":
        listed = h_files(run, rel)
        return listed[0], rel, None, "", None, None, "dir", listed[-1]
    meta, text, video, image, kind, err = h_preview(run, rel)
    return gr.update(), folder, meta, text, video, image, kind, err


# ---- layout ----

def build_app(run_dir) -> gr.Blocks:
    run = views.Run(run_dir)
    # Run files are served where they are, never copied into Gradio's cache (disk is the binding
    # constraint); they stay behind the login like every other file route.
    gr.set_static_paths([str(run.files.root)])

    def bind(fn):
        bound = functools.partial(fn, run)
        bound.__name__ = fn.__name__           # for Gradio's function names; the signature stays the partial's
        return bound
    nodes = [n["node_id"] for n in run.nodes()]
    with gr.Blocks(title=f"Run {run.name}", fill_width=True, delete_cache=(3600, 3600)) as app:
        gr.Markdown(f"## Run `{run.name}` — read-only panel")
        node = gr.Dropdown(choices=nodes, value=nodes[0] if nodes else None, label="Node (shared by node tabs)",
                           allow_custom_value=True)
        with gr.Tabs():
            with gr.Tab("Overview"):
                ov_head = gr.Markdown()
                ov_nodes = gr.Dataframe(headers=COLUMNS["nodes"], label="Nodes", wrap=True, interactive=False)
                with gr.Row():
                    ov_scores, ov_gpu = gr.Plot(), gr.Plot()
                ov_alerts = gr.Dataframe(headers=COLUMNS["alerts"], label="Alerts (newest first)", wrap=True, interactive=False)
                ov_problems = gr.Dataframe(headers=COLUMNS["problem_counts"], interactive=False,
                                           label="Problems: the run's last hour / total (details in the Problems tab)")
                ov_recent = gr.Dataframe(headers=COLUMNS["events"], label="Last 50 events (newest first)", wrap=True, interactive=False)
                ov_err = gr.Markdown()
            with gr.Tab("Problems"):
                with gr.Row():
                    pr_kind = gr.Dropdown(PROBLEM_KINDS, multiselect=True, label="Kinds (none = all)")
                    pr_source = gr.Dropdown(["any", "recorded", "inferred"], value="any", label="Source")
                    pr_load = gr.Button("Load problems")
                pr_count = gr.Markdown()
                pr_rows = gr.Dataframe(headers=COLUMNS["problems"], interactive=False, wrap=True,
                                       label="Every failure (select one)")
                pr_state = gr.State([])
                with gr.Row():
                    pr_event, pr_payload = gr.JSON(label="Event"), gr.JSON(label="Payload")
                pr_text = gr.Code(label="Tool output (inferred problems)", lines=15)
                pr_err = gr.Markdown()
            with gr.Tab("Node"):
                nd_load = gr.Button("Load node")
                nd_head = gr.Markdown()
                with gr.Row():
                    nd_edit, nd_commit = gr.JSON(label="edit.json"), gr.JSON(label="Data commit")
                nd_rationale = gr.Markdown(label="rationale.md")
                nd_recipe = gr.HTML(label="Recipe vs parent")
                nd_metrics = gr.Dataframe(headers=COLUMNS["metrics"], label="Metrics: node, parent, root", interactive=False)
                nd_attempts = gr.Dataframe(headers=COLUMNS["attempts"], label="Attempts", wrap=True, interactive=False)
                with gr.Row():
                    nd_timings, nd_row = gr.JSON(label="Phase timings"), gr.JSON(label="Archive row")
                nd_err = gr.Markdown()
            with gr.Tab("Trace"):
                choices = views.trace_choices(run)
                with gr.Row():
                    tr_node = gr.Dropdown([""] + choices["node"], value="", label="Event source (a node, run or gpu)",
                                          allow_custom_value=True)
                    tr_phase = gr.Dropdown([""] + choices["phase"], value="", label="Phase", allow_custom_value=True)
                    tr_attempt = gr.Dropdown(["any"] + choices["attempt"], value="any",
                                             label="Attempt (0 = a kernel step, not an agent attempt)")
                    tr_types = gr.Dropdown(choices["type"], multiselect=True, label="Event types",
                                           allow_custom_value=True)
                    tr_comp = gr.Dropdown([""] + choices["component"], value="", label="Component",
                                          allow_custom_value=True)
                with gr.Row():
                    tr_text = gr.Textbox(label="Search the event records")
                    tr_gpu = gr.Checkbox(label="GPU samples", value=False)
                    tr_page = gr.Number(value=1, precision=0, minimum=1, label="Page")
                    tr_go = gr.Button("Search")
                tr_count = gr.Markdown()
                tr_rows = gr.Dataframe(headers=COLUMNS["events"], interactive=False, wrap=True)
                with gr.Row():
                    tr_event, tr_payload = gr.JSON(label="Event"), gr.JSON(label="Payload")
                tr_chat = gr.Chatbot(label="As a chat", height=600)
                tr_err = gr.Markdown()
            with gr.Tab("Conversations"):
                with gr.Row():
                    cv_attempt = gr.Dropdown(label="Phase-attempt")
                    cv_conv = gr.Dropdown(label="Conversation (role labels are inferred)")
                cv_chat = gr.Chatbot(height=900, label="Conversation")
                cv_log = gr.Dropdown(label="Full run_command outputs (tool_output/)")
                cv_log_text = gr.Code(label="Output", lines=20)
                cv_err = gr.Markdown()
            with gr.Tab("Code edits"):
                ce_load = gr.Button("Load code edits")
                ce_head, ce_diff = gr.Markdown(), gr.HTML()
                ce_attempt = gr.Dropdown(label="edit_self attempt")
                ce_att_head, ce_att_diff = gr.Markdown(), gr.HTML()
                ce_contract = gr.JSON(label="Contract report")
                with gr.Row():
                    ce_commit = gr.Dropdown(label="Agent commit")
                    ce_path = gr.Dropdown(label="File", allow_custom_value=True)
                ce_file = gr.Code(label="File content", lines=25)
                ce_err = gr.Markdown()
            with gr.Tab("Training data"):
                td_load = gr.Button("Load training data")
                td_head = gr.Markdown()
                td_sets = gr.Dataframe(headers=COLUMNS["datasets"], label="Datasets", interactive=False)
                with gr.Row():
                    td_dataset = gr.Dropdown(label="Dataset")
                    td_page = gr.Number(value=1, precision=0, minimum=1, label="Page")
                td_count = gr.Markdown()
                td_clips = gr.Dataframe(headers=COLUMNS["clips"], label="Clips (select one)", interactive=False, wrap=True)
                with gr.Row():
                    td_video = gr.Video(label="Clip")
                    td_plot = gr.Plot()
                with gr.Row():
                    td_caption, td_info = gr.JSON(label="Caption"), gr.JSON(label="Clip record")
                td_ingest = gr.Dataframe(headers=COLUMNS["ingests"], label="Every ingest candidate", wrap=True, interactive=False)
                td_staging = gr.Dataframe(headers=COLUMNS["staging"], label="Kept staging files", interactive=False)
                td_err = gr.Markdown()
            with gr.Tab("Training"):
                tn_attempt = gr.Dropdown(label="improve_recipe attempt")
                tn_head = gr.Markdown()
                tn_loss = gr.Plot()
                tn_config = gr.Code(label="Resolved train config", language="yaml", lines=20)
                tn_gates = gr.Dataframe(headers=COLUMNS["gates"], label="Gate checks", wrap=True, interactive=False)
                tn_gpu = gr.Dataframe(headers=COLUMNS["gpu"], label="GPU during training", interactive=False)
                tn_err = gr.Markdown()
            with gr.Tab("Eval"):
                with gr.Row():
                    ev_load = gr.Button("Load eval")
                    ev_other = gr.Dropdown(choices=nodes, value="root" if "root" in nodes else None,
                                           label="Compare with", allow_custom_value=True)
                ev_cases = gr.Dataframe(headers=COLUMNS["cases"], label="Per-case scores", interactive=False)
                ev_case = gr.Dropdown(label="Case")
                with gr.Row():
                    ev_video, ev_video2 = gr.Video(label="This node"), gr.Video(label="Compared node")
                ev_table = gr.Dataframe(headers=COLUMNS["case_scores"], label="Case scores", interactive=False)
                with gr.Row():
                    ev_meta, ev_prompt = gr.JSON(label="Case"), gr.JSON(label="Prompt schedule")
                ev_agg = gr.JSON(label="aggregates.json")
                ev_err = gr.Markdown()
            with gr.Tab("Selection"):
                se_load = gr.Button("Load selection history")
                se_rows = gr.Dataframe(headers=COLUMNS["selection"], label="Parent picks (select one)", interactive=False)
                se_state = gr.State([])
                se_cands = gr.Dataframe(headers=COLUMNS["candidates"], label="Candidates at that pick", interactive=False)
                se_err = gr.Markdown()
            with gr.Tab("Cost"):
                co_load = gr.Button("Compute")
                co_phase = gr.Dataframe(headers=COLUMNS["by_phase"], label="By node and phase", interactive=False)
                co_role = gr.Dataframe(headers=COLUMNS["by_role"], label="By role (inferred)", interactive=False)
                co_errs = gr.Dataframe(headers=COLUMNS["errors"], label="Non-200 responses", interactive=False)
                co_err = gr.Markdown()
            with gr.Tab("Files"):
                with gr.Row():
                    fi_path = gr.Textbox(value="", label="Folder (run-relative)")
                    fi_open, fi_up = gr.Button("Open"), gr.Button("Up")
                fi_list = gr.Dataframe(headers=COLUMNS["files"], label="Entries (select one)", interactive=False)
                fi_meta = gr.JSON(label="File")
                fi_text = gr.Code(label="Text", lines=30)
                with gr.Row():
                    fi_video, fi_image = gr.Video(label="Video"), gr.Image(label="Image")
                fi_kind = gr.Textbox(visible=False)
                fi_err = gr.Markdown()

        # overview: on load and every 15 s
        ov_out = [ov_head, ov_nodes, ov_scores, ov_gpu, ov_alerts, ov_recent, ov_problems, node, ev_other, ov_err]
        app.load(bind(h_overview), None, ov_out)
        gr.Timer(15).tick(bind(h_overview), None, ov_out)

        nd_out = [nd_head, nd_edit, nd_rationale, nd_recipe, nd_commit, nd_metrics, nd_attempts, nd_timings,
                  nd_row, nd_err]
        nd_load.click(bind(h_node), node, nd_out)
        pr_out = [pr_rows, pr_state, pr_count, pr_err]
        app.load(bind(h_problems), [pr_kind, pr_source], pr_out)
        pr_load.click(bind(h_problems), [pr_kind, pr_source], pr_out)

        def on_problem_select(rows: list, evt: gr.SelectData):
            return select_problem(run, rows, evt)
        pr_rows.select(on_problem_select, pr_state, [pr_event, pr_payload, pr_text, pr_err])
        app.load(bind(h_files), fi_path, [fi_list, fi_path, fi_err])

        tr_in = [tr_node, tr_phase, tr_attempt, tr_types, tr_comp, tr_text, tr_gpu, tr_page]
        tr_go.click(bind(h_trace), tr_in, [tr_rows, tr_count, tr_page, tr_err])

        def on_trace_select(evt: gr.SelectData):
            return select_trace(run, evt)
        tr_rows.select(on_trace_select, None, [tr_event, tr_payload, tr_chat, tr_err])

        # Gradio 6 fires .change only when a value differs, so a step that sets a dropdown to the
        # same value as before would stop a .change cascade and leave the previous node's data up.
        # Programmatic steps are chained with .then; user picks listen on .input.
        conv_out, chat_out = [cv_conv, cv_log, cv_err], [cv_chat, cv_err]
        ce_out = [ce_head, ce_diff, ce_attempt, ce_commit, ce_path, ce_err]
        ce_att_out = [ce_att_head, ce_att_diff, ce_contract, ce_err]
        td_out = [td_head, td_sets, td_dataset, td_ingest, td_staging, td_video, td_caption, td_plot, td_info, td_err]
        clips_out = [td_clips, td_count, td_page, td_err]
        node_eval_in, ev_out = [node, ev_other], [ev_cases, ev_case, ev_agg, ev_err]
        ev_case_out = [ev_video, ev_video2, ev_table, ev_meta, ev_prompt, ev_err]
        tn_out = [tn_head, tn_loss, tn_config, tn_gates, tn_gpu, tn_err]
        for trigger in (app.load, node.change):
            trigger(bind(h_attempts), node, [cv_attempt, cv_err]) \
                .then(bind(h_conversations), [node, cv_attempt], conv_out) \
                .then(bind(h_chat), [node, cv_attempt, cv_conv], chat_out)
            trigger(bind(h_training_attempts), node, [tn_attempt, tn_err]) \
                .then(bind(h_training), [node, tn_attempt], tn_out)
            trigger(bind(h_node), node, nd_out)
            trigger(bind(h_code), node, ce_out).then(bind(h_code_attempt), [node, ce_attempt], ce_att_out)
            trigger(bind(h_training_data), node, td_out).then(bind(h_clips), [node, td_dataset, td_page], clips_out)
            trigger(bind(h_eval), node_eval_in, ev_out).then(bind(h_eval_case), [node, ev_other, ev_case], ev_case_out)
        cv_attempt.input(bind(h_conversations), [node, cv_attempt], conv_out) \
            .then(bind(h_chat), [node, cv_attempt, cv_conv], chat_out)
        cv_conv.input(bind(h_chat), [node, cv_attempt, cv_conv], chat_out)
        cv_log.input(bind(h_tool_log), cv_log, [cv_log_text, cv_err])
        tn_attempt.input(bind(h_training), [node, tn_attempt], tn_out)

        ce_load.click(bind(h_code), node, ce_out).then(bind(h_code_attempt), [node, ce_attempt], ce_att_out)
        ce_attempt.input(bind(h_code_attempt), [node, ce_attempt], ce_att_out)
        ce_path.input(bind(h_agent_file), [ce_commit, ce_path], [ce_file, ce_err])
        ce_commit.input(bind(h_commit_tree), ce_commit, [ce_path, ce_file, ce_err])

        td_load.click(bind(h_training_data), node, td_out).then(bind(h_clips), [node, td_dataset, td_page], clips_out)
        td_dataset.input(bind(h_clips), [node, td_dataset, td_page], clips_out)
        td_page.submit(bind(h_clips), [node, td_dataset, td_page], clips_out)

        def on_clip_select(evt: gr.SelectData):
            return select_clip(run, evt)
        td_clips.select(on_clip_select, None, [td_video, td_caption, td_plot, td_info, td_err])

        ev_load.click(bind(h_eval), node_eval_in, ev_out).then(bind(h_eval_case), [node, ev_other, ev_case], ev_case_out)
        ev_case.input(bind(h_eval_case), [node, ev_other, ev_case], ev_case_out)
        ev_other.input(bind(h_eval_case), [node, ev_other, ev_case], ev_case_out)

        se_load.click(bind(h_selection), None, [se_rows, se_state, se_err])

        def on_selection_select(rows: list, evt: gr.SelectData):
            return select_selection(run, rows, evt)
        se_rows.select(on_selection_select, se_state, [se_cands, se_err])

        co_load.click(bind(h_cost), None, [co_phase, co_role, co_errs, co_err])

        fi_open.click(bind(h_files), fi_path, [fi_list, fi_path, fi_err])
        fi_up.click(lambda p: h_files(run, str(Path(p or ".").parent) if p else ""), fi_path,
                    [fi_list, fi_path, fi_err])

        def on_file_select(folder: str, evt: gr.SelectData):
            return select_file(run, folder, evt)
        fi_list.select(on_file_select, fi_path,
                       [fi_list, fi_path, fi_meta, fi_text, fi_video, fi_image, fi_kind, fi_err])
    return app
