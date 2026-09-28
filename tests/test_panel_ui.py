"""Runs in the panel env: .envs/panel/bin/python -m pytest tests/test_panel_ui.py"""
import os
from pathlib import Path

import pytest

# Gradio's file cache stays inside the project (.cache/ is git-ignored), never in /tmp.
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "panel-tests"))
gr = pytest.importorskip("gradio")

from fixtures.panel_run import make_run  # noqa: E402
from panel import ui, views  # noqa: E402


@pytest.fixture
def run(tmp_path):
    return views.Run(make_run(tmp_path))


def _ok(out):
    assert out[-1] == "", out[-1]
    return out


def test_every_handler_runs_on_the_fake_run(run, tmp_path):
    assert isinstance(ui.build_app(run.files.root), gr.Blocks)
    _ok(ui.h_overview(run))
    rows = _ok(ui.h_trace(run, "n1", "", None, ["llm.request"], "", "", False, 1))[0]
    _ok(ui.h_trace_detail(run, int(rows.iloc[0]["seq"])))
    _ok(ui.h_node(run, "n1")), _ok(ui.h_node(run, "root"))
    attempts = _ok(ui.h_attempts(run, "n1"))[0]["choices"]
    key = next(c for c in (c[0] if isinstance(c, tuple) else c for c in attempts) if c.startswith("improve_recipe"))
    convs = _ok(ui.h_conversations(run, "n1", key))[0]["choices"]
    conv = convs[-1][0] if isinstance(convs[-1], tuple) else convs[-1]
    messages = _ok(ui.h_chat(run, "n1", key, conv))[0]
    assert any("Compaction #1" in (m.get("metadata") or {}).get("title", "") or "Compaction #1" in m["content"]
               for m in messages)
    _ok(ui.h_tool_log(run, "nodes/n1/attempts/improve_recipe-1/workspace/tool_output/run_command-20260927-120000-0001.log"))
    _ok(ui.h_code(run, "n1")), _ok(ui.h_code_attempt(run, "n1", 1)), _ok(ui.h_code(run, "root"))
    _ok(ui.h_training_data(run, "n1")), _ok(ui.h_clips(run, "n1", "ds1", 1)), _ok(ui.h_clip(run, "clip1"))
    _ok(ui.h_training(run, "n1", 1)), _ok(ui.h_eval(run, "n1", "root")), _ok(ui.h_eval_case(run, "n1", "root", "7"))
    _ok(ui.h_selection(run)), _ok(ui.h_cost(run)), _ok(ui.h_files(run, "")), _ok(ui.h_preview(run, "config/run.json"))


def test_a_broken_view_reports_in_its_tab(run, monkeypatch):
    monkeypatch.setattr(views, "node_detail", lambda *a: 1 / 0)
    out = ui.h_node(run, "n1")
    assert "ZeroDivisionError" in out[-1]


def test_chat_items_become_chatbot_messages():
    msgs = ui.to_messages([
        {"kind": "system", "title": "System prompt", "text": "sys"},
        {"kind": "user", "title": None, "text": "hi"},
        {"kind": "compaction_request", "title": "Compaction #1: ~9 → ~2 tokens", "text": "summarize"},
        {"kind": "compaction_summary", "title": "Compaction #1 summary", "text": "sum"},
        {"kind": "tool_output", "title": "Tool output: x", "text": "```inner```"},
    ])
    assert msgs[0]["metadata"] == {"title": "System prompt", "status": "done"}      # folded (closed) in Gradio 6
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2]["role"] == "user" and "Compaction #1" in msgs[2]["content"]
    assert msgs[3]["role"] == "assistant" and "Compaction #1 summary" in msgs[3]["content"]
    assert msgs[4]["content"].startswith("````")                 # fence longer than any inside


def test_diff_html_colours_lines():
    html = ui.diff_html("--- a\n+++ b\n@@ -1 +1 @@\n-old <x>\n+new\n")
    assert "&lt;x&gt;" in html and "#cf222e" in html and "#1a7f37" in html
    assert "no changes" in ui.diff_html("")


def test_launcher_refuses_without_login(tmp_path, monkeypatch, capsys):
    from panel import __main__ as launcher
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    make_run(tmp_path)
    (tmp_path / ".env").write_text("PANEL_USER=\nPANEL_PASSWORD=\n")
    monkeypatch.delenv("PANEL_USER", raising=False)
    monkeypatch.delenv("PANEL_PASSWORD", raising=False)
    assert launcher.main(["--run-id", "r1"]) == 2 and "PANEL_USER" in capsys.readouterr().err
    (tmp_path / ".env").write_text("PANEL_USER=u\nPANEL_PASSWORD=p\n")
    assert launcher.main(["--run-id", "nope"]) == 2 and "no run" in capsys.readouterr().err


# ---- final review fixes ----

def _by_label(app, label):
    return next(b for b in app.blocks.values() if getattr(b, "label", None) == label)


def _chains(app, component, event):
    """Function names run, in order, for each listener of `event` on `component` (following .then)."""
    deps = app.config["dependencies"]
    out = []
    for start in [d for d in deps if (component._id, event) in [tuple(t) for t in d["targets"]]]:
        names, cur = [app.fns[start["id"]].name], start
        while nxt := [d for d in deps if d.get("trigger_after") == cur["id"]]:
            cur = nxt[0]
            names.append(app.fns[cur["id"]].name)
        out.append(names)
    return out


def test_node_level_tabs_follow_the_node_even_when_a_default_repeats(run):
    """Gradio 6 fires .change only when a value differs, so a cascade may not rely on it: a new
    node whose default attempt key equals the old one would keep the old node's data."""
    app = ui.build_app(run.files.root)
    node = _by_label(app, "Node (shared by node tabs)")
    chains = _chains(app, node, "change")
    assert ["h_attempts", "h_conversations", "h_chat"] in chains
    assert ["h_training_attempts", "h_training"] in chains
    button = lambda text: next(b for b in app.blocks.values() if getattr(b, "value", None) == text)
    assert _chains(app, button("Load code edits"), "click") == [["h_code", "h_code_attempt"]]
    assert _chains(app, button("Load eval"), "click") == [["h_eval", "h_eval_case"]]
    assert _chains(app, button("Load training data"), "click") == [["h_training_data", "h_clips"]]
    assert _chains(app, _by_label(app, "Compare with"), "input") == [["h_eval_case"]]
    for label in ("Phase-attempt", "Conversation (role labels are inferred)", "improve_recipe attempt",
                  "edit_self attempt", "Case", "Dataset", "Full run_command outputs (tool_output/)", "File",
                  "Compare with"):
        assert _chains(app, _by_label(app, label), "change") == [], label        # user changes use .input
        assert _chains(app, _by_label(app, label), "input"), label


def test_row_selection_uses_the_selected_row_values(run):
    """A sorted table's evt.index need not be the frame's row: select by the row's own values."""
    from types import SimpleNamespace
    seq = next(e for e in run.log.events() if e["type"] == "llm.response")["_seq"]
    out = ui.select_trace(run, SimpleNamespace(index=[0, 0], row_value=[seq, "t", "n1"]))
    assert out[0]["type"] == "llm.response" and out[-1] == ""
    out = ui.select_clip(run, SimpleNamespace(index=[7, 0], row_value=["clip1", "moving"]))
    assert out[0].endswith("store/blobs/video/vd1.mp4") and out[-1] == ""
    rows = views.selection(run)
    out = ui.select_selection(run, rows, SimpleNamespace(index=[9, 0], row_value=[rows[0]["time"], "n1", "root"]))
    assert list(out[0]["node_id"]) == ["root"]
    out = ui.select_file(run, "config", SimpleNamespace(index=[5, 0], row_value=["run.json", "file", 10, "t"]))
    assert out[1] == "config" and out[2]["kind"] == "text" and '"run_id"' in out[3]


def test_run_files_are_served_in_place_not_copied(run):
    ui.build_app(run.files.root)
    video = str(run.files.path("nodes/n1/eval/work_dirs/ar_r1_nn1/videos/case_7_combined.mp4"))
    from gradio import processing_utils
    block = gr.Video()
    served = processing_utils.move_files_to_cache(block.postprocess(video), block, postprocess=True)
    assert (served if isinstance(served, dict) else served.model_dump())["path"] == video


# ---- owner's first look (2026-09-28) ----

def test_trace_attempt_defaults_to_any(run):
    app = ui.build_app(run.files.root)
    attempt = _by_label(app, "Attempt (0 = a kernel step, not an agent attempt)")
    assert isinstance(attempt, gr.Dropdown) and attempt.value == "any"
    rows = ui.h_trace(run, "n1", "", "any", ["llm.request"], "", "", False, 1)
    assert "7 events" in rows[1]


def test_every_table_has_named_columns_even_when_empty(run):
    app = ui.build_app(run.files.root)
    for block in app.blocks.values():
        if isinstance(block, gr.Dataframe):
            assert block.value["headers"] != ["1", "2", "3"], block.label
    assert list(ui.h_node(run, "root")[6].columns) == ["phase", "attempt", "outcome", "time", "detail"]
    assert list(ui.h_selection_row(run, [], -1)[0].columns)[:1] == ["node_id"]


def test_switching_node_reloads_every_node_tab_and_clears_the_open_clip(run):
    app = ui.build_app(run.files.root)
    chains = _chains(app, _by_label(app, "Node (shared by node tabs)"), "change")
    for chain in (["h_node"], ["h_code", "h_code_attempt"], ["h_training_data", "h_clips"], ["h_eval", "h_eval_case"]):
        assert chain in chains, chain
    out = ui.h_training_data(run, "root")
    assert out[-5:-1] == (None, None, None, None)              # clip video, caption, plot, record cleared


def test_files_tab_lists_the_run_folder_on_load(run):
    app = ui.build_app(run.files.root)
    loads = [app.fns[d["id"]].name for d in app.config["dependencies"]
             if any(t[1] == "load" for t in d["targets"])]
    assert "h_files" in loads


def test_agent_commit_is_a_labelled_choice(run):
    out = ui.h_code(run, "n1")
    labels = [c[0] for c in out[3]["choices"]]
    assert labels[0] == "n1 · this node" and "root · parent" in labels
    assert any(label.startswith("edit_self attempt 1 · contract_failed") for label in labels)
    app = ui.build_app(run.files.root)
    assert isinstance(_by_label(app, "Agent commit"), gr.Dropdown)
    assert _chains(app, _by_label(app, "Agent commit"), "input") == [["h_commit_tree"]]


# ---- problems view (2026-09-28) ----

def test_problems_tab_and_overview_counts(tmp_path):
    from test_panel_data import _add_problems
    run_dir = make_run(tmp_path)
    _add_problems(run_dir)
    run = views.Run(run_dir)
    frame, rows, count, err = ui.h_problems(run, [], "any")
    assert err == "" and len(frame) == len(rows) == 15 and "15 problems" in count
    assert list(frame.columns) == ui.COLUMNS["problems"]
    only = ui.h_problems(run, ["agent_tool"], "any")[1]
    assert {r["kind"] for r in only} == {"agent_tool"} and len(only) == 2
    assert len(ui.h_problems(run, [], "inferred")[1]) == 2
    from types import SimpleNamespace
    recorded = next(i for i, r in enumerate(rows) if r["kind"] == "tool_error")
    event, payload, text, err = ui.select_problem(run, rows, SimpleNamespace(index=[0, 0], row_value=[recorded]))
    assert event["type"] == "tool.error" and payload["error"] == "bad filter" and err == ""
    guessed = next(i for i, r in enumerate(rows) if r["kind"] == "agent_tool")
    event, payload, text, err = ui.select_problem(run, rows, SimpleNamespace(index=[0, 0], row_value=[guessed]))
    assert event is None and text.startswith(("exit 2", "Error: nope"))
    counts = ui.h_overview(run)[6]
    assert list(counts.columns) == ui.COLUMNS["problem_counts"] and counts["total"].sum() == 15
    app = ui.build_app(run_dir)
    assert _chains(app, next(b for b in app.blocks.values() if getattr(b, "value", None) == "Load problems"),
                   "click") == [["h_problems"]]


def test_tool_errors_stay_open_in_chats():
    msgs = ui.to_messages([{"kind": "tool_error", "title": "⚠ Tool error: x", "text": "Error: boom"}])
    title = msgs[0]["metadata"]["title"]
    assert "status" not in msgs[0]["metadata"]                                   # no status: shown open
    assert title.startswith('<span class="panel-tool-error" style="color:#cf222e">⚠ Tool error</span>')
    assert title.endswith(": x") and ".panel-tool-error" in ui.CSS
