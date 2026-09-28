"""Runs in the panel env: .envs/panel/bin/python -m pytest tests/test_panel_ui.py"""
import pytest

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
