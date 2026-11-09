"""python -m simple_panel --run-id <run>: a scrubbed snapshot of the run on a gradio share link, behind a login
read from AutoResearcher/.simple_panel_auth (one `user:password` line, readable by its owner only)."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from ar_kernel.subproc import tmp_link
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

REPO = Path(__file__).resolve().parents[1]
AUTH = REPO / ".simple_panel_auth"
CACHE = REPO / ".cache" / "simple_panel"
CODE_KINDS = {"tools", "tool_call", "tool_output", "tool_error"}
USER_KINDS = {"user", "context", "compaction_request"}


def read_auth(path: Path = AUTH) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"error: no login file {path}; write one `user:password` line to it, chmod 600")
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"error: {path} is readable by others; chmod 600 it")
    user, _, password = path.read_text().strip().partition(":")
    if not user or not password:
        raise SystemExit(f"error: {path} must hold one `user:password` line")
    return user, password


def fence(text: str, lang: str = "") -> str:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def chat(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        if it["kind"] == "tokens":
            out.append({"role": "assistant", "content": f"*{it['text']}*"})
            continue
        text = fence(it["text"]) if it["kind"] in CODE_KINDS else it["text"]
        message = {"role": "user" if it["kind"] in USER_KINDS else "assistant", "content": text or "(empty)"}
        if it["title"] and it["kind"] != "assistant":
            message["metadata"] = {"title": it["title"]}
        out.append(message)
    return out


def table(rows: list[dict]):
    import pandas as pd
    return pd.DataFrame(rows).round(4) if rows else pd.DataFrame()


def scores_md(scores: dict) -> str:
    return "\n".join(f"- {m}: {v:.4f}" if isinstance(v, float) else f"- {m}: {v}" for m, v in scores.items())


def add_clip_route(app, snap: Path) -> None:
    """Videos by snapshot path: gradio's own file route would show clients the host path. Logged-in users only,
    by gradio's own check (its routes.get_current_user)."""
    videos = (snap / "videos").resolve()

    @app.get("/clip/videos/{rel:path}")
    def clip(rel: str, request: Request):
        token = (request.cookies.get(f"access-token-{app.cookie_id}")
                 or request.cookies.get(f"access-token-unsecure-{app.cookie_id}"))
        if app.tokens.get(token) is None:
            raise HTTPException(401, "not logged in")
        path = (videos / rel).resolve()
        if not path.is_relative_to(videos) or not path.is_file():
            raise HTTPException(404, "no such video")
        return FileResponse(path, media_type="video/mp4")


def build(run_dir: Path, snap: Path):
    import gradio as gr
    from .export import export, tokens_text

    summary = lambda: json.loads((snap / "summary.json").read_text())

    def detail(node: str | None) -> dict | None:
        n = next((n for n in summary()["nodes"] if n["node"] == node), None)
        return json.loads((snap / n["file"]).read_text()) if n else None

    def show(current: str | None, version: int):
        s = summary()
        names = [n["node"] for n in s["nodes"]]
        head = (f"## Run {s['run']}\nSnapshot taken {s['exported']} · {len(names)} nodes · "
                f"agent tokens: {tokens_text(s['tokens'])}")
        return (head, table([{k: v for k, v in n.items() if k != "file"} for n in s["nodes"]]),
                table(s["scores"]), table(s["alerts"]),
                gr.update(choices=names, value=current if current in names else (names[-1] if names else None)),
                version + 1)

    def refresh(current, version):
        export(run_dir, snap)
        return show(current, version)

    def conversations(node):
        labels = [c["label"] for c in (detail(node) or {}).get("conversations", [])]
        return gr.update(choices=labels, value=labels[0] if labels else None)

    def conversation(node, label):
        convs = (detail(node) or {}).get("conversations", [])
        return chat(next((c["items"] for c in convs if c["label"] == label), []))

    with gr.Blocks(title="Run snapshot") as app:
        version = gr.State(0)
        with gr.Row():
            head = gr.Markdown()
            refresh_btn = gr.Button("Refresh", size="sm", scale=0)
        with gr.Tab("Nodes"):
            nodes_df = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Scores"):
            scores_df = gr.Dataframe(interactive=False)
        with gr.Tab("Alerts"):
            alerts_df = gr.Dataframe(interactive=False, wrap=True)
        node = gr.Dropdown(label="Node (for the tabs below)")
        with gr.Tab("Videos"):
            @gr.render(inputs=[node, version])
            def videos(node_id, _):
                d = detail(node_id) or {}
                for title, key in (("Training data", "training_videos"), ("Eval renders", "eval_videos")):
                    gr.Markdown(f"### {title}" + ("" if d.get(key) else "\nnone"))
                    for v in d.get(key, []):
                        with gr.Row():
                            gr.HTML(f'<b>{v["label"]}</b><video controls preload="metadata" style="width:100%" '
                                    f'src="clip/{v["file"]}"></video>' if v["file"]
                                    else f"{v['label']}: video file missing from the run")
                            text = v.get("caption") or ""
                            if key == "eval_videos":
                                text = f"perspective: {v['perspective']}\n\n{fence(v['prompt'])}\n\n{scores_md(v['scores'])}"
                            gr.Markdown(text)
        with gr.Tab("Conversations"):
            conv = gr.Dropdown(label="Conversation")
            chatbot = gr.Chatbot(height=800, group_consecutive_messages=False)
        with gr.Tab("Edit-self diffs"):
            @gr.render(inputs=[node, version])
            def diffs(node_id, _):
                e = (detail(node_id) or {}).get("edits") or {}
                gr.Markdown("### This node vs its parent")
                gr.Markdown(fence(e["node_diff"], "diff") if e.get("node_diff") else "(no change)")
                for a in e.get("attempts", []):
                    gr.Markdown(f"### edit_self attempt {a['attempt']}: {a['outcome']}"
                                + (f"\n{fence(str(a['detail']))}" if a["detail"] else ""))
                    gr.Markdown(fence(a["diff"], "diff") if a["diff"] else "(no commit)")

        outputs = [head, nodes_df, scores_df, alerts_df, node, version]
        app.load(show, [node, version], outputs)
        refresh_btn.click(refresh, [node, version], outputs)
        node.change(conversations, node, conv)
        conv.change(conversation, [node, conv], chatbot)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m simple_panel", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-share", action="store_true", help="serve on this machine's network only")
    args = parser.parse_args(argv)
    run_dir = REPO / "runs" / args.run_id
    if not (run_dir / "config" / "run.json").is_file():
        raise SystemExit(f"error: no run {args.run_id!r} under {REPO / 'runs'}")
    auth = read_auth()
    os.environ.update(GRADIO_ANALYTICS_ENABLED="False", GRADIO_TEMP_DIR=str(CACHE / "gradio"),
                      HF_HOME=str(CACHE / "hf"), TMPDIR=str(tmp_link()))   # gradio's tunnel binary goes under HF_HOME
    from .export import export
    snaps = CACHE / "snapshots"
    snap = export(run_dir, snaps / args.run_id)
    blocks = build(run_dir, snap)
    app, _, _ = blocks.launch(auth=auth, share=not args.no_share, server_name="0.0.0.0", server_port=args.port,
                              footer_links=[], mcp_server=False, prevent_thread_lock=True)
    add_clip_route(app, snap)
    blocks.block_thread()


if __name__ == "__main__":
    main()
