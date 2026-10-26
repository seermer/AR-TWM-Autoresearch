"""The panel's HTTP server: a JSON API over the data layer, the run's files, and the built web app."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import media, problems, timeline, views
from .runfiles import fmt_ts

DIST = Path(__file__).resolve().parent / "web" / "dist"


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def reply(data) -> Response:
    try:
        body = json.dumps(data, default=str, allow_nan=False, ensure_ascii=False)
    except ValueError:                         # NaN or inf somewhere (a diverged loss): sent as null
        body = json.dumps(_clean(data), default=str, ensure_ascii=False)
    return Response(body, media_type="application/json")


def status(run: views.Run) -> dict:
    events = run.log.events()
    pid = run.files.loop_pid()
    responses = [e for e in events if e.get("type") == "llm.response"]
    costs = [e["cost_usd"] for e in responses if e.get("cost_usd") is not None]
    config = run.files.read_json("config/run.json") or {}
    judge = config.get("judge")
    return {"name": run.name, "alive": pid is not None, "pid": pid, "state": run.files.read_json("control/state.json"),
            "max_nodes": (run.files.read_json("control/run_args.json") or {}).get("max_nodes"),
            "nodes": len(run.nodes()), "judge": f"{judge['kind']} {judge['model']}" if judge else None,
            "calls": len(responses), "tokens": sum((e.get("usage") or {}).get("total_tokens") or 0 for e in responses),
            "usd": round(sum(costs), 4) if costs else None,
            "disk_free_gb": round(shutil.disk_usage(run.files.root).free / 1e9, 1),
            "events": len(events), "last_event": events[-1]["ts_wall"] if events else None,
            "last_seq": max((e["_seq"] for e in events), default=None), "now": time.time(), "config": config}


def problem_id(p: dict) -> str:
    """Stable across refreshes: the event's seq, or a digest of an inferred problem's own fields."""
    if p["seq"] is not None:
        return f"e{p['seq']}"
    key = json.dumps([p["ts"], p["node"], p["phase"], p["attempt"], p["where"], p["summary"]], default=str)
    return "i" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _conversation_meta(conv: dict) -> dict:
    return {k: v for k, v in conv.items() if k != "chain"} | {
        "conversation_ids": [s.conversation_id for s in conv["chain"]]}


def create_app(run_dir: Path | str) -> FastAPI:
    run = views.Run(run_dir)
    app = FastAPI(title=f"Run {run.name}", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def error(exc: Exception, code: int) -> Response:      # shown in the view that asked; the panel stays up
        return Response(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), status_code=code, media_type="application/json")

    @app.exception_handler(ValueError)
    async def refused(request: Request, exc: ValueError):  # a path outside the run, a bad argument
        return error(exc, 400)

    @app.exception_handler(Exception)
    async def failed(request: Request, exc: Exception):
        return error(exc, 500)

    @app.get("/api/status")
    def get_status():
        return reply(status(run))

    @app.get("/api/overview")
    def get_overview():
        ov = views.overview(run)
        return reply({**ov, "problem_counts": problems.problem_counts(run)})

    @app.get("/api/gpu")
    def get_gpu():
        return reply(views.gpu_full(run))

    @app.get("/api/timeline")
    def get_timeline(node: str | None = None):
        return reply(timeline.timeline(run, node or None))

    @app.get("/api/node/{node}")
    def get_node(node: str):
        d = views.node_detail(run, node)
        if d is None:
            raise HTTPException(404, f"no node {node!r}")
        attempts = []
        for phase, attempt in views.agent_attempts(run, node):
            convs = views.conversations(run, node, phase, attempt)
            attempts.append({"phase": phase, "attempt": attempt, "conversations": [_conversation_meta(c) for c in convs]})
        children = [n["node_id"] for n in run.nodes() if n["parent_id"] == node]
        return reply({**d, "agent_attempts": attempts, "children": children,
                      "training_attempts": views.training_attempts(run, node)})

    @app.get("/api/agents/{node}/{phase}/{attempt}")
    def get_agents(node: str, phase: str, attempt: int):
        convs = views.conversations(run, node, phase, attempt)
        return reply({"conversations": [_conversation_meta(c) for c in convs],
                      "graph": views.tasks(run, node, phase, attempt),
                      "tool_logs": views.tool_logs(run, node, phase, attempt)})

    @app.get("/api/chat/{node}/{phase}/{attempt}/{index}")
    def get_chat(node: str, phase: str, attempt: int, index: int):
        return reply(views.conversation_chat(run, node, phase, attempt, index))

    @app.get("/api/task/{node}/{phase}/{attempt}/{index}")
    def get_task(node: str, phase: str, attempt: int, index: int):
        return reply(views.task_chat(run, node, phase, attempt, index))

    @app.get("/api/event-choices")
    def get_event_choices():
        return reply(views.trace_choices(run))

    @app.get("/api/events")
    def get_events(node: str = "", phase: str = "", attempt: str = "", types: str = "", component: str = "",
                   text: str = "", gpu: bool = False, offset: int = Query(0, ge=0), limit: int = Query(500, ge=1)):
        return reply(views.trace(run, node=node, phase=phase, attempt=int(attempt) if attempt not in ("", "any") else None,
                                 types=[t for t in types.split(",") if t], component=component, text=text,
                                 include_gpu=gpu, offset=offset, limit=limit))

    @app.get("/api/event/{seq}")
    def get_event(seq: int):
        d = views.trace_detail(run, seq)
        if d is None:
            raise HTTPException(404, f"no event {seq}")
        return reply(d)

    @app.get("/api/problems")
    def get_problems():
        return reply([dict(p, id=problem_id(p)) for p in problems.problems(run)])

    @app.get("/api/code/{node}")
    def get_code(node: str):
        d = views.code_edits(run, node)
        n = run.node(node)
        files = views.changed_files(run, d["parent_commit"], d["agent_commit"]) if d["parent_commit"] and d["agent_commit"] else []
        for a in d["attempts"]:
            a["files"] = views.changed_files(run, a["base"], a["commit"]) if a["base"] and a["commit"] else []
        return reply({**d, "node_files": files, "commits": views.commit_choices(run, node),
                      "status": n["status"] if n else None})

    @app.get("/api/agent/tree")
    def get_agent_tree(commit: str):
        return reply(views.agent_tree(run, commit))

    @app.get("/api/agent/file")
    def get_agent_file(commit: str, path: str):
        return reply({"text": views.agent_file(run, commit, path)})

    @app.get("/api/data/{node}")
    def get_data(node: str):
        return reply(media.training_data(run, node))

    @app.get("/api/clips/{node}/{dataset}")
    def get_clips(node: str, dataset: str):
        got = media.clips(run, node, dataset, page=0, page_size=10 ** 9)
        return reply([{"clip_id": r["clip_id"], "video": r["video"], "derived_from": r["derived_from"],
                       "origin": media.origin(r["derived_from"]), "metadata": r["metadata"], "warnings": r["warnings"],
                       "leakage": r["leakage"], "ingested_by": r["ingested_by"]} for r in got["rows"]])

    @app.get("/api/clip/{clip_id}")
    def get_clip(clip_id: str):
        c = media.clip_detail(run, clip_id)
        if c is None:
            raise HTTPException(404, f"no clip {clip_id!r}")
        try:
            path = media.camera_path(run, c["pose"])
        except (OSError, KeyError, ValueError):
            path = None
        return reply({**c, "camera_path": path})

    @app.get("/api/training/{node}")
    def get_training(node: str):
        return reply(views.training_all(run, node))

    @app.get("/api/training-compare")
    def get_training_compare():
        return reply(views.training_compare(run))

    @app.get("/api/eval/{node}")
    def get_eval(node: str):
        return reply(media.eval_view(run, node))

    @app.get("/api/eval/{node}/{case}")
    def get_eval_case(node: str, case: str, other: str = ""):
        def one(n):
            if not n:
                return None
            c = next((c for c in media.eval_view(run, n)["cases"] if c["case"] == case), None)
            return {**c, "judge": media.judge_view(run, n, case)} if c else None
        return reply({"node": one(node), "other": one(other)})

    @app.get("/api/eval-matrix")
    def get_eval_matrix():
        return reply(views.eval_matrix(run))

    @app.get("/api/selection")
    def get_selection():
        return reply(views.selection(run))

    @app.get("/api/cost")
    def get_cost():
        return reply({**views.cost(run), "calls": views.llm_calls(run)})

    @app.get("/api/files")
    def get_files(path: str = ""):
        return reply(media.list_dir(run, path))

    @app.get("/api/preview")
    def get_preview(path: str):
        return reply(media.preview(run, path))

    @app.get("/api/tail")
    def get_tail(path: str, offset: int = 0, limit: int = 2_000_000):
        """Bytes of a growing file from `offset` (negative: the last -offset bytes), at most `limit`."""
        p = run.files.path(path)
        if not p.is_file():
            raise HTTPException(404, f"no file {path!r}")
        size = p.stat().st_size
        start = max(0, size + offset) if offset < 0 else min(offset, size)
        with p.open("rb") as handle:
            handle.seek(start)
            data = handle.read(limit)
        return reply({"text": data.decode("utf-8", errors="replace"), "start": start, "end": start + len(data),
                      "size": size, "modified": fmt_ts(p.stat().st_mtime)})

    @app.get("/raw/{path:path}")
    def get_raw(path: str):
        p = run.files.path(path)
        if not p.is_file():
            raise HTTPException(404, f"no file {path!r}")
        return FileResponse(p, headers={"Content-Security-Policy": "sandbox"})   # agent-written HTML runs no script here

    @app.get("/api/stream")
    async def stream(request: Request):
        """Server-sent events: the run's status whenever it changes (new events, loop start or stop)."""
        async def ticks():
            last = None
            while not await request.is_disconnected():
                s = await run_in_threadpool(status, run)
                key = (s["last_seq"], s["alive"], json.dumps(s["state"], default=str), s["nodes"])
                if key != last:
                    last = key
                    yield f"data: {json.dumps(s, default=str)}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(2)
        return StreamingResponse(ticks(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/{path:path}")
    def web(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, f"no API route /{path}")
        target = (DIST / path).resolve()
        if path and target.is_file() and target.is_relative_to(DIST):
            return FileResponse(target)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})

    return app
