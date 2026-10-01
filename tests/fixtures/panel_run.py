"""A small but complete fake run for the panel tests, written with the kernel's own writers
(Recorder, open_db/NodeStore, AgentsRepo), so the file formats are the real ones."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.vcs.agents_repo import AgentsRepo

PLANNER = "You plan the data for this node.\n"
BUILDER = "You build the training data for this node.\n"
COMPACT = "Your task is to write a detailed summary of the conversation so far.\n"
SUMMARY = "  Summary: fetched 3 clips, ingested 1, v2.mp4 rejected for its aspect ratio.  \n"
CONTINUATION = ("This session is being continued from a previous conversation that ran out of context. "
                "The summary below covers the earlier portion of the conversation.\n\nSummary:\n{summary}\n\n"
                "Continue the work from where it left off.")


def chat_request(messages, tools=None):
    return {"body": {"messages": messages, "model": "m", "tools": tools or []}}


def chat_response(content="", reasoning=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"body": {"choices": [{"index": 0, "finish_reason": "stop", "message": message}]}}


def tool_call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Llm:
    """Writes llm.request / llm.response pairs the way the gateway does."""

    def __init__(self, rec: Recorder, node: str, phase: str, attempt: int) -> None:
        self.rec, self.base = rec, dict(node=node, phase=phase, attempt=attempt, component="gateway")

    def call(self, conv, turn, messages, reply=None, tools=None, prompt_tokens=100) -> str:
        call_id = uuid.uuid4().hex
        self.rec.event("llm.request", payload=chat_request(messages, tools), call_id=call_id,
                       conversation_id=conv, turn_index=turn, parent_call_id=None, model="m", **self.base)
        if reply is not None:
            self.rec.event("llm.response", payload=chat_response(**reply), call_id=call_id,
                           conversation_id=conv, status=200, latency_s=1.5, attempts=1,
                           usage={"prompt_tokens": prompt_tokens, "completion_tokens": 10,
                                  "total_tokens": prompt_tokens + 10},
                           cost_usd=0.001, mock=False, **self.base)
        return call_id


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def make_run(base: Path, name: str = "r1") -> Path:
    run = Path(base) / "runs" / name
    _write(run / "config" / "run.json", {"run_id": name, "metric_set": ["m1", "m2"]})
    _write(run / "control" / "run_args.json", {"max_nodes": 3})
    _write(run / "control" / "state.json", {"node": "n1", "phase": "eval", "attempt": 0, "since": 1.0})

    # agent code: the seed, two edit_self attempts of n1 (the second on top of the first)
    seed = Path(base) / "seed"
    _write(seed / "agent" / "prompts" / "planner.md", PLANNER)
    _write(seed / "agent" / "prompts" / "data_builder.md", BUILDER)
    _write(seed / "agent" / "entry.py", "X = 1\n")
    repo = AgentsRepo(run / "agents.git")
    root_commit = repo.init(seed)
    _write(seed / "agent" / "entry.py", "X = 2\n")
    edit1 = repo.commit_tree(seed, root_commit, "n1 edit_self attempt 1")
    repo.set_ref(repo.attempt_ref("n1", "edit_self", 1), edit1)
    _write(seed / "agent" / "entry.py", "X = 3\n")
    edit2 = repo.commit_tree(seed, edit1, "n1 edit_self attempt 2")
    repo.set_ref(repo.attempt_ref("n1", "edit_self", 2), edit2)
    repo.set_ref(repo.branch_ref("n1"), edit2)

    # archive
    conn = open_db(run)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root_commit)
    nodes.record_score("root", 0.70, ["m1", "m2"], {"m1": 0.6, "m2": 0.8})
    nodes.create("n1", "root", 1)
    nodes.set_fields("n1", agent_commit=edit2, data_commit="dc1",
                     recipe_path="nodes/n1/recipe.yaml", rationale_path="nodes/n1/rationale.md",
                     phase_timings=json.dumps({"total_s": 600.0}),
                     attempt_counts=json.dumps({"edit_self": 2, "improve_recipe": 1}))
    nodes.record_score("n1", 0.75, ["m1", "m2"], {"m1": 0.7, "m2": 0.8})
    nodes.add_attempt("n1", "edit_self", 1, "contract_failed", {"error": "import failed"})
    nodes.add_attempt("n1", "edit_self", 2, "passed", {"commit": edit2})
    nodes.add_attempt("n1", "improve_recipe", 1, "passed", {"checkpoint": "nodes/n1/x"})
    nodes.create("n2", "n1", 2)
    nodes.add_attempt("n2", "edit_self", 1, "failed", {"error": "cannot check out the agent code"})
    conn.execute("INSERT INTO data_commits VALUES (?,?,?,?,?,?,?)",
                 ("dc1", None, "n1", 1, json.dumps({"datasets": {"ds1": {"clips": ["clip1"]}}}),
                  "ds1: one clip", 1.0))
    conn.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("clip1", "vd1", "cd1", "pd1", "moving",
                  json.dumps({"frames": 5, "fps": 24.0, "width": 1280, "height": 720}),
                  json.dumps(["video_caption_camera"]), json.dumps([]),
                  json.dumps({"kind": "hf", "repo": "org/ds"}), "mit", json.dumps([]), "n1", 1.0))
    conn.execute("INSERT INTO selection_events (child_id, chosen, seed, candidates, created_at) "
                 "VALUES (?,?,?,?,?)", ("n1", "root", 7, json.dumps(
                     [{"node_id": "root", "value": 0.7, "penalty": 1.0, "P": 1.0}]), 1.0))
    conn.close()

    # stored clip blobs
    _write(run / "store" / "blobs" / "video" / "vd1.mp4", b"\x00mp4")
    _write(run / "store" / "blobs" / "caption" / "cd1.json",
           {"caption": "a hallway", "segments": [{"time_range_s": [0, 1], "text": "walk"}]})
    c2w = np.tile(np.eye(4), (5, 1, 1))
    c2w[:, 0, 3] = np.arange(5)
    c2w[:, 2, 3] = np.arange(5) * 2
    (run / "store" / "blobs" / "pose").mkdir(parents=True)
    np.savez(run / "store" / "blobs" / "pose" / "pd1.npz", cam_c2w=c2w)

    # node files
    _write(run / "nodes" / "n1" / "edit.json", {"summary": "tighter planner"})
    _write(run / "nodes" / "n1" / "rationale.md", "Why this recipe.\n")
    _write(run / "nodes" / "n1" / "recipe.yaml", "optimizer.lr: 1.0e-05\n")
    attempt = run / "nodes" / "n1" / "attempts" / "improve_recipe-1"
    _write(attempt / "train_config.yaml", "optimizer:\n  lr: 1.0e-05\n")
    _write(attempt / "train" / "train.log",
           "[SpatialMaskDbg] step=0 B_loss_valid=28.9\n"
           "[Train] step=1 epoch=0 source=ds1 loss=0.347656 grad=0.0291 lr=1.00e-05 time=39.36s\n"
           "[Train] step=2 epoch=0 source=ds1 loss=0.275391 grad=0.0310 lr=1.50e-05 time=8.08s\n")
    _write(attempt / "workspace" / "tool_output" / "run_command-20260927-120000-0001.log",
           "$ seq 3\nexit 0\n1\n2\n3\n")
    _write(run / "staging" / "n1" / "improve_recipe-1" / "work" / "v2.mp4", b"\x00rejected")
    for node, score in (("root", 0.6), ("n1", 0.7)):
        work = run / "nodes" / node / "eval" / "work_dirs" / f"ar_{name}_{node}"
        _write(work / "videos" / "case_7_combined.mp4", b"\x00video")
        _write(work / "videos" / "case_7_combined.json",
               {"case_id": "7", "perspective": "first_person", "actions": ["W"],
                "prompt_schedule": [{"prompt": "a corridor"}]})
        _write(work / "evaluation" / "m1" / "case_7.json", {"case_id": "7", "summary": {"m1": score}})
        _write(work / "evaluation" / "m2" / "case_7.json", {"case_id": "7", "summary": {"m2": 0.8}})
        _write(run / "nodes" / node / "eval" / "aggregates.json", {"metrics": {"m1": score, "m2": 0.8}})

    # telemetry, in time order
    rec = Recorder(run)
    rec.event("alert", payload={"kind": "stall"}, kind="stall", level="warning", message="quiet 30 min")
    rec.event("phase.start", node="n1", phase="edit_self", attempt=1, payload={"code_commit": root_commit})
    rec.event("contract.report", node="n1", phase="contract", attempt=1,
              payload={"ok": False, "failures": ["import failed"]})
    rec.event("phase.start", node="n1", phase="edit_self", attempt=2, payload={"code_commit": root_commit})
    rec.event("contract.report", node="n1", phase="contract", attempt=2, payload={"ok": True, "failures": []})
    rec.event("phase.start", node="n1", phase="improve_recipe", attempt=1, payload={"code_commit": edit2})

    llm = Llm(rec, "n1", "improve_recipe", 1)
    plan_sys = {"role": "system", "content": PLANNER}
    plan_user = {"role": "user", "content": "CONTEXT"}
    first = {"content": "", "reasoning": "look first", "tool_calls": [tool_call("c1", "list_dir", {"path": "."})]}
    tools = [{"type": "function", "function": {"name": "list_dir", "description": "List a directory"}}]
    llm.call("p1", 0, [plan_sys, plan_user], first, tools=tools)
    # the harness round-trips reasoning with the assistant message it resends
    llm.call("p1", 1, [plan_sys, plan_user, {"role": "assistant", "content": "", "reasoning": "look first",
                                             "tool_calls": first["tool_calls"]},
                       {"role": "tool", "tool_call_id": "c1", "content": "a\nb"}], {"content": "plan done"},
             tools=tools)

    build_sys = {"role": "system", "content": BUILDER + "\n\n# Reference: data_building.md\n\nref"}
    ingest_call = tool_call("c2", "data_ingest", {"candidates": []})
    history = [build_sys, {"role": "user", "content": "PLAN"}]
    llm.call("b1", 0, history, {"content": "", "tool_calls": [ingest_call]})
    history += [{"role": "assistant", "content": "", "tool_calls": [ingest_call]},
                {"role": "tool", "tool_call_id": "c2", "content": "[{\"accepted\": true}]"},
                {"role": "user", "content": COMPACT}]
    llm.call("b1", 2, history, {"content": "", "tool_calls": [ingest_call]}, prompt_tokens=900)   # compaction, tool call
    llm.call("b1", 2, history, {"content": SUMMARY}, prompt_tokens=900)                          # forced retry
    cont = [build_sys, {"role": "user", "content": CONTINUATION.format(summary=SUMMARY.strip())}]
    llm.call("b2", 0, cont, {"content": "continuing"}, prompt_tokens=80)
    llm.call("b2", 1, cont + [{"role": "assistant", "content": "continuing"},
                              {"role": "user", "content": "go on"}], None)                        # no response

    span = rec.event("tool.call", node="n1", phase="improve_recipe", attempt=1, component="tools",
                     tool="data_ingest", payload={"tool": "data_ingest", "args": {"candidates": [
                         {"video": "/workspace/staging/work/v1.mp4", "caption": "/workspace/staging/work/v1.json",
                          "camera_motion": "moving"},
                         {"video": "/workspace/staging/work/v2.mp4", "caption": "/workspace/staging/work/v2.json",
                          "camera_motion": "static"}]}})
    staged = str(run / "staging" / "n1" / "improve_recipe-1" / "work")
    cand1 = {"video": f"{staged}/v1.mp4", "caption": f"{staged}/v1.json", "pose": None}
    cand2 = {"video": f"{staged}/v2.mp4", "caption": f"{staged}/v2.json", "pose": None}
    rec.event("ingest.leakage", node="n1", phase="ingest", payload={"candidate": cand1, "matches": [], "near": [{"case_id": "7"}]})
    rec.event("ingest.accepted", node="n1", phase="ingest", payload={"candidate": cand1, "clip_id": "clip1",
                                                                    "formats": ["video_caption_camera"], "warnings": []})
    rec.event("ingest.rejected", node="n1", phase="ingest", payload={"candidate": cand2, "reasons": ["aspect 4:3"]})
    rec.event("tool.result", node="n1", phase="improve_recipe", attempt=1, component="tools", tool="data_ingest",
              parent_span_id=span, duration_s=2.0, payload={"tool": "data_ingest", "result": [
                  {"accepted": True, "clip_id": "clip1", "formats": ["video_caption_camera"], "warnings": [], "reasons": []},
                  {"accepted": False, "clip_id": None, "formats": [], "warnings": [], "reasons": ["aspect 4:3"]}]})
    rec.event("gate.failed", node="n1", phase="gate", payload={"failures": ["resolution 1x1 is not allowed"], "recipe": {}})
    rec.event("phase.end", node="n1", phase="improve_recipe", attempt=1)
    rec.event("gate.passed", node="n1", phase="gate", payload={"failures": [], "recipe": {}})
    rec.event("train.start", node="n1", phase="train")
    rec.event("gpu.sample", node="gpu", gpus={"0": {"util": 50.0, "memory_mib": 2048.0},
                                              "1": {"util": 70.0, "memory_mib": 4096.0}})
    rec.event("train.end", node="n1", phase="train")
    rec.event("node.end", payload={"node": "n1", "status": "scored", "error": None}, child="n1", status="scored")
    return run
