"""A small but complete fake run for the panel tests, written with the kernel's own writers
(telemetry, open_db/NodeStore, AgentsRepo), so the file formats are the real ones."""
from __future__ import annotations

import importlib
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ar_kernel import telemetry
from ar_kernel.archive.activity import add_call
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.vcs.agents_repo import AgentsRepo

PLANNER = "You plan the data for this node.\n"
BUILDER = "You build the training data for this node.\n"
COMPACT = "Your task is to write a detailed summary of the conversation so far.\n"
SUMMARY = "  Summary: fetched 3 clips, ingested 1, v2.mp4 rejected for its aspect ratio.  \n"
CONTINUATION = ("This session is being continued from a previous conversation that ran out of context. "
                "The summary below covers the earlier portion of the conversation.\n\nSummary:\n{summary}\n\n"
                "Continue the work from where it left off.")
LINKS = ["previous response", "conversation", "prefix", "new"]


def chat_request(messages, tools=None):
    return {"messages": messages, "model": "m", "tools": tools or []}


def chat_response(content="", reasoning=None, tool_calls=None, prompt_tokens=100):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10, "total_tokens": prompt_tokens + 10}}


def tool_call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """Records written with the kernel's telemetry API, each at the time `t` (seconds) says. The writer's process
    has ended unless `alive`."""

    def __init__(self, run: Path, t: float | None = None, alive: bool = False) -> None:
        self.t = round(time.time()) - 3600 if t is None else t
        self.writer = importlib.import_module("ar_kernel.telemetry.writer")
        self.real_time = self.writer.time
        start_time = self.writer._start_time
        if not alive:
            self.writer._start_time = lambda: "0"      # never this process's start: its pid reads as another's
        try:
            telemetry.open(run)
        finally:
            self.writer._start_time = start_time
        self.writer.time = SimpleNamespace(time=lambda: self.t, monotonic=lambda: self.t, strftime=time.strftime)
        self.root = telemetry.start("ar.run")

    def at(self, t: float) -> "Scripted":
        self.t = t
        return self

    def close(self, end: bool = True) -> None:
        """End the invocation (unless not `end`) and detach the writer: the test reads the run as the panel does."""
        if end:
            self.root.end("ok", {"exit_code": 0})
        self.writer.time = self.real_time
        self.writer._writer = None
        importlib.import_module("ar_kernel.telemetry.api")._root = None

    def llm_start(self, parent, conv, turn, request, role=None, tool=None):
        sp = telemetry.start("llm.request", {"endpoint": "/chat/completions", "role": role,
                                             "body": json.dumps(request)}, parent=parent)
        with telemetry.use(sp):
            meta = {"call_id": uuid.uuid4().hex, "conversation_id": conv, "turn_index": turn, "parent_call_id": None,
                    **({"role": role} if role else {}), **({"tool": tool} if tool else {})}
            telemetry.decision("llm.link", meta, LINKS, "new", "no earlier call of this caller matches")
            with telemetry.span("llm.censor") as step:
                step.output = {"body": request}
        return sp

    @staticmethod
    def llm_end(sp, response, status=200, cost=0.001):
        sp.end("ok" if status < 400 else "error", {"status": status, "body": response, "cost_usd": cost, "mock": False})

    def llm(self, parent, conv, turn, messages, reply=None, tools=None, prompt_tokens=100, role=None):
        """One call that gets its reply a second later, or none (`reply` None: its span stays open)."""
        sp = self.llm_start(parent, conv, turn, chat_request(messages, tools), role=role)
        if reply is not None:
            self.t += 1
            self.llm_end(sp, chat_response(**reply, prompt_tokens=prompt_tokens))
        self.t += 1
        return sp

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
    conn.execute("INSERT INTO data_commits (commit_id, node_id, attempt, status, manifest, message, created_at) "
                 "VALUES (?,?,?,?,?,?,?)", ("dc1", "n1", 1, "accepted", json.dumps({"datasets": {"ds1": {"clips": ["clip1"]}}}),
                                            "ds1: one clip", 1.0))
    conn.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("clip1", "vd1", "cd1", "ad1", "pd1",
                  json.dumps({"frames": 5, "fps": 24.0, "width": 960, "height": 544, "turns": 1}),
                  json.dumps([]), json.dumps({"kind": "dataset", "name": "org/ds"}), "n1", 1, 1.0, None))
    conn.execute("INSERT INTO selection_events (child_id, chosen, seed, candidates, created_at) "
                 "VALUES (?,?,?,?,?)", ("n1", "root", 7, json.dumps(
                     [{"node_id": "root", "value": 0.7, "penalty": 1.0, "P": 1.0}]), 1.0))
    conn.close()

    # stored clip blobs
    _write(run / "store" / "blobs" / "video" / "vd1.mp4", b"\x00mp4")
    _write(run / "store" / "blobs" / "caption" / "cd1.json",
           {"caption": "a hallway", "segments": [{"time_range_s": [0, 1], "prompt": "a hallway"}]})
    _write(run / "store" / "blobs" / "annotation" / "ad1.json",
           {"scene_prompt": "a hallway", "turns": [{"start_s": 0.0, "perspective": "first_person"}]})
    c2w = np.tile(np.eye(4), (5, 1, 1))
    c2w[:, 0, 3] = np.arange(5)
    c2w[:, 2, 3] = np.arange(5) * 2
    (run / "store" / "blobs" / "pose").mkdir(parents=True)
    np.savez(run / "store" / "blobs" / "pose" / "pd1.npz", cam_c2w=c2w)

    # node files
    _write(run / "nodes" / "n1" / "edit.json", {"summary": "tighter planner"})
    _write(run / "nodes" / "n1" / "rationale.md", "Why this recipe.\n")
    _write(run / "nodes" / "n1" / "summary.md", "Tested pool clips.\n")
    _write(run / "nodes" / "n1" / "recipe.yaml", "optimizer.lr: 1.0e-05\n")
    attempt = run / "nodes" / "n1" / "attempts" / "improve_recipe-1"
    _write(attempt / "train_config.yaml", "optimizer:\n  lr: 1.0e-05\n")
    _write(attempt / "train" / "train.log",
           "[SpatialMaskDbg] step=0 B_loss_valid=28.9\n"
           "[Train] step=1 epoch=0 source=ds1 loss=0.347656 grad=0.0291 lr=1.00e-05 time=39.36s\n"
           "[Train] step=2 epoch=0 source=ds1 loss=0.275391 grad=0.0310 lr=1.50e-05 time=8.08s\n")
    _write(attempt / "workspace" / "tool_output" / "run_command-20260927-120000-0001.log",
           "$ seq 3\nexit 0\n1\n2\n3\n")
    _write(attempt / "workspace" / "tool_output" / "data_engineer-data_query-20260927-120001-2.txt", "$ data_query {}\n[]\n")
    _write(run / "staging" / "n1" / "improve_recipe-1" / "work" / "v2.mp4", b"\x00rejected")
    for node, score in (("root", 0.6), ("n1", 0.7)):
        work = run / "nodes" / node / "eval" / "work_dirs" / f"ar_{name}_{node}"
        _write(work / "videos" / "case_7_combined.mp4", b"\x00video")
        _write(work / "videos" / "case_7_combined.json",
               {"case_id": "7", "perspective": "first_person", "actions": ["W"],
                "prompt_schedule": [{"prompt": "a corridor"}]})
        seen = node == "n1"                     # the judge saw the event at n1 only
        _write(work / "evaluation" / "report.json",
               {"per_case": {"7": {"m1": score, "m2": 0.8, "event_edit_adherence": 1.0 if seen else 0.5}}})
        _write(work / "evaluation" / "event_edit_adherence" / "case_7.json",
               {"case_id": "7", "turn_details": [{"turn": 2, "action": "Rain begins.", "questions": [
                   {"sub_id": "Q1", "correct": True, "answer": "no", "expected": "no", "reason": "r"},
                   {"sub_id": "Q2", "correct": seen, "answer": "yes" if seen else "no", "expected": "yes",
                    "reason": "r"}]}]})
        _write(run / "nodes" / node / "eval" / "aggregates.json", {"metrics": {"m1": score, "m2": 0.8}})

    rec = Scripted(run)
    try:
        _records(rec, run, root_commit, edit2)
    finally:
        rec.close()
    return run


def _records(rec: Scripted, run: Path, root_commit: str, edit2: str) -> None:
    """The telemetry, in time order."""
    telemetry.alert("monitor.stall", "warning", "quiet 30 min")
    rec.t += 1
    with telemetry.span("node", {"node": "n1", "resumed": False, "phase": None, "attempt": None}) as node:
        for k, ok in ((1, False), (2, True)):
            with telemetry.span("phase.attempt", {"node": "n1", "phase": "edit_self", "attempt": k}) as sp:
                telemetry.decision("contract.step", {"step": "import"}, ["pass", "fail"], "pass" if ok else "fail",
                                   "passed" if ok else "import failed")
                sp.output = {}
            rec.t += 1
        attempt = telemetry.start("phase.attempt", {"node": "n1", "phase": "improve_recipe", "attempt": 1})
        with telemetry.use(attempt):
            plan_sys = {"role": "system", "content": PLANNER}
            plan_user = {"role": "user", "content": "CONTEXT"}
            first = {"content": "", "reasoning": "look first", "tool_calls": [tool_call("c1", "list_dir", {"path": "."})]}
            tools = [{"type": "function", "function": {"name": "list_dir", "description": "List a directory"}}]
            rec.llm(attempt, "p1", 0, [plan_sys, plan_user], first, tools=tools, role="planner")
            # the harness round-trips reasoning with the assistant message it resends
            rec.llm(attempt, "p1", 1, [plan_sys, plan_user, {"role": "assistant", "content": "", "reasoning": "look first",
                                                            "tool_calls": first["tool_calls"]},
                                       {"role": "tool", "tool_call_id": "c1", "content": "a\nb"}], {"content": "plan done"},
                    tools=tools, role="planner")
            build_sys = {"role": "system", "content": BUILDER + "\n\n# Reference: data_building.md\n\nref"}
            ingest_call = tool_call("c2", "data_ingest", {"candidates": []})
            history = [build_sys, {"role": "user", "content": "PLAN"}]
            rec.llm(attempt, "b1", 0, history, {"content": "", "tool_calls": [ingest_call]}, role="data_builder")
            history += [{"role": "assistant", "content": "", "tool_calls": [ingest_call]},
                        {"role": "tool", "tool_call_id": "c2", "content": "[{\"accepted\": true}]"},
                        {"role": "user", "content": COMPACT}]
            rec.llm(attempt, "b1", 2, history, {"content": "", "tool_calls": [ingest_call]}, prompt_tokens=900,
                    role="data_builder")                                             # compaction, tool call
            rec.llm(attempt, "b1", 2, history, {"content": SUMMARY}, prompt_tokens=900, role="data_builder")   # forced retry
            cont = [build_sys, {"role": "user", "content": CONTINUATION.format(summary=SUMMARY.strip())}]
            rec.llm(attempt, "b2", 0, cont, {"content": "continuing"}, prompt_tokens=80, role="data_builder")
            rec.llm(attempt, "b2", 1, cont + [{"role": "assistant", "content": "continuing"},
                                              {"role": "user", "content": "go on"}], None, role="data_builder")  # no response
            add_call(run, node="n1", phase="improve_recipe", attempt=1, role="planner", tool=None, conversation_id="p1",
                     turn=0, transcript=None, status=200, tokens=(100, 0, 10), cost_usd=0.001, mock=False, audit_hits=[])
            _ingest(rec, run)
            telemetry.decision("recipe.gate", {"recipe": {}}, ["pass", "refuse"], "refuse", "resolution 1x1 is not allowed")
            rec.t += 1
            telemetry.decision("recipe.gate", {"recipe": {}}, ["pass", "refuse"], "pass", "no findings")
        attempt.end("ok", {})
        rec.t += 1
        with telemetry.span("phase.attempt", {"node": "n1", "phase": "train", "attempt": 1}) as sp:
            with telemetry.span("train", {"config": "optimizer.lr: 1.0e-05"}) as train:
                rec.t += 30
                telemetry.sample(gpus={"0": {"util": 50.0, "memory_mib": 2048.0, "pids": []},
                                       "1": {"util": 70.0, "memory_mib": 4096.0, "pids": []}})
                rec.t += 30
                train.output = {"checkpoint": "nodes/n1/x"}
            sp.output = {}
        telemetry.decision("node.status", {"node": "n1"}, ["scored"], "scored", "a score")
        node.output = {"status": "scored"}


def _ingest(rec: Scripted, run: Path) -> None:
    """A data_ingest call as the tool server records it: two candidates, one accepted, and the result file."""
    staged = "/workspace/staging/work"
    with telemetry.span("tool.data_ingest", {"candidates": 2}) as tool:
        for name, accepted in (("v1", True), ("v2", False)):
            named = {"video": f"{staged}/{name}.mp4", "annotation": f"{staged}/{name}.json", "pose": f"{staged}/{name}.npz"}
            with telemetry.span("ingest.candidate", {"named": named, "derived_from": {"kind": "dataset", "name": "org/ds"}}) as c:
                telemetry.decision("ingest.check", {"check": "held-out cases", "leakage": {
                    "rejected": False, "matches": [], "near_matches": [{"case_id": "7"}]}, "excluded": []},
                    ["pass", "reject"], "pass", "passed")
                if not accepted:
                    telemetry.decision("ingest.check", {"check": "shape", "width": 4, "height": 3}, ["pass", "reject"],
                                       "reject", "aspect 4:3")
                c.output = {"accepted": accepted, "reasons": [] if accepted else ["aspect 4:3"]}
        rows = [{"index": 0, "video": f"{staged}/v1.mp4", "accepted": True, "clip_id": "clip1", "warnings": [],
                 "reasons": []},
                {"index": 1, "video": f"{staged}/v2.mp4", "accepted": False, "clip_id": None, "warnings": [],
                 "reasons": ["aspect 4:3"]}]
        telemetry.keep("/workspace/staging/results/data_ingest-1.json", json.dumps(rows).encode(), "handed")
        rec.t += 2
        tool.output = {"accepted": 1, "rejected": 1, "result_file": "/workspace/staging/results/data_ingest-1.json"}
