"""The loop: one node at a time. Nothing pauses (user decision 2026-09-27): a
failure the agent can act on goes back to it as the next attempt's retry report; any other
failure ends the node with a status and an error that later agents see in the lineage."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .agent_phase import PhaseEnv, run_edit_self, run_improve_recipe, smoke_contexts
from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.nodes import NodeStore, run_rel
from .contract.verify import verify_contract
from .guards import alert
from .eval.score import weighted_score
from .run import record_root_counts, root_key, score_node
from .selection import select_parent, selection_seed, update_values
from .subproc import file_tail
from .train.gate import Gate
from .train.recipe import lora_of
from .train.runner import TrainOutcome, TrainRunner
from .transcripts import write_transcripts


def save_root(cache: Path, eval_dir: Path, metrics: dict, expected_n: dict, run_id: str) -> None:
    """Keep a scored root for later runs with the same root_key (files hard-linked, not copied)."""
    tmp = cache.with_name(cache.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(eval_dir, tmp / "eval", copy_function=os.link)
    (tmp / "root.json").write_text(json.dumps({
        "metrics": metrics, "expected_n": expected_n, "run_id": run_id,
        "aggregates": json.loads((eval_dir / "aggregates.json").read_text())}, indent=1))
    tmp.rename(cache)


def _submitted(result: dict) -> dict:
    """What a failed improve_recipe attempt submitted: its data commit is still valid for a retry."""
    return {k: result[k] for k in ("data_commit", "recipe", "rationale")}


class StopRun(Exception):
    """End the run now (budget spent); resume marks the running node `interrupted`."""


def _gate(loop, recipe, data_commit, parent_commit, node, attempt_dir):
    conn = loop.ctx.conn
    store = CommitStore(conn, BlobStore(loop.ctx.run_dir, conn), ClipStore(conn))
    with loop.kit.gpu_lock:
        return Gate(loop.cfg, store, loop.ctx.recorder).check(
            recipe, data_commit, parent_commit, node, attempt_dir, loop.ctx.run_dir, loop.ctx.gpus)


def _contract(*, loop, node, parent_id, commit, attempt):
    contexts = smoke_contexts(loop.env, conn=loop.ctx.conn, node=node, parent_id=parent_id,
                              nodes_remaining=loop.max_nodes - loop._children())
    return verify_contract(cfg=loop.cfg, run_dir=loop.ctx.run_dir, run_id=Path(loop.ctx.run_dir).name,
                           repo=loop.repo, commit=commit, harness=loop.kit.harness,
                           recorder=loop.ctx.recorder, node=node, attempt=attempt, contexts=contexts)


def _train(loop, resolved, node, attempt_dir):
    runner = TrainRunner(loop.cfg, loop.ctx.recorder)
    with loop.kit.gpu_lock:
        try:
            runner.precache(resolved, loop.ctx.gpus, node)
        except RuntimeError as exc:
            return TrainOutcome(None, "infra", Path(attempt_dir) / "train" / "train.log",
                                detail=f"prompt precache failed: {exc}")
        return runner.train(resolved, loop.ctx.gpus, node, attempt_dir)


def _score(loop, node, checkpoint, resolved):
    rank = lora_of(resolved)[0] if resolved is not None else 0
    with loop.kit.gpu_lock:
        return score_node(loop.cfg, loop.ctx, node, checkpoint, rank)


@dataclass
class Phases:
    edit_self: Callable = run_edit_self
    contract: Callable = _contract
    improve_recipe: Callable = run_improve_recipe
    gate: Callable = _gate
    train: Callable = _train
    score: Callable = _score


class Loop:
    def __init__(self, cfg, ctx, kit, repo, *, max_nodes: int, phases: Phases | None = None,
                 root_cache: Path | None = None) -> None:
        self.cfg, self.ctx, self.kit, self.repo, self.max_nodes = cfg, ctx, kit, repo, int(max_nodes)
        self.root_cache = root_cache            # None: always score the root
        self.phases = phases or Phases()
        self.nodes = NodeStore(ctx.conn)
        self.graceful = threading.Event()
        self.state_path = Path(ctx.run_dir) / "control" / "state.json"
        self.env = PhaseEnv(cfg=cfg, run_dir=ctx.run_dir, run_id=Path(ctx.run_dir).name,
                            recorder=ctx.recorder, registry=kit.registry, queue=kit.queue, repo=repo,
                            socket_dir=kit.socket_dir, gpus=ctx.gpus, default_model=kit.default_model)

    # -- bookkeeping -------------------------------------------------------------------------
    def _state(self, node: str | None, phase: str, attempt: int = 0) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"node": node, "phase": phase, "attempt": attempt, "since": time.time()}))
        tmp.replace(self.state_path)

    def _children(self) -> int:
        """Nodes counted toward max_nodes: every child except `interrupted` ones (user decision)."""
        return sum(1 for n in self.nodes.all() if n["parent_id"] is not None and n["status"] != "interrupted")

    def _next_id(self) -> str:
        """From the archive, not telemetry: rows are never deleted (interrupted nodes stay), and a
        kill between nodes.create and the node.created event must not hand out the id again."""
        nums = [int(n["node_id"][1:]) for n in self.nodes.all()
                if n["node_id"].startswith("n") and n["node_id"][1:].isdigit()]
        return f"n{max(nums, default=0) + 1}"

    def _check_budget(self) -> None:
        why = self.kit.budget.exhausted()
        if why:
            alert(self.ctx.recorder, "budget", why)
            raise StopRun(why)

    def _check_failed_attempt(self) -> None:
        """After a failed agent attempt: a spent budget (the gateway's 402) or an LLM provider outage
        is not the agent's fault, so it stops the run instead of charging the attempt (user decision
        2026-09-27, option A: a stop, never a pause)."""
        self._check_budget()
        rate, calls = self.kit.budget.error_rate(60 * float(self.cfg.get("gateway.outage_window_min")))
        if calls >= int(self.cfg.get("gateway.outage_min_calls")) and \
                rate >= float(self.cfg.get("gateway.outage_error_rate")):
            why = f"LLM provider outage: {rate:.0%} of the last {calls} calls failed upstream; resume later"
            alert(self.ctx.recorder, "llm_outage", why)
            raise StopRun(why)

    def _node_dir(self, node: str) -> Path:
        return Path(self.ctx.run_dir) / "nodes" / node

    # -- the run -----------------------------------------------------------------------------
    def run(self) -> str:
        self.ensure_root()
        while True:
            if self.graceful.is_set():
                return "graceful stop"
            if self._children() >= self.max_nodes:
                return "max_nodes reached"
            try:
                self._check_budget()
                self.cycle()
            except StopRun as exc:
                return str(exc)

    def ensure_root(self) -> None:
        """Create and score the root of a new run. A run whose root never got a score is not
        resumable (cli refuses it), so a resumed run always finds a scored root here."""
        try:
            if self.nodes.get("root")["status"] == "scored":
                return
        except KeyError:
            pass
        self.nodes.create("root", None, 0)
        commit = self.repo.resolve(self.repo.branch_ref("root")) or \
            self.repo.init(self.cfg.repo_root / "seed_agent")
        self.nodes.set_fields("root", agent_commit=commit)
        self._state("root", "eval")
        cache = self.root_cache / root_key(self.ctx) if self.root_cache else None
        if cache is not None and (cache / "root.json").exists():
            self._reuse_root(cache)
            return
        try:
            score, detail = self._eval("root", None, None)
        except Exception as exc:                # ends the run; the run cannot be resumed
            error = f"{type(exc).__name__}: {exc}"
            self.nodes.set_status("root", "eval_failed")
            self.nodes.set_fields("root", error=error)
            alert(self.ctx.recorder, "root_failed", error, node_id="root")
            raise
        record_root_counts(self.ctx, detail["counts"])     # every later node must match the root's
        self._record_score("root", score, detail)
        if cache is not None:
            save_root(cache, self._node_dir("root") / "eval", detail["metrics"], self.ctx.expected_n,
                      Path(self.ctx.run_dir).name)

    def _reuse_root(self, cache: Path) -> None:
        """The root is the unedited seed agent's model: scoring it again would repeat hours of eval."""
        saved = json.loads((cache / "root.json").read_text())
        shutil.copytree(cache / "eval", self._node_dir("root") / "eval", copy_function=os.link,
                        ignore=shutil.ignore_patterns("aggregates.json"))   # rewritten below, never through a link
        record_root_counts(self.ctx, saved["expected_n"])
        # The cache holds metric means, not a score: each run weighs them its own way.
        score = weighted_score(saved["metrics"], self.cfg.get("eval.score_weights"))
        self._record_score("root", score, {"metrics": saved["metrics"], "aggregates": saved["aggregates"]})
        self.ctx.recorder.event("root.reused", node="root", phase="eval",
                                payload={"cache": str(cache), "scored_by_run": saved["run_id"]})

    def cycle(self) -> None:
        child = self._next_id()
        self._state(child, "select")
        parent_id = select_parent(self.ctx.conn, self.cfg, child,
                                  selection_seed(Path(self.ctx.run_dir).name, child), self.ctx.recorder)
        parent = self.nodes.get(parent_id)
        self.nodes.create(child, parent_id, parent["depth"] + 1)
        self.ctx.recorder.event("node.created", payload={"node": child, "parent": parent_id}, child=child)
        started, status, error = time.time(), "scored", None
        try:
            code, error = self._edit(child, parent)
            if code is None:
                status = "invalid_code"
                return
            trained, status, error = self._recipe(child, parent, code)
            if trained is None:
                return
            checkpoint, resolved = trained
            self._state(child, "eval")
            try:
                score, detail = self._eval(child, checkpoint, resolved)
            except Exception as exc:                              # noqa: BLE001 -- LoRA concat/render/WBench
                status, error = "eval_failed", f"{type(exc).__name__}: {exc}"
                return
            self._record_score(child, score, detail)
        except StopRun:
            status = None                                         # left running: interrupted on resume
            raise
        except Exception as exc:                                  # noqa: BLE001
            status, error = "crashed", f"{type(exc).__name__}: {exc}"
        except BaseException:                                     # ForceStop / KeyboardInterrupt
            status = None                                         # left running: interrupted on resume
            raise
        finally:
            if status is not None:
                self._finish(child, status, error, started)

    def _eval(self, node: str, checkpoint, resolved):
        """Score a node; a failed eval is run once more before it counts (a judge or GPU hiccup
        must not cost a trained node)."""
        try:
            return self.phases.score(self, node, checkpoint, resolved)
        except Exception as exc:                                  # noqa: BLE001 -- LoRA concat/render/WBench
            alert(self.ctx.recorder, "eval_retry", f"{node}: {type(exc).__name__}: {exc}; evaluating once more",
                  level="warning", node_id=node)
        return self.phases.score(self, node, checkpoint, resolved)

    # -- steps 3-4 ---------------------------------------------------------------------------
    def _edit(self, child: str, parent: dict) -> tuple[str | None, str | None]:
        n = int(self.cfg.get("retries.edit_self"))
        base, retry, prev_ws = parent["agent_commit"], None, None
        for k in range(1, n + 1):
            self._check_budget()
            self._state(child, "edit_self", k)
            out = self.phases.edit_self(self.env, conn=self.ctx.conn, node=child, parent_id=parent["node_id"],
                                        base_commit=base, runner_commit=parent["agent_commit"], attempt=k,
                                        max_attempts=n, retry=retry,
                                        nodes_remaining=self.max_nodes - self._children(),
                                        previous_workspace=prev_ws)
            base = out.commit or base
            prev_ws = Path(out.attempt_dir) / "workspace"
            if not out.ok:
                self._check_failed_attempt()
                retry = {"kind": "edit_self", "error": out.error}
                self.nodes.add_attempt(child, "edit_self", k, "failed", retry)
                continue
            self._state(child, "contract", k)
            report = self.phases.contract(loop=self, node=child, parent_id=parent["node_id"],
                                          commit=out.commit, attempt=k)
            if not report.ok:
                retry = report.to_retry()
                self.nodes.add_attempt(child, "edit_self", k, "contract_failed", retry)
                continue
            self.nodes.add_attempt(child, "edit_self", k, "passed", {"commit": out.commit})
            self.repo.set_ref(self.repo.branch_ref(child), out.commit)
            self._node_dir(child).mkdir(parents=True, exist_ok=True)
            (self._node_dir(child) / "edit.json").write_text(json.dumps(out.result, indent=1))
            self.nodes.set_fields(child, agent_commit=out.commit, edit_component=out.result.get("component"))
            return out.commit, None
        return None, f"edit_self retries exhausted; last failure: {json.dumps(retry)[:2000]}"

    # -- steps 5-7 ---------------------------------------------------------------------------
    def _recipe(self, child: str, parent: dict, code: str):
        n = int(self.cfg.get("retries.improve_recipe"))
        retry, prev_ws, status = None, None, "invalid_recipe"
        for k in range(1, n + 1):
            self._check_budget()
            self._state(child, "improve_recipe", k)
            out = self.phases.improve_recipe(self.env, conn=self.ctx.conn, node=child,
                                             parent_id=parent["node_id"], agent_commit=code, attempt=k,
                                             max_attempts=n, retry=retry,
                                             nodes_remaining=self.max_nodes - self._children(),
                                             previous_workspace=prev_ws)
            adir = Path(out.attempt_dir)
            prev_ws = adir / "workspace"
            if not out.ok:
                self._check_failed_attempt()
                retry, status = {"kind": "improve_recipe", "error": out.error}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "failed", retry)
                continue
            res = out.result
            self._state(child, "gate", k)
            gate = self.phases.gate(self, res["recipe"], res["data_commit"], parent["data_commit"], child, adir)
            if not gate.ok:
                retry, status = {"kind": "gate", "failures": gate.failures, **_submitted(res)}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "gate_failed", retry)
                continue
            self._record_recipe(child, res, gate.resolved_path)
            self._state(child, "train", k)
            trained = self.phases.train(self, gate.resolved_path, child, adir)
            if trained.checkpoint is None or trained.failure != "none":   # e.g. nan loss after a checkpoint
                retry = {"kind": "train", "failure": trained.failure, "detail": trained.detail,
                         "log_tail": file_tail(trained.log_path, 20_000), **_submitted(res)}
                status = "train_failed"
                self.nodes.add_attempt(child, "improve_recipe", k, "train_failed", retry)
                alert(self.ctx.recorder, "train_failed", f"{child} attempt {k}: {trained.failure} "
                      f"{trained.detail}".strip(), level="warning", node_id=child)
                continue
            self.nodes.add_attempt(child, "improve_recipe", k, "passed",
                                   {"checkpoint": run_rel(self.ctx.run_dir, trained.checkpoint)})
            self.nodes.set_fields(child, checkpoint_path=run_rel(self.ctx.run_dir, trained.checkpoint))
            return (trained.checkpoint, gate.resolved_path), "scored", None
        # The node's error line: what failed, without the log tail and the submission (both are in its files).
        cause = {k: v for k, v in retry.items() if k not in ("log_tail", "data_commit", "recipe", "rationale")}
        return None, status, f"improve_recipe retries exhausted; last failure: {json.dumps(cause)[:2000]}"

    def _record_recipe(self, child: str, res: dict, resolved: Path) -> None:
        d = self._node_dir(child)
        d.mkdir(parents=True, exist_ok=True)
        (d / "recipe.yaml").write_text(yaml.safe_dump(res["recipe"], sort_keys=True))
        (d / "rationale.md").write_text(res["rationale"])
        rank, alpha = lora_of(resolved)
        self.nodes.set_fields(
            child, data_commit=res["data_commit"],
            recipe_hash=hashlib.sha256(json.dumps(res["recipe"], sort_keys=True).encode()).hexdigest(),
            recipe_path=run_rel(self.ctx.run_dir, d / "recipe.yaml"),
            rationale_path=run_rel(self.ctx.run_dir, d / "rationale.md"),
            resolved_config_path=run_rel(self.ctx.run_dir, resolved), lora_rank=rank, lora_alpha=alpha)

    # -- steps 8-9 ---------------------------------------------------------------------------
    def _record_score(self, node: str, score: float, detail: dict) -> None:
        self.nodes.record_score(node, score, self.ctx.metric_set, detail["metrics"])
        agg = self._node_dir(node) / "eval" / "aggregates.json"
        agg.parent.mkdir(parents=True, exist_ok=True)
        agg.write_text(json.dumps(detail["aggregates"], indent=1))
        update_values(self.ctx.conn, self.cfg)

    def _finish(self, node: str, status: str, error: str | None, started: float) -> None:
        if status != "scored":
            self.nodes.set_status(node, status)
            alert(self.ctx.recorder, "node_failed", f"{node} ended {status}: {error}", node_id=node)
        counts = {p: len(self.nodes.attempts(node, p)) for p in ("edit_self", "improve_recipe")}
        self.nodes.set_fields(node, error=error, attempt_counts=json.dumps(counts),
                              phase_timings=json.dumps({"total_s": time.time() - started}))
        # Only raw downloads go (their source and files are in the hf_download events); every
        # candidate the agent or a GPU job prepared stays, ingested or not.
        for raw in (Path(self.ctx.run_dir) / "staging" / node).glob("*/hf"):
            shutil.rmtree(raw, ignore_errors=True)
        write_transcripts(self.ctx.run_dir, node)
        self.ctx.recorder.event("node.end", payload={"node": node, "status": status, "error": error},
                                child=node, status=status)
        self._state(None, "idle")
