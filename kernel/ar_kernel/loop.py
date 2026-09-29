"""The loop (spec 7.1-7.2): one node at a time. Nothing pauses (user decision 2026-09-27): a
failure the agent can act on goes back to it as the next attempt's retry report; any other
failure ends the node with a status and an error that later agents see in the lineage."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .agent_phase import PhaseEnv, run_edit_self, run_improve_recipe
from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.nodes import NodeStore, run_rel
from .contract.verify import verify_contract
from .guards import alert
from .run import score_node
from .selection import select_parent, selection_seed, update_values
from .subproc import file_tail
from .train.gate import Gate
from .train.recipe import lora_of
from .train.runner import TrainOutcome, TrainRunner


class StopRun(Exception):
    """End the run now (budget spent); resume marks the running node `interrupted`."""


def _gate(loop, recipe, data_commit, parent_commit, node, attempt_dir):
    conn = loop.ctx.conn
    store = CommitStore(conn, BlobStore(loop.ctx.run_dir, conn), ClipStore(conn))
    with loop.kit.gpu_lock:
        return Gate(loop.cfg, store, loop.ctx.recorder).check(
            recipe, data_commit, parent_commit, node, attempt_dir, loop.ctx.run_dir, loop.ctx.gpus)


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
    rank, alpha = lora_of(resolved) if resolved is not None else (0, 0)
    with loop.kit.gpu_lock:
        return score_node(loop.cfg, loop.ctx, node, checkpoint, rank, alpha)


@dataclass
class Phases:
    edit_self: Callable = run_edit_self
    contract: Callable = verify_contract
    improve_recipe: Callable = run_improve_recipe
    gate: Callable = _gate
    train: Callable = _train
    score: Callable = _score


class Loop:
    def __init__(self, cfg, ctx, kit, repo, *, max_nodes: int, phases: Phases | None = None) -> None:
        self.cfg, self.ctx, self.kit, self.repo, self.max_nodes = cfg, ctx, kit, repo, int(max_nodes)
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

    def _check_outage(self) -> None:
        """After a failed agent attempt: if the LLM provider is down, stop the run instead of
        charging the attempt (user decision 2026-09-27, option A: a stop, never a pause)."""
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
        try:
            score, detail = self.phases.score(self, "root", None, None)
        except Exception as exc:                # ends the run; the run cannot be resumed
            error = f"{type(exc).__name__}: {exc}"
            self.nodes.set_status("root", "eval_failed")
            self.nodes.set_fields("root", error=error)
            alert(self.ctx.recorder, "root_failed", error, node_id="root")
            raise
        self._record_score("root", score, detail)

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
                score, detail = self.phases.score(self, child, checkpoint, resolved)
            except Exception as exc:                              # noqa: BLE001 -- merge/render/WBench
                status, error = "eval_failed", f"{type(exc).__name__}: {exc}"
                return
            self._record_score(child, score, detail)
        except StopRun:
            status = None                                         # left running: interrupted on resume
            raise
        except Exception as exc:                                  # noqa: BLE001 -- spec 14.2 last row
            status, error = "crashed", f"{type(exc).__name__}: {exc}"
        except BaseException:                                     # ForceStop / KeyboardInterrupt
            status = None                                         # left running: interrupted on resume
            raise
        finally:
            if status is not None:
                self._finish(child, status, error, started)

    # -- steps 3-4 ---------------------------------------------------------------------------
    def _edit(self, child: str, parent: dict) -> tuple[str | None, str | None]:
        n = int(self.cfg.get("retries.edit_self"))
        base, retry, prev_ws = parent["agent_commit"], None, None
        for k in range(1, n + 1):
            self._check_budget()
            self._state(child, "edit_self", k)
            out = self.phases.edit_self(self.env, conn=self.ctx.conn, node=child, parent_id=parent["node_id"],
                                        base_commit=base, attempt=k, max_attempts=n, retry=retry,
                                        nodes_remaining=self.max_nodes - self._children(),
                                        previous_workspace=prev_ws)
            base = out.commit or base
            prev_ws = Path(out.attempt_dir) / "workspace"
            if not out.ok:
                self._check_budget()        # a 402 from the gateway is a spent budget, not an agent failure
                self._check_outage()        # a provider outage is not an agent failure either
                retry = {"kind": "edit_self", "error": out.error}
                self.nodes.add_attempt(child, "edit_self", k, "failed", retry)
                continue
            self._state(child, "contract", k)
            report = self.phases.contract(cfg=self.cfg, run_dir=self.ctx.run_dir,
                                          run_id=Path(self.ctx.run_dir).name, repo=self.repo,
                                          commit=out.commit, harness=self.kit.harness,
                                          recorder=self.ctx.recorder, node=child, attempt=k)
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
                self._check_budget()        # a 402 from the gateway is a spent budget, not an agent failure
                self._check_outage()        # a provider outage is not an agent failure either
                retry, status = {"kind": "improve_recipe", "error": out.error}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "failed", retry)
                continue
            res = out.result
            self._state(child, "gate", k)
            gate = self.phases.gate(self, res["recipe"], res["data_commit"], parent["data_commit"], child, adir)
            if not gate.ok:
                retry, status = {"kind": "gate", "failures": gate.failures}, "invalid_recipe"
                self.nodes.add_attempt(child, "improve_recipe", k, "gate_failed", retry)
                continue
            self._record_recipe(child, res, gate.resolved_path)
            self._state(child, "train", k)
            trained = self.phases.train(self, gate.resolved_path, child, adir)
            if trained.checkpoint is None or trained.failure != "none":   # e.g. nan loss after a checkpoint
                retry = {"kind": "train", "failure": trained.failure, "detail": trained.detail,
                         "log_tail": file_tail(trained.log_path, 4000)}
                status = "train_failed"
                self.nodes.add_attempt(child, "improve_recipe", k, "train_failed", retry)
                alert(self.ctx.recorder, "train_failed", f"{child} attempt {k}: {trained.failure} "
                      f"{trained.detail}".strip(), level="warning", node_id=child)
                continue
            self.nodes.add_attempt(child, "improve_recipe", k, "passed",
                                   {"checkpoint": run_rel(self.ctx.run_dir, trained.checkpoint)})
            self.nodes.set_fields(child, checkpoint_path=run_rel(self.ctx.run_dir, trained.checkpoint))
            return (trained.checkpoint, gate.resolved_path), "scored", None
        return None, status, f"improve_recipe retries exhausted; last failure: {json.dumps(retry)[:2000]}"

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
        # candidate the agent or a GPU job prepared stays, ingested or not (spec 15).
        for raw in (Path(self.ctx.run_dir) / "staging" / node).glob("*/hf"):
            shutil.rmtree(raw, ignore_errors=True)
        self.ctx.recorder.event("node.end", payload={"node": node, "status": status, "error": error},
                                child=node, status=status)
        self._state(None, "idle")
