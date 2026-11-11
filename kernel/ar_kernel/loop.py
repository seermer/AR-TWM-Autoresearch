"""The loop: one node at a time. Nothing pauses: a failure the agent can act on costs an attempt, and the
next one starts clean. A phase that fails otherwise runs once more after a cleanup; failing again stops
the run (KernelFault, or any other exception), or ends the node when either failure was a NodeFault."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from . import fs, telemetry
from .agent_phase import run_edit_self, run_improve_recipe
from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.db import transaction
from .archive.nodes import NOT_SHOWN, NodeStore, run_rel
from .faults import KernelFault, NodeFault, disk_fault
from .process_digest import attempt_process
from .isolation import audit, host_roots, scrub_train_files
from .eval.score import weighted_score
from .run import key_of, record_root_counts, root_key_fields, score_node
from .selection import select_parent, selection_seed, update_values
from .train.gate import resolve_config
from .train.recipe import lora_of
from .tools.hf_tools import clear_hf_tmp
from .tools.jobs import attempt_jobs_dir
from .train.runner import TrainRunner


def save_root(cache: Path, eval_dir: Path, metrics: dict, expected_n: dict, run_id: str) -> None:
    """Keep a scored root for later runs with the same root_key (files hard-linked, not copied)."""
    tmp = cache.with_name(cache.name + ".tmp")
    fs.rmtree(tmp)
    fs.copytree(eval_dir, tmp / "eval", link=True)
    fs.write_text(tmp / "root.json", json.dumps({
        "metrics": metrics, "expected_n": expected_n, "run_id": run_id,
        "aggregates": json.loads((eval_dir / "aggregates.json").read_text())}, indent=1))
    if (cache / "root.json").exists():          # another run with the same key saved its root meanwhile
        fs.rmtree(tmp)
    else:
        fs.rename(tmp, cache)


NATURAL_END = "max_nodes reached"             # the one ending with exit code 0


class StopRun(Exception):
    """End the run now; the node it leaves running starts its stopped phase again on resume."""


class Quarantined(Exception):
    """An agent phase reached for the evaluation set: the node ends here, with no retry."""


def _resolve(loop, recipe: dict, data_commit: str, node: str, attempt_dir: Path) -> Path:
    """The training config of an accepted snapshot (the gate passed it at submit): the data commit's view,
    and the recipe over the run's base recipe."""
    conn = loop.ctx.conn
    store = CommitStore(conn, BlobStore(loop.ctx.run_dir, conn), ClipStore(conn))
    return resolve_config(loop.cfg, store, loop.ctx.run_dir, recipe, data_commit, node, attempt_dir)


def _attempt_of(node: dict) -> int:
    """The improve_recipe attempt whose recipe the node recorded."""
    return int(re.search(r"improve_recipe-(\d+)", node["resolved_config_path"]).group(1))


def _remove(dirs: list[Path]) -> None:
    for path in dirs:
        fs.rmtree(path)


def _train(loop, resolved, node, attempt_dir) -> Path:
    runner = TrainRunner(loop.cfg)
    with loop.kit.gpu_lock:
        runner.precache(resolved, loop.ctx.gpus)
        return runner.train(resolved, loop.ctx.gpus, attempt_dir)


def _score(loop, node, checkpoint, resolved):
    rank = lora_of(resolved)[0] if resolved is not None else 0
    with loop.kit.gpu_lock:
        return score_node(loop.cfg, loop.ctx, node, checkpoint, rank)


@dataclass
class Phases:
    edit_self: Callable = run_edit_self
    improve_recipe: Callable = run_improve_recipe
    resolve: Callable = _resolve
    train: Callable = _train
    score: Callable = _score


def _scored(result, sp):
    """An eval try's output: its score."""
    sp.output = {"score": result[0]}
    return result


class Loop:
    def __init__(self, cfg, ctx, kit, repo, *, max_nodes: int, phases: Phases | None = None,
                 root_cache: Path | None = None, upload: Callable | None = None) -> None:
        """`upload(node, phase, attempt)`: called after every phase (HfUpload)."""
        self.cfg, self.ctx, self.kit, self.repo, self.max_nodes = cfg, ctx, kit, repo, int(max_nodes)
        self.upload = upload
        self._current: tuple | None = None      # the phase running, for the upload when it ends
        self.root_cache = root_cache            # None: always score the root
        self.phases = phases or Phases()
        self.nodes = NodeStore(ctx.conn)
        self.graceful = threading.Event()
        self.state_path = Path(ctx.run_dir) / "control" / "state.json"
        self._node_faults = 0                   # nodes in a row ended by a NodeFault
        self.env = kit.env

    # -- bookkeeping -------------------------------------------------------------------------
    def _state(self, node: str | None, phase: str, attempt: int = 0) -> None:
        """A phase ends when the state moves on: its upload first."""
        if self._current is not None and self._current != (node, phase, attempt):
            self._phase_ended()
        self._current = None if phase in ("select", "idle") else (node, phase, attempt)
        fs.mkdir(self.state_path.parent)
        fs.write_text(self.state_path, json.dumps({"node": node, "phase": phase, "attempt": attempt,
                                                   "since": time.time()}), atomic=True)

    def _phase_ended(self) -> None:
        current, self._current = self._current, None
        if self.upload is not None:
            self.upload(*current)

    def _children(self) -> int:
        """Nodes counted toward max_nodes: every child."""
        return sum(1 for n in self.nodes.all() if n["parent_id"] is not None)

    def _next_id(self) -> str:
        """From the archive: an id a killed run created is never handed out again."""
        nums = [int(n["node_id"][4:]) for n in self.nodes.all()
                if n["node_id"].startswith("node") and n["node_id"][4:].isdigit()]
        return f"node{max(nums, default=0) + 1:04d}"

    def _check_budget(self) -> None:
        budget = self.kit.budget
        why = budget.exhausted()
        telemetry.decision("loop.budget", {"usd": budget.usd, "max_usd": budget.max_usd}, ["go on", "stop the run"],
                           "stop the run" if why else "go on", why or "budget left")
        if why:
            telemetry.alert("loop.budget", "error", why)
            raise StopRun(why)

    def _node_dir(self, node: str) -> Path:
        return Path(self.ctx.run_dir) / "nodes" / node

    def _audit(self, node: str, phase: str, attempt: int) -> None:
        hits = audit(self.ctx.conn, node, phase, attempt)
        telemetry.decision("phase.audit", {"node": node, "phase": phase, "attempt": attempt, "hits": hits},
                           ["go on", "quarantine"], "quarantine" if hits else "go on",
                           "the model wrote audited names" if hits else "no audit hits")
        if hits:
            raise Quarantined(f"the model wrote {', '.join(hits)}")

    # -- phase retry -------------------------------------------------------------------------
    def _phase(self, node: str, phase: str, attempt: int, run: Callable, cleanup: Callable, settle: Callable,
               node_fault_status: str | None = None):
        """Each try is a `phase.attempt` span: `run()`, then `settle(result, span)`, which sets the span's output.
        After a failure that is not the agent's, `cleanup()` and one more try. A second failure raises StopRun, or
        NodeFault(status=node_fault_status) when either failure was a NodeFault."""
        first = None
        for t in (1, 2):
            sp = telemetry.start("phase.attempt", {"node": node, "phase": phase, "attempt": attempt, "try": t})
            try:
                with telemetry.use(sp):
                    try:
                        result = run()
                    except (StopRun, Quarantined):
                        raise
                    except Exception as exc:                      # noqa: BLE001 -- retried, then stops the run
                        self._failed(node, phase, attempt, exc, first, cleanup, node_fault_status)
                        sp.end("error", None, exc)
                        first = exc
                        continue
                    result = settle(result, sp)
            except BaseException as exc:
                sp.fail(exc)
                raise
            sp.end("ok", sp.output)
            return result

    def _failed(self, node: str, phase: str, attempt: int, exc: Exception, first: Exception | None,
                cleanup: Callable, node_fault_status: str | None) -> None:
        """A try failed: the decision; returns after `cleanup()` when the phase runs once more, otherwise raises."""
        hits = audit(self.ctx.conn, node)                 # a phase that raised was not audited
        why = f"{node} {phase}{f' attempt {attempt}' if attempt else ''}: {type(exc).__name__}: {exc}"
        node_fault = isinstance(first, NodeFault) or isinstance(exc, NodeFault)
        chosen, reason = (("quarantine", f"the model wrote {', '.join(hits)}") if hits else
                          ("retry", "its first failure: clean up and run it once more") if first is None else
                          ("node fault", "failed twice, once by the node's fault") if node_fault else
                          ("stop the run", "failed twice, neither by the node's fault"))
        telemetry.error(exc, chosen)
        telemetry.decision("phase.failure", {"node": node, "phase": phase, "attempt": attempt, "error": why,
                                             "first": None if first is None else repr(first), "hits": hits},
                           ["retry", "quarantine", "node fault", "stop the run"], chosen, reason)
        if hits:
            raise Quarantined(f"the model wrote {', '.join(hits)}") from exc
        where = {"node": node, "phase": phase, "attempt": attempt}
        if first is None:
            telemetry.alert("loop.phase_retry", "warning", f"{why}; running the phase once more", where)
            cleanup()
            return
        if node_fault:
            raise NodeFault(f"{why} (failed twice)", node_fault_status) from exc
        telemetry.alert("loop.phase_failed", "error", f"{why}; failed twice", where)
        raise StopRun(f"{why} (failed twice; fix it, then resume)") from exc

    def _clean_recipe(self, node: str, attempt: int) -> None:
        """What an abandoned improve_recipe attempt made outside its folders (which `_run` clears)."""
        conn, run_dir = self.ctx.conn, self.ctx.run_dir
        ClipStore(conn).remove_ingested(node, attempt)
        CommitStore(conn, None, None).remove(node, attempt)
        BlobStore(run_dir, conn).remove_unreferenced()
        clear_hf_tmp(run_dir)
        _remove([attempt_jobs_dir(run_dir, node, "improve_recipe", attempt)])

    def _clean_train(self, node: str, attempt: int, attempt_dir: Path) -> None:
        _remove([attempt_dir / "train" / "outputs", attempt_dir / "train" / "logs", attempt_dir / "dataset_cache"])

    def _clean_eval(self, node: str) -> None:
        _remove([self._node_dir(node) / "eval"])

    # -- the run -----------------------------------------------------------------------------
    def run(self) -> str:
        with telemetry.span("loop", {"max_nodes": self.max_nodes}) as sp:
            try:
                why = self._run()
            except StopRun as exc:
                telemetry.error(exc, "the loop's stop reason")
                why = str(exc)
            if self._current is not None:           # the phase a StopRun ended
                self._phase_ended()
            sp.output = {"reason": why}
        return why

    def _run(self) -> str:
        self.ensure_root()
        self.resume()                           # before the stop, max_nodes and budget checks: it is one of the nodes
        while True:
            with telemetry.span("loop.iteration") as sp:
                stop = self.graceful.is_set()
                telemetry.decision("loop.graceful", {}, ["go on", "stop"], "stop" if stop else "go on",
                                   "a stop was requested" if stop else "no stop request")
                if stop:
                    sp.output = {"stop": "graceful stop"}
                    return "graceful stop"
                children = self._children()
                telemetry.decision("loop.max_nodes", {"children": children, "max_nodes": self.max_nodes},
                                   ["go on", "stop"], "stop" if children >= self.max_nodes else "go on",
                                   "max_nodes reached" if children >= self.max_nodes else "below max_nodes")
                if children >= self.max_nodes:
                    sp.output = {"stop": NATURAL_END}
                    return NATURAL_END
                self._check_budget()
                sp.output = {"node": self.cycle()}

    def ensure_root(self) -> None:
        with telemetry.span("root.ensure") as sp:
            self._ensure_root()
            sp.output = {"status": self.nodes.get("root")["status"]}

    def _ensure_root(self) -> None:
        """Create and score the root of a new run; a root a stopped run left unscored is evaluated again."""
        try:
            root = self.nodes.get("root")
        except KeyError:
            root = None
        fields = root_key_fields(self.ctx, self.cfg) if self.root_cache else None
        cache = self.root_cache / key_of(fields) if fields is not None else None
        reasons = {"scored": "the root is scored", "reuse": "a root with the same key is cached",
                   "evaluate": "the run has no root yet", "evaluate again": "a stopped run left the root unscored"}
        chosen = ("scored" if root is not None and root["status"] == "scored" else
                  "reuse" if cache is not None and (cache / "root.json").exists() else
                  "evaluate" if root is None else "evaluate again")
        telemetry.decision("root.how", {"root": root, "cache": None if cache is None else str(cache), "key_fields": fields},
                           list(reasons), chosen, reasons[chosen])
        if chosen == "scored":
            return
        if root is None:
            self.nodes.create("root", None, 0)
        else:
            self.nodes.set_status("root", "running")
            self._clean_eval("root")
        commit = self.repo.resolve(self.repo.branch_ref("root")) or \
            self.repo.init(self.cfg.repo_root / "seed_agent")
        self.nodes.set_fields("root", agent_commit=commit)
        self._state("root", "eval")
        if chosen == "reuse":
            self._reuse_root(cache)
            return
        try:
            score, detail = self._phase("root", "eval", 0, lambda: self.phases.score(self, "root", None, None),
                                        lambda: self._clean_eval("root"), _scored)
        except (StopRun, NodeFault) as exc:     # ends the run; resume evaluates the root again
            error = str(exc)
            self.nodes.set_status("root", "eval_failed")
            self.nodes.set_fields("root", error=error)
            telemetry.alert("loop.root_failed", "error", error)
            raise StopRun(f"the root could not be evaluated: {error}") from exc
        record_root_counts(self.ctx, detail["counts"])     # every later node must match the root's
        self._record_score("root", score, detail)
        if cache is not None:
            save_root(cache, self._node_dir("root") / "eval", detail["metrics"], self.ctx.expected_n,
                      Path(self.ctx.run_dir).name)

    def _reuse_root(self, cache: Path) -> None:
        """The root is the unedited seed agent's model: scoring it again would repeat hours of eval."""
        saved = json.loads((cache / "root.json").read_text())
        fs.copytree(cache / "eval", self._node_dir("root") / "eval", link=True,
                    ignore=shutil.ignore_patterns("aggregates.json"))   # rewritten below, never through a link
        record_root_counts(self.ctx, saved["expected_n"])
        # The cache holds metric means, not a score: each run weighs them its own way.
        score = weighted_score(saved["metrics"], self.cfg.get("eval.score_weights"))
        self._record_score("root", score, {"metrics": saved["metrics"], "aggregates": saved["aggregates"]})

    def resume(self) -> None:
        """The node a stopped run left running starts again at the phase it stopped in, from scratch (after
        the cleanup a phase retry does); a restarted attempt keeps its number."""
        running = [n for n in self.nodes.all() if n["parent_id"] is not None and n["status"] == "running"]
        if len(running) > 1:
            raise RuntimeError(f"more than one node was left running: {[n['node_id'] for n in running]}")
        if not running:
            return
        node = running[0]
        phase, k, why = self._resume_point(node)
        telemetry.decision("run.resume_point", {"node": node, "attempts": {
            p: [a["idx"] for a in self.nodes.attempts(node["node_id"], p)] for p in ("edit_self", "improve_recipe")}},
                           ["edit_self", "improve_recipe", "train", "eval"],
                           {"node": node["node_id"], "phase": phase, "attempt": k}, why)
        nid, run_dir = node["node_id"], Path(self.ctx.run_dir)
        if phase == "improve_recipe":
            self._clean_recipe(nid, k)
        elif phase == "train":
            self._clean_train(nid, k, (run_dir / node["resolved_config_path"]).parent)
        elif phase == "eval":
            self._clean_eval(nid)
        telemetry.alert("loop.node_resumed", "warning", f"{nid} starts again at {phase}" + (f" attempt {k}" if k else ""))
        self.cycle(node, phase, k)

    def _resume_point(self, node: dict) -> tuple[str, int, str]:
        """Where `node` stopped, from its records, and why: eval once it has a checkpoint; train k when attempt
        k's recipe is recorded but has no attempt row; otherwise the next edit_self or improve_recipe attempt."""
        nid = node["node_id"]
        if node["checkpoint_path"]:
            return "eval", 0, "it has a checkpoint"
        if node["agent_commit"] is None:
            return "edit_self", len(self.nodes.attempts(nid, "edit_self")) + 1, "it has no agent commit"
        done = {a["idx"] for a in self.nodes.attempts(nid, "improve_recipe")}
        if node["resolved_config_path"] and _attempt_of(node) not in done:
            return "train", _attempt_of(node), "its recipe is resolved and that attempt has no row"
        return "improve_recipe", len(done) + 1, "it has an agent commit and no recipe left to train"

    def cycle(self, resumed: dict | None = None, phase: str = "edit_self", k: int = 1) -> str:
        """One node: a new one, or `resumed` from `phase` attempt `k`; its id."""
        child = self._next_id() if resumed is None else resumed["node_id"]
        with telemetry.span("node", {"node": child, "resumed": resumed is not None, "phase": phase, "attempt": k}) as sp:
            sp.output = {"status": self._cycle(child, resumed, phase, k)}
        return child

    def _cycle(self, child: str, resumed: dict | None, phase: str, k: int) -> str:
        """The node's final status."""
        if resumed is None:
            self._state(child, "select")
            parent_id = select_parent(self.ctx.conn, self.cfg, child, selection_seed(Path(self.ctx.run_dir).name, child))
            parent = self.nodes.get(parent_id)
            self.nodes.create(child, parent_id, parent["depth"] + 1)
        else:
            parent = self.nodes.get(resumed["parent_id"])
        started, status, error, scored = time.time(), "scored", None, None
        run_dir = Path(self.ctx.run_dir)
        try:
            if phase == "edit_self":
                code, error = self._edit(child, parent, start=k)
                if code is None:
                    status = "invalid_code"
                    return status
                phase, k = "improve_recipe", 1
            else:
                code = resumed["agent_commit"]
            if phase == "eval":
                checkpoint, resolved = run_dir / resumed["checkpoint_path"], run_dir / resumed["resolved_config_path"]
            else:
                trained, status, error = self._recipe(
                    child, parent, code, start=k,
                    resolved=run_dir / resumed["resolved_config_path"] if phase == "train" else None)
                if trained is None:
                    return status
                checkpoint, resolved = trained
            self._state(child, "eval")
            scored = self._phase(child, "eval", 0, lambda: self._evaluate(child, checkpoint, resolved),
                                 lambda: self._clean_eval(child), _scored, "eval_failed")
        except Quarantined as exc:
            telemetry.error(exc, "the node ends quarantined")
            status, error = "quarantined", f"quarantined: {exc}"
        except NodeFault as exc:
            status, error = exc.status, str(exc)
            self._node_faults += 1
            stop = self._node_faults >= 2
            telemetry.decision("node.fault", {"node": child, "error": error, "in_a_row": self._node_faults},
                               ["the node ends", "stop the run"], "stop the run" if stop else "the node ends",
                               "two nodes in a row ended by a fault" if stop else "the first node in a row ended by a fault")
            if stop:
                why = f"two nodes in a row ended by a fault, the last: {exc}; fix it, then resume"
                telemetry.alert("loop.node_faults", "error", why)
                raise StopRun(why) from exc
        except BaseException:                                     # ForceStop / KeyboardInterrupt, or a loop error
            status = None                                         # left running: restarted on resume
            raise
        finally:
            if status is not None:
                self._finish(child, status, error, started, scored)
        return status

    def _evaluate(self, node: str, checkpoint, resolved):
        """A node's eval failing is the node's (its LoRA renders or scores badly), unless a disk or kernel
        fault says otherwise; the root's always stops the run."""
        try:
            return self.phases.score(self, node, checkpoint, resolved)
        except (KernelFault, StopRun, Quarantined):
            raise
        except Exception as exc:                                  # noqa: BLE001 -- LoRA concat/render/WBench
            disk = disk_fault(exc)
            telemetry.decision("eval.failure", {"node": node, "error": f"{type(exc).__name__}: {exc}"},
                               ["node fault", "disk fault"], "disk fault" if disk else "node fault",
                               "a disk error is the machine's" if disk else "the node's LoRA renders or scores badly")
            if disk:
                raise
            raise NodeFault(f"the evaluation failed: {type(exc).__name__}: {exc}") from exc

    # -- steps 3-4 ---------------------------------------------------------------------------
    def _last_failure(self, child: str, phase: str) -> dict | None:
        """A resumed node's last failed attempt of `phase`, for its error if no attempt is left."""
        failed = [a["detail"] for a in self.nodes.attempts(child, phase) if a["outcome"] == "failed"]
        return failed[-1] if failed else None

    def _edit(self, child: str, parent: dict, start: int = 1) -> tuple[str | None, str | None]:
        # Root runs no roles, so before any other node is shown there is no evidence to edit from.
        skip = not any(n["node_id"] != "root" and n["status"] not in NOT_SHOWN for n in self.nodes.all())
        telemetry.decision("edit_self.skip", {"node": child}, ["run edit_self", "skip"], "skip" if skip else "run edit_self",
                           "no node other than the root is shown yet" if skip else "a node other than the root is shown")
        if skip:
            self.repo.set_ref(self.repo.branch_ref(child), parent["agent_commit"])
            self.nodes.set_fields(child, agent_commit=parent["agent_commit"])
            return parent["agent_commit"], None
        n = int(self.cfg.get("retries.edit_self"))
        failure = self._last_failure(child, "edit_self")
        for k in range(start, n + 1):
            self._check_budget()

            def attempt(k=k):
                self._state(child, "edit_self", k)
                out = self.phases.edit_self(self.env, conn=self.ctx.conn, node=child, parent_id=parent["node_id"],
                                            agent_commit=parent["agent_commit"], attempt=k)
                self._audit(child, "edit_self", k)
                return out

            def settle(out, sp, k=k):
                nonlocal failure
                if not out.ok:
                    sp.output = {"outcome": "failed"}
                    self._check_budget()                     # a spent budget is not the agent's: stop, not charge
                    failure = {"kind": "edit_self", "error": out.error,
                               "process": attempt_process(self.ctx.conn, child, "edit_self", k)}
                    self.nodes.add_attempt(child, "edit_self", k, "failed", failure)
                    return out
                sp.output = {"outcome": "passed"}
                self.repo.set_ref(self.repo.branch_ref(child), out.commit)
                fs.mkdir(self._node_dir(child))
                fs.write_text(self._node_dir(child) / "edit.json", json.dumps(out.result, indent=1))
                self.nodes.set_fields(child, agent_commit=out.commit)                # first: resume reads it
                self.nodes.add_attempt(child, "edit_self", k, "passed", {"commit": out.commit})
                return out
            out = self._phase(child, "edit_self", k, attempt, lambda: None, settle)
            if out.ok:
                return out.commit, None
        cause = {key: v for key, v in failure.items() if key != "process"}
        return None, f"edit_self retries exhausted; last failure: {json.dumps(cause)[:2000]}"

    # -- steps 5-7 ---------------------------------------------------------------------------
    def _recipe(self, child: str, parent: dict, code: str, start: int = 1, resolved: Path | None = None):
        """improve_recipe attempts from `start` until one trains: an attempt that continues ends a subphase, and
        the next attempt is the next subphase. `resolved`: attempt `start`'s recipe is already recorded (a resumed
        node starts at its training)."""
        if resolved is not None:
            return self._train(child, start, resolved)
        n = int(self.cfg.get("retries.improve_recipe"))
        failure = self._last_failure(child, "improve_recipe")
        failures = sum(a["outcome"] == "failed" for a in self.nodes.attempts(child, "improve_recipe"))
        k = start
        while failures < n:
            self._check_budget()

            def attempt(k=k):
                self._state(child, "improve_recipe", k)
                out = self.phases.improve_recipe(self.env, conn=self.ctx.conn, node=child,
                                                 parent_id=parent["node_id"], agent_commit=code, attempt=k)
                self._audit(child, "improve_recipe", k)
                if not out.ok or not out.result["train"]:
                    return out, None
                return out, self.phases.resolve(self, out.result["recipe"], out.result["data_commit"], child,
                                                Path(out.attempt_dir))

            def settle(result, sp, k=k):
                nonlocal failure
                out, resolved = result
                if not out.ok:
                    sp.output = {"outcome": "failed"}
                    self._check_budget()                     # a spent budget is not the agent's: stop, not charge
                    failure = {"kind": "improve_recipe", "error": out.error,
                               "process": attempt_process(self.ctx.conn, child, "improve_recipe", k)}
                    with transaction(self.ctx.conn):
                        self.nodes.add_attempt(child, "improve_recipe", k, "failed", failure)
                        self._commits.fail(child, k)
                        ClipStore(self.ctx.conn).hide(child, f"fail{k}", k)
                elif not out.result["train"]:
                    sp.output = {"outcome": "continued"}
                    with transaction(self.ctx.conn):
                        self.nodes.add_attempt(child, "improve_recipe", k, "continued", {
                            "snapshot_id": out.result["snapshot_id"], "summary": out.result["summary"]})
                        self._commits.settle(child, k, out.result["data_commit"])
                else:
                    sp.output = {"outcome": "train"}
                    self._record_recipe(child, k, out.result, resolved)
                return result
            out, resolved = self._phase(child, "improve_recipe", k, attempt, lambda k=k: self._clean_recipe(child, k),
                                        settle)
            if not out.ok:
                failures, k = failures + 1, k + 1
            elif not out.result["train"]:
                k += 1
            else:
                return self._train(child, k, resolved)
        with transaction(self.ctx.conn):                  # the continued subphases' commits and clips
            self._commits.fail(child)
            ClipStore(self.ctx.conn).hide(child, f"fail{k - 1}")
        cause = {key: v for key, v in failure.items() if key != "process"}
        return None, "invalid_recipe", f"improve_recipe retries exhausted; last failure: {json.dumps(cause)[:2000]}"

    def _train(self, child: str, k: int, resolved: Path):
        """Train attempt `k`'s recorded recipe (`resolved` lies in the attempt's folder)."""
        adir = resolved.parent

        def settle(checkpoint, sp):
            self.nodes.set_fields(child, checkpoint_path=run_rel(self.ctx.run_dir, checkpoint))  # first: resume reads it
            self.nodes.add_attempt(child, "improve_recipe", k, "passed",
                                   {"checkpoint": run_rel(self.ctx.run_dir, checkpoint)})
            sp.output = {"checkpoint": run_rel(self.ctx.run_dir, checkpoint)}
            return checkpoint
        self._state(child, "train", k)
        checkpoint = self._phase(child, "train", k, lambda: self.phases.train(self, resolved, child, adir),
                                 lambda: self._clean_train(child, k, adir), settle, "train_failed")
        return (checkpoint, resolved), "scored", None

    def _record_recipe(self, child: str, k: int, res: dict, resolved: Path) -> None:
        d = self._node_dir(child)
        fs.mkdir(d)
        fs.write_text(d / "recipe.yaml", yaml.safe_dump(res["recipe"], sort_keys=True))
        fs.write_text(d / "rationale.md", res["rationale"])
        fs.write_text(d / "summary.md", res["summary"])
        rank, alpha = lora_of(resolved)
        with transaction(self.ctx.conn):
            self.nodes.set_fields(
                child, data_commit=res["data_commit"],
                resolved_config_path=run_rel(self.ctx.run_dir, resolved), lora_rank=rank, lora_alpha=alpha)
            self._commits.settle(child, k, res["data_commit"])

    # -- steps 8-9 ---------------------------------------------------------------------------
    def _record_score(self, node: str, score: float, detail: dict) -> None:
        self._node_faults = 0
        self.nodes.record_score(node, score, self.ctx.metric_set, detail["metrics"])
        agg = self._node_dir(node) / "eval" / "aggregates.json"
        fs.mkdir(agg.parent)
        fs.write_text(agg, json.dumps(detail["aggregates"], indent=1))
        update_values(self.ctx.conn, self.cfg)

    @property
    def _commits(self) -> CommitStore:
        return CommitStore(self.ctx.conn, None, None)

    def _finish(self, node: str, status: str, error: str | None, started: float, scored: tuple | None) -> None:
        """`scored`: (score, detail) of a scored node."""
        with telemetry.span("node.finish", {"node": node}) as sp:
            telemetry.decision("node.status", {"node": node, "error": error},
                               ["scored", "invalid_code", "invalid_recipe", "train_failed", "eval_failed", "quarantined"],
                               status, error or "it was scored")
            if scored is not None:
                self._record_score(node, *scored)
            self._end_node(node, status, error, started)
            sp.output = {"status": self.nodes.get(node)["status"]}

    def _end_node(self, node: str, status: str, error: str | None, started: float) -> None:
        if status == "quarantined":
            with transaction(self.ctx.conn):
                self._commits.fail(node)
                ClipStore(self.ctx.conn).hide(node, "quarantine")
                self.nodes.set_status(node, status)
        elif status != "scored":
            self.nodes.set_status(node, status)
        if status != "scored":
            telemetry.alert("loop.node_failed", "error", f"{node} ended {status}: {error}")
        counts = {p: len(self.nodes.attempts(node, p)) for p in ("edit_self", "improve_recipe")}
        self.nodes.set_fields(node, error=error, attempt_counts=json.dumps(counts),
                              phase_timings=json.dumps({"total_s": time.time() - started}))
        # Only raw downloads go (their source and files are in the hf_download events); every
        # candidate the agent or a GPU job prepared stays, ingested or not.
        for raw in (Path(self.ctx.run_dir) / "staging" / node).glob("*/hf"):
            fs.rmtree(raw, ignore_errors="raw downloads of an ended node; a leftover only takes disk")
        scrub_train_files(self._node_dir(node), self.cfg.get("isolation.blocked_names") or [], host_roots(self.cfg))
        self._state(None, "idle")
