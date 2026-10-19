"""The loop: one node at a time. Nothing pauses: a failure the agent can act on costs an attempt, and the
next one starts clean. A phase that fails otherwise runs once more after a cleanup; failing again stops
the run (KernelFault, or any other exception), or ends the node when either failure was a NodeFault."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .agent_phase import run_edit_self, run_improve_recipe
from .archive.blobs import BlobStore
from .archive.clips import ClipStore, hide_clips
from .archive.commits import CommitStore
from .archive.nodes import NOT_SHOWN, NodeStore, run_rel
from .faults import KernelFault, NodeFault, disk_fault
from .guards import alert
from .process_digest import attempt_process
from .isolation import audit, host_roots, scrub_train_files
from .eval.score import weighted_score
from .run import record_root_counts, root_key, score_node
from .selection import select_parent, selection_seed, update_values
from .train.gate import resolve_config
from .train.recipe import lora_of
from .tools.hf_tools import clear_hf_tmp
from .tools.jobs import attempt_jobs_dir
from .train.runner import TrainRunner
from .transcripts import write_transcripts


def save_root(cache: Path, eval_dir: Path, metrics: dict, expected_n: dict, run_id: str) -> None:
    """Keep a scored root for later runs with the same root_key (files hard-linked, not copied)."""
    tmp = cache.with_name(cache.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(eval_dir, tmp / "eval", copy_function=os.link)
    (tmp / "root.json").write_text(json.dumps({
        "metrics": metrics, "expected_n": expected_n, "run_id": run_id,
        "aggregates": json.loads((eval_dir / "aggregates.json").read_text())}, indent=1))
    if (cache / "root.json").exists():          # another run with the same key saved its root meanwhile
        shutil.rmtree(tmp)
    else:
        tmp.rename(cache)


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


def _train(loop, resolved, node, attempt_dir) -> Path:
    runner = TrainRunner(loop.cfg, loop.ctx.recorder)
    with loop.kit.gpu_lock:
        runner.precache(resolved, loop.ctx.gpus, node)
        return runner.train(resolved, loop.ctx.gpus, node, attempt_dir)


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


class Loop:
    def __init__(self, cfg, ctx, kit, repo, *, max_nodes: int, phases: Phases | None = None,
                 root_cache: Path | None = None) -> None:
        self.cfg, self.ctx, self.kit, self.repo, self.max_nodes = cfg, ctx, kit, repo, int(max_nodes)
        self.root_cache = root_cache            # None: always score the root
        self.phases = phases or Phases()
        self.nodes = NodeStore(ctx.conn)
        self.graceful = threading.Event()
        self.state_path = Path(ctx.run_dir) / "control" / "state.json"
        self._node_faults = 0                   # nodes in a row ended by a NodeFault
        self.env = kit.env

    # -- bookkeeping -------------------------------------------------------------------------
    def _state(self, node: str | None, phase: str, attempt: int = 0) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"node": node, "phase": phase, "attempt": attempt, "since": time.time()}))
        tmp.replace(self.state_path)

    def _process(self, node: str, phase: str, attempt: int) -> dict:
        """The failed attempt's process, for the record."""
        return attempt_process(self.ctx.run_dir, node, phase, attempt)

    def _children(self) -> int:
        """Nodes counted toward max_nodes: every child."""
        return sum(1 for n in self.nodes.all() if n["parent_id"] is not None)

    def _next_id(self) -> str:
        """From the archive, not telemetry: a kill between nodes.create and the node.created event must
        not hand out the id again."""
        nums = [int(n["node_id"][4:]) for n in self.nodes.all()
                if n["node_id"].startswith("node") and n["node_id"][4:].isdigit()]
        return f"node{max(nums, default=0) + 1:04d}"

    def _check_budget(self) -> None:
        why = self.kit.budget.exhausted()
        if why:
            alert(self.ctx.recorder, "budget", why)
            raise StopRun(why)

    def _node_dir(self, node: str) -> Path:
        return Path(self.ctx.run_dir) / "nodes" / node

    def _audit(self, node: str, phase: str, attempt: int) -> None:
        hits = audit(self.cfg, self.ctx.recorder, node, phase, attempt)
        if hits:
            self.ctx.recorder.event("isolation.audit", node=node, phase=phase, attempt=attempt, payload={"hits": hits})
            raise Quarantined(f"the model wrote {', '.join(hits)}")

    # -- phase retry -------------------------------------------------------------------------
    def _phase(self, node: str, phase: str, attempt: int, run: Callable, cleanup: Callable,
               node_fault_status: str | None = None):
        """`run()`; after a failure that is not the agent's, `cleanup()` and `run()` once more. A second
        failure raises StopRun, or NodeFault(status=node_fault_status) when either failure was a NodeFault."""
        first = None
        while True:
            try:
                return run()
            except (StopRun, Quarantined):
                raise
            except Exception as exc:                              # noqa: BLE001 -- retried, then stops the run
                hits = audit(self.cfg, self.ctx.recorder, node)  # a phase that raised was not audited
                if hits:
                    raise Quarantined(f"the model wrote {', '.join(hits)}") from exc
                why = f"{node} {phase}{f' attempt {attempt}' if attempt else ''}: {type(exc).__name__}: {exc}"
                if first is None:
                    first = exc
                    alert(self.ctx.recorder, "phase_retry", f"{why}; running the phase once more", level="warning",
                          node_id=node, traceback=traceback.format_exc())
                    cleanup()
                    continue
                if isinstance(first, NodeFault) or isinstance(exc, NodeFault):
                    raise NodeFault(f"{why} (failed twice)", node_fault_status) from exc
                alert(self.ctx.recorder, "phase_failed", f"{why}; failed twice", node_id=node,
                      traceback=traceback.format_exc())
                raise StopRun(f"{why} (failed twice; fix it, then resume)") from exc

    def _clean_recipe(self, node: str, attempt: int) -> None:
        """What an abandoned improve_recipe attempt made outside its folders (which `_run` clears)."""
        conn, run_dir = self.ctx.conn, self.ctx.run_dir
        clips = ClipStore(conn).remove_ingested(node)
        commits = CommitStore(conn, None, None).remove(node, attempt)
        blobs = BlobStore(run_dir, conn).remove_unreferenced()
        self._clean_dirs(node, "improve_recipe", attempt, [attempt_jobs_dir(run_dir, node, "improve_recipe", attempt)],
                         clips=clips, data_commits=commits, blobs=blobs, hf_tmp=clear_hf_tmp(run_dir))

    def _clean_dirs(self, node: str, phase: str, attempt: int, dirs: list[Path], **payload) -> None:
        removed = [p for p in dirs if p.exists()]
        for path in removed:
            shutil.rmtree(path)
        self.ctx.recorder.event("phase.cleanup", node=node, phase=phase, attempt=attempt,
                                payload={**payload, "removed": [str(p) for p in removed]})

    def _clean_train(self, node: str, attempt: int, attempt_dir: Path) -> None:
        self._clean_dirs(node, "train", attempt, [attempt_dir / "train" / "outputs", attempt_dir / "train" / "logs",
                                                  attempt_dir / "dataset_cache"])

    def _clean_eval(self, node: str) -> None:
        self._clean_dirs(node, "eval", 0, [self._node_dir(node) / "eval"])

    # -- the run -----------------------------------------------------------------------------
    def run(self) -> str:
        try:
            self.ensure_root()
            self.resume()                       # before the stop, max_nodes and budget checks: it is one of the nodes
            while True:
                if self.graceful.is_set():
                    return "graceful stop"
                if self._children() >= self.max_nodes:
                    return "max_nodes reached"
                self._check_budget()
                self.cycle()
        except StopRun as exc:
            return str(exc)

    def ensure_root(self) -> None:
        """Create and score the root of a new run; a root a stopped run left unscored is evaluated again."""
        try:
            root = self.nodes.get("root")
        except KeyError:
            root = None
        if root is not None and root["status"] == "scored":
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
        cache = self.root_cache / root_key(self.ctx, self.cfg) if self.root_cache else None
        if cache is not None and (cache / "root.json").exists():
            self._reuse_root(cache)
            return
        try:
            score, detail = self._phase("root", "eval", 0, lambda: self.phases.score(self, "root", None, None),
                                        lambda: self._clean_eval("root"))
        except (StopRun, NodeFault) as exc:     # ends the run; resume evaluates the root again
            error = str(exc)
            self.nodes.set_status("root", "eval_failed")
            self.nodes.set_fields("root", error=error)
            alert(self.ctx.recorder, "root_failed", error, node_id="root")
            raise StopRun(f"the root could not be evaluated: {error}") from exc
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

    def resume(self) -> None:
        """The node a stopped run left running starts again at the phase it stopped in, from scratch (after
        the cleanup a phase retry does); a restarted attempt keeps its number."""
        running = [n for n in self.nodes.all() if n["parent_id"] is not None and n["status"] == "running"]
        if len(running) > 1:
            raise RuntimeError(f"more than one node was left running: {[n['node_id'] for n in running]}")
        if not running:
            return
        node = running[0]
        phase, k = self._resume_point(node)
        nid, run_dir = node["node_id"], Path(self.ctx.run_dir)
        if phase == "improve_recipe":
            self._clean_recipe(nid, k)
        elif phase == "train":
            self._clean_train(nid, k, (run_dir / node["resolved_config_path"]).parent)
        elif phase == "eval":
            self._clean_eval(nid)
        alert(self.ctx.recorder, "node_resumed", f"{nid} starts again at {phase}" + (f" attempt {k}" if k else ""),
              level="warning", node_id=nid)
        self.cycle(node, phase, k)

    def _resume_point(self, node: dict) -> tuple[str, int]:
        """Where `node` stopped, from its records: eval once it has a checkpoint; train k when attempt k's
        recipe is recorded but has no attempt row; otherwise the next edit_self or improve_recipe attempt."""
        nid = node["node_id"]
        if node["checkpoint_path"]:
            return "eval", 0
        if node["agent_commit"] is None:
            return "edit_self", len(self.nodes.attempts(nid, "edit_self")) + 1
        done = {a["idx"] for a in self.nodes.attempts(nid, "improve_recipe")}
        if node["resolved_config_path"] and _attempt_of(node) not in done:
            return "train", _attempt_of(node)
        return "improve_recipe", len(done) + 1

    def cycle(self, resumed: dict | None = None, phase: str = "edit_self", k: int = 1) -> None:
        """One node: a new one, or `resumed` from `phase` attempt `k`."""
        if resumed is None:
            child = self._next_id()
            self._state(child, "select")
            parent_id = select_parent(self.ctx.conn, self.cfg, child,
                                      selection_seed(Path(self.ctx.run_dir).name, child), self.ctx.recorder)
            parent = self.nodes.get(parent_id)
            self.nodes.create(child, parent_id, parent["depth"] + 1)
            self.ctx.recorder.event("node.created", payload={"node": child, "parent": parent_id}, child=child)
        else:
            child, parent = resumed["node_id"], self.nodes.get(resumed["parent_id"])
        started, status, error, hide = time.time(), "scored", None, None
        self._recipe_attempt = _attempt_of(resumed) if phase == "eval" else 0 if phase == "edit_self" else k
        run_dir = Path(self.ctx.run_dir)
        try:
            if phase == "edit_self":
                code, error = self._edit(child, parent, start=k)
                if code is None:
                    status = "invalid_code"
                    return
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
                    return
                checkpoint, resolved = trained
            self._state(child, "eval")
            score, detail = self._phase(child, "eval", 0, lambda: self._evaluate(child, checkpoint, resolved),
                                        lambda: self._clean_eval(child), "eval_failed")
            self._record_score(child, score, detail)
        except StopRun:
            status = None                                         # left running: restarted on resume
            raise
        except Quarantined as exc:
            status, error = "quarantined", f"quarantined: {exc}"
        except NodeFault as exc:
            status, error = exc.status, str(exc)
            hide = f"fail{self._recipe_attempt}" if self._recipe_attempt else None
            self._node_faults += 1
            if self._node_faults >= 2:
                why = f"two nodes in a row ended by a fault, the last: {exc}; fix it, then resume"
                alert(self.ctx.recorder, "node_faults", why, node_id=child)
                raise StopRun(why) from exc
        except BaseException:                                     # ForceStop / KeyboardInterrupt, or a loop error
            status = None                                         # left running: restarted on resume
            raise
        finally:
            if status is not None:
                self._finish(child, status, error, started, hide)

    def _evaluate(self, node: str, checkpoint, resolved):
        """A node's eval failing is the node's (its LoRA renders or scores badly), unless a disk or kernel
        fault says otherwise; the root's always stops the run."""
        try:
            return self.phases.score(self, node, checkpoint, resolved)
        except (KernelFault, StopRun, Quarantined):
            raise
        except Exception as exc:                                  # noqa: BLE001 -- LoRA concat/render/WBench
            if disk_fault(exc):
                raise
            raise NodeFault(f"the evaluation failed: {type(exc).__name__}: {exc}") from exc

    # -- steps 3-4 ---------------------------------------------------------------------------
    def _last_failure(self, child: str, phase: str) -> dict | None:
        """A resumed node's last failed attempt of `phase`, for its error if no attempt is left."""
        failed = [a["detail"] for a in self.nodes.attempts(child, phase) if a["outcome"] == "failed"]
        return failed[-1] if failed else None

    def _edit(self, child: str, parent: dict, start: int = 1) -> tuple[str | None, str | None]:
        if not any(n["node_id"] != "root" and n["status"] not in NOT_SHOWN for n in self.nodes.all()):
            # Root runs no roles, so before any other node is shown there is no evidence to edit from.
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
            out = self._phase(child, "edit_self", k, attempt, lambda: None)
            if not out.ok:
                self._check_budget()                     # a spent budget is not the agent's: stop, not charge
                failure = {"kind": "edit_self", "error": out.error, "process": self._process(child, "edit_self", k)}
                self.nodes.add_attempt(child, "edit_self", k, "failed", failure)
                continue
            self.repo.set_ref(self.repo.branch_ref(child), out.commit)
            self._node_dir(child).mkdir(parents=True, exist_ok=True)
            (self._node_dir(child) / "edit.json").write_text(json.dumps(out.result, indent=1))
            self.nodes.set_fields(child, agent_commit=out.commit)                # first: resume reads it
            self.nodes.add_attempt(child, "edit_self", k, "passed", {"commit": out.commit})
            return out.commit, None
        cause = {key: v for key, v in failure.items() if key != "process"}
        return None, f"edit_self retries exhausted; last failure: {json.dumps(cause)[:2000]}"

    # -- steps 5-7 ---------------------------------------------------------------------------
    def _recipe(self, child: str, parent: dict, code: str, start: int = 1, resolved: Path | None = None):
        """improve_recipe attempts from `start`, then training; `resolved`: attempt `start`'s recipe is
        already recorded (a resumed node starts at its training)."""
        n = int(self.cfg.get("retries.improve_recipe"))
        failure = self._last_failure(child, "improve_recipe")
        for k in range(start, n + 1):
            self._recipe_attempt = k
            if resolved is not None:
                return self._train(child, k, resolved)
            self._check_budget()

            def attempt(k=k):
                self._state(child, "improve_recipe", k)
                out = self.phases.improve_recipe(self.env, conn=self.ctx.conn, node=child,
                                                 parent_id=parent["node_id"], agent_commit=code, attempt=k)
                self._audit(child, "improve_recipe", k)
                if not out.ok:
                    return out, None
                return out, self.phases.resolve(self, out.result["recipe"], out.result["data_commit"], child,
                                                Path(out.attempt_dir))
            out, resolved = self._phase(child, "improve_recipe", k, attempt, lambda k=k: self._clean_recipe(child, k))
            if not out.ok:
                self._check_budget()                     # a spent budget is not the agent's: stop, not charge
                failure = {"kind": "improve_recipe", "error": out.error,
                           "process": self._process(child, "improve_recipe", k)}
                self.nodes.add_attempt(child, "improve_recipe", k, "failed", failure)
                hide_clips(self.ctx.conn, self.ctx.recorder, child, f"fail{k}")
                continue
            self._record_recipe(child, out.result, resolved)
            return self._train(child, k, resolved)
        cause = {key: v for key, v in failure.items() if key != "process"}
        return None, "invalid_recipe", f"improve_recipe retries exhausted; last failure: {json.dumps(cause)[:2000]}"

    def _train(self, child: str, k: int, resolved: Path):
        """Train attempt `k`'s recorded recipe (`resolved` lies in the attempt's folder)."""
        adir = resolved.parent
        self._state(child, "train", k)
        checkpoint = self._phase(child, "train", k, lambda: self.phases.train(self, resolved, child, adir),
                                 lambda: self._clean_train(child, k, adir), "train_failed")
        self.nodes.set_fields(child, checkpoint_path=run_rel(self.ctx.run_dir, checkpoint))      # first: resume reads it
        self.nodes.add_attempt(child, "improve_recipe", k, "passed", {"checkpoint": run_rel(self.ctx.run_dir, checkpoint)})
        return (checkpoint, resolved), "scored", None

    def _record_recipe(self, child: str, res: dict, resolved: Path) -> None:
        d = self._node_dir(child)
        d.mkdir(parents=True, exist_ok=True)
        (d / "recipe.yaml").write_text(yaml.safe_dump(res["recipe"], sort_keys=True), encoding="utf-8")
        (d / "rationale.md").write_text(res["rationale"], encoding="utf-8")
        (d / "summary.md").write_text(res["summary"], encoding="utf-8")
        rank, alpha = lora_of(resolved)
        self.nodes.set_fields(
            child, data_commit=res["data_commit"],
            recipe_hash=hashlib.sha256(json.dumps(res["recipe"], sort_keys=True).encode()).hexdigest(),
            recipe_path=run_rel(self.ctx.run_dir, d / "recipe.yaml"),
            rationale_path=run_rel(self.ctx.run_dir, d / "rationale.md"),
            resolved_config_path=run_rel(self.ctx.run_dir, resolved), lora_rank=rank, lora_alpha=alpha)

    # -- steps 8-9 ---------------------------------------------------------------------------
    def _record_score(self, node: str, score: float, detail: dict) -> None:
        self._node_faults = 0
        self.nodes.record_score(node, score, self.ctx.metric_set, detail["metrics"])
        agg = self._node_dir(node) / "eval" / "aggregates.json"
        agg.parent.mkdir(parents=True, exist_ok=True)
        agg.write_text(json.dumps(detail["aggregates"], indent=1))
        update_values(self.ctx.conn, self.cfg)

    def _finish(self, node: str, status: str, error: str | None, started: float, hide: str | None = None) -> None:
        """`hide`: the mark for the node's clips when a fault ended it."""
        if status == "quarantined":
            hide_clips(self.ctx.conn, self.ctx.recorder, node, "quarantine")
        elif hide:
            hide_clips(self.ctx.conn, self.ctx.recorder, node, hide)
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
        write_transcripts(self.ctx.run_dir, node, self.ctx.recorder)
        scrub_train_files(self._node_dir(node), self.cfg.get("isolation.blocked_names") or [], host_roots(self.cfg))
        self.ctx.recorder.event("node.end", payload={"node": node, "status": status, "error": error},
                                child=node, status=status)
        self._state(None, "idle")
