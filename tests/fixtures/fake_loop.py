"""Scripted loop phases (Task 9) and a runnable fake loop for the integration tests: the real
`drive` in a subprocess, with no GPU, LLM or Docker.
Usage: python tests/fixtures/fake_loop.py RUN_DIR MAX_NODES none|pace|<phase>"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import ar_kernel.control
from ar_kernel.agent_phase import PhaseOutcome
from ar_kernel.archive.db import open_db
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractReport
from ar_kernel.control import Control, drive
from ar_kernel.loop import Loop, Phases
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import GateResult
from ar_kernel.train.runner import TrainOutcome
from ar_kernel.vcs.agents_repo import AgentsRepo


def outcome(ok, attempt_dir, result=None, error=None, commit=None):
    return PhaseOutcome(ok, result, error, 0, False, commit, attempt_dir, 1.0)


class Script:
    """Scripted phases: each list is consumed per call; default is success.
    slow: None, a phase name (that phase sleeps up to 600 s) or "pace" (every phase sleeps 1.2 s)."""
    def __init__(self, tmp, edit=(), contract=(), recipe=(), gate=(), train=(), score=(), slow=None):
        self.tmp, self.q = tmp, {k: list(v) for k, v in dict(edit=edit, contract=contract, recipe=recipe,
                                                               gate=gate, train=train, score=score).items()}
        self.retries, self.slow = [], slow

    def _next(self, key, default):
        return self.q[key].pop(0) if self.q[key] else default

    def _wait(self, phase):
        if self.slow == "pace":            # 6 phases: ~7 s per node, well over the 2 s stop-file poll
            time.sleep(1.2)
        elif self.slow == phase:
            for _ in range(1200):                    # short steps: a signal is handled within 0.5 s
                time.sleep(0.5)

    def edit_self(self, env, *, node, base_commit, attempt, retry, **kw):
        self.retries.append(("edit_self", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"edit_self-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        self._wait("edit_self")
        ok = self._next("edit", True)
        return outcome(ok, d, {"summary": "s", "component": "prompts"} if ok else None,
                       None if ok else "agent failed", commit=base_commit)

    def contract(self, *, node, attempt, **kw):
        self._wait("contract")
        return ContractReport(ok=self._next("contract", True))

    def improve_recipe(self, env, *, node, attempt, retry, **kw):
        self.retries.append(("improve_recipe", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"improve_recipe-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        self._wait("improve_recipe")
        ok = self._next("recipe", True)
        return outcome(ok, d, {"data_commit": "c" * 64, "recipe": {"optimizer.max_steps": 2},
                               "rationale": "why"} if ok else None, None if ok else "no commit")

    def gate(self, loop, recipe, data_commit, parent_commit, node, attempt_dir):
        self._wait("gate")
        if not self._next("gate", True):
            return GateResult(ok=False, failures=["dataset d has 1 clips, fewer than 4 GPUs"])
        resolved = attempt_dir / "train_config.yaml"
        resolved.write_text("lora: {rank: 16, alpha: 16}\n")
        return GateResult(ok=True, resolved_path=resolved)

    def train(self, loop, resolved, node, attempt_dir):
        log = attempt_dir / "train" / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA out of memory\n")
        self._wait("train")
        if not self._next("train", True):
            return TrainOutcome(None, "recipe", log, detail="CUDA OOM")
        ck = attempt_dir / "train" / "outputs" / "checkpoint-2"
        ck.mkdir(parents=True)
        return TrainOutcome(ck, "none", log)

    def score(self, loop, node, checkpoint, resolved):
        self._wait("score")
        s = self._next("score", 0.8)
        if isinstance(s, Exception):
            raise s
        return s, {"metrics": {"m": s}, "aggregates": {"metrics": {"m": s}}}

    def phases(self):
        return Phases(edit_self=self.edit_self, contract=self.contract, improve_recipe=self.improve_recipe,
                      gate=self.gate, train=self.train, score=self.score)


SCENARIO = dict(contract=[False, True], gate=[False, True],           # n1: both retry loops, then scored
                edit=[True, True, False, False, False],               # n2: invalid_code (after n1's 2 edits)
                train=[True, False, False, False],                    # n3: train_failed
                score=[0.78, 0.80, RuntimeError("wbench gpu failed")])  # root, n1, n4 -> eval_failed


def build(run: Path, max_nodes: int, script):
    rec = Recorder(run)
    ctx = RunContext(run_dir=run, conn=open_db(run), recorder=rec, gpus=[0, 1, 2, 3],
                     metric_set=["m"], case_ids=["1"], versions={})
    kit = SimpleNamespace(registry=None, queue=None, gpu_lock=threading.Lock(), budget=Budget(),
                          harness=None, socket_dir=run / "sock", default_model="mock-model",
                          start=lambda: None, stop=lambda: None)
    return Loop(KernelConfig.load(), ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=max_nodes,
                phases=script.phases()), kit


def main() -> None:
    run, max_nodes, slow = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ar_kernel.control.kill_run_containers = lambda run_id, node=None: []    # no Docker in tests
    script = Script(run, slow=None if slow == "none" else slow, **(SCENARIO if slow == "none" else {}))
    loop, kit = build(run, max_nodes, script)
    drive(loop, kit, Control(run), loop.ctx.recorder)


if __name__ == "__main__":
    main()
