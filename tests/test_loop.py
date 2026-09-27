import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ar_kernel.agent_phase import PhaseOutcome
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractReport
from ar_kernel.loop import Loop, Phases
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.gate import GateResult
from ar_kernel.train.runner import TrainOutcome
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
SEED = Path(__file__).resolve().parents[1] / "seed_agent"


def outcome(ok, attempt_dir, result=None, error=None, commit=None):
    return PhaseOutcome(ok, result, error, 0, False, commit, attempt_dir, 1.0)


class Script:
    """Scripted phases: each list is consumed per call; default is success."""
    def __init__(self, tmp, edit=(), contract=(), recipe=(), gate=(), train=(), score=()):
        self.tmp, self.q = tmp, {k: list(v) for k, v in dict(edit=edit, contract=contract, recipe=recipe,
                                                               gate=gate, train=train, score=score).items()}
        self.retries = []

    def _next(self, key, default):
        return self.q[key].pop(0) if self.q[key] else default

    def edit_self(self, env, *, node, base_commit, attempt, retry, **kw):
        self.retries.append(("edit_self", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"edit_self-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        ok = self._next("edit", True)
        return outcome(ok, d, {"summary": "s", "component": "prompts"} if ok else None,
                       None if ok else "agent failed", commit=base_commit)

    def contract(self, *, node, attempt, **kw):
        return ContractReport(ok=self._next("contract", True))

    def improve_recipe(self, env, *, node, attempt, retry, **kw):
        self.retries.append(("improve_recipe", node, attempt, retry))
        d = self.tmp / "nodes" / node / "attempts" / f"improve_recipe-{attempt}"
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        ok = self._next("recipe", True)
        return outcome(ok, d, {"data_commit": "c" * 64, "recipe": {"optimizer.max_steps": 2},
                               "rationale": "why"} if ok else None, None if ok else "no commit")

    def gate(self, loop, recipe, data_commit, parent_commit, node, attempt_dir):
        if not self._next("gate", True):
            return GateResult(ok=False, failures=["dataset d has 1 clips, fewer than 4 GPUs"])
        resolved = attempt_dir / "train_config.yaml"
        resolved.write_text("lora: {rank: 16, alpha: 16}\n")
        return GateResult(ok=True, resolved_path=resolved)

    def train(self, loop, resolved, node, attempt_dir):
        log = attempt_dir / "train" / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA out of memory\n")
        if not self._next("train", True):
            return TrainOutcome(None, "recipe", log, detail="CUDA OOM")
        ck = attempt_dir / "train" / "outputs" / "checkpoint-2"
        ck.mkdir(parents=True)
        return TrainOutcome(ck, "none", log)

    def score(self, loop, node, checkpoint, resolved):
        s = self._next("score", 0.8)
        if isinstance(s, Exception):
            raise s
        return s, {"metrics": {"m": s}, "aggregates": {"metrics": {"m": s}}}

    def phases(self):
        return Phases(edit_self=self.edit_self, contract=self.contract, improve_recipe=self.improve_recipe,
                      gate=self.gate, train=self.train, score=self.score)


@pytest.fixture
def make_loop(tmp_path):
    run = tmp_path / "run"

    def make(script, max_nodes=1, budget=None):
        rec = Recorder(run)
        ctx = RunContext(run_dir=run, conn=open_db(run), recorder=rec, gpus=[0, 1, 2, 3],
                         metric_set=["m"], case_ids=["1"], versions={})
        kit = SimpleNamespace(registry=None, queue=None, gpu_lock=threading.Lock(),
                              budget=budget or Budget(), harness=None, socket_dir=tmp_path / "sock",
                              default_model="mock-model")
        return Loop(CFG, ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=max_nodes,
                    phases=script.phases())
    return run, make


def test_root_then_one_scored_child(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])                      # root 0.7, n1 0.9
    loop = make(script, max_nodes=1)
    assert loop.run() == "max_nodes reached"
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert nodes["root"]["status"] == "scored" and nodes["n1"]["status"] == "scored"
    assert nodes["n1"]["edit_component"] == "prompts" and nodes["n1"]["checkpoint_path"].startswith("nodes/n1/")
    assert json.loads((run / "nodes" / "n1" / "edit.json").read_text())["summary"] == "s"
    assert (run / "nodes" / "n1" / "recipe.yaml").exists() and (run / "nodes" / "n1" / "rationale.md").exists()
    assert json.loads((run / "nodes" / "n1" / "eval" / "aggregates.json").read_text())["metrics"]["m"] == 0.9
    assert nodes["root"]["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.5 * 0.9) / 1.5)
    assert not (run / "staging" / "n1").exists()


def test_contract_failure_retries_with_the_report(make_loop):
    run, make = make_loop
    script = Script(run, contract=[False, True])
    make(script).run()
    edits = [r for r in script.retries if r[0] == "edit_self"]
    assert [a for _, _, a, _ in edits] == [1, 2] and edits[1][3]["kind"] == "contract"


def test_edit_exhaustion_is_invalid_code(make_loop):
    run, make = make_loop
    loop = make(Script(run, edit=[False, False, False]))
    loop.run()
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "invalid_code" and "agent failed" in n1["error"]


def test_training_failure_goes_back_to_the_agent_then_train_failed(make_loop):
    run, make = make_loop
    script = Script(run, train=[False, False, False])
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert [r[3]["kind"] if r[3] else None for r in recipes] == [None, "train", "train"]
    assert "CUDA out of memory" in recipes[1][3]["log_tail"]
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "train_failed"


def test_gate_failure_then_success(make_loop):
    run, make = make_loop
    script = Script(run, gate=[False, True])
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert recipes[1][3] == {"kind": "gate", "failures": ["dataset d has 1 clips, fewer than 4 GPUs"]}
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "scored"


def test_eval_failure_is_eval_failed_and_the_loop_continues(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("wbench gpu failed")]), max_nodes=2)
    loop.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert nodes["n1"]["status"] == "eval_failed" and "wbench gpu failed" in nodes["n1"]["error"]
    assert nodes["n2"]["status"] == "scored"
    kinds = [e["kind"] for e in loop.ctx.recorder.read_events() if e["type"] == "alert"]
    assert "node_failed" in kinds


def test_unexpected_exception_is_crashed(make_loop):
    run, make = make_loop
    script = Script(run)
    script.gate = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
    loop = make(script, max_nodes=1)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "crashed"


def test_provider_outage_stops_the_run_instead_of_failing_the_node(make_loop):
    run, make = make_loop
    budget = Budget()
    script = Script(run, edit=[False, False, False])
    real_edit = script.edit_self

    def outage(env, **kw):
        for _ in range(3):
            budget.record(503, None)                    # what the gateway records during an outage
        return real_edit(env, **kw)
    loop = make(script, max_nodes=3, budget=budget)
    loop.phases.edit_self = outage
    assert "outage" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"      # -> interrupted on resume
    assert NodeStore(loop.ctx.conn).attempts("n1", "edit_self") == []     # the attempt was not charged


def test_agent_side_400s_are_not_an_outage(make_loop):
    run, make = make_loop
    budget = Budget()
    script = Script(run, edit=[False, True])
    real_edit = script.edit_self

    def bad_requests(env, **kw):
        for _ in range(3):
            budget.record(400, None)
        return real_edit(env, **kw)
    loop = make(script, max_nodes=1, budget=budget)
    loop.phases.edit_self = bad_requests
    assert loop.run() == "max_nodes reached"


def test_budget_spent_during_the_last_attempt_is_a_stop_not_invalid_code(make_loop):
    run, make = make_loop
    budget = Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})
    script = Script(run, edit=[False, False, False])
    real_edit = script.edit_self

    def spend_on_third(env, **kw):
        if kw["attempt"] == 3:
            budget.record(200, {"prompt_tokens": 5, "completion_tokens": 0})
        return real_edit(env, **kw)
    loop = make(script, max_nodes=3, budget=budget)
    loop.phases.edit_self = spend_on_third
    assert "budget" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"


def test_budget_exhaustion_stops_the_run_and_leaves_the_node_running(make_loop):
    run, make = make_loop
    budget = Budget(max_usd=1e-6, prices={"input": 1, "cached_input": 1, "output": 1})
    script = Script(run)
    real_edit = script.edit_self

    def spend(env, **kw):
        budget.record(200, {"prompt_tokens": 5, "completion_tokens": 0})
        return real_edit(env, **kw)
    script.edit_self = spend
    loop = make(script, max_nodes=3, budget=budget)
    assert "budget" in loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "running"


def test_graceful_flag_finishes_the_current_node(make_loop):
    run, make = make_loop
    script = Script(run)
    loop = make(script, max_nodes=5)
    real_score = script.score

    def score_and_stop(*a, **k):
        loop.graceful.set()
        return real_score(*a, **k)
    script.score = score_and_stop
    loop.phases.score = score_and_stop
    assert loop.run() == "graceful stop"
    # root's score call sets the flag, so no child is started
    assert [n["node_id"] for n in NodeStore(loop.ctx.conn).all()] == ["root"]


def test_interrupted_nodes_do_not_count_and_ids_are_never_reused(make_loop):
    run, make = make_loop
    loop = make(Script(run), max_nodes=1)
    loop.run()
    NodeStore(loop.ctx.conn).set_status("n1", "interrupted")   # as resume marks an unfinished node
    loop2 = make(Script(run), max_nodes=1)
    loop2.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop2.ctx.conn).all()}
    assert {k: v["status"] for k, v in nodes.items()} == {"root": "scored", "n1": "interrupted", "n2": "scored"}
    assert nodes["n2"]["parent_id"] == "root"                  # an interrupted node is never a parent
    assert (run / "nodes" / "n1" / "edit.json").exists()       # and keeps its files
