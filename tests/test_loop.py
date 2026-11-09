import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ar_kernel.archive.activity import add_activity, add_call
from ar_kernel.archive.clips import ClipStore
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.faults import KernelFault, NodeFault
from ar_kernel.loop import Loop
from ar_kernel.run import RunContext
from ar_kernel.vcs.agents_repo import AgentsRepo
from fixtures.fake_loop import Script

CFG = KernelConfig.load()
SEED = Path(__file__).resolve().parents[1] / "seed_agent"


def _model_wrote(run, node, phase, attempt, hits, turn=0, role=None):
    """One finished model call as the gateway records it, `hits` being the audit patterns it found."""
    add_call(run, node=node, phase=phase, attempt=attempt, role=role, tool=None, conversation_id="c", turn=turn,
             transcript=f"transcripts/{phase}-{attempt}/01-{role or 'unlabeled'}.jsonl", status=200,
             tokens=(1, 0, 1), cost_usd=None, mock=False, audit_hits=hits)


def _alerts(run) -> list[tuple[str, str, str]]:
    from conftest import telemetry_records
    return [(r["name"], r["level"], r["message"]) for r in telemetry_records(run) if r["kind"] == "alert"]


def _recorded(run):
    from ar_kernel import telemetry
    telemetry.open(run)
    telemetry.start("ar.run")


@pytest.fixture
def make_loop(tmp_path):
    run = tmp_path / "run"

    def make(script, max_nodes=1, budget=None, upload=None):
        (run / "config").mkdir(parents=True, exist_ok=True)
        (run / "config" / "run.json").write_text("{}")
        ctx = RunContext(run_dir=run, conn=open_db(run), gpus=[0, 1, 2, 3],
                         metric_set=["m"], case_ids=["1"], versions={})
        kit = SimpleNamespace(env=None, gpu_lock=threading.Lock(), budget=budget or Budget(), harness=None)
        return Loop(CFG, ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=max_nodes,
                    phases=script.phases(), upload=upload)
    return run, make


def test_root_then_one_scored_child(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])                      # root 0.7, node0001 0.9
    loop = make(script, max_nodes=1)
    assert loop.run() == "max_nodes reached"
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert nodes["root"]["status"] == "scored" and nodes["node0001"]["status"] == "scored"
    assert nodes["node0001"]["checkpoint_path"].startswith("nodes/node0001/")
    assert (run / "nodes" / "node0001" / "recipe.yaml").exists() and (run / "nodes" / "node0001" / "rationale.md").exists()
    assert (run / "nodes" / "node0001" / "summary.md").read_text() == "tested"
    assert json.loads((run / "nodes" / "node0001" / "eval" / "aggregates.json").read_text())["metrics"]["m"] == 0.9
    assert nodes["root"]["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.5 * 0.9) / 1.5)
    passed = NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")[-1]["detail"]
    assert passed["checkpoint"].startswith("nodes/node0001/")          # run-relative, never absolute


def test_the_first_child_runs_its_parents_code_without_edit_self(make_loop):
    """Root runs no roles: before another node is shown, an edit has no evidence to start from."""
    run, make = make_loop
    script = Script(run)
    loop = make(script, max_nodes=2)                            # node0001 skips edit_self, node0002 edits
    loop.run()
    nodes = NodeStore(loop.ctx.conn)
    root, n1 = nodes.get("root")["agent_commit"], nodes.get("node0001")
    assert n1["status"] == "scored" and n1["agent_commit"] == root
    assert loop.repo.resolve(loop.repo.branch_ref("node0001")) == root
    assert not (run / "nodes" / "node0001" / "edit.json").exists() and nodes.attempts("node0001", "edit_self") == []
    assert [r[:3] for r in script.calls if r[0] == "edit_self"] == [("edit_self", "node0002", 1)]
    assert json.loads((run / "nodes" / "node0002" / "edit.json").read_text())["summary"] == "s"


def test_every_edit_attempt_starts_from_the_parents_code(make_loop):
    run, make = make_loop
    script = Script(run, edit=[False, True])
    loop = make(script, max_nodes=2)
    loop.run()
    nodes = NodeStore(loop.ctx.conn)
    parent = nodes.get(nodes.get("node0002")["parent_id"])["agent_commit"]
    assert [r[3] for r in script.calls if r[0] == "edit_self"] == [parent] * 2


def test_edit_exhaustion_is_invalid_code(make_loop):
    run, make = make_loop
    loop = make(Script(run, edit=[False, False, False]), max_nodes=2)
    loop.run()
    n2 = NodeStore(loop.ctx.conn).get("node0002")
    assert n2["status"] == "invalid_code" and "agent failed" in n2["error"]


def test_a_failed_recipe_attempt_then_success(make_loop):
    run, make = make_loop
    loop = make(Script(run, recipe=[False, True]))
    loop.run()
    assert NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")[0]["detail"] == {
        "kind": "improve_recipe", "error": "no commit",
        "process": {"conversations": [], "snapshots": [], "accepted": None}}
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "scored"


def test_a_failed_attempt_records_its_conversations_and_snapshots(make_loop):
    run, make = make_loop
    script = Script(run, recipe=[False, True])
    loop = make(script)
    for turn in range(2):
        _model_wrote(run, "node0001", "improve_recipe", 1, [], turn=turn, role="recipe_planner")
    add_activity(run, node="node0001", phase="improve_recipe", attempt=1, kind="snapshot", name="s1")
    loop.run()
    first = NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")[0]["detail"]
    assert first["process"] == {"conversations": [{"role": "recipe_planner", "turns": 2}], "snapshots": ["s1"],
                                "accepted": None}
    assert "process" not in (NodeStore(loop.ctx.conn).get("node0001")["error"] or "")


def test_a_failed_eval_is_run_once_more_after_its_cleanup(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("judge down"), 0.9]))
    real = loop.phases.score

    def score(loop_, node, *a):
        if node == "node0001":
            assert not (run / "nodes" / node / "eval").exists()         # the failed try's files are gone
            (run / "nodes" / node / "eval").mkdir(parents=True)
        return real(loop_, node, *a)
    loop.phases.score = score
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["score"] == 0.9


def test_a_known_machine_error_trains_once_more_without_the_agents(make_loop):
    run, make = make_loop
    script = Script(run, train=["infra", True])
    loop = make(script, max_nodes=1)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "scored"
    assert [r[:3] for r in script.calls if r[0] == "improve_recipe"] == [("improve_recipe", "node0001", 1)]


def test_a_second_machine_error_stops_the_run_without_charging_the_attempt(make_loop):
    run, make = make_loop
    script = Script(run, train=["infra", "infra", True])
    loop = make(script, max_nodes=1)
    assert "NCCL error" in loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "running"
    assert NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe") == []


def _raising(script, phase, *errors):
    """`phase` raises the next of `errors` per call, then runs as scripted."""
    real, queue = getattr(script, phase), list(errors)

    def fn(*a, **k):
        if queue:
            raise queue.pop(0)
        return real(*a, **k)
    return fn


def test_a_kernel_fault_once_reruns_the_phase(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script, max_nodes=1)
    loop.phases.improve_recipe = _raising(script, "improve_recipe", KernelFault("sandbox did not start"))
    assert loop.run() == "max_nodes reached"
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "scored"
    assert [r[2] for r in script.calls if r[0] == "improve_recipe"] == [1]       # the rerun keeps attempt 1
    assert NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")[0]["outcome"] == "passed"


def test_a_node_fault_twice_ends_the_node_and_keeps_its_clips_and_trained_commit(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7])
    loop = make(script, max_nodes=1)
    _ingesting(script, loop)
    loop.phases.train = _raising(script, "train", NodeFault("loss=nan"), NodeFault("loss=nan"))
    assert loop.run() == "max_nodes reached"
    n1 = NodeStore(loop.ctx.conn).get("node0001")
    assert n1["status"] == "train_failed" and "loss=nan" in n1["error"]
    assert [c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()] == ["node0001"]
    assert _status(loop, "dc_node0001_1") == "accepted"


def test_a_kernel_then_a_node_fault_ends_the_node(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7])
    loop = make(script, max_nodes=1)
    real = loop.phases.score
    errors = [KernelFault("judge down"), NodeFault("metrics are nan")]

    def score(loop_, node, *a):
        if node != "root" and errors:
            raise errors.pop(0)
        return real(loop_, node, *a)
    loop.phases.score = score
    assert loop.run() == "max_nodes reached"
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "eval_failed"


def test_two_nodes_in_a_row_ended_by_faults_stop_the_run_and_a_score_resets_the_count(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.8])
    loop = make(script, max_nodes=5)
    real = script.train
    # node0001 ends by a fault, node0002 is scored, node0003 and node0004 end by faults
    plan = {"node0001": 2, "node0003": 2, "node0004": 2}

    def train(loop_, resolved, node, attempt_dir):
        if plan.get(node):
            plan[node] -= 1
            raise NodeFault("loss=inf")
        return real(loop_, resolved, node, attempt_dir)
    loop.phases.train = train
    _recorded(run)
    assert "two nodes in a row" in loop.run()
    [(level, message)] = [(lv, m) for n, lv, m in _alerts(run) if n == "loop.node_faults"]
    assert level == "error" and message.startswith("two nodes in a row ended by a fault, the last:")
    statuses = {n["node_id"]: n["status"] for n in NodeStore(loop.ctx.conn).all()}
    assert statuses == {"root": "scored", "node0001": "train_failed", "node0002": "scored",
                        "node0003": "train_failed", "node0004": "train_failed"}


def test_a_retried_recipe_attempt_leaves_nothing_of_the_first_try(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script, max_nodes=1)
    conn = loop.ctx.conn
    ClipStore(conn).add(_clip("old", "node0009"))
    ClipStore(conn).hide("node0009", "fail1")
    job = run / "jobs" / "node0001" / "improve_recipe-1" / "j1"
    real, tries = script.improve_recipe, []

    def improve_recipe(env, *, node, attempt, **kw):
        tries.append(attempt)
        if len(tries) == 1:
            assert ClipStore(conn).add(_clip("old", node)) == "node0009fail1"      # taken from a hidden mark
            ClipStore(conn).add(_clip("new", node))
            conn.execute("INSERT INTO data_commits (commit_id, node_id, attempt, status, manifest, message, created_at) "
                         "VALUES ('c1', ?, 1, 'draft', '{}', 'm', 0)", (node,))
            job.mkdir(parents=True)
            raise KernelFault("the agent sandbox did not start")
        assert {c["clip_id"]: c["ingested_by"] for c in ClipStore(conn).all()} == {"old": "node0009fail1"}
        assert conn.execute("SELECT COUNT(*) FROM data_commits").fetchone()[0] == 0 and not job.exists()
        return real(env, node=node, attempt=attempt, **kw)
    loop.phases.improve_recipe = improve_recipe
    assert loop.run() == "max_nodes reached" and tries == [1, 1]


def test_a_training_crash_trains_once_more_without_charging_an_attempt(make_loop):
    run, make = make_loop
    script = Script(run, train=[False, True])
    loop = make(script, max_nodes=1)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "scored"
    assert len([r for r in script.calls if r[0] == "improve_recipe"]) == 1


def test_a_training_crash_twice_ends_the_node_train_failed(make_loop):
    run, make = make_loop
    script = Script(run, train=[False, False])
    loop = make(script, max_nodes=1)
    loop.run()
    n1 = NodeStore(loop.ctx.conn).get("node0001")
    assert n1["status"] == "train_failed" and "exit code 1" in n1["error"]
    assert [a["outcome"] for a in NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")] == []
    assert len([r for r in script.calls if r[0] == "improve_recipe"]) == 1


def test_a_node_whose_eval_crashes_twice_ends_eval_failed_and_the_run_goes_on(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("render failed"), RuntimeError("render failed"), 0.8]),
                max_nodes=2)
    assert loop.run() == "max_nodes reached"
    statuses = {n["node_id"]: n["status"] for n in NodeStore(loop.ctx.conn).all()}
    assert statuses == {"root": "scored", "node0001": "eval_failed", "node0002": "scored"}


def test_a_disk_error_in_eval_is_the_kernels(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, OSError(28, "No space left on device"), OSError(28, "No space left")]))
    assert "No space left" in loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "running"


def test_eval_failing_twice_by_a_kernel_fault_stops_the_run(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, KernelFault("wbench gpu failed"), KernelFault("wbench gpu failed")]),
                max_nodes=2)
    assert "wbench gpu failed" in loop.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert set(nodes) == {"root", "node0001"} and nodes["node0001"]["status"] == "running"      # restarted on resume


def test_an_exception_in_a_phase_twice_stops_the_run(make_loop):
    run, make = make_loop
    script = Script(run)
    script.resolve = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
    loop = make(script, max_nodes=1)
    assert "KeyError" in loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "running"


def test_every_phase_is_uploaded_once_when_it_ends(make_loop):
    run, make = make_loop
    uploads = []
    loop = make(Script(run, edit=[False, True], recipe=[False, True]), max_nodes=2,
                upload=lambda *phase: uploads.append((*phase, NodeStore(loop.ctx.conn).all()[-1]["status"])))
    loop.run()
    assert [u[:3] for u in uploads] == [
        ("root", "eval", 0), ("node0001", "improve_recipe", 1), ("node0001", "improve_recipe", 2),
        ("node0001", "train", 2), ("node0001", "eval", 0), ("node0002", "edit_self", 1), ("node0002", "edit_self", 2),
        ("node0002", "improve_recipe", 1), ("node0002", "train", 1), ("node0002", "eval", 0)]
    assert uploads[4][3] == "scored"                 # after the node's own bookkeeping


def test_the_phase_a_stop_ends_is_uploaded(make_loop):
    run, make = make_loop
    uploads = []
    script = Script(run)
    script.resolve = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
    loop = make(script, max_nodes=1, upload=lambda *phase: uploads.append(phase))
    assert "KeyError" in loop.run()
    assert uploads == [("root", "eval", 0), ("node0001", "improve_recipe", 1)]


def test_an_exception_in_loop_code_outside_a_phase_is_raised(make_loop):
    run, make = make_loop
    loop = make(Script(run), max_nodes=1)
    loop._record_recipe = lambda *a: (_ for _ in ()).throw(OSError("disk gone"))
    with pytest.raises(OSError):
        loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "running"


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
    assert NodeStore(loop.ctx.conn).get("node0002")["status"] == "running"


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
    _recorded(run)
    assert "budget" in loop.run()
    assert [n for n, level, _ in _alerts(run) if level == "error"] == ["loop.budget"]
    assert NodeStore(loop.ctx.conn).get("node0002")["status"] == "running"


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


def _stopped_in(make, run, phase, max_nodes=1, node="node0001", **script):
    """A run stopped (StopRun) in `phase` of `node`, at the phase's first call for a node; returns its loop."""
    from ar_kernel.loop import StopRun
    first = Script(run, **script)
    loop = make(first, max_nodes=max_nodes)
    real = getattr(loop.phases, phase)

    def stop(*a, **k):
        if getattr(stop, "done", False) or (phase == "score" and a[1] == "root"):
            return real(*a, **k)
        stop.done = True
        raise StopRun("stopped for the test")
    setattr(loop.phases, phase, stop)
    assert loop.run() == "stopped for the test"
    assert NodeStore(loop.ctx.conn).get(node)["status"] == "running"
    return loop


@pytest.mark.parametrize("phase, calls", [
    ("improve_recipe", [("improve_recipe", 1), ("train", 1), ("score", 1)]),
    ("train", [("train", 1), ("score", 1)]),
    ("score", [("score", 1)])])
def test_resume_restarts_the_phase_the_run_stopped_in(make_loop, phase, calls):
    run, make = make_loop
    _stopped_in(make, run, phase)
    script = Script(run, score=[0.9])
    loop = make(script, max_nodes=1)                    # the running node is the max_nodes-th: it still finishes
    seen = []
    for name in ("improve_recipe", "train", "score"):
        real = getattr(loop.phases, name)

        def spy(*a, _name=name, _real=real, **k):
            seen.append((_name, k.get("attempt", 1)))
            return _real(*a, **k)
        setattr(loop.phases, name, spy)
    assert loop.run() == "max_nodes reached"
    nodes = NodeStore(loop.ctx.conn)
    assert nodes.get("node0001")["status"] == "scored" and seen == calls
    assert [a["idx"] for a in nodes.attempts("node0001", "improve_recipe")] == [1]     # not charged again


def test_resume_after_a_failed_attempt_restarts_the_next_one_with_its_cleanup(make_loop):
    run, make = make_loop
    from ar_kernel.loop import StopRun
    script = Script(run, recipe=[False])
    loop = make(script, max_nodes=1)
    _ingesting(script, loop)
    real = loop.phases.improve_recipe

    def stop_in_second(env, *, node, attempt, **kw):
        out = real(env, node=node, attempt=attempt, **kw)       # ingests clip <node>-<attempt>
        if attempt == 2:
            raise StopRun("stopped in attempt 2")
        return out
    loop.phases.improve_recipe = stop_in_second
    assert loop.run() == "stopped in attempt 2"
    clips = ClipStore(loop.ctx.conn)
    assert {c["clip_id"]: c["ingested_by"] for c in clips.all()} == {"node0001-1": "node0001fail1", "node0001-2": "node0001"}
    script2 = Script(run)
    loop2 = make(script2, max_nodes=1)
    loop2.run()
    assert [r[2] for r in script2.calls if r[0] == "improve_recipe"] == [2]
    assert {c["clip_id"]: c["ingested_by"] for c in clips.all()} == {"node0001-1": "node0001fail1"}   # attempt 2's: gone
    assert NodeStore(loop2.ctx.conn).get("node0001")["status"] == "scored"


def test_resume_in_edit_self_reruns_the_attempt_and_a_node_stopped_before_any_phase_starts_it(make_loop):
    run, make = make_loop
    make(Script(run, score=[0.7, 0.8]), max_nodes=1).run()            # node0001 scored: node0002 edits
    loop = _stopped_in(make, run, "edit_self", max_nodes=2, node="node0002")
    script = Script(run)
    make(script, max_nodes=2).run()
    assert [r[:3] for r in script.calls if r[0] == "edit_self"] == [("edit_self", "node0002", 1)]
    nodes = NodeStore(loop.ctx.conn)
    nodes.create("node0003", "root", 1)                                 # stopped right after it was created
    script = Script(run)
    make(script, max_nodes=3).run()
    assert [r[:3] for r in script.calls if r[0] == "edit_self"] == [("edit_self", "node0003", 1)]
    assert nodes.get("node0003")["status"] == "scored"


def test_a_root_left_unscored_is_evaluated_again(make_loop):
    run, make = make_loop
    assert "wbench down" in make(Script(run, score=[RuntimeError("wbench down")] * 2)).run()
    assert NodeStore(open_db(run)).get("root")["status"] == "eval_failed"
    loop = make(Script(run, score=[0.7]), max_nodes=0)
    assert loop.run() == "max_nodes reached"
    assert NodeStore(loop.ctx.conn).get("root")["status"] == "scored"


def test_root_score_failure_ends_the_run_with_the_root_eval_failed(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[RuntimeError("wbench down")] * 2))
    _recorded(run)
    assert "wbench down" in loop.run()
    assert [(n, lv) for n, lv, m in _alerts(run) if "wbench down" in m][-1] == ("loop.root_failed", "error")
    root = NodeStore(loop.ctx.conn).get("root")
    assert root["status"] == "eval_failed" and "wbench down" in root["error"]


def test_node_end_deletes_only_raw_downloads_from_staging(make_loop):
    run, make = make_loop
    attempt = run / "staging" / "node0001" / "improve_recipe-1"
    for rel in ("hf/org/ds/clip.mp4", "rollouts/job1/r.mp4", "work/rejected.mp4", "annotations/job2/p.npz"):
        (attempt / rel).parent.mkdir(parents=True, exist_ok=True)
        (attempt / rel).write_bytes(b"x")
    make(Script(run, score=[0.7, 0.9]), max_nodes=1).run()
    assert not (attempt / "hf").exists()                                   # raw downloads: recorded, deleted
    for rel in ("rollouts/job1/r.mp4", "work/rejected.mp4", "annotations/job2/p.npz"):
        assert (attempt / rel).exists()                                    # prepared candidates are kept


def test_a_scored_root_is_reused_by_the_next_run(tmp_path):
    cache = tmp_path / "root_cache"

    def run_once(name, scores):
        run = tmp_path / name
        (run / "config").mkdir(parents=True)
        (run / "config" / "run.json").write_text("{}")
        ctx = RunContext(run_dir=run, conn=open_db(run), gpus=[0, 1, 2, 3],
                         metric_set=["m"], case_ids=["1"], versions={})
        kit = SimpleNamespace(env=None, gpu_lock=threading.Lock(), budget=Budget(), harness=None)
        script = Script(run, score=scores)
        loop = Loop(CFG, ctx, kit, AgentsRepo(run / "agents.git"), max_nodes=0, phases=script.phases(),
                    root_cache=cache)
        if name == "a":                                    # what scoring the root leaves behind
            (run / "nodes" / "root" / "eval").mkdir(parents=True)
            (run / "nodes" / "root" / "eval" / "video.mp4").write_text("v")
        loop.run()
        return run, loop, script

    first, _, _ = run_once("a", [0.7])
    second, loop, script = run_once("b", [0.1])          # would score 0.1 if the root were scored again
    assert NodeStore(loop.ctx.conn).get("root")["score"] == 0.7
    assert json.loads((second / "config" / "run.json").read_text())["expected_n"] == loop.ctx.expected_n
    assert (second / "nodes" / "root" / "eval" / "video.mp4").read_text() == "v"
    assert json.loads((second / "nodes" / "root" / "eval" / "aggregates.json").read_text())["metrics"]["m"] == 0.7


def test_a_phase_that_reaches_for_the_evaluation_set_quarantines_the_node(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script, max_nodes=2)
    real_edit = script.edit_self

    def reaching_edit(env, *, node, attempt, **kw):
        _model_wrote(run, node, "edit_self", attempt, ["wbench"])
        return real_edit(env, node=node, attempt=attempt, **kw)
    loop.phases.edit_self = reaching_edit
    loop.run()
    n2 = NodeStore(loop.ctx.conn).get("node0002")
    assert n2["status"] == "quarantined" and n2["error"] == "quarantined: the model wrote wbench"
    assert [r[0] for r in script.calls if r[1] == "node0002"] == ["edit_self"]   # one attempt, no retry, no data phase


def test_a_phase_that_crashes_after_reaching_for_the_evaluation_set_is_still_quarantined(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script, max_nodes=2)

    def crashing_edit(env, *, node, attempt, **kw):
        _model_wrote(run, node, "edit_self", attempt, ["wbench"])
        raise RuntimeError("sandbox died")
    loop.phases.edit_self = crashing_edit
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0002")["status"] == "quarantined"


def test_saving_a_root_another_run_already_saved_keeps_the_first(tmp_path):
    from ar_kernel.loop import save_root
    eval_dir, cache = tmp_path / "eval", tmp_path / "cache" / "key"
    eval_dir.mkdir()
    (eval_dir / "aggregates.json").write_text("{}")
    cache.parent.mkdir()
    save_root(cache, eval_dir, {"m": 0.5}, {"m": 4}, "first")
    save_root(cache, eval_dir, {"m": 0.6}, {"m": 4}, "second")           # raised: the cache folder exists
    assert json.loads((cache / "root.json").read_text())["run_id"] == "first"
    assert not cache.with_name("key.tmp").exists()


def test_the_root_key_follows_what_changes_the_roots_score(tmp_path):
    """A root scored under other judge settings was reused silently against children judged with the new ones."""
    from types import SimpleNamespace
    from ar_kernel.config import KernelConfig
    from ar_kernel.eval.judge import Judge
    from ar_kernel.run import root_key
    real = KernelConfig.load()
    ctx = SimpleNamespace(metric_set=["m"], case_ids=["1"], versions={"worldmodel_sha": "a", "wbench_sha": "b"},
                          judge=Judge("local", "model", None))
    changed = lambda **raw: KernelConfig(raw={**real.raw, **raw}, repo_root=real.repo_root)
    key = root_key(ctx, real)
    assert root_key(ctx, changed()) == key
    # another benchmark with the same case ids, or this one edited in place, is another root
    for part, name in (("cases", "case_1.json"), ("images", "case_1.jpg"), ("masks", "case_1_mask.png")):
        (tmp_path / part).mkdir()
        (tmp_path / part / name).write_bytes(b"other")
    other = changed(eval={**real.raw["eval"], "data": str(tmp_path)})
    assert root_key(ctx, other) != key
    moved = root_key(ctx, other)
    (tmp_path / "masks" / "case_1_mask.png").write_bytes(b"edited")
    assert root_key(ctx, other) != moved
    assert root_key(ctx, changed(eval={**real.raw["eval"], "judge": {"max_images": 8}})) != key
    assert root_key(ctx, changed(captioner={**real.raw["captioner"], "max_model_len": 4096})) != key
    render = {"spatial_memory": {"vigeo_cache_budget": 262144}}
    assert root_key(ctx, changed(eval={**real.raw["eval"], "render_overrides": render})) != key
    # what does not change the root's metric means leaves the key alone
    assert root_key(ctx, changed(eval={**real.raw["eval"], "score_weights": {}})) == key
    assert root_key(ctx, changed(captioner={**real.raw["captioner"], "clip_timeout_s": 5})) == key
    assert root_key(ctx, changed(captioner={**real.raw["captioner"], "concurrent_requests": 3})) == key
    assert root_key(ctx, changed(eval={**real.raw["eval"], "local_judge_workers": 32})) == key
    api = SimpleNamespace(**{**vars(ctx), "judge": Judge("api", "model", "http://x")})
    assert root_key(api, changed(captioner={**real.raw["captioner"], "max_model_len": 4096})) == root_key(api, real)


def _clip(clip_id: str, node: str, attempt: int = 1) -> dict:
    return {"clip_id": clip_id, "video_digest": "v", "caption_digest": "c", "annotation_digest": "a",
            "pose_digest": "p", "metadata": {}, "warnings": [], "derived_from": {"kind": "model", "name": "x"},
            "ingested_by": node, "attempt": attempt}


def _ingesting(script, loop):
    """Each improve_recipe attempt ingests one clip, named after its node and attempt."""
    real = script.improve_recipe

    def improve_recipe(env, *, node, attempt, **kw):
        ClipStore(loop.ctx.conn).add(_clip(f"{node}-{attempt}", node, attempt))
        return real(env, node=node, attempt=attempt, **kw)
    loop.phases.improve_recipe = improve_recipe


def test_a_failed_attempts_clips_leave_the_pool_and_a_passing_ones_stay(make_loop):
    run, make = make_loop
    script = Script(run, recipe=[False, True])
    loop = make(script)
    _ingesting(script, loop)
    loop.run()
    assert {c["clip_id"]: c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()} == {
        "node0001-1": "node0001fail1", "node0001-2": "node0001"}


def test_a_node_ended_by_a_fault_in_eval_keeps_its_clips(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7])
    loop = make(script)
    _ingesting(script, loop)
    real = loop.phases.score

    def score(loop_, node, *a):
        if node != "root":
            raise NodeFault("the score is nan")
        return real(loop_, node, *a)
    loop.phases.score = score
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "eval_failed"
    assert [c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()] == ["node0001"]
    assert _status(loop, "dc_node0001_1") == "accepted"



def test_a_node_resumed_at_eval_and_ended_by_a_fault_keeps_its_clips_too(make_loop):
    run, make = make_loop
    from ar_kernel.loop import StopRun
    script = Script(run)
    loop = make(script)
    _ingesting(script, loop)
    real = loop.phases.score

    def stop(loop_, node, *a):
        if node != "root":
            raise StopRun("stopped in eval")
        return real(loop_, node, *a)
    loop.phases.score = stop
    assert loop.run() == "stopped in eval"
    loop2 = make(Script(run))

    def fault(loop_, node, *a):
        raise NodeFault("the score is nan")
    loop2.phases.score = fault
    loop2.run()
    assert NodeStore(loop2.ctx.conn).get("node0001")["status"] == "eval_failed"
    assert [c["ingested_by"] for c in ClipStore(loop2.ctx.conn).all()] == ["node0001"]

def test_a_quarantined_nodes_clips_leave_the_pool_with_its_failed_attempts(make_loop):
    run, make = make_loop
    script = Script(run, recipe=[False, True])
    loop = make(script)
    _ingesting(script, loop)

    def reaching_train(loop_, resolved, node, attempt_dir):
        _model_wrote(run, node, "improve_recipe", 2, ["wbench"])
        raise RuntimeError("sandbox died")
    loop.phases.train = reaching_train
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "quarantined"
    assert [c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()] == ["node0001quarantine"] * 2
    assert _status(loop, "dc_node0001_2") == "failed"          # trained on, but its clips are hidden


# --- improve_recipe in subphases ----------------------------------------------------------------------


def _outcomes(loop, node="node0001") -> list[str]:
    return [a["outcome"] for a in NodeStore(loop.ctx.conn).attempts(node, "improve_recipe")]


def test_subphases_continue_until_one_trains_and_only_that_one_is_resolved(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", "continue", True])
    loop = make(script)
    loop.run()
    assert _outcomes(loop) == ["continued", "continued", "passed"]
    assert [r for r in script.calls if r[0] == "resolve"] == [("resolve", "node0001", "improve_recipe-3")]
    first = NodeStore(loop.ctx.conn).attempts("node0001", "improve_recipe")[0]["detail"]
    assert first == {"snapshot_id": "1" * 32, "summary": "subphase 1"}
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "scored"


def test_a_failed_subphase_hides_only_its_own_clips(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, True])
    loop = make(script)
    _ingesting(script, loop)
    loop.run()
    assert _outcomes(loop) == ["continued", "failed", "passed"]
    assert {c["clip_id"]: c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()} == {
        "node0001-1": "node0001", "node0001-2": "node0001fail2", "node0001-3": "node0001"}


def test_failures_across_subphases_reach_the_retry_cap(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, "continue", False, False])
    loop = make(script)
    loop.run()
    assert _outcomes(loop) == ["continued", "failed", "continued", "failed", "failed"]
    n1 = NodeStore(loop.ctx.conn).get("node0001")
    assert n1["status"] == "invalid_recipe" and "no commit" in n1["error"]


def test_a_subphase_stopped_by_the_machine_restarts_on_resume_and_loses_only_its_own_clips(make_loop):
    from ar_kernel.loop import StopRun
    run, make = make_loop
    script = Script(run, recipe=["continue", "continue"])
    loop = make(script)
    _ingesting(script, loop)
    real = loop.phases.improve_recipe

    def stop_in_subphase_3(env, *, node, attempt, **kw):
        out = real(env, node=node, attempt=attempt, **kw)
        if attempt == 3:
            raise StopRun("the machine went away")
        return out
    loop.phases.improve_recipe = stop_in_subphase_3
    assert loop.run() == "the machine went away"
    assert {c["clip_id"] for c in ClipStore(loop.ctx.conn).all()} == {"node0001-1", "node0001-2", "node0001-3"}
    script2 = Script(run)
    loop2 = make(script2)
    seen = []

    def improve_recipe(env, *, node, attempt, **kw):
        seen.append((attempt, sorted(c["clip_id"] for c in ClipStore(loop2.ctx.conn).all())))
        return script2.improve_recipe(env, node=node, attempt=attempt, **kw)
    loop2.phases.improve_recipe = improve_recipe
    loop2.run()
    assert seen == [(3, ["node0001-1", "node0001-2"])]
    assert _outcomes(loop2) == ["continued", "continued", "passed"]


def test_a_node_fault_after_training_keeps_every_subphases_clips(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", True], train=[False, False])
    loop = make(script)
    _ingesting(script, loop)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "train_failed"
    assert {c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()} == {"node0001"}


def test_a_node_out_of_retries_hides_its_continued_subphases_clips_too(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, False, False])
    loop = make(script)
    _ingesting(script, loop)
    loop.run()
    assert _outcomes(loop) == ["continued", "failed", "failed", "failed"]
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "invalid_recipe"
    assert {c["clip_id"]: c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()} == {
        "node0001-1": "node0001fail4", "node0001-2": "node0001fail2", "node0001-3": "node0001fail3",
        "node0001-4": "node0001fail4"}


def test_a_stop_in_training_after_a_continued_subphase_trains_that_subphase_again_on_resume(make_loop):
    run, make = make_loop
    uploads = []
    script = Script(run, recipe=["continue", True], train=["infra", "infra"])
    loop = make(script, upload=lambda *phase: uploads.append(phase))
    _ingesting(script, loop)
    loop.run()
    assert _outcomes(loop) == ["continued"]
    assert ("node0001", "improve_recipe", 1) in uploads and ("node0001", "improve_recipe", 2) in uploads
    loop2 = make(Script(run))
    loop2.run()
    assert _outcomes(loop2) == ["continued", "passed"]
    assert NodeStore(loop2.ctx.conn).get("node0001")["status"] == "scored"
    assert {c["clip_id"]: c["ingested_by"] for c in ClipStore(loop2.ctx.conn).all()} == {
        "node0001-1": "node0001", "node0001-2": "node0001"}


# --- data commit status -----------------------------------------------------------------------------


def _status(loop, commit_id: str) -> str:
    return CommitStore(loop.ctx.conn, None, None).get(commit_id)["status"]


def _committing(script, loop):
    """Each improve_recipe attempt stores drafts a and b; a continue names a."""
    real = script.improve_recipe

    def improve_recipe(env, *, node, attempt, **kw):
        for name in "ab":
            loop.ctx.conn.execute("INSERT INTO data_commits (commit_id, node_id, attempt, status, manifest, message, "
                                  "created_at) VALUES (?,?,?,'draft','{\"datasets\": {}}','m',0)",
                                  (f"{node}-{attempt}{name}", node, attempt))
        out = real(env, node=node, attempt=attempt, **kw)
        if out.ok and not out.result["train"]:
            out.result["data_commit"] = f"{node}-{attempt}a"
        return out
    loop.phases.improve_recipe = improve_recipe


def test_a_subphase_end_keeps_the_named_commit_and_makes_its_other_drafts_obsolete(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, True])
    loop = make(script)
    _committing(script, loop)
    loop.run()
    assert _outcomes(loop) == ["continued", "failed", "passed"]
    assert {i: _status(loop, i) for i in ("node0001-1a", "node0001-1b", "node0001-2a", "node0001-2b",
                                          "node0001-3a", "node0001-3b", "dc_node0001_3")} == {
        "node0001-1a": "accepted", "node0001-1b": "obsolete", "node0001-2a": "failed", "node0001-2b": "failed",
        "node0001-3a": "obsolete", "node0001-3b": "obsolete", "dc_node0001_3": "accepted"}


def test_a_continue_naming_a_snapshot_without_a_data_commit_keeps_none(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", True])
    loop = make(script)
    _committing(script, loop)
    real = loop.phases.improve_recipe

    def without_commit(env, **kw):
        out = real(env, **kw)
        if not out.result["train"]:
            out.result["data_commit"] = None
        return out
    loop.phases.improve_recipe = without_commit
    loop.run()
    assert [_status(loop, i) for i in ("node0001-1a", "node0001-1b")] == ["obsolete", "obsolete"]


def test_a_node_out_of_retries_fails_every_commit_it_made(make_loop):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, False, False])
    loop = make(script)
    _committing(script, loop)
    loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["status"] == "invalid_recipe"
    statuses = {r["status"] for r in loop.ctx.conn.execute("SELECT status FROM data_commits WHERE node_id='node0001'")}
    assert statuses == {"failed"}


def test_a_subphase_end_is_recorded_whole_or_not_at_all(make_loop, monkeypatch):
    run, make = make_loop
    script = Script(run, recipe=["continue"])
    loop = make(script)
    _committing(script, loop)

    def broken(self, *a):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(CommitStore, "settle", broken)
    with pytest.raises(RuntimeError, match="disk gone"):
        loop.run()
    assert _outcomes(loop) == [] and _status(loop, "node0001-1a") == "draft"


def _breaking(monkeypatch, cls, name, when=lambda *a, **k: True):
    real = getattr(cls, name)

    def broken(self, *a, **k):
        if when(*a, **k):
            raise RuntimeError("disk gone")
        return real(self, *a, **k)
    monkeypatch.setattr(cls, name, broken)


def test_a_failed_subphase_is_recorded_whole_or_not_at_all(make_loop, monkeypatch):
    run, make = make_loop
    script = Script(run, recipe=[False])
    loop = make(script)
    _committing(script, loop)
    _breaking(monkeypatch, ClipStore, "hide")
    with pytest.raises(RuntimeError, match="disk gone"):
        loop.run()
    assert _outcomes(loop) == [] and _status(loop, "node0001-1a") == "draft"


def test_a_recorded_recipe_is_written_whole_or_not_at_all(make_loop, monkeypatch):
    run, make = make_loop
    script = Script(run, recipe=[True])
    loop = make(script)
    _breaking(monkeypatch, CommitStore, "settle")
    with pytest.raises(RuntimeError, match="disk gone"):
        loop.run()
    assert NodeStore(loop.ctx.conn).get("node0001")["resolved_config_path"] is None
    assert _status(loop, "dc_node0001_1") == "draft"


def test_running_out_of_retries_is_recorded_whole_or_not_at_all(make_loop, monkeypatch):
    run, make = make_loop
    script = Script(run, recipe=["continue", False, False, False])
    loop = make(script)
    _committing(script, loop)
    _breaking(monkeypatch, ClipStore, "hide", when=lambda node, mark, attempt=None: attempt is None)
    with pytest.raises(RuntimeError, match="disk gone"):
        loop.run()
    assert _status(loop, "node0001-1a") == "accepted"


def test_a_quarantine_is_recorded_whole_or_not_at_all(make_loop, monkeypatch):
    run, make = make_loop
    script = Script(run)
    loop = make(script)
    _ingesting(script, loop)

    def reaching_train(loop_, resolved, node, attempt_dir):
        _model_wrote(run, node, "improve_recipe", 1, ["wbench"])
        raise RuntimeError("sandbox died")
    loop.phases.train = reaching_train
    _breaking(monkeypatch, NodeStore, "set_status", when=lambda node, status: status == "quarantined")
    with pytest.raises(RuntimeError, match="disk gone"):
        loop.run()
    assert [c["ingested_by"] for c in ClipStore(loop.ctx.conn).all()] == ["node0001"]
    assert _status(loop, "dc_node0001_1") == "accepted"


@pytest.mark.parametrize("exc, chosen, raised", [
    (ValueError("bad render"), "node fault", NodeFault), (OSError(28, "No space left on device"), "disk fault", OSError)])
def test_a_failed_eval_is_the_nodes_unless_the_disk_is_full(tmp_path, exc, chosen, raised):
    from ar_kernel import telemetry
    from ar_kernel.telemetry import tree
    from conftest import telemetry_records
    telemetry.open(tmp_path)
    telemetry.start("ar.run")

    def score(*a):
        raise exc
    with pytest.raises(raised):
        Loop._evaluate(SimpleNamespace(phases=SimpleNamespace(score=score)), "n1", None, None)
    assert [r["chosen"] for r in telemetry_records(tmp_path) if r["kind"] == "decision"] == [chosen]
    assert tree.check(tree.load(tmp_path)) == []
