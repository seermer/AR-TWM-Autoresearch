import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.loop import Loop
from ar_kernel.run import RunContext
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.train.runner import TrainOutcome
from ar_kernel.vcs.agents_repo import AgentsRepo
from fixtures.fake_loop import Script

CFG = KernelConfig.load()
SEED = Path(__file__).resolve().parents[1] / "seed_agent"


@pytest.fixture
def make_loop(tmp_path):
    run = tmp_path / "run"

    def make(script, max_nodes=1, budget=None):
        (run / "config").mkdir(parents=True, exist_ok=True)
        (run / "config" / "run.json").write_text("{}")
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
    assert nodes["n1"]["checkpoint_path"].startswith("nodes/n1/")
    assert json.loads((run / "nodes" / "n1" / "edit.json").read_text())["summary"] == "s"
    assert (run / "nodes" / "n1" / "recipe.yaml").exists() and (run / "nodes" / "n1" / "rationale.md").exists()
    assert json.loads((run / "nodes" / "n1" / "eval" / "aggregates.json").read_text())["metrics"]["m"] == 0.9
    assert nodes["root"]["subtree_value"] == pytest.approx(0.7 * 0.7 + 0.3 * (0.7 + 0.5 * 0.9) / 1.5)
    passed = NodeStore(loop.ctx.conn).attempts("n1", "improve_recipe")[-1]["detail"]
    assert passed["checkpoint"].startswith("nodes/n1/")          # run-relative, never absolute


def test_contract_failure_retries_with_the_report(make_loop):
    run, make = make_loop
    script = Script(run, contract=[False, True])
    make(script).run()
    edits = [r for r in script.retries if r[0] == "edit_self"]
    assert [a for _, _, a, _ in edits] == [1, 2] and edits[1][3]["kind"] == "contract"


def test_every_edit_attempt_runs_on_the_parents_code(make_loop):
    run, make = make_loop
    script = Script(run, contract=[False, True])
    loop = make(script)
    loop.run()
    assert script.runners == [NodeStore(loop.ctx.conn).get("root")["agent_commit"]] * 2


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
    assert recipes[1][3]["data_commit"] == "c" * 64 and recipes[1][3]["recipe"] == {"optimizer.max_steps": 2}
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "train_failed" and '"detail": "CUDA OOM"' in n1["error"]
    assert "log_tail" not in n1["error"] and "rationale" not in n1["error"]   # the node's error names the failure


def test_failed_training_that_left_a_checkpoint_is_not_scored(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    real_train = script.train

    def diverged_first(loop, resolved, node, attempt_dir):     # nan loss (exit 0) after checkpoint-2
        out = real_train(loop, resolved, node, attempt_dir)
        if attempt_dir.name.endswith("-1"):
            return TrainOutcome(out.checkpoint, "recipe", out.log_path, detail="loss became nan")
        return out
    script.train = diverged_first
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert recipes[1][3]["kind"] == "train" and recipes[1][3]["detail"] == "loss became nan"
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "scored" and "improve_recipe-2" in n1["checkpoint_path"]


def test_gate_failure_then_success(make_loop):
    run, make = make_loop
    script = Script(run, gate=[False, True])
    loop = make(script)
    loop.run()
    recipes = [r for r in script.retries if r[0] == "improve_recipe"]
    assert recipes[1][3] == {"kind": "gate", "failures": ["dataset d has 1 clips, fewer than 4 GPUs"],
                             "data_commit": "c" * 64, "recipe": {"optimizer.max_steps": 2}, "rationale": "why"}
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "scored"


def test_a_failed_eval_is_run_once_more(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("judge down"), 0.9]))
    loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["score"] == 0.9
    alerts = [e["kind"] for e in loop.ctx.recorder.read_events() if e["type"] == "alert"]
    assert alerts == ["eval_retry"]


def test_eval_failing_twice_stops_the_run(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[0.7, RuntimeError("wbench gpu failed"), RuntimeError("wbench gpu failed")]),
                max_nodes=2)
    assert "wbench gpu failed" in loop.run()
    nodes = {n["node_id"]: n for n in NodeStore(loop.ctx.conn).all()}
    assert set(nodes) == {"root", "n1"} and nodes["n1"]["status"] == "running"      # -> interrupted on resume
    kinds = [e["kind"] for e in loop.ctx.recorder.read_events() if e["type"] == "alert"]
    assert kinds == ["eval_retry", "eval_failed"]


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


def test_root_score_failure_ends_the_run_with_the_root_eval_failed(make_loop):
    run, make = make_loop
    loop = make(Script(run, score=[RuntimeError("wbench down")] * 2))
    with pytest.raises(RuntimeError):
        loop.run()
    root = NodeStore(loop.ctx.conn).get("root")
    assert root["status"] == "eval_failed" and "wbench down" in root["error"]
    alerts = [e for e in loop.ctx.recorder.read_events() if e["type"] == "alert"]
    assert alerts[-1]["kind"] == "root_failed" and "wbench down" in alerts[-1]["message"]


def test_node_end_deletes_only_raw_downloads_from_staging(make_loop):
    run, make = make_loop
    attempt = run / "staging" / "n1" / "improve_recipe-1"
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
        rec = Recorder(run)
        ctx = RunContext(run_dir=run, conn=open_db(run), recorder=rec, gpus=[0, 1, 2, 3],
                         metric_set=["m"], case_ids=["1"], versions={})
        kit = SimpleNamespace(registry=None, queue=None, gpu_lock=threading.Lock(), budget=Budget(),
                              harness=None, socket_dir=tmp_path / "sock", default_model="mock-model")
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
    assert [e["type"] for e in loop.ctx.recorder.read_events("root")].count("root.reused") == 1


def test_a_phase_that_reaches_for_the_evaluation_set_quarantines_the_node(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script)
    real_edit = script.edit_self

    def reaching_edit(env, *, node, attempt, **kw):
        call = {"id": "c1", "type": "function",
                "function": {"name": "run_command", "arguments": '{"command": "curl hf.co/datasets/x/WBench"}'}}
        loop.ctx.recorder.event("llm.response", node=node, phase="edit_self", attempt=attempt, conversation_id="c",
                                payload={"body": {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}})
        return real_edit(env, node=node, attempt=attempt, **kw)
    loop.phases.edit_self = reaching_edit
    loop.run()
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "quarantined" and n1["error"] == "quarantined: the model wrote wbench"
    assert [r[0] for r in script.retries] == ["edit_self"]              # one attempt, no retry, no data phase
    assert any(e["type"] == "isolation.audit" for e in loop.ctx.recorder.read_events("n1"))


def test_a_phase_that_crashes_after_reaching_for_the_evaluation_set_is_still_quarantined(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script)

    def crashing_edit(env, *, node, attempt, **kw):
        message = {"role": "assistant", "content": "I will fetch WBench."}
        loop.ctx.recorder.event("llm.response", node=node, phase="edit_self", attempt=attempt, conversation_id="c",
                                payload={"body": {"choices": [{"message": message}]}})
        raise RuntimeError("docker died")
    loop.phases.edit_self = crashing_edit
    loop.run()
    assert NodeStore(loop.ctx.conn).get("n1")["status"] == "quarantined"


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
    # what does not change the root's metric means leaves the key alone
    assert root_key(ctx, changed(eval={**real.raw["eval"], "score_weights": {}})) == key
    assert root_key(ctx, changed(captioner={**real.raw["captioner"], "startup_timeout_s": 5})) == key
    api = SimpleNamespace(**{**vars(ctx), "judge": Judge("api", "model", "http://x")})
    assert root_key(api, changed(captioner={**real.raw["captioner"], "max_model_len": 4096})) == root_key(api, real)
