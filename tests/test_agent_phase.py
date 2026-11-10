import json
import subprocess
import threading

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import telemetry
from ar_kernel.agent_phase import PhaseEnv, attempt_dirs, run_edit_self, run_improve_recipe
from ar_kernel.archive.activity import activity
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig, merged
from ar_kernel.faults import KernelFault
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.telemetry import tree
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.snapshot import SnapshotStore, register_snapshot_tool
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()


def snapshot_id(penv, env, **recipe) -> str:
    """What the agent's call of the snapshot tool returns, taken from inside the 'container'."""
    caller = penv.registry.lookup(env["AR_TOKEN"])
    return SnapshotStore(penv.run_dir, penv.repo, lambda c: None).take(
        caller, recipe.get("data_commit"), recipe.get("recipe"), recipe.get("rationale"))["snapshot_id"]


class FakeRunner:
    """Stands in for the container; with `snap`, the result names a snapshot taken after the edit."""
    def __init__(self, result, edit=None, exit_code=0, staged=None, raises=None, snap=None):
        self.result, self.edit, self.exit_code = result, edit, exit_code
        self.staged = staged
        self.raises = raises
        self.snap = snap
        self.penv = None                # set by the fixture's make()
        self.calls = []

    def __call__(self, *, mounts, env, **kw):
        self.calls.append({"mounts": mounts, "env": env, **kw})
        if self.raises is not None:
            raise self.raises
        if self.edit:
            (mounts.agent / "agent" / "entry.py").write_text(self.edit)
        if self.staged:                         # what hf_download would leave in /workspace/staging
            (mounts.staging / self.staged).parent.mkdir(parents=True, exist_ok=True)
            (mounts.staging / self.staged).write_text("x")
        if self.result is not None:
            body = json.loads(json.dumps(self.result))
            if self.snap is not None:
                body["result"]["snapshot_id"] = snapshot_id(self.penv, env, **self.snap)
            (mounts.workspace / "result.json").write_text(json.dumps(body))
        return RunResult(self.exit_code, "", "", 1.0, "c")


@pytest.fixture
def env(tmp_path, monkeypatch):
    seed = tmp_path / "seed" / "agent"
    seed.mkdir(parents=True)
    (seed / "entry.py").write_text("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    repo = AgentsRepo(tmp_path / "run" / "agents.git")
    root = repo.init(tmp_path / "seed")
    conn = open_db(tmp_path / "run")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root)
    nodes.create("n1", "root", 1)
    queue = JobQueue(tmp_path / "run", threading.Lock(), wait_cap_s=1)

    def make(runner):
        penv = PhaseEnv(cfg=CFG, run_dir=tmp_path / "run", run_id="t",
                        registry=TokenRegistry(), queue=queue, repo=repo,
                        socket_dir=tmp_path / "sock", gpus=[0, 1, 2, 3], default_model="gpt-x",
                        runner=runner)
        if isinstance(runner, FakeRunner):
            runner.penv = penv
        return penv
    yield make, conn, root, queue
    queue.shutdown()


def test_attempt_layout(tmp_path):
    d = attempt_dirs(tmp_path, "n1", "edit_self", 2)
    assert d["agent"] == tmp_path / "nodes" / "n1" / "attempts" / "edit_self-2" / "agent"
    assert d["staging"] == tmp_path / "staging" / "n1" / "edit_self-2"


def test_edit_self_commits_the_edited_code_to_an_attempt_ref(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "tightened prompts"}},
                        edit="# v2\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", snap={})
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert out.ok and out.result == {"snapshot_id": out.commit, "summary": "tightened prompts"}
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert "# v2" in penv.repo.read_file(out.commit, "agent/entry.py")
    assert runner.calls[0]["mounts"].agent_readonly is False


def test_failed_edit_attempt_is_still_committed(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": False, "error": "RuntimeError: boom"}, edit="broken(\n", exit_code=1)
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "boom" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_token_is_revoked_and_jobs_cancelled_after_the_phase(env, monkeypatch):
    make, conn, root, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, snap={})
    penv = make(runner)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None and cancelled == [token]


def test_the_sandbox_gets_hfk_as_its_hf_token_only_when_it_is_set(env, monkeypatch):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, snap={})
    penv = make(runner)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    monkeypatch.setenv("HFK", "hf-key")
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=2)
    assert "HF_TOKEN" not in runner.calls[0]["env"] and runner.calls[1]["env"]["HF_TOKEN"] == "hf-key"


def test_runner_gets_the_agent_env_shared_memory_and_the_runs_gpus(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    call = runner.calls[0]
    assert call["env_dir"] == CFG.repo_root / ".envs" / "agent" and call["shm_gb"] == CFG.get("sandbox.shm_gb") == 64
    assert call["gpus"] == [0, 1, 2, 3] and "timeout_s" not in call and "liveness" not in call


def test_improve_recipe_mounts_code_read_only_and_rejects_unknown_commits(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s", "train": True}},
                        snap={"data_commit": "f" * 64, "recipe": {}, "rationale": "r"})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    assert runner.calls[0]["mounts"].agent_readonly is True
    assert not out.ok and "data commit" in out.error


def test_improve_recipe_lists_the_caption_tool_when_its_backend_is_registered(env):
    from ar_kernel.tools.captioner import CaptionBackend
    make, conn, root, queue = env
    penv = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    queue.register(CaptionBackend(CFG, penv.run_dir, penv.gpus, penv.registry))
    run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=1)
    tools = json.loads((penv.runner.calls[0]["mounts"].context / "context.json").read_text())["tools"]
    assert "caption_videos" in tools and "job_wait" in tools and "hf_list_files" in tools


def test_a_retry_starts_clean(env):
    make, conn, root, _ = env
    first = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1, staged="hf/big.mp4"))
    out1 = run_improve_recipe(first, conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=1)
    (out1.attempt_dir / "workspace" / "notes.md").write_text("half-built dataset")
    runner = FakeRunner({"ok": False, "error": "y"}, exit_code=1)
    run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=2)
    mounts = runner.calls[0]["mounts"]
    assert [f.name for f in mounts.workspace.iterdir()] == ["result.json"]          # the fake's own
    assert list(mounts.staging.iterdir()) == []
    assert [f.name for f in mounts.context.iterdir()] == ["context.json"]
    assert "retry" not in json.loads((mounts.context / "context.json").read_text())
    assert (out1.attempt_dir / "workspace" / "notes.md").exists()                   # the failed attempt's files stay
    assert (mounts.staging.parent / "improve_recipe-1" / "hf" / "big.mp4").exists()


# --- Edge cases (each gets its own test) ---------------------------


def test_invalid_edit_result_is_a_failed_attempt_not_a_crash(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {}})   # missing snapshot_id and summary
    out = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None and out.error
    assert [(r["kind"], r["name"]) for r in activity(conn, "n1")] == [("accepted", None)]


def test_invalid_recipe_result_missing_snapshot_id_is_a_failed_attempt_not_a_keyerror(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}})   # no snapshot_id
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    assert not out.ok and out.result is None and out.error


# --- Hostile or broken agent output -----------------------------------------------------


def test_malformed_json_result_is_a_failed_attempt_and_still_committed(env):
    """Bad JSON in result.json must not raise a JSONDecodeError out
    of the kernel, and edit_self must still commit the attempt."""
    make, conn, root, _ = env

    def bad_json(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_text("{not valid json")
        return RunResult(0, "", "", 1.0, "c")

    penv = make(bad_json)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None and out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_non_utf8_result_is_a_failed_attempt_not_a_crash(env):
    """Non-UTF-8 bytes in result.json must not raise a
    UnicodeDecodeError out of the kernel."""
    make, conn, root, _ = env

    def bad_bytes(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_bytes(b"\xff\xfe\x00\x01")
        return RunResult(0, "", "", 1.0, "c")

    out = run_edit_self(make(bad_bytes), conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=1)
    assert not out.ok and out.result is None and out.error


def test_non_object_json_result_is_a_failed_attempt_not_a_crash(env):
    """A JSON value that isn't an object (e.g. a list) must not
    raise an AttributeError out of the kernel on body.get(...)."""
    make, conn, root, _ = env

    def array_body(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_text(json.dumps([1, 2, 3]))
        return RunResult(0, "", "", 1.0, "c")

    out = run_improve_recipe(make(array_body), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    assert not out.ok and "not a JSON object" in out.error


def test_symlinked_result_json_is_not_followed_onto_the_host(env, tmp_path):
    """A symlinked result.json must be treated as "no result", not
    resolved and read -- the target could be any host path the container's
    user can reach."""
    make, conn, root, _ = env
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"ok": True, "result": {"summary": "leaked"}}))

    def symlink_result(*, mounts, env, **kw):
        (mounts.workspace / "result.json").symlink_to(secret)
        return RunResult(0, "", "", 1.0, "c")

    penv = make(symlink_result)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "no result.json" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_rerunning_the_same_attempt_number_clears_stale_staging(env, tmp_path):
    """Staging lives outside the attempt dir, so re-running the same
    attempt number must clear it too, or stale files pollute the new run."""
    make, conn, root, _ = env
    run_dir = tmp_path / "run"
    run_improve_recipe(make(FakeRunner({"ok": False, "error": "x"}, exit_code=1, staged="old/file.bin")),
                       conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    staging = attempt_dirs(run_dir, "n1", "improve_recipe", 1)["staging"]
    assert (staging / "old" / "file.bin").exists()
    runner2 = FakeRunner({"ok": False, "error": "y"}, exit_code=1)   # nothing staged this time
    run_improve_recipe(make(runner2), conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=1)   # same attempt number
    assert not (staging / "old" / "file.bin").exists()
    assert runner2.calls[0]["mounts"].staging == staging


def test_runner_exception_still_revokes_token_and_cancels_jobs(env, monkeypatch):
    """An exception raised by the container call (or anything after
    the token is issued) must still revoke the token and cancel its jobs."""
    make, conn, root, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner(None, raises=RuntimeError("sandbox exploded"))
    penv = make(runner)
    with pytest.raises(RuntimeError, match="sandbox exploded"):
        run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None
    assert cancelled == [token]


def test_a_summary_that_is_not_utf8_fails_the_attempt(env):
    """The result models refuse a lone surrogate, so it never reaches summary.md."""
    make, conn, root, _ = env
    out = run_edit_self(make(FakeRunner({"ok": True, "result": {"summary": "fixed \ud800"}}, snap={})), conn=conn,
                        node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "string_unicode" in out.error


def test_ok_must_be_the_literal_true_not_merely_truthy(env):
    """`{"ok": 1}` (truthy but not True) must not be treated as success."""
    make, conn, root, _ = env
    out = run_edit_self(make(FakeRunner({"ok": 1, "result": {"summary": "x"}})), conn=conn, node="n1",
                        parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None


def test_error_field_is_coerced_to_a_string_with_a_fallback(env):
    """A non-string (or missing) error field must not crash the PhaseOutcome; a missing error gets a clear
    fallback message."""
    make, conn, root, _ = env
    out1 = run_edit_self(make(FakeRunner({"ok": False})), conn=conn, node="n1", parent_id="root",
                         agent_commit=root, attempt=1)
    assert out1.error == "agent reported failure without an error"

    out2 = run_edit_self(make(FakeRunner({"ok": False, "error": 42})), conn=conn, node="n1",
                         parent_id="root", agent_commit=root, attempt=2)
    assert out2.error == "42"


# --- Final review: agent-hostile filesystem states are failed attempts, never kernel exceptions ---


def _edit_then_run_on(make, conn, root, mutate):
    """Attempt 1 lets `mutate(agent_dir)` run inside the 'container'; the next node edits its commit."""
    def runner(*, mounts, env, **kw):
        mutate(mounts.agent / "agent")
        result = {"snapshot_id": snapshot_id(penv, env), "summary": "x"}
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": result}))
        return RunResult(0, "", "", 1.0, "c")

    penv = make(runner)
    out1 = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    second = FakeRunner({"ok": True, "result": {"summary": "unused"}}, snap={})
    out2 = run_edit_self(make(second), conn=conn, node="n2", parent_id="root", agent_commit=out1.commit, attempt=1)
    return out1, out2, second


def test_committed_symlink_leaving_the_tree_fails_the_phase_instead_of_raising(env):
    make, conn, root, _ = env
    out1, out2, second = _edit_then_run_on(make, conn, root,
                                                lambda agent: (agent / "ctx").symlink_to("/context/x"))
    assert out1.ok and out1.commit
    assert not out2.ok and "check out" in out2.error and second.calls == []
    assert out2.commit is None


@pytest.mark.parametrize("mutate", [
    lambda agent: ((agent / "requirements.txt").mkdir(), (agent / "requirements.txt" / "x").write_text("")),
    lambda agent: (agent / "requirements.txt").write_bytes(b"\xff\xfe not utf-8"),
], ids=["directory", "non_utf8"])
def test_unreadable_requirements_txt_fails_the_phase_instead_of_raising(env, mutate):
    make, conn, root, _ = env
    _, out2, second = _edit_then_run_on(make, conn, root, mutate)
    assert not out2.ok and "cannot read agent/requirements.txt" in out2.error and second.calls == []


class InstallingRunner(FakeRunner):
    """The install sandbox (log_name="install") leaves a package in the workspace's home; `code` is its exit code."""
    def __init__(self, *a, code=0, **k):
        super().__init__(*a, **k)
        self.code = code

    def __call__(self, *, mounts, env, **kw):
        if kw.get("log_name") != "install":
            return super().__call__(mounts=mounts, env=env, **kw)
        self.calls.append({"mounts": mounts, "env": env, **kw})
        (mounts.workspace / ".home" / ".local" / "pkg").mkdir(parents=True)
        return RunResult(self.code, "", "No matching distribution found for nope", 1.0, "c")


def _with_requirements(make, root, text, tmp):
    """A commit of the seed whose agent/requirements.txt holds `text`."""
    repo = make(None).repo
    repo.checkout(root, tmp)
    (tmp / "agent" / "requirements.txt").write_text(text)
    return repo.commit_tree(tmp, root, "requirements")


def test_listed_requirements_are_installed_before_the_phase(env, tmp_path):
    make, conn, root, _ = env
    commit = _with_requirements(make, root, "# a comment\nrich==13.0\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    out = run_edit_self(make(runner), conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    install, phase = runner.calls
    assert install["command"][-2:] == ["-r", "/code/agent/requirements.txt"] and "--user" in install["command"]
    assert install["mounts"].agent_readonly and "gpus" not in install and install["env"] == {}
    assert phase.get("log_name") is None and phase["mounts"].workspace == install["mounts"].workspace
    assert out.ok
    recipe = InstallingRunner({"ok": True, "result": {"summary": "s"}})
    run_improve_recipe(make(recipe), conn=conn, node="n3", parent_id="root", agent_commit=commit, attempt=1)
    assert recipe.calls[0]["command"][-1] == "/agent/agent/requirements.txt"


def test_a_failed_install_of_contract_checked_code_is_a_kernel_fault(env, tmp_path):
    make, conn, root, _ = env
    commit = _with_requirements(make, root, "nope==0.0\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "unused"}}, code=1)
    with pytest.raises(KernelFault, match="No matching distribution"):
        run_edit_self(make(runner), conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    assert [c.get("log_name") for c in runner.calls] == ["install"]          # the phase never started


def test_a_caller_failed_mid_phase_cancels_the_sandbox_and_raises(env):
    make, conn, root, _ = env

    class FailingRunner(FakeRunner):
        def __call__(self, **kw):
            caller = self.penv.registry.lookup(kw["env"]["AR_TOKEN"])
            caller.fail("disk full while ingesting")
            assert kw["cancel"].is_set()
            return super().__call__(**kw)
    with pytest.raises(KernelFault, match="disk full while ingesting"):
        run_edit_self(make(FailingRunner({"ok": True, "result": {"summary": "x"}}, snap={})), conn=conn,
                      node="n1", parent_id="root", agent_commit=root, attempt=1)


def test_an_agent_killed_after_start_is_a_failed_attempt(env):
    make, conn, root, _ = env
    out = run_edit_self(make(FakeRunner(None, exit_code=137)), conn=conn, node="n1", parent_id="root",
                        agent_commit=root, attempt=1)
    assert not out.ok and "exit code 137" in out.error


def test_a_requirements_file_of_comments_installs_nothing(env, tmp_path):
    make, conn, root, _ = env
    commit = _with_requirements(make, root, "# nothing yet\n\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    run_edit_self(make(runner), conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    assert [c.get("log_name") for c in runner.calls] == [None]


def test_uncommittable_agent_tree_is_a_failed_attempt(env):
    """A chmod-000 file the agent leaves in /agent makes `git add` fail."""
    make, conn, root, _ = env

    def lock(agent):
        (agent / "locked.py").write_text("x = 1\n")
        (agent / "locked.py").chmod(0)

    def runner(*, mounts, env, **kw):
        lock(mounts.agent / "agent")
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {"summary": "x"}}))
        return RunResult(0, "", "", 1.0, "c")

    out = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    (out.attempt_dir / "agent" / "agent" / "locked.py").chmod(0o644)
    assert not out.ok and "commit" in out.error and out.commit is None


@pytest.mark.sandbox
def test_container_commits_data_through_the_real_tool_server(tmp_path):
    """No LLM: the fixture agent calls data_query and data_commit over the socket;
    the kernel creates a real commit and the phase validates it."""
    from pathlib import Path
    from ar_kernel.gateway.app import create_gateway_app
    from ar_kernel.gateway.mock import MockBook
    from ar_kernel.gateway.store import CallStore
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.tools.data_tools import DataTools, register_data_tools
    from ar_kernel.tools.jobs import register_job_tools
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
    from ar_kernel.tools.context import TokenRegistry
    from conftest import DERIVED, NAMED, staged_clip
    from ar_kernel.data.ingest import Candidate, Ingestor

    run = tmp_path / "run"
    telemetry.open(run)
    telemetry.start("ar.run")
    conn = open_db(run)
    for i in range(4):                                   # seed the pool with 4 distinct clips
        d = run / "staging" / "seed" / f"c{i}"
        video, annotation, pose = staged_clip(d, 4.0 + 0.5 * i)
        Ingestor(CFG, run, conn).ingest([Candidate(video=video, annotation=annotation, pose=pose,
                                                       derived_from=DERIVED, named=NAMED)], node_id="root", attempt=1)
    repo = AgentsRepo(run / "agents.git")
    commit = repo.init(Path(__file__).parent / "fixtures" / "agents" / "tool_user")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.set_fields("root", agent_commit=commit), nodes.create("n1", "root", 1)
    reg, queue = TokenRegistry(), JobQueue(run, threading.Lock(), wait_cap_s=5)
    kit, mcp = ToolKit(reg, run), new_mcp(reg)
    data = DataTools(CFG, run, [0, 1, 2, 3], threading.Lock())
    register_data_tools(mcp, kit, data)
    register_snapshot_tool(mcp, kit, SnapshotStore(run, repo, data.usable))
    register_job_tools(mcp, kit, queue)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=reg, store=CallStore(run), allowed_models={"mock-model"},
                                      upstream=None, mocks=MockBook.default()), build_tool_app(mcp))
    try:
        penv = PhaseEnv(cfg=CFG, run_dir=run, run_id="e2e", registry=reg, queue=queue,
                        repo=repo, socket_dir=services.socket_dir, gpus=[], default_model="mock-model")
        out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=commit,
                                 attempt=1)
    finally:
        services.stop(), queue.shutdown()
    assert out.ok, out.error
    assert out.result["rationale"] == "4 clips" and out.result["summary"] == "the pool's first clips"
    kinds = {r["kind"] for r in activity(conn, "n1")}
    assert "tool" in kinds and "accepted" in kinds
    records = telemetry_records(run)
    [sandbox] = [r for r in records if r["kind"] == "end" and r["name"] == "sandbox"]
    assert sandbox["status"] == "ok" and "rusage" in telemetry_payload(run, sandbox["output"])
    assert {"sandbox.changes", "phase.close"} <= {r["name"] for r in records if r["kind"] == "end"}
    assert tree.check(tree.load(run)) == []


def test_agents_see_every_finished_node_under_nodes(env, tmp_path):
    make, conn, root, _ = env
    NodeStore(conn).set_status("root", "scored"), NodeStore(conn).set_status("n1", "train_failed")
    NodeStore(conn).create("n2", "n1", 2)
    for node in ("root", "n1", "n2"):
        (tmp_path / "run" / "nodes" / node / "eval").mkdir(parents=True)
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    run_edit_self(make(runner), conn=conn, node="n2", parent_id="n1", agent_commit=root, attempt=1)
    assert runner.calls[0]["mounts"].nodes == {"n1": tmp_path / "run" / "nodes" / "n1",
                                               "root": tmp_path / "run" / "nodes" / "root"}


def test_smoke_contexts_are_the_nodes_real_contexts_marked_dry_run(env):
    from ar_kernel.agent_phase import smoke_contexts
    make, conn, _, _ = env
    out = smoke_contexts(make(FakeRunner({})), conn=conn, node="n1", parent_id="root")
    assert sorted(out) == ["edit_self", "improve_recipe"]
    for ctx in out.values():
        assert ctx["dry_run"] is True and [n["node_id"] for n in ctx["lineage"]] == ["root"]
        json.dumps(ctx)
    assert out["improve_recipe"]["tools"] and out["improve_recipe"]["tunable_rules"]


# --- Results name a kernel snapshot -------------------------------------------------------------


def test_the_accepted_edit_is_the_snapshot_not_the_tree_at_the_end(env):
    make, conn, root, _ = env

    def runner(*, mounts, env, **kw):
        entry = mounts.agent / "agent" / "entry.py"
        entry.write_text("# snapshotted\n" + entry.read_text())
        result = {"snapshot_id": snapshot_id(penv, env), "summary": "s"}
        entry.write_text("# changed after the snapshot\n")
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": result}))
        return RunResult(0, "", "", 1.0, "c")

    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=1)
    assert out.ok and out.commit == out.result["snapshot_id"]
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert [(r["phase"], r["attempt"], r["kind"], r["name"]) for r in activity(conn, "n1")] == [
        ("edit_self", 1, "accepted", out.commit)]
    assert penv.repo.read_file(out.commit, "agent/entry.py").startswith("# snapshotted")


@pytest.mark.parametrize("snapshot", ["0" * 40, "../../heads/node/root", "another attempt's"])
def test_an_edit_result_naming_no_snapshot_of_this_attempt_fails_and_commits_the_tree(env, snapshot):
    make, conn, root, _ = env
    if snapshot == "another attempt's":
        first = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
        snapshot = run_edit_self(make(first), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1).commit
    runner = FakeRunner({"ok": True, "result": {"snapshot_id": snapshot, "summary": "s"}}, edit="# end tree\n")
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=2)
    assert not out.ok and out.result is None and "snapshot" in out.error
    assert penv.repo.read_file(out.commit, "agent/entry.py") == "# end tree\n"
    assert penv.repo.resolve("refs/attempts/n1/edit_self-2") == out.commit


def test_an_end_tree_that_cannot_be_committed_is_added_to_the_error(env, monkeypatch):
    make, conn, root, _ = env
    penv = make(FakeRunner({"ok": True, "result": {"snapshot_id": "0" * 40, "summary": "s"}}, edit="# end tree\n"))
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr=b"disk full")
    monkeypatch.setattr(penv.repo, "commit_tree", fail)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=1)
    assert not out.ok and out.commit is None
    assert out.error.endswith("\nThe agent tree could not be committed either: disk full")
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") is None


def _data_commit(conn, commit_id="dc1", status="draft", clips=(), node="n1"):
    manifest = json.dumps({"datasets": {"cam": {"weight": 1.0, "clips": list(clips)}}})
    conn.execute("INSERT INTO data_commits (commit_id, node_id, attempt, status, manifest, message, created_at) "
                 "VALUES (?,?,?,?,?,?,?)", (commit_id, node, 1, status, manifest, "m", 0.0))
    return commit_id


def test_a_recipe_result_resolves_to_its_snapshot(env):
    make, conn, root, _ = env
    penv = make(None)
    commit = _data_commit(conn)
    snap = {"data_commit": commit, "recipe": {"lr": 0.1}, "rationale": "why"}
    penv.runner = runner = FakeRunner({"ok": True, "result": {"summary": "tested x", "train": True}}, snap=snap)
    runner.penv = penv
    out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    sid = json.loads((out.attempt_dir / "workspace" / "result.json").read_text())["result"]["snapshot_id"]
    assert out.ok and out.result == {"train": True, "snapshot_id": sid, **snap, "summary": "tested x"}


@pytest.mark.parametrize("snap,error", [({}, "holds no data commit"), (None, "not a snapshot of n1")])
def test_a_recipe_result_without_a_usable_snapshot_is_a_failed_attempt(env, snap, error):
    make, conn, root, _ = env
    result = {"summary": "s", "train": True} if snap is not None else {"snapshot_id": "f" * 32, "summary": "s", "train": True}
    out = run_improve_recipe(make(FakeRunner({"ok": True, "result": result}, snap=snap)), conn=conn, node="n1",
                             parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None and error in out.error


def test_rerunning_an_attempt_clears_its_snapshots(env):
    make, conn, root, _ = env
    first = FakeRunner({"ok": True, "result": {"summary": "s", "train": False}}, snap={})
    out1 = run_improve_recipe(make(first), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    stale = json.loads((out1.attempt_dir / "workspace" / "result.json").read_text())["result"]["snapshot_id"]
    edit = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    penv = make(edit)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    old = penv.repo.resolve("refs/attempts/n1/edit_self-1")
    again = FakeRunner({"ok": True, "result": {"snapshot_id": stale, "summary": "s", "train": False}})
    out = run_improve_recipe(make(again), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "not a snapshot" in out.error
    rerun = FakeRunner({"ok": True, "result": {"snapshot_id": old, "summary": "s"}})
    assert not run_edit_self(make(rerun), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1).ok


@pytest.mark.parametrize("snap", [{}, {"data_commit": "dc1", "recipe": {"lr": 0.1}, "rationale": "why"}])
def test_a_continue_result_names_any_snapshot_of_this_subphase(env, snap):
    make, conn, root, _ = env
    _data_commit(conn)
    runner = FakeRunner({"ok": True, "result": {"summary": "half done", "train": False}}, snap=snap)
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    sid = json.loads((out.attempt_dir / "workspace" / "result.json").read_text())["result"]["snapshot_id"]
    assert out.ok and out.result == {"train": False, "snapshot_id": sid, "data_commit": snap.get("data_commit"),
                                     "summary": "half done"}


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("status,error", [("obsolete", "a result was submitted that did not name it"),
                                          ("failed", "the work that made it failed")])
def test_a_result_naming_a_data_commit_that_can_no_longer_be_used_fails(env, train, status, error):
    make, conn, root, _ = env
    snap = {"data_commit": _data_commit(conn, status=status), "recipe": {"lr": 0.1}, "rationale": "why"}
    runner = FakeRunner({"ok": True, "result": {"summary": "s", "train": train}}, snap=snap)
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and f"data commit dc1 can no longer be used: {error}" in out.error


@pytest.mark.parametrize("train", [True, False])
def test_a_result_naming_a_data_commit_that_does_not_exist_fails(env, train):
    make, conn, root, _ = env
    snap = {"data_commit": "dc9", "recipe": {"lr": 0.1}, "rationale": "why"}
    runner = FakeRunner({"ok": True, "result": {"summary": "s", "train": train}}, snap=snap)
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "names data commit 'dc9', which does not exist" in out.error


@pytest.mark.parametrize("train", [True, False])
def test_a_result_whose_data_commit_holds_too_many_new_clips_fails(env, train):
    make, conn, root, _ = env
    _data_commit(conn, "dc1", "accepted", ["kept"], node="n0")
    _data_commit(conn, "dc2", clips=["kept", "a", "b"])
    _data_commit(conn, "dc3", clips=["kept", "a", "b", "c"])

    def run(commit):
        penv = make(None)
        penv.cfg = KernelConfig(raw=merged(CFG.raw, {"improve_recipe": {"new_clips_per_commit": 2}}),
                                repo_root=CFG.repo_root)
        penv.runner = FakeRunner({"ok": True, "result": {"summary": "s", "train": train}},
                                 snap={"data_commit": commit, "recipe": {"lr": 0.1}, "rationale": "why"})
        penv.runner.penv = penv
        return run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert run("dc2").ok
    out = run("dc3")
    assert not out.ok and "3 new clips, more than the 2 a data commit may hold" in out.error


def test_a_continue_result_naming_no_snapshot_of_this_subphase_fails(env):
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"snapshot_id": "f" * 32, "summary": "s", "train": False}})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "not a snapshot of n1" in out.error


def test_earlier_subphases_are_mounted_and_in_the_context_and_failed_ones_are_not(env):
    make, conn, root, _ = env
    nodes = NodeStore(conn)
    for k, outcome in ((1, "continued"), (2, "failed"), (3, "continued")):
        runner = FakeRunner({"ok": True, "result": {"summary": f"subphase {k}", "train": False}}, snap={})
        out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=k)
        nodes.add_attempt("n1", "improve_recipe", k, outcome,
                          {"snapshot_id": out.result["snapshot_id"], "summary": out.result["summary"]})
    runner = FakeRunner({"ok": False, "error": "x"}, exit_code=1)
    penv = make(runner)
    run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=4)
    call = runner.calls[0]
    assert call["mounts"].subphases == [attempt_dirs(penv.run_dir, "n1", "improve_recipe", k)["workspace"] for k in (1, 3)]
    assert "AR_ATTEMPT" not in call["env"]
    ctx = json.loads((call["mounts"].context / "context.json").read_text())
    assert [(p["path"], p["summary"], p["snapshot"]) for p in ctx["earlier_subphases"]] == [
        ("/earlier/1", "subphase 1", None), ("/earlier/2", "subphase 3", None)]
    assert ctx["new_clips_per_commit"] == CFG.get("improve_recipe.new_clips_per_commit")


def test_a_subphase_run_again_after_cleanup_gets_the_same_context_and_mounts(env):
    from ar_kernel.archive.clips import ClipStore
    make, conn, root, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "subphase 1", "train": False}}, snap={})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    NodeStore(conn).add_attempt("n1", "improve_recipe", 1, "continued",
                                {"snapshot_id": out.result["snapshot_id"], "summary": "subphase 1"})

    def subphase_2():
        runner = FakeRunner({"ok": False, "error": "x"}, exit_code=1)
        run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=2)
        call = runner.calls[0]
        return call["mounts"].subphases, json.loads((call["mounts"].context / "context.json").read_text())
    fresh = subphase_2()
    ClipStore(conn).add({"clip_id": "half", "video_digest": "v", "caption_digest": "c", "annotation_digest": "a",
                         "pose_digest": "p", "metadata": {}, "warnings": [], "derived_from": {"kind": "model", "name": "x"},
                         "ingested_by": "n1", "attempt": 2})
    assert subphase_2()[1]["clip_pool_size"] == fresh[1]["clip_pool_size"] + 1
    ClipStore(conn).remove_ingested("n1", 2)                  # what the loop's cleanup of subphase 2 does to the pool
    assert subphase_2() == fresh


def test_the_smoke_context_holds_an_earlier_subphase(env):
    from ar_kernel.agent_phase import smoke_contexts
    make, conn, _, _ = env
    ctx = smoke_contexts(make(FakeRunner({})), conn=conn, node="n1", parent_id="root")["improve_recipe"]
    subphases = ctx["earlier_subphases"]
    assert [p["path"] for p in subphases] == ["/earlier/1"] and subphases[0]["snapshot"]["rationale"] == "dry run"


def test_a_phase_records_its_start_hand_over_close_and_result(env, tmp_path):
    make, conn, root, _ = env
    run = tmp_path / "run"
    telemetry.open(run)
    telemetry.start("ar.run")
    NodeStore(conn).set_status("root", "scored")
    (run / "nodes" / "root").mkdir(parents=True)
    seen = []

    class Seeing(FakeRunner):
        def __call__(self, **kw):
            seen.append((self.penv.registry.lookup(kw["env"]["AR_TOKEN"]).span, kw["env"]["AR_TOKEN"]))
            return super().__call__(**kw)
    attempts = []
    for result in ({"ok": False, "error": "gave up"}, {"ok": True, "result": {"summary": "s"}}):
        with telemetry.span("phase.attempt", {"node": "n1", "phase": "edit_self", "attempt": 1}) as sp:
            out = run_edit_self(make(Seeing(result, edit="x = 2\n", snap={} if result["ok"] else None)), conn=conn, node="n1",
                                parent_id="root", agent_commit=root, attempt=1)
            sp.output = {"outcome": "passed" if out.ok else "failed"}
        attempts.append(sp)
    assert out.ok and [s for s, _ in seen] == attempts and seen[1][1] in telemetry.writer().secrets
    records = telemetry_records(run)

    def under(sp, kind="start"):
        return [r for r in records if r["kind"] == kind and (r["parent"] if kind == "start" else r["span"]) == sp.id]
    first, second = (under(sp) for sp in attempts)
    assert "fs.rmtree" not in [r["name"] for r in first] and "fs.rmtree" in [r["name"] for r in second]
    for starts in (first, second):
        names = [r["name"] for r in starts]
        assert "process" in names and "phase.close" in names           # the checkout; the token's end
        [changes] = [r for r in records if r["kind"] == "end" and r["span"] in {x["span"] for x in starts}
                     and r["name"] == "sandbox.changes"]
        assert telemetry_payload(run, changes["output"])["/agent"]["changed"] == [["agent/entry.py", 6]]
        assert [r for r in starts if r["name"] == "fs.write"
                and telemetry_payload(run, r["inputs"])["path"].endswith("context/context.json")]
    verdicts = [(d["chosen"], d["reason"]) for sp in attempts for d in under(sp, "decision")
                if d["name"] == "phase.result"]
    assert verdicts == [("fail", "gave up"), ("accept", "a valid result that names a snapshot of this attempt")]
    assert [telemetry_payload(run, r["output"]) for r in records if r["kind"] == "end" and r["name"] == "phase.close"] \
        == [{"jobs_cancelled": 0}] * 2
    assert json.loads((run / "control" / "hf_upload.json").read_text())["mounted"] == ["nodes/root"]
    assert tree.check(tree.load(run)) == []
