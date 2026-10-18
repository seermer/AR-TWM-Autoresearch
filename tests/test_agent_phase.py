import json
import subprocess
import threading

import pytest

from ar_kernel.agent_phase import PhaseEnv, attempt_dirs, run_edit_self, run_improve_recipe
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.snapshot import SnapshotStore, register_snapshot_tool
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()


def snapshot_id(penv, env, **recipe) -> str:
    """What the agent's call of the snapshot tool returns, taken from inside the 'container'."""
    caller = penv.registry.lookup(env["AR_TOKEN"])
    return SnapshotStore(penv.run_dir, penv.repo, lambda c: True).take(
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
    rec = Recorder(tmp_path / "run")
    conn = open_db(tmp_path / "run")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root)
    nodes.create("n1", "root", 1)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=1)

    def make(runner):
        penv = PhaseEnv(cfg=CFG, run_dir=tmp_path / "run", run_id="t", recorder=rec,
                        registry=TokenRegistry(rec), queue=queue, repo=repo,
                        socket_dir=tmp_path / "sock", gpus=[0, 1, 2, 3], default_model="gpt-x",
                        runner=runner)
        if isinstance(runner, FakeRunner):
            runner.penv = penv
        return penv
    yield make, conn, root, rec, queue
    queue.shutdown()


def test_attempt_layout(tmp_path):
    d = attempt_dirs(tmp_path, "n1", "edit_self", 2)
    assert d["agent"] == tmp_path / "nodes" / "n1" / "attempts" / "edit_self-2" / "agent"
    assert d["staging"] == tmp_path / "staging" / "n1" / "edit_self-2"


def test_edit_self_commits_the_edited_code_to_an_attempt_ref(env):
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "tightened prompts"}},
                        edit="# v2\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", snap={})
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert out.ok and out.result == {"snapshot_id": out.commit, "summary": "tightened prompts"}
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert "# v2" in penv.repo.read_file(out.commit, "agent/entry.py")
    assert runner.calls[0]["mounts"].agent_readonly is False


def test_failed_edit_attempt_is_still_committed(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": False, "error": "RuntimeError: boom"}, edit="broken(\n", exit_code=1)
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "boom" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_token_is_revoked_and_jobs_cancelled_after_the_phase(env, monkeypatch):
    make, conn, root, _, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, snap={})
    penv = make(runner)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None and cancelled == [token]


def test_staged_files_appear_in_the_recorded_diff(env):
    """/workspace/staging is a separate host directory (a nested mount), so the
    workspace snapshot alone would miss every staged download."""
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, staged="hf/clip.mp4", snap={})
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    end = [e for e in rec.read_events("n1") if e["type"] == "phase.end"][0]
    assert rec.load_payload(end["payload"])["diffs"]["staging"]["added"] == ["hf/clip.mp4"]


def test_runner_gets_the_agent_env_shared_memory_and_the_runs_gpus(env):
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    call = runner.calls[0]
    assert call["env_dir"] == CFG.repo_root / ".envs" / "agent" and call["shm_gb"] == CFG.get("sandbox.shm_gb") == 64
    assert call["gpus"] == [0, 1, 2, 3] and "timeout_s" not in call and "liveness" not in call


def test_improve_recipe_mounts_code_read_only_and_rejects_unknown_commits(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}},
                        snap={"data_commit": "f" * 64, "recipe": {}, "rationale": "r"})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    assert runner.calls[0]["mounts"].agent_readonly is True
    assert not out.ok and "data commit" in out.error


def test_improve_recipe_lists_the_caption_tool_when_its_backend_is_registered(env):
    from ar_kernel.tools.captioner import CaptionBackend
    make, conn, root, rec, queue = env
    penv = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    queue.register(CaptionBackend(CFG, penv.run_dir, penv.gpus, penv.registry, rec))
    run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=1)
    tools = json.loads((penv.runner.calls[0]["mounts"].context / "context.json").read_text())["tools"]
    assert "caption_videos" in tools and "job_wait" in tools and "hf_list_files" in tools


def test_a_retry_starts_clean(env):
    make, conn, root, _, _ = env
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
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {}})   # missing snapshot_id and summary
    out = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None and out.error


def test_invalid_recipe_result_missing_snapshot_id_is_a_failed_attempt_not_a_keyerror(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "s"}})   # no snapshot_id
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    assert not out.ok and out.result is None and out.error


def test_token_redaction_falls_back_to_env_recorder_when_registry_uses_a_different_one(env, tmp_path):
    """Ruling 5: TokenRegistry.issue() only redacts on the recorder it was built
    with. In the normal wiring that IS env.recorder, so nothing extra is needed
    (verified here); but if it were a different Recorder, env.recorder must
    still get the redaction, or a token printed by the agent would leak."""
    make, conn, root, rec, _ = env
    penv = make(FakeRunner({"ok": True, "result": {"summary": "x"}}))
    assert penv.registry is not None
    # Normal wiring: registry and env share one recorder, so issue() already redacts it.
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    token1 = penv.runner.calls[0]["env"]["AR_TOKEN"]
    assert token1 in rec.redact

    # Now simulate a registry built against a *different* recorder (as the
    # contract harness does): env.recorder must still see the redaction.
    other_recorder = Recorder(tmp_path / "other")
    penv.registry = TokenRegistry(other_recorder)
    runner2 = FakeRunner({"ok": True, "result": {"summary": "y"}})
    penv.runner = runner2
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=2)
    token2 = runner2.calls[0]["env"]["AR_TOKEN"]
    assert token2 in rec.redact and token2 in other_recorder.redact


# --- Hostile or broken agent output -----------------------------------------------------


def test_malformed_json_result_is_a_failed_attempt_and_still_committed(env):
    """Bad JSON in result.json must not raise a JSONDecodeError out
    of the kernel, and edit_self must still commit the attempt."""
    make, conn, root, _, _ = env

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
    make, conn, root, _, _ = env

    def bad_bytes(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_bytes(b"\xff\xfe\x00\x01")
        return RunResult(0, "", "", 1.0, "c")

    out = run_edit_self(make(bad_bytes), conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=1)
    assert not out.ok and out.result is None and out.error


def test_non_object_json_result_is_a_failed_attempt_not_a_crash(env):
    """A JSON value that isn't an object (e.g. a list) must not
    raise an AttributeError out of the kernel on body.get(...)."""
    make, conn, root, _, _ = env

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
    make, conn, root, _, _ = env
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
    make, conn, root, _, _ = env
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
    make, conn, root, _, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner(None, raises=RuntimeError("docker exploded"))
    penv = make(runner)
    with pytest.raises(RuntimeError, match="docker exploded"):
        run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None
    assert cancelled == [token]


def test_ok_must_be_the_literal_true_not_merely_truthy(env):
    """`{"ok": 1}` (truthy but not True) must not be treated as success."""
    make, conn, root, _, _ = env
    out = run_edit_self(make(FakeRunner({"ok": 1, "result": {"summary": "x"}})), conn=conn, node="n1",
                        parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None


def test_error_field_is_coerced_to_a_string_with_a_fallback(env):
    """A non-string (or missing) error field must not crash telemetry
    or PhaseOutcome; a missing error gets a clear fallback message."""
    make, conn, root, _, _ = env
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
    make, conn, root, _, _ = env
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
    make, conn, root, _, _ = env
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


def test_listed_requirements_are_installed_before_the_phase_and_kept_out_of_its_diff(env, tmp_path):
    make, conn, root, rec, _ = env
    commit = _with_requirements(make, root, "# a comment\nrich==13.0\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    out = run_edit_self(make(runner), conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    install, phase = runner.calls
    assert install["command"][-2:] == ["-r", "/code/agent/requirements.txt"] and "--user" in install["command"]
    assert install["mounts"].agent_readonly and "gpus" not in install and install["env"] == {}
    assert phase.get("log_name") is None and phase["mounts"].workspace == install["mounts"].workspace
    assert out.ok
    end = [e for e in rec.read_events("n2") if e["type"] == "phase.end"][0]
    assert not any(".local" in p for p in rec.load_payload(end["payload"])["diffs"]["workspace"]["added"])
    recipe = InstallingRunner({"ok": True, "result": {"summary": "s"}})
    run_improve_recipe(make(recipe), conn=conn, node="n3", parent_id="root", agent_commit=commit, attempt=1)
    assert recipe.calls[0]["command"][-1] == "/agent/agent/requirements.txt"


def test_a_failed_install_is_a_failed_attempt_and_edit_self_still_commits(env, tmp_path):
    make, conn, root, rec, _ = env
    commit = _with_requirements(make, root, "nope==0.0\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "unused"}}, code=1)
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    assert not out.ok and "failed to install (exit code 1)" in out.error and "No matching distribution" in out.error
    assert [c.get("log_name") for c in runner.calls] == ["install"]          # the phase never started
    assert penv.repo.resolve("refs/attempts/n2/edit_self-1") == out.commit
    assert [e for e in rec.read_events("n2") if e["type"] == "phase.end"][0]["ok"] is False


def test_a_requirements_file_of_comments_installs_nothing(env, tmp_path):
    make, conn, root, _, _ = env
    commit = _with_requirements(make, root, "# nothing yet\n\n", tmp_path / "src")
    runner = InstallingRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    run_edit_self(make(runner), conn=conn, node="n2", parent_id="root", agent_commit=commit, attempt=1)
    assert [c.get("log_name") for c in runner.calls] == [None]


def test_uncommittable_agent_tree_is_a_failed_attempt(env):
    """A chmod-000 file the agent leaves in /agent makes `git add` fail."""
    make, conn, root, _, _ = env

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
    from conftest import DERIVED, staged_clip
    from ar_kernel.data.ingest import Candidate, Ingestor

    run = tmp_path / "run"
    rec, conn = Recorder(run), open_db(run)
    for i in range(4):                                   # seed the pool with 4 distinct clips
        d = run / "staging" / "seed" / f"c{i}"
        video, annotation, pose = staged_clip(d, 4.0 + 0.5 * i)
        Ingestor(CFG, run, conn, rec).ingest([Candidate(video=video, annotation=annotation, pose=pose,
                                                       derived_from=DERIVED)], node_id="root")
    repo = AgentsRepo(run / "agents.git")
    commit = repo.init(Path(__file__).parent / "fixtures" / "agents" / "tool_user")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.set_fields("root", agent_commit=commit), nodes.create("n1", "root", 1)
    reg, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5)
    kit, mcp = ToolKit(reg, rec), new_mcp()
    data = DataTools(CFG, run, rec, [0, 1, 2, 3], threading.Lock())
    register_data_tools(mcp, kit, data)
    register_snapshot_tool(mcp, kit, SnapshotStore(run, repo, data.commit_exists))
    register_job_tools(mcp, kit, queue)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=reg, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=MockBook.default()), build_tool_app(mcp))
    try:
        penv = PhaseEnv(cfg=CFG, run_dir=run, run_id="e2e", recorder=rec, registry=reg, queue=queue,
                        repo=repo, socket_dir=services.socket_dir, gpus=[], default_model="mock-model")
        out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=commit,
                                 attempt=1)
    finally:
        services.stop(), queue.shutdown()
    assert out.ok, out.error
    assert out.result["rationale"] == "4 clips" and out.result["summary"] == "the pool's first clips"
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert "tool.call" in kinds and "phase.end" in kinds


def test_agents_see_every_finished_node_under_nodes(env, tmp_path):
    make, conn, root, _, _ = env
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
    make, conn, _, _, _ = env
    out = smoke_contexts(make(FakeRunner({})), conn=conn, node="n1", parent_id="root")
    assert sorted(out) == ["edit_self", "improve_recipe"]
    for ctx in out.values():
        assert ctx["dry_run"] is True and [n["node_id"] for n in ctx["lineage"]] == ["root"]
        json.dumps(ctx)
    assert out["improve_recipe"]["tools"] and out["improve_recipe"]["tunable_rules"]


# --- Results name a kernel snapshot -------------------------------------------------------------


def test_the_accepted_edit_is_the_snapshot_not_the_tree_at_the_end(env):
    make, conn, root, _, _ = env

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
    assert penv.repo.read_file(out.commit, "agent/entry.py").startswith("# snapshotted")


@pytest.mark.parametrize("snapshot", ["0" * 40, "../../heads/node/root", "another attempt's"])
def test_an_edit_result_naming_no_snapshot_of_this_attempt_fails_and_commits_the_tree(env, snapshot):
    make, conn, root, _, _ = env
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
    make, conn, root, _, _ = env
    penv = make(FakeRunner({"ok": True, "result": {"snapshot_id": "0" * 40, "summary": "s"}}, edit="# end tree\n"))
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr=b"disk full")
    monkeypatch.setattr(penv.repo, "commit_tree", fail)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                        attempt=1)
    assert not out.ok and out.commit is None
    assert out.error.endswith("\nThe agent tree could not be committed either: disk full")
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") is None


def _data_commit(conn):
    conn.execute("INSERT INTO data_commits (commit_id, parent_commit, node_id, attempt, manifest, message, "
                 "created_at) VALUES (?,?,?,?,?,?,?)", ("c" * 64, None, "n1", 1, "{}", "m", 0.0))
    return "c" * 64


def test_a_recipe_result_resolves_to_its_snapshot(env):
    make, conn, root, rec, _ = env
    penv = make(None)
    commit = _data_commit(conn)
    snap = {"data_commit": commit, "recipe": {"lr": 0.1}, "rationale": "why"}
    penv.runner = runner = FakeRunner({"ok": True, "result": {"summary": "tested x"}}, snap=snap)
    runner.penv = penv
    out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1)
    sid = json.loads((out.attempt_dir / "workspace" / "result.json").read_text())["result"]["snapshot_id"]
    assert out.ok and out.result == {"snapshot_id": sid, **snap, "summary": "tested x"}
    end = [e for e in rec.read_events("n1") if e["type"] == "phase.end"][0]
    assert rec.load_payload(end["payload"])["result"] == out.result


@pytest.mark.parametrize("snap,error", [({}, "holds no data commit"), (None, "not a snapshot of n1")])
def test_a_recipe_result_without_a_usable_snapshot_is_a_failed_attempt(env, snap, error):
    make, conn, root, _, _ = env
    result = {"summary": "s"} if snap is not None else {"snapshot_id": "f" * 32, "summary": "s"}
    out = run_improve_recipe(make(FakeRunner({"ok": True, "result": result}, snap=snap)), conn=conn, node="n1",
                             parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and out.result is None and error in out.error


def test_rerunning_an_attempt_clears_its_snapshots(env):
    make, conn, root, _, _ = env
    first = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    out1 = run_improve_recipe(make(first), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    stale = json.loads((out1.attempt_dir / "workspace" / "result.json").read_text())["result"]["snapshot_id"]
    edit = FakeRunner({"ok": True, "result": {"summary": "s"}}, snap={})
    penv = make(edit)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    old = penv.repo.resolve("refs/attempts/n1/edit_self-1")
    again = FakeRunner({"ok": True, "result": {"snapshot_id": stale, "summary": "s"}})
    out = run_improve_recipe(make(again), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1)
    assert not out.ok and "not a snapshot" in out.error
    rerun = FakeRunner({"ok": True, "result": {"snapshot_id": old, "summary": "s"}})
    assert not run_edit_self(make(rerun), conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1).ok
