import json
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
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()


class FakeRunner:
    """Stands in for the container: edits /agent and writes /workspace/result.json."""
    def __init__(self, result, edit=None, exit_code=0, timed_out=False, staged=None, raises=None):
        self.result, self.edit, self.exit_code, self.timed_out = result, edit, exit_code, timed_out
        self.staged = staged
        self.raises = raises
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
            (mounts.workspace / "result.json").write_text(json.dumps(self.result))
        return RunResult(self.exit_code, self.timed_out, "", "", 1.0, [], "c")


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
    monkeypatch.setattr("ar_kernel.agent_phase.ensure_image", lambda cfg, reqs, **k: "img:test")

    def make(runner):
        return PhaseEnv(cfg=CFG, run_dir=tmp_path / "run", run_id="t", recorder=rec,
                        registry=TokenRegistry(rec), queue=queue, repo=repo,
                        socket_dir=tmp_path / "sock", gpus=[0, 1, 2, 3], default_model="gpt-x",
                        runner=runner)
    yield make, conn, root, rec, queue
    queue.shutdown()


def test_attempt_layout(tmp_path):
    d = attempt_dirs(tmp_path, "n1", "edit_self", 2)
    assert d["agent"] == tmp_path / "nodes" / "n1" / "attempts" / "edit_self-2" / "agent"
    assert d["staging"] == tmp_path / "staging" / "n1" / "edit_self-2"


def test_edit_self_commits_the_edited_code_to_an_attempt_ref(env):
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "tightened prompts"}},
                        edit="# v2\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert out.ok and out.result == {"summary": "tightened prompts", "component": None}
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert "# v2" in penv.repo.read_file(out.commit, "agent/entry.py")
    assert runner.calls[0]["mounts"].agent_readonly is False


def test_failed_edit_attempt_is_still_committed(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": False, "error": "RuntimeError: boom"}, edit="broken(\n", exit_code=1)
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and "boom" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_token_is_revoked_and_jobs_cancelled_after_the_phase(env, monkeypatch):
    make, conn, root, _, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}})
    penv = make(runner)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                  max_attempts=3, retry=None, nodes_remaining=5)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None and cancelled == [token]


def test_staged_files_appear_in_the_recorded_diff(env):
    """/workspace/staging is a separate host directory (a nested mount), so the
    workspace snapshot alone would miss every staged download."""
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, staged="hf/clip.mp4")
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                  max_attempts=3, retry=None, nodes_remaining=5)
    end = [e for e in rec.read_events("n1") if e["type"] == "phase.end"][0]
    assert rec.load_payload(end["payload"])["diffs"]["staging"]["added"] == ["hf/clip.mp4"]


def test_timeout_is_a_failed_attempt(env):
    make, conn, root, _, _ = env
    out = run_edit_self(make(FakeRunner(None, timed_out=True, exit_code=None)), conn=conn, node="n1",
                        parent_id="root", base_commit=root, attempt=1, max_attempts=3, retry=None,
                        nodes_remaining=5)
    assert not out.ok and out.timed_out and "timed out" in out.error


def test_improve_recipe_mounts_code_read_only_and_rejects_unknown_commits(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {"data_commit": "f" * 64, "recipe": {}, "rationale": "r"}})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert runner.calls[0]["mounts"].agent_readonly is True
    assert not out.ok and "data commit" in out.error


def test_improve_recipe_lists_the_caption_tool_when_its_backend_is_registered(env):
    from ar_kernel.tools.captioner import CaptionBackend
    make, conn, root, rec, queue = env
    penv = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    queue.register(CaptionBackend(CFG, penv.run_dir, penv.gpus, penv.registry, rec))
    run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    tools = json.loads((penv.runner.calls[0]["mounts"].context / "context.json").read_text())["tools"]
    assert "caption_videos" in tools and "job_wait" in tools


def test_retry_continues_from_the_previous_workspace(env):
    make, conn, root, _, _ = env
    first = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    out1 = run_improve_recipe(first, conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    (out1.attempt_dir / "workspace" / "notes.md").write_text("half-built dataset")
    runner = FakeRunner({"ok": False, "error": "y"}, exit_code=1)
    run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=2, max_attempts=3, retry={"failures": ["x"]}, nodes_remaining=5,
                       previous_workspace=out1.attempt_dir / "workspace")
    assert (runner.calls[0]["mounts"].workspace / "notes.md").read_text() == "half-built dataset"
    assert json.loads((runner.calls[0]["mounts"].context / "retry.json").read_text()) == {"failures": ["x"]}


# --- Controller rulings (each gets its own test) ---------------------------


def test_invalid_edit_result_is_a_failed_attempt_not_a_crash(env):
    """Ruling 1: an untrusted `ok: true` result is re-validated against the
    phase's pydantic model; a missing required field is a clean failure."""
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {}})   # missing required 'summary'
    out = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and out.result is None and out.error


def test_invalid_recipe_result_missing_data_commit_is_a_failed_attempt_not_a_keyerror(env):
    """Ruling 1: run_improve_recipe must not index body['result']['data_commit']
    before validation -- a missing field is ok=False, never a KeyError."""
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {"recipe": {}, "rationale": "r"}})   # no data_commit
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and out.result is None and out.error


def test_stale_result_json_is_removed_before_the_retry_container_runs(env):
    """Ruling 2: workspace/result.json copied over from previous_workspace must
    not be mistaken for the new attempt's own result."""
    make, conn, root, _, _ = env
    out1 = run_improve_recipe(
        make(FakeRunner({"ok": True, "result": {"data_commit": "f" * 64, "recipe": {}, "rationale": "r"}})),
        conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1, max_attempts=3,
        retry=None, nodes_remaining=5)
    assert (out1.attempt_dir / "workspace" / "result.json").exists()
    runner2 = FakeRunner(None)                      # does NOT write result.json on the retry
    out2 = run_improve_recipe(make(runner2), conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=2, max_attempts=3, retry={"failures": ["x"]}, nodes_remaining=5,
                              previous_workspace=out1.attempt_dir / "workspace")
    assert not out2.ok and "no result.json" in out2.error


def test_edit_self_passes_previous_workspace_through_on_retry(env):
    """Ruling 3: run_edit_self accepts previous_workspace and forwards it to
    _run, so edit_self retries keep /workspace/edit_plan.json."""
    make, conn, root, _, _ = env
    out1 = run_edit_self(make(FakeRunner({"ok": False, "error": "x"}, exit_code=1)), conn=conn, node="n1",
                         parent_id="root", base_commit=root, attempt=1, max_attempts=3, retry=None,
                         nodes_remaining=5)
    (out1.attempt_dir / "workspace" / "edit_plan.json").write_text('{"component": "prompts"}')
    runner2 = FakeRunner({"ok": True, "result": {"summary": "continued"}})
    run_edit_self(make(runner2), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=2,
                  max_attempts=3, retry={"failures": ["x"]}, nodes_remaining=5,
                  previous_workspace=out1.attempt_dir / "workspace")
    kept = runner2.calls[0]["mounts"].workspace / "edit_plan.json"
    assert kept.read_text() == '{"component": "prompts"}'


def test_staging_directory_is_moved_not_copied_on_retry(env, tmp_path):
    """Ruling 4: the previous attempt's staging dir is MOVEd (not copied) into
    the new attempt's staging path -- a partial download can be tens of GiB."""
    make, conn, root, _, _ = env
    run_dir = tmp_path / "run"
    out1 = run_improve_recipe(make(FakeRunner({"ok": False, "error": "x"}, exit_code=1, staged="hf/big.mp4")),
                              conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1,
                              max_attempts=3, retry=None, nodes_remaining=5)
    old_staging = attempt_dirs(run_dir, "n1", "improve_recipe", 1)["staging"]
    assert (old_staging / "hf" / "big.mp4").exists()
    run_improve_recipe(make(FakeRunner({"ok": False, "error": "y"}, exit_code=1)), conn=conn, node="n1",
                       parent_id="root", agent_commit=root, attempt=2, max_attempts=3,
                       retry={"failures": ["x"]}, nodes_remaining=5,
                       previous_workspace=out1.attempt_dir / "workspace")
    new_staging = attempt_dirs(run_dir, "n1", "improve_recipe", 2)["staging"]
    assert (new_staging / "hf" / "big.mp4").exists()
    assert not old_staging.exists()          # moved, not copied: nothing left behind


def test_token_redaction_falls_back_to_env_recorder_when_registry_uses_a_different_one(env, tmp_path):
    """Ruling 5: TokenRegistry.issue() only redacts on the recorder it was built
    with. In the normal wiring that IS env.recorder, so nothing extra is needed
    (verified here); but if it were a different Recorder, env.recorder must
    still get the redaction, or a token printed by the agent would leak."""
    make, conn, root, rec, _ = env
    penv = make(FakeRunner({"ok": True, "result": {"summary": "x"}}))
    assert penv.registry is not None
    # Normal wiring: registry and env share one recorder, so issue() already redacts it.
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    token1 = penv.runner.calls[0]["env"]["AR_TOKEN"]
    assert token1 in rec.redact

    # Now simulate a registry built against a *different* recorder (as the
    # contract harness does): env.recorder must still see the redaction.
    other_recorder = Recorder(tmp_path / "other")
    penv.registry = TokenRegistry(other_recorder)
    runner2 = FakeRunner({"ok": True, "result": {"summary": "y"}})
    penv.runner = runner2
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=2,
                  max_attempts=3, retry=None, nodes_remaining=5)
    token2 = runner2.calls[0]["env"]["AR_TOKEN"]
    assert token2 in rec.redact and token2 in other_recorder.redact


# --- Review fix round 1 -----------------------------------------------------


def test_malformed_json_result_is_a_failed_attempt_and_still_committed(env):
    """Finding 1: bad JSON in result.json must not raise a JSONDecodeError out
    of the kernel, and edit_self must still commit the attempt."""
    make, conn, root, _, _ = env

    def bad_json(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_text("{not valid json")
        return RunResult(0, False, "", "", 1.0, [], "c")

    penv = make(bad_json)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and out.result is None and out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_non_utf8_result_is_a_failed_attempt_not_a_crash(env):
    """Finding 1: non-UTF-8 bytes in result.json must not raise a
    UnicodeDecodeError out of the kernel."""
    make, conn, root, _, _ = env

    def bad_bytes(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_bytes(b"\xff\xfe\x00\x01")
        return RunResult(0, False, "", "", 1.0, [], "c")

    out = run_edit_self(make(bad_bytes), conn=conn, node="n1", parent_id="root", base_commit=root,
                        attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and out.result is None and out.error


def test_non_object_json_result_is_a_failed_attempt_not_a_crash(env):
    """Finding 1: a JSON value that isn't an object (e.g. a list) must not
    raise an AttributeError out of the kernel on body.get(...)."""
    make, conn, root, _, _ = env

    def array_body(*, mounts, env, **kw):
        (mounts.workspace / "result.json").write_text(json.dumps([1, 2, 3]))
        return RunResult(0, False, "", "", 1.0, [], "c")

    out = run_improve_recipe(make(array_body), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and "not a JSON object" in out.error


def test_symlinked_result_json_is_not_followed_onto_the_host(env, tmp_path):
    """Finding 1: a symlinked result.json must be treated as "no result", not
    resolved and read -- the target could be any host path the container's
    user can reach."""
    make, conn, root, _, _ = env
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"ok": True, "result": {"summary": "leaked"}}))

    def symlink_result(*, mounts, env, **kw):
        (mounts.workspace / "result.json").symlink_to(secret)
        return RunResult(0, False, "", "", 1.0, [], "c")

    penv = make(symlink_result)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and "no result.json" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_image_build_failure_is_a_failed_attempt_and_edit_self_still_commits(env, monkeypatch):
    """Finding 2: an agent-authored requirements.txt that fails to build must
    become a failed PhaseOutcome (with phase.end recorded), not an exception
    that skips edit_self's commit -- a retry needs that commit as its
    base_commit."""
    from ar_kernel.sandbox.image import ImageBuildError

    make, conn, root, rec, _ = env

    def boom(cfg, reqs, **k):
        raise ImageBuildError("pip install exploded")

    monkeypatch.setattr("ar_kernel.agent_phase.ensure_image", boom)
    runner = FakeRunner({"ok": True, "result": {"summary": "unused"}})
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and "pip install exploded" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert runner.calls == []                        # the container never started
    end = [e for e in rec.read_events("n1") if e["type"] == "phase.end"][0]
    assert end["ok"] is False


def test_rerunning_the_same_attempt_number_clears_stale_staging(env, tmp_path):
    """Finding 3: staging lives outside the attempt dir, so re-running the same
    attempt number must clear it too, or stale files pollute the new run."""
    make, conn, root, _, _ = env
    run_dir = tmp_path / "run"
    run_improve_recipe(make(FakeRunner({"ok": False, "error": "x"}, exit_code=1, staged="old/file.bin")),
                       conn=conn, node="n1", parent_id="root", agent_commit=root, attempt=1,
                       max_attempts=3, retry=None, nodes_remaining=5)
    staging = attempt_dirs(run_dir, "n1", "improve_recipe", 1)["staging"]
    assert (staging / "old" / "file.bin").exists()
    runner2 = FakeRunner({"ok": False, "error": "y"}, exit_code=1)   # nothing staged this time
    run_improve_recipe(make(runner2), conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=1, max_attempts=3, retry=None, nodes_remaining=5)   # same attempt number
    assert not (staging / "old" / "file.bin").exists()
    assert runner2.calls[0]["mounts"].staging == staging


def test_runner_exception_still_revokes_token_and_cancels_jobs(env, monkeypatch):
    """Finding 4: an exception raised by the container call (or anything after
    the token is issued) must still revoke the token and cancel its jobs."""
    make, conn, root, _, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner(None, raises=RuntimeError("docker exploded"))
    penv = make(runner)
    with pytest.raises(RuntimeError, match="docker exploded"):
        run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                      max_attempts=3, retry=None, nodes_remaining=5)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None
    assert cancelled == [token]


def test_ok_must_be_the_literal_true_not_merely_truthy(env):
    """Cheap fix: `{"ok": 1}` (truthy but not True) must not be treated as success."""
    make, conn, root, _, _ = env
    out = run_edit_self(make(FakeRunner({"ok": 1, "result": {"summary": "x"}})), conn=conn, node="n1",
                        parent_id="root", base_commit=root, attempt=1, max_attempts=3, retry=None,
                        nodes_remaining=5)
    assert not out.ok and out.result is None


def test_error_field_is_coerced_to_a_string_with_a_fallback(env):
    """Cheap fix: a non-string (or missing) error field must not crash telemetry
    or PhaseOutcome; a missing error gets a clear fallback message."""
    make, conn, root, _, _ = env
    out1 = run_edit_self(make(FakeRunner({"ok": False})), conn=conn, node="n1", parent_id="root",
                         base_commit=root, attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert out1.error == "agent reported failure without an error"

    out2 = run_edit_self(make(FakeRunner({"ok": False, "error": 42})), conn=conn, node="n1",
                         parent_id="root", base_commit=root, attempt=2, max_attempts=3, retry=None,
                         nodes_remaining=5)
    assert out2.error == "42"


# --- Final review: agent-hostile filesystem states are failed attempts, never kernel exceptions ---


def _edit_then_retry(make, conn, root, mutate):
    """Attempt 1 lets `mutate(agent_dir)` run inside the 'container'; attempt 2 retries from its commit."""
    def runner(*, mounts, env, **kw):
        mutate(mounts.agent / "agent")
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {"summary": "x"}}))
        return RunResult(0, False, "", "", 1.0, [], "c")

    out1 = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                         max_attempts=3, retry=None, nodes_remaining=5)
    retry_runner = FakeRunner({"ok": True, "result": {"summary": "unused"}})
    out2 = run_edit_self(make(retry_runner), conn=conn, node="n1", parent_id="root", base_commit=out1.commit,
                         attempt=2, max_attempts=3, retry={"failures": []}, nodes_remaining=5)
    return out1, out2, retry_runner


def test_committed_symlink_leaving_the_tree_fails_the_retry_instead_of_raising(env):
    make, conn, root, _, _ = env
    out1, out2, retry_runner = _edit_then_retry(make, conn, root,
                                                lambda agent: (agent / "ctx").symlink_to("/context/x"))
    assert out1.ok and out1.commit
    assert not out2.ok and "check out" in out2.error and retry_runner.calls == []
    assert out2.commit is None


@pytest.mark.parametrize("mutate", [
    lambda agent: ((agent / "requirements.txt").mkdir(), (agent / "requirements.txt" / "x").write_text("")),
    lambda agent: (agent / "requirements.txt").write_bytes(b"\xff\xfe not utf-8"),
], ids=["directory", "non_utf8"])
def test_unreadable_requirements_txt_is_a_failed_attempt(env, mutate):
    make, conn, root, _, _ = env
    _, out2, retry_runner = _edit_then_retry(make, conn, root, mutate)
    assert not out2.ok and "requirements.txt" in out2.error and retry_runner.calls == []
    assert out2.commit                                   # the (unchanged) attempt tree is still committed


def test_uncommittable_agent_tree_is_a_failed_attempt(env):
    """A chmod-000 file the agent leaves in /agent makes `git add` fail."""
    make, conn, root, _, _ = env

    def lock(agent):
        (agent / "locked.py").write_text("x = 1\n")
        (agent / "locked.py").chmod(0)

    def runner(*, mounts, env, **kw):
        lock(mounts.agent / "agent")
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {"summary": "x"}}))
        return RunResult(0, False, "", "", 1.0, [], "c")

    out = run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    (out.attempt_dir / "agent" / "agent" / "locked.py").chmod(0o644)
    assert not out.ok and "commit" in out.error and out.commit is None


def test_previous_workspace_with_a_fifo_is_a_failed_attempt(env):
    import os

    make, conn, root, _, _ = env
    first = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    out1 = run_improve_recipe(first, conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    os.mkfifo(out1.attempt_dir / "workspace" / "pipe")
    runner = FakeRunner({"ok": False, "error": "y"}, exit_code=1)
    out2 = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=2, max_attempts=3, retry={"failures": ["x"]}, nodes_remaining=5,
                              previous_workspace=out1.attempt_dir / "workspace")
    assert not out2.ok and "previous workspace" in out2.error and runner.calls == []


@pytest.mark.docker
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
    from conftest import make_mp4, write_caption, write_poses
    from ar_kernel.data.ingest import Candidate, Ingestor

    run = tmp_path / "run"
    rec, conn = Recorder(run), open_db(run)
    for i in range(4):                                   # seed the pool with 4 distinct clips
        d = run / "staging" / "seed" / f"c{i}"
        secs = 4.0 + 0.5 * i
        make_mp4(d / "v.mp4", seconds=secs), write_caption(d / "c.json"), write_poses(d / "p.npz", int(secs * 30))
        Ingestor(CFG, run, conn, rec).ingest([Candidate(video=d / "v.mp4", caption=d / "c.json",
                                                       pose=d / "p.npz", camera_motion="moving",
                                                       provenance={"kind": "derived", "from": [], "transform": "t"})],
                                             node_id="root")
    repo = AgentsRepo(run / "agents.git")
    commit = repo.init(Path(__file__).parent / "fixtures" / "agents" / "tool_user")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.set_fields("root", agent_commit=commit), nodes.create("n1", "root", 1)
    reg, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5)
    kit, mcp = ToolKit(reg, rec), new_mcp()
    register_data_tools(mcp, kit, DataTools(CFG, run, rec, [0, 1, 2, 3], threading.Lock()))
    register_job_tools(mcp, kit, queue)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=reg, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=MockBook.default()), build_tool_app(mcp))
    try:
        penv = PhaseEnv(cfg=CFG, run_dir=run, run_id="e2e", recorder=rec, registry=reg, queue=queue,
                        repo=repo, socket_dir=services.socket_dir, gpus=[0, 1, 2, 3],
                        default_model="mock-model")
        out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=commit,
                                 attempt=1, max_attempts=3, retry=None, nodes_remaining=1)
    finally:
        services.stop(), queue.shutdown()
    assert out.ok, out.error
    assert out.result["rationale"] == "4 clips"
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert "tool.call" in kinds and "phase.end" in kinds
