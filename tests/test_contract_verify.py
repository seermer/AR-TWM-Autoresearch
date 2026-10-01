import json
from pathlib import Path

import pytest
from ar_contract.models import EditContext, RecipeContext

from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractHarness, static_check, verify_contract
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
FIXTURES = Path(__file__).parent / "fixtures" / "agents"
_COMMON = {"nodes_remaining": 1, "attempt": 1, "max_attempts": 1, "dry_run": True}
SMOKE = {"edit_self": EditContext(**_COMMON).model_dump(mode="json"),
         "improve_recipe": RecipeContext(**_COMMON).model_dump(mode="json")}


# 20,000 nested binary operations exhaust the parser's own AST-construction
# recursion budget (RecursionError), not Python's own call stack -- distinct
# from the "too many nested parentheses" SyntaxError the parser raises for
# nested brackets/calls/parens. Deliberately not a SyntaxError.
_DEEPLY_NESTED_SRC = ("def edit_self(ctx):\n    return " + "1+" * 20000 + "1\n"
                     "def improve_recipe(ctx): ...\n")


@pytest.mark.parametrize("src,ok,fragment", [
    ("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", True, ""),
    ("async def edit_self(ctx): ...\nasync def improve_recipe(ctx): ...\n", True, ""),
    ("def edit_self(ctx): ...\n", False, "improve_recipe"),
    ("def edit_self(ctx, extra): ...\ndef improve_recipe(ctx): ...\n", False, "exactly one"),
    ("def edit_self(*args): ...\ndef improve_recipe(ctx): ...\n", False, "exactly one"),
    ("def outer():\n    def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", False, "edit_self"),
    ("def edit_self(ctx):\n  return (\n", False, "syntax"),
    pytest.param(_DEEPLY_NESTED_SRC, False, "cannot parse", id="deeply_nested_recursion_error"),
])
def test_static_check(src, ok, fragment):
    step = static_check(src)
    assert step.ok is ok and fragment in step.detail


def test_each_container_token_is_revoked_and_its_jobs_cancelled(tmp_path, monkeypatch):
    """Like the phase runner: a smoke run must not leave GPU jobs behind on the harness queue.
    No Docker: the image build and the container are faked."""
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    cancelled, tokens = [], []
    monkeypatch.setattr(h.queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")

    networks = []

    def runner(*, mounts, env, **kw):
        tokens.append(env["AR_TOKEN"])
        networks.append(kw["network"])
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, False, "", "", 0.1, [], "c")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h, recorder=Recorder(run),
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    assert report.ok, report.steps
    assert networks and set(networks) == {"none"}          # verification containers stay offline
    assert len(tokens) == 3 and cancelled == tokens          # import + two smoke runs
    assert all(h.registry.lookup(t) is None for t in tokens)


def test_non_utf8_entry_py_fails_static_instead_of_raising(tmp_path, monkeypatch):
    """entry.read_text() defaults to UTF-8; a non-UTF-8 agent/entry.py is an agent
    fault and must fail the `static` step, not raise out of
    verify_contract. No Docker: the runner would never even be called."""
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")
    agent_src = tmp_path / "agent_src" / "agent"
    agent_src.mkdir(parents=True)
    (agent_src / "__init__.py").write_text("")
    (agent_src / "entry.py").write_bytes(b"\xff\xfe\x00not valid utf-8")

    def runner(**kw):
        pytest.fail("the runner must not be invoked once static_check has failed")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(agent_src.parent), harness=h, recorder=Recorder(run),
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "static" and "utf-8" in failed.detail.lower()


@pytest.mark.parametrize("write_result_json,fragment", [
    (lambda ws: (ws / "result.json").write_text("{not valid json"), "could not be parsed"),
    (lambda ws: (ws / "result.json").write_text(json.dumps(["not", "a", "dict"])), "not a JSON object"),
])
def test_malformed_result_json_fails_the_smoke_step_instead_of_raising(
        tmp_path, monkeypatch, write_result_json, fragment):
    """A smoke run's /workspace/result.json is agent-controlled output (written by
    ar_contract.run, but the agent code inside the container could in principle
    corrupt or replace it): malformed JSON (JSONDecodeError) or valid JSON that
    isn't an object (body.get would raise AttributeError) must fail the
    smoke:<kind> step, not raise out of verify_contract. No Docker: the runner
    is faked."""
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")

    def runner(*, mounts, command, **kw):
        if command[-1] == "edit_self":
            write_result_json(mounts.workspace)
        else:
            (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, False, "", "", 0.1, [], "c")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h, recorder=Recorder(run),
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "smoke:edit_self" and fragment in failed.detail


def _verify_tree(tmp_path, monkeypatch, mutate, runner=None):
    """verify_contract on a copy of the good fixture changed by `mutate(agent_dir)`; no Docker."""
    import shutil
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")
    src = tmp_path / "src"
    shutil.copytree(FIXTURES / "good", src)
    mutate(src / "agent")

    def no_runner(**kw):
        pytest.fail("the runner must not be invoked")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        return verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo, commit=repo.init(src), harness=h,
                               recorder=Recorder(run), node="n1", attempt=1, contexts=SMOKE, runner=runner or no_runner)
    finally:
        h.queue.shutdown()


def _replace_with_dir(path):
    path.unlink(missing_ok=True)
    path.mkdir()
    (path / "x").write_text("")


@pytest.mark.parametrize("mutate,step,fragment", [
    (lambda a: (a / "ctx").symlink_to("/context/x"), "checkout", "ctx"),
    (lambda a: _replace_with_dir(a / "requirements.txt"), "build", "requirements.txt"),
    (lambda a: (a / "requirements.txt").write_bytes(b"\xff\xfe"), "build", "requirements.txt"),
    (lambda a: _replace_with_dir(a / "entry.py"), "static", "entry.py"),
], ids=["symlink_out_of_tree", "requirements_dir", "requirements_non_utf8", "entry_dir"])
def test_agent_hostile_trees_fail_a_step_instead_of_raising(tmp_path, monkeypatch, mutate, step, fragment):
    report = _verify_tree(tmp_path, monkeypatch, mutate)
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == step and fragment in failed.detail


@pytest.mark.parametrize("plant", ["symlink", "fifo"])
def test_smoke_result_json_symlink_or_fifo_is_not_followed(tmp_path, monkeypatch, plant):
    """A symlink onto a host file must not be read (its JSON would reach the agent); a FIFO must not hang."""
    import os
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"ok": True, "result": {}}))

    def runner(*, mounts, command, **kw):
        out = mounts.workspace / "result.json"
        if command[-1] == "edit_self":
            out.symlink_to(secret) if plant == "symlink" else os.mkfifo(out)
        return RunResult(0, False, "", "", 0.1, [], "c")

    report = _verify_tree(tmp_path, monkeypatch, lambda a: None, runner=runner)
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "smoke:edit_self" and "no result.json" in failed.detail


def test_container_token_printed_by_agent_is_redacted_from_telemetry(tmp_path, monkeypatch):
    """verify_contract's own telemetry (the sandbox run and the final
    contract.report) goes through the `recorder` argument, which in real usage
    (and in `test_each_container_token_is_revoked_and_its_jobs_cancelled` above)
    is a *different* Recorder instance from `harness.recorder` -- the one
    TokenRegistry.issue() already redacts on. If verify_contract only relied on
    that, a token an agent printed to stdout/stderr would leak into whatever
    `recorder` records. No Docker: the runner is faked, standing in for what a
    real run_container would record via `recorder.event("sandbox.end", ...)`."""
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")
    recorder = Recorder(run)
    seen_tokens: list[str] = []

    def runner(*, mounts, command, env, recorder, node, phase, attempt, **kw):
        token = env["AR_TOKEN"]
        seen_tokens.append(token)
        if command[-1] == "edit_self":
            stderr = f"agent stderr leaked its own token: {token}"
            recorder.event("sandbox.end", node=node, phase=phase, attempt=attempt,
                           component="sandbox", payload={"stdout": "", "stderr": stderr})
            return RunResult(1, False, "", stderr, 0.1, [], "c")
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, False, "", "", 0.1, [], "c")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h, recorder=recorder,
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()

    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "smoke:edit_self"
    assert seen_tokens and all(len(t) > 10 for t in seen_tokens)

    events = recorder.read_events("n1")
    assert events, "expected telemetry events to have been recorded for node n1"
    saw_redacted = False
    for event in events:
        assert all(token not in json.dumps(event) for token in seen_tokens), \
            f"a container token leaked into an unscrubbed event: {event}"
        if event.get("payload"):
            body = recorder.load_payload(event["payload"])
            dumped = json.dumps(body)
            assert all(token not in dumped for token in seen_tokens), \
                f"a container token leaked into a recorded payload: {body}"
            saw_redacted = saw_redacted or "[REDACTED]" in dumped
    assert saw_redacted, "expected the scrubbed placeholder to appear where the token was"


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    h = ContractHarness(CFG, run, Recorder(run))
    h.start()
    yield h, run
    h.stop()


def _verify(harness, tmp_path, fixture, **timeouts):
    h, run = harness
    repo = AgentsRepo(tmp_path / "agents.git")
    commit = repo.init(FIXTURES / fixture)
    return verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo, commit=commit, harness=h,
                           recorder=Recorder(run), node="n1", attempt=1, contexts=SMOKE, **timeouts)


def _failed_step(report):
    return next((s for s in report.steps if not s.ok), None)


@pytest.mark.docker
def test_good_agent_passes_every_step(harness, tmp_path):
    report = _verify(harness, tmp_path, "good")
    assert report.ok, report.steps
    assert [s.name for s in report.steps] == ["build", "static", "import", "smoke:edit_self",
                                              "smoke:improve_recipe"]


@pytest.mark.docker
@pytest.mark.parametrize("fixture,step", [
    ("bad_requirements", "build"), ("no_improve", "static"), ("two_params", "static"),
    ("import_error", "import"), ("invalid_result", "smoke:edit_self"),
])
def test_broken_agents_fail_at_the_right_step(harness, tmp_path, fixture, step):
    report = _verify(harness, tmp_path, fixture)
    assert not report.ok and _failed_step(report).name == step
    assert report.to_retry()["failed_step"] == step


@pytest.mark.docker
def test_hanging_smoke_run_times_out(harness, tmp_path):
    report = _verify(harness, tmp_path, "hangs", smoke_timeout_s=15)
    failed = _failed_step(report)
    assert failed.name == "smoke:edit_self" and "timed out" in failed.detail
