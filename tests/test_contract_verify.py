import json
import os
from pathlib import Path

import pytest
from ar_contract.models import EditContext, RecipeContext

from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractHarness, verify_contract
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
FIXTURES = Path(__file__).parent / "fixtures" / "agents"
_COMMON = {"dry_run": True}
SMOKE = {"edit_self": EditContext(**_COMMON).model_dump(mode="json"),
         "improve_recipe": RecipeContext(**_COMMON).model_dump(mode="json")}


def test_each_container_token_is_revoked_and_its_jobs_cancelled(tmp_path, monkeypatch):
    """Like the phase runner: a smoke run must not leave GPU jobs behind on the harness queue.
    The sandbox is faked."""
    run = tmp_path / "run"
    repo = AgentsRepo(tmp_path / "agents.git")
    h = ContractHarness(CFG, run, repo)
    cancelled, tokens = [], []
    monkeypatch.setattr(h.queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)

    calls = []

    def runner(*, mounts, env, **kw):
        tokens.append(env["AR_TOKEN"])
        calls.append(kw)
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, "", "", 0.1, "c")

    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h,
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    assert report.ok, report.steps
    assert all(c["shm_gb"] == 8 and c["env_dir"] == CFG.repo_root / ".envs" / "agent" and "gpus" not in c
               for c in calls)                            # no GPUs in a contract run
    assert len(tokens) == 3 and cancelled == tokens          # import + two smoke runs
    assert all(h.registry.lookup(t) is None for t in tokens)


def test_non_utf8_entry_py_fails_static_instead_of_raising(tmp_path, monkeypatch):
    """entry.read_text() defaults to UTF-8; a non-UTF-8 agent/entry.py is an agent
    fault and must fail the `static` step, not raise out of
    verify_contract. The runner is never even called."""
    from ar_kernel import telemetry
    from conftest import telemetry_records
    run = tmp_path / "run"
    telemetry.open(run)
    telemetry.start("ar.run")
    repo = AgentsRepo(tmp_path / "agents.git")
    h = ContractHarness(CFG, run, repo)
    agent_src = tmp_path / "agent_src" / "agent"
    agent_src.mkdir(parents=True)
    (agent_src / "__init__.py").write_text("")
    (agent_src / "entry.py").write_bytes(b"\xff\xfe\x00not valid utf-8")

    def runner(**kw):
        pytest.fail("the runner must not be invoked once static_check has failed")

    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(agent_src.parent), harness=h,
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "static" and "utf-8" in failed.detail.lower()
    steps = [(r["chosen"], r["reason"][:20]) for r in telemetry_records(run)
             if r["kind"] == "decision" and r["name"] == "contract.step"]
    assert [c for c, _ in steps] == ["pass", "fail"] and steps[1][1].startswith("cannot read agent/")


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
    smoke:<kind> step, not raise out of verify_contract. The runner is faked."""
    run = tmp_path / "run"
    repo = AgentsRepo(tmp_path / "agents.git")
    h = ContractHarness(CFG, run, repo)

    def runner(*, mounts, command, **kw):
        if command[-1] == "edit_self":
            write_result_json(mounts.workspace)
        else:
            (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, "", "", 0.1, "c")

    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h,
                                 node="n1", attempt=1, contexts=SMOKE, runner=runner)
    finally:
        h.queue.shutdown()
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "smoke:edit_self" and fragment in failed.detail


def _verify_tree(tmp_path, monkeypatch, mutate, runner=None):
    """verify_contract on a copy of the good fixture changed by `mutate(agent_dir)`; the runner is faked."""
    import shutil
    run = tmp_path / "run"
    repo = AgentsRepo(tmp_path / "agents.git")
    h = ContractHarness(CFG, run, repo)
    src = tmp_path / "src"
    shutil.copytree(FIXTURES / "good", src)
    mutate(src / "agent")

    def no_runner(**kw):
        pytest.fail("the runner must not be invoked")

    try:
        return verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo, commit=repo.init(src), harness=h, node="n1", attempt=1, contexts=SMOKE, runner=runner or no_runner)
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
        return RunResult(0, "", "", 0.1, "c")

    report = _verify_tree(tmp_path, monkeypatch, lambda a: None, runner=runner)
    failed = next(s for s in report.steps if not s.ok)
    assert not report.ok and failed.name == "smoke:edit_self" and "no result.json" in failed.detail


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    repo = AgentsRepo(run / "agents.git")
    repo.init(FIXTURES / "good")
    h = ContractHarness(CFG, run, repo)
    h.start()
    yield h, run, repo
    h.stop()


def _verify(harness, tmp_path, fixture):
    h, run, repo = harness
    commit = repo.commit_tree(FIXTURES / fixture, None, fixture)
    return verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo, commit=commit, harness=h, node="n1", attempt=1, contexts=SMOKE)


def _failed_step(report):
    return next((s for s in report.steps if not s.ok), None)


@pytest.mark.sandbox
def test_good_agent_passes_every_step(harness, tmp_path):
    report = _verify(harness, tmp_path, "good")
    assert report.ok, report.steps
    assert [s.name for s in report.steps] == ["build", "static", "mentions", "import", "smoke:edit_self",
                                              "smoke:improve_recipe"]


@pytest.mark.sandbox
@pytest.mark.parametrize("fixture,step", [
    ("bad_requirements", "build"), ("no_improve", "static"), ("two_params", "static"),
    ("import_error", "import"), ("invalid_result", "smoke:edit_self"),
])
def test_broken_agents_fail_at_the_right_step(harness, tmp_path, fixture, step):
    report = _verify(harness, tmp_path, fixture)
    assert not report.ok and _failed_step(report).name == step
    assert report.to_record()["failed_step"] == step


@pytest.mark.parametrize("text,hits", [
    ("# Mission\nYou choose the data idea.\n", []),
    ("x = 1\n# Keep in mind that node0012 scored best.\n", ["agent/tools.py:2 mentions node0012"]),
    ("STEPS = '\\n1. look\\n2. act'  # n1, path/n3.py\n", []),
    ("PRIORITY = 'follows_event_instruction'\n", ["agent/tools.py:1 mentions follows_event_instruction"]),
    ("# The plan needs no nine-step list; turn 1 comes first; n_steps = 3.\n", []),
])
def test_mentions_finds_nodes_and_metrics_in_any_text_file(tmp_path, text, hits):
    from ar_contract.static import mentions
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "tools.py").write_text(text)
    assert mentions(tmp_path) == hits


def test_mentions_reads_subfolders_and_skips_binaries_symlinks_and_fifos(tmp_path):
    from ar_contract.static import mentions
    notes = tmp_path / "agent" / "memory"
    notes.mkdir(parents=True)
    (notes / "odd.md").mkdir()                                     # a directory named like a file
    (notes / "blob.bin").write_bytes(b"\xff\xfe node0004 \x00")           # not text
    (tmp_path / "agent" / "link.md").symlink_to(tmp_path / "outside.md")
    (tmp_path / "outside.md").write_text("node0009\n")
    os.mkfifo(notes / "pipe.md")                                   # reading it would block
    assert mentions(tmp_path / "agent") == []
    (notes / "lessons.md").write_text("ok\nnode0007 did well with camera_path_accuracy.\n")
    assert mentions(tmp_path / "agent") == ["memory/lessons.md:2 mentions camera_path_accuracy",
                                            "memory/lessons.md:2 mentions node0007"]


def test_the_seed_mentions_no_node_or_metric():
    from ar_contract.static import mentions
    assert mentions(Path(__file__).parents[1] / "seed_agent") == []


def test_requirements_install_once_and_every_run_sees_the_packages(tmp_path, monkeypatch):
    calls = []

    def runner(*, mounts, command, **kw):
        calls.append((kw.get("log_name"), command, mounts))
        if kw.get("log_name") == "install":
            (mounts.workspace / ".home" / ".local" / "lib").mkdir(parents=True)
            (mounts.workspace / ".home" / ".local" / "lib" / "pkg.py").write_text("x = 1\n")
        else:
            assert (mounts.workspace / ".home" / ".local" / "lib" / "pkg.py").read_text() == "x = 1\n"
            (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, "", "", 0.1, "c")

    report = _verify_tree(tmp_path, monkeypatch, lambda a: (a / "requirements.txt").write_text("rich==13.0\n"), runner)
    assert report.ok, report.steps
    assert [c[0] for c in calls] == ["install", None, None, None]
    assert calls[0][1][-1] == "/agent/agent/requirements.txt" and calls[0][2].agent_readonly


def test_code_naming_a_node_fails_verification_before_any_sandbox(tmp_path, monkeypatch):
    def mutate(agent):
        (agent / "notes.md").write_text("Remember: node0003 used too many static clips.\n")
    report = _verify_tree(tmp_path, monkeypatch, mutate)
    failed = next(s for s in report.steps if not s.ok)
    assert failed.name == "mentions" and failed.detail == "agent/notes.md:1 mentions node0003"
