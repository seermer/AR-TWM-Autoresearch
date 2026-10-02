"""Run one agent phase attempt end to end."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ar_contract.models import RESULT_MODELS
from pydantic import ValidationError

from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.nodes import NOT_SHOWN, NodeStore
from .context_bundle import build_edit_context, build_recipe_context, write_bundle
from .isolation import scrub
from .liveness import Liveness, tree_mark
from .sandbox.image import ImageBuildError, ensure_image
from .sandbox.runner import Mounts, RunResult, container_name, diff, run_container, snapshot
from .vcs.agents_repo import CheckoutError


@dataclass
class PhaseEnv:
    cfg: object
    run_dir: Path
    run_id: str
    recorder: object
    registry: object
    queue: object
    repo: object
    socket_dir: Path
    gpus: list[int]
    default_model: str
    runner: object = run_container


@dataclass
class PhaseOutcome:
    ok: bool
    result: dict | None
    error: str | None
    exit_code: int | None
    timed_out: bool
    commit: str | None
    attempt_dir: Path
    duration_s: float


def attempt_dirs(run_dir: Path, node: str, phase: str, attempt: int) -> dict[str, Path]:
    base = Path(run_dir) / "nodes" / node / "attempts" / f"{phase}-{attempt}"
    return {"attempt": base, "agent": base / "agent", "code": base / "code", "workspace": base / "workspace",
            "context": base / "context", "staging": Path(run_dir) / "staging" / node / f"{phase}-{attempt}"}


def finished_node_dirs(conn, run_dir: Path) -> dict[str, Path]:
    """The node dir of every finished node, by node id: not the node being built, nor an interrupted one."""
    out = {n["node_id"]: Path(run_dir) / "nodes" / n["node_id"] for n in NodeStore(conn).all()
           if n["status"] not in NOT_SHOWN}
    return {node: path for node, path in out.items() if path.is_dir()}


def _failed_before_start(dirs: dict, started: float, error: str) -> tuple:
    """An attempt that fails before its container starts: a failed attempt recorded like
    any other, never an exception that skips edit_self's commit of the attempt tree."""
    empty = {"added": [], "removed": [], "changed": []}
    result = RunResult(None, False, "", error, time.monotonic() - started, [], "")
    diffs = {"agent": dict(empty), "workspace": dict(empty), "staging": dict(empty)}
    return dirs, result, {"ok": False, "error": error}, diffs, time.monotonic() - started


def _run(env: PhaseEnv, *, phase: str, node: str, attempt: int, code_commit: str, ctx,
         mock_script: str | None, agent_readonly: bool, previous_workspace: Path | None,
         nodes: dict[str, Path], runner_commit: str | None = None) -> tuple:
    """`code_commit` is checked out at /agent. `runner_commit`, when given, is the code that runs
    instead, checked out at /code."""
    dirs = attempt_dirs(env.run_dir, node, phase, attempt)
    started = time.monotonic()
    if dirs["attempt"].exists():
        shutil.rmtree(dirs["attempt"])
    if dirs["staging"].exists():            # outside the attempt dir: a rerun must clear it too
        shutil.rmtree(dirs["staging"])
    try:
        env.repo.checkout(code_commit, dirs["agent"])
    except CheckoutError as exc:            # e.g. a committed symlink out of the tree
        shutil.rmtree(dirs["agent"])        # no tree: edit_self commits nothing
        return _failed_before_start(dirs, started, f"cannot check out the agent code: {exc}")
    running = dirs["agent"]
    if runner_commit is not None:
        running = dirs["code"]
        env.repo.checkout(runner_commit, running)       # a commit that already passed the contract check
    if previous_workspace is not None and Path(previous_workspace).exists():
        prev_ws = Path(previous_workspace)
        try:
            shutil.copytree(prev_ws, dirs["workspace"], symlinks=True)
        except (shutil.Error, OSError) as exc:     # FIFOs or files the agent made unreadable
            return _failed_before_start(dirs, started, f"cannot copy the previous workspace: {exc}")
        # Move (never copy) the previous attempt's staging: a partial hf_download can be tens of GiB.
        prev_staging = Path(env.run_dir) / "staging" / node / prev_ws.parent.name
        if prev_staging.exists():
            shutil.move(str(prev_staging), str(dirs["staging"]))
    for key in ("workspace", "context", "staging"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    # A retry's workspace is copied from the previous attempt, whose result.json must never
    # pass for this one's. A symlink is unlinked, never followed.
    result_file = dirs["workspace"] / "result.json"
    if result_file.is_symlink() or result_file.is_file():
        result_file.unlink()
    elif result_file.is_dir():
        shutil.rmtree(result_file)
    (Path(env.run_dir) / "store").mkdir(exist_ok=True)
    write_bundle(ctx, dirs["context"], env.cfg.get("isolation.blocked_names") or [])
    reqs = running / "agent" / "requirements.txt"
    try:
        reqs_text = reqs.read_text(encoding="utf-8") if reqs.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:     # e.g. a directory, or not UTF-8
        return _failed_before_start(dirs, started, f"cannot read agent/requirements.txt: {exc}")
    try:
        image = ensure_image(env.cfg, reqs_text, recorder=env.recorder, node=node)
    except ImageBuildError as exc:
        return _failed_before_start(dirs, started, f"agent image failed to build: {exc}")

    caller = env.registry.issue(node=node, phase=phase, attempt=attempt, workspace_host=dirs["workspace"],
                                staging_host=dirs["staging"], mock_script=mock_script)
    # From here on, any raise must still revoke the token and cancel its jobs, or an ended
    # phase could keep the GPUs.
    try:
        env.recorder.add_redaction(caller.token)   # idempotent, regardless of who redacted it first
        soft = float(env.cfg.get(f"timeouts.{phase}_s"))
        liveness = Liveness.from_config(env.cfg, soft, signals=[
            lambda: tree_mark(dirs["workspace"], dirs["staging"]),          # workspace changes
            lambda: tree_mark(env.recorder.events_path(node)),               # gateway + tool calls
            lambda: env.queue.active_for_token(caller.token) and time.monotonic()])  # own GPU jobs
        # /workspace/staging is mounted from its own host directory, so it is snapshotted separately.
        before = {"agent": snapshot(dirs["agent"], True), "workspace": snapshot(dirs["workspace"], False),
                  "staging": snapshot(dirs["staging"], False)}
        env.recorder.event("phase.start", node=node, phase=phase, attempt=attempt, component="kernel",
                           payload={"code_commit": code_commit, "image": image})
        result = env.runner(
            image=image, name=container_name(env.run_id, node, phase, attempt),
            mounts=Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"],
                          context=dirs["context"], store=Path(env.run_dir) / "store",
                          contract=env.cfg.repo_root / "contract", sockets=env.socket_dir,
                          agent_readonly=agent_readonly, nodes=nodes, hide_scores=phase == "edit_self",
                          code=running if runner_commit is not None else None),
            command=["python", "-m", "ar_contract.run", phase],
            env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": env.default_model, "AR_NODE": node,
                 "AR_PHASE": phase, "AR_ATTEMPT": str(attempt),
                 "AR_CONTEXT_WINDOW": str(env.cfg.get("agents.context_window_tokens")),
                 "AR_COMPACT_AT": str(env.cfg.get("agents.compact_at"))},
            cpus=env.cfg.get("sandbox.cpus"), memory_gb=env.cfg.get("sandbox.memory_gb"),
            network=env.cfg.get("sandbox.network"),
            timeout_s=4 * soft,                 # hard cap
            liveness=liveness,
            recorder=env.recorder, node=node, phase=phase, attempt=attempt)
    finally:
        try:
            env.registry.revoke(caller.token)
        finally:
            env.queue.cancel_for_token(caller.token)     # an ended phase must not keep the GPUs
    out_file = dirs["workspace"] / "result.json"
    body = read_result(out_file)
    diffs = {"agent": diff(before["agent"], snapshot(dirs["agent"], True)),
             "workspace": diff(before["workspace"], snapshot(dirs["workspace"], False)),
             "staging": diff(before["staging"], snapshot(dirs["staging"], False))}
    return dirs, result, body, diffs, time.monotonic() - started


def read_result(out_file: Path) -> dict | None:
    """The agent-written result.json: None when missing, a directory or a symlink (never
    followed); a failed body when it is not a JSON object. Never raises."""
    if out_file.is_symlink() or not out_file.is_file():
        return None
    try:
        body = json.loads(out_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"ok": False, "error": f"result.json could not be parsed: {type(exc).__name__}: {exc}"}
    if not isinstance(body, dict):
        return {"ok": False, "error": "result.json is not a JSON object"}
    return body


def _validate_result(phase: str, body: dict) -> tuple[bool, dict | None, str | None]:
    """(ok, result, error): an `ok: true` body is re-validated against the phase's model before
    any field is used, so a malformed result is a failed attempt. Never raises."""
    if body.get("ok") is not True:                 # truthy-but-not-True is still a failure
        error = body.get("error")
        return False, None, str(error) if error is not None else "agent reported failure without an error"
    try:
        model = RESULT_MODELS[phase](**(body.get("result") or {}))
    except (TypeError, ValidationError) as exc:
        return False, None, f"the agent's result failed {RESULT_MODELS[phase].__name__} validation: {exc}"
    return True, model.model_dump(), None


def _outcome(env, phase, node, attempt, dirs, result, body, diffs, duration, commit, extra_check=None):
    if result.timed_out:
        ok, res, error = False, None, "the agent call timed out and was killed"
    elif body is None:
        ok, res, error = False, None, f"no result.json (exit code {result.exit_code}): {result.stderr[-6000:]}"
    else:
        ok, res, error = _validate_result(phase, body)
        if ok and extra_check is not None:
            problem = extra_check(res)
            if problem:
                ok, res, error = False, None, problem
    outcome = PhaseOutcome(ok, res, error, result.exit_code, result.timed_out, commit, dirs["attempt"], duration)
    env.recorder.event("phase.end", node=node, phase=phase, attempt=attempt, component="kernel",
                       ok=ok, timed_out=result.timed_out, exit_code=result.exit_code,
                       payload={"result": outcome.result, "error": error, "diffs": diffs, "commit": commit})
    return outcome


def run_edit_self(env: PhaseEnv, *, conn, node: str, parent_id: str, base_commit: str, runner_commit: str,
                  attempt: int, max_attempts: int, retry: dict | None, nodes_remaining: int,
                  previous_workspace: Path | None = None, mock_script: str | None = None,
                  dry_run: bool = False) -> PhaseOutcome:
    """`base_commit`: the tree to edit (the parent's, or a failed attempt's on a retry).
    `runner_commit`: the parent's code, which runs the phase, so an edit that broke the agent can
    still be repaired on a retry."""
    ctx = build_edit_context(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id,
                             attempt=attempt, max_attempts=max_attempts, retry=retry,
                             nodes_remaining=nodes_remaining, dry_run=dry_run)
    dirs, result, body, diffs, duration = _run(env, phase="edit_self", node=node, attempt=attempt,
                                               code_commit=base_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=False, previous_workspace=previous_workspace,
                                               nodes=finished_node_dirs(conn, env.run_dir),
                                               runner_commit=runner_commit)
    # The edited code is committed to an attempt ref whether or not the attempt succeeded.
    # No tree, or an uncommittable one, means no commit and a failed attempt.
    commit = None
    if dirs["agent"].exists():
        try:
            commit = env.repo.commit_tree(dirs["agent"], base_commit, f"{node} edit_self attempt {attempt}")
        except subprocess.CalledProcessError as exc:
            body = {"ok": False, "error": f"cannot commit the agent tree: {exc.stderr.decode(errors='replace')}"}
        else:
            env.repo.set_ref(env.repo.attempt_ref(node, "edit_self", attempt), commit)
    return _outcome(env, "edit_self", node, attempt, dirs, result, body, diffs, duration, commit)


def _recipe_tools(env: PhaseEnv) -> list[str]:
    return ["video_probe", "data_ingest", "data_query", "data_commit", "recipe_check",
            "hf_search", "hf_list_files", "hf_download", "job_status", "job_wait", "job_cancel", "ask", "read_skill",
            *sorted(b.tool for b in env.queue.backends.values())]


def smoke_contexts(env: PhaseEnv, *, conn, node: str, parent_id: str, nodes_remaining: int) -> dict[str, dict]:
    """The contexts this node's two phases get, marked dry_run: what the contract check feeds the
    edited code, so code that cannot digest the node's real context fails there."""
    common = dict(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id, attempt=1,
                  max_attempts=1, retry=None, nodes_remaining=nodes_remaining, dry_run=True)
    recipe = build_recipe_context(cfg=env.cfg, node_id=node, n_gpus=len(env.gpus), tools=_recipe_tools(env),
                                  **common)
    return scrub({"edit_self": build_edit_context(**common).model_dump(mode="json"),
                  "improve_recipe": recipe.model_dump(mode="json")}, env.cfg.get("isolation.blocked_names") or [])


def run_improve_recipe(env: PhaseEnv, *, conn, node: str, parent_id: str, agent_commit: str, attempt: int,
                       max_attempts: int, retry: dict | None, nodes_remaining: int,
                       previous_workspace: Path | None = None, mock_script: str | None = None,
                       dry_run: bool = False) -> PhaseOutcome:
    ctx = build_recipe_context(cfg=env.cfg, conn=conn, run_dir=env.run_dir, repo=env.repo, node_id=node,
                               parent_id=parent_id, attempt=attempt, max_attempts=max_attempts,
                               retry=retry, nodes_remaining=nodes_remaining, n_gpus=len(env.gpus),
                               tools=_recipe_tools(env), dry_run=dry_run)
    # improve_recipe mounts /agent read-only: only edit_self changes code.
    dirs, result, body, diffs, duration = _run(env, phase="improve_recipe", node=node, attempt=attempt,
                                               code_commit=agent_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=True, previous_workspace=previous_workspace,
                                               nodes=finished_node_dirs(conn, env.run_dir))

    def check_data_commit(res: dict) -> str | None:           # `res` is already validated
        try:
            CommitStore(conn, BlobStore(env.run_dir, conn), ClipStore(conn)).get(res["data_commit"])
        except KeyError:
            return f"result names data commit {res['data_commit']!r}, which does not exist"
        return None

    return _outcome(env, "improve_recipe", node, attempt, dirs, result, body, diffs, duration, None,
                    extra_check=check_data_commit)
