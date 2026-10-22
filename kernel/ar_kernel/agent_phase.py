"""Run one agent phase attempt end to end."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

from ar_contract.models import RESULT_MODELS
from pydantic import ValidationError

from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .archive.nodes import NOT_SHOWN, NodeStore
from .context_bundle import build_edit_context, build_recipe_context, write_bundle
from .faults import KernelFault
from .isolation import host_roots, scrub
from .sandbox.runner import Mounts, RunResult, container_name, diff, run_container, snapshot
from .tools import snapshot as snapshots
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
    commit: str | None
    attempt_dir: Path
    duration_s: float


def attempt_dirs(run_dir: Path, node: str, phase: str, attempt: int) -> dict[str, Path]:
    base = Path(run_dir) / "nodes" / node / "attempts" / f"{phase}-{attempt}"
    return {"attempt": base, "agent": base / "agent", "code": base / "code", "workspace": base / "workspace",
            "context": base / "context", "staging": Path(run_dir) / "staging" / node / f"{phase}-{attempt}"}


def finished_node_dirs(conn, run_dir: Path) -> dict[str, Path]:
    """The node dir of every finished node, by node id: not the node being built."""
    out = {n["node_id"]: Path(run_dir) / "nodes" / n["node_id"] for n in NodeStore(conn).all()
           if n["status"] not in NOT_SHOWN}
    return {node: path for node, path in out.items() if path.is_dir()}


def _failed_before_start(dirs: dict, started: float, error: str) -> tuple:
    """An attempt that fails before its sandbox starts: a failed attempt recorded like
    any other, never an exception that skips edit_self's commit of the attempt tree."""
    empty = {"added": [], "removed": [], "changed": []}
    result = RunResult(None, "", error, time.monotonic() - started, "")
    diffs = {"agent": dict(empty), "workspace": dict(empty), "staging": dict(empty)}
    return dirs, result, {"ok": False, "error": error}, diffs, time.monotonic() - started


def requirements_listed(code: Path) -> bool:
    """Whether `code`/agent/requirements.txt names a package. Raises OSError or UnicodeDecodeError
    when it cannot be read (e.g. a directory, or not UTF-8)."""
    path = Path(code) / "agent" / "requirements.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return any(line.split("#")[0].strip() for line in text.splitlines())


def install_requirements(runner, *, name: str, mounts: Mounts, path: str, cfg, recorder, node: str, phase: str,
                         attempt: int) -> RunResult:
    """`pip install --user -r path` in a sandbox with `mounts`, without GPUs: the packages land in the
    workspace's home, where the sandboxes that follow on the same workspace find them."""
    return runner(name=name, mounts=mounts, env={}, env_dir=cfg.repo_root / ".envs" / "agent", shm_gb=1,
                  command=["python", "-m", "pip", "install", "--user", "--no-warn-script-location", "-r", path],
                  recorder=recorder, node=node, phase=phase, attempt=attempt, log_name="install")


def _run(env: PhaseEnv, *, phase: str, node: str, attempt: int, code_commit: str, ctx,
         mock_script: str | None, agent_readonly: bool, nodes: dict[str, Path]) -> tuple:
    """`code_commit` is checked out at /agent; when /agent is writable, the code runs from a second
    checkout at /code, so an edit that breaks the agent mid-phase does not stop the roles."""
    dirs = attempt_dirs(env.run_dir, node, phase, attempt)
    started = time.monotonic()
    if dirs["attempt"].exists():
        shutil.rmtree(dirs["attempt"])
    if dirs["staging"].exists():            # outside the attempt dir: a rerun must clear it too
        shutil.rmtree(dirs["staging"])
    snapshots.clear(env.run_dir, env.repo, node, phase, attempt)
    try:
        env.repo.checkout(code_commit, dirs["agent"])
    except CheckoutError as exc:            # e.g. a committed symlink out of the tree
        shutil.rmtree(dirs["agent"])        # no tree: edit_self commits nothing
        return _failed_before_start(dirs, started, f"cannot check out the agent code: {exc}")
    running = dirs["agent"]
    if not agent_readonly:
        running = dirs["code"]
        env.repo.checkout(code_commit, running)
    for key in ("workspace", "context", "staging"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    (Path(env.run_dir) / "store").mkdir(exist_ok=True)
    write_bundle(ctx, dirs["context"], env.cfg.get("isolation.blocked_names") or [], host_roots(env.cfg))
    mounts = Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"], context=dirs["context"],
                    contract=env.cfg.repo_root / "contract", sockets=env.socket_dir, agent_readonly=agent_readonly,
                    nodes=nodes, code=None if agent_readonly else running)
    try:
        listed = requirements_listed(running)
    except (OSError, UnicodeDecodeError) as exc:
        return _failed_before_start(dirs, started, f"cannot read agent/requirements.txt: {exc}")
    if listed:          # before the snapshot below: the installed packages are not the agent's changes
        installed = install_requirements(
            env.runner, name=container_name(env.run_id, node, f"{phase}-install", attempt),
            mounts=replace(mounts, agent_readonly=True),
            path=f"{'/agent' if agent_readonly else '/code'}/agent/requirements.txt", cfg=env.cfg,
            recorder=env.recorder, node=node, phase=phase, attempt=attempt)
        if installed.exit_code != 0:        # the contract installed this commit's requirements before
            raise KernelFault(f"agent/requirements.txt failed to install (exit code {installed.exit_code}):\n"
                              f"{installed.stderr[-6000:]}")

    caller = env.registry.issue(node=node, phase=phase, attempt=attempt, workspace_host=dirs["workspace"],
                                staging_host=dirs["staging"], mock_script=mock_script,
                                agent_host=None if agent_readonly else dirs["agent"],
                                base_commit=None if agent_readonly else code_commit)
    # From here on, any raise must still revoke the token and cancel its jobs, or an ended
    # phase could keep the GPUs.
    try:
        env.recorder.add_redaction(caller.token)   # idempotent, regardless of who redacted it first
        # /workspace/staging is mounted from its own host directory, so it is snapshotted separately.
        before = {"agent": snapshot(dirs["agent"], True), "workspace": snapshot(dirs["workspace"], False),
                  "staging": snapshot(dirs["staging"], False)}
        env.recorder.event("phase.start", node=node, phase=phase, attempt=attempt, component="kernel",
                           payload={"code_commit": code_commit})
        result = env.runner(
            name=container_name(env.run_id, node, phase, attempt), mounts=mounts,
            command=["python", "-m", "ar_contract.run", phase],
            env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": env.default_model, "AR_NODE": node,
                 "AR_PHASE": phase, "AR_ATTEMPT": str(attempt),
                 "AR_CONTEXT_WINDOW": str(env.cfg.get("agents.context_window_tokens")),
                 "AR_COMPACT_AT": str(env.cfg.get("agents.compact_at")),
                 **({"HF_TOKEN": os.environ["HFK"]} if os.environ.get("HFK", "").strip() else {})},  # gated repos
            env_dir=env.cfg.repo_root / ".envs" / "agent", shm_gb=env.cfg.get("sandbox.shm_gb"), gpus=env.gpus,
            recorder=env.recorder, node=node, phase=phase, attempt=attempt, cancel=caller.fault.event,
            on_start=lambda pid: caller.sandbox.update(pid=pid))
    finally:
        try:
            env.registry.revoke(caller.token)
        finally:
            env.queue.cancel_for_token(caller.token)     # an ended phase must not keep the GPUs
            env.registry.wait_idle(caller.token)         # nothing of this attempt writes after it ends
            env.queue.drain(caller.token)
    if caller.fault.event.is_set():
        raise KernelFault(caller.fault.reason)
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


def _judge(phase, result, body, resolve) -> tuple[bool, dict | None, str | None]:
    """(ok, result, error); `resolve` maps the validated result to (result, None) or (None, error)."""
    if body is None:
        return False, None, f"no result.json (exit code {result.exit_code}): {result.stderr[-6000:]}"
    ok, res, error = _validate_result(phase, body)
    if not ok:
        return False, None, error
    res, error = resolve(res)
    return error is None, res, error


def _outcome(env, phase, node, attempt, dirs, result, verdict, diffs, duration, commit):
    ok, res, error = verdict
    outcome = PhaseOutcome(ok, res, error, result.exit_code, commit, dirs["attempt"], duration)
    env.recorder.event("phase.end", node=node, phase=phase, attempt=attempt, component="kernel",
                       ok=ok, exit_code=result.exit_code,
                       payload={"result": res, "error": error, "diffs": diffs, "commit": commit})
    return outcome


def _unknown(snapshot_id: str, node: str, phase: str, attempt: int) -> str:
    return (f"result names snapshot {snapshot_id!r}, which is not a snapshot of {node} {phase} attempt {attempt}: "
            "snapshot_id must be an id the snapshot tool returned in this attempt")


def run_edit_self(env: PhaseEnv, *, conn, node: str, parent_id: str, agent_commit: str, attempt: int,
                  mock_script: str | None = None, dry_run: bool = False) -> PhaseOutcome:
    ctx = build_edit_context(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id, dry_run=dry_run)
    dirs, result, body, diffs, duration = _run(env, phase="edit_self", node=node, attempt=attempt,
                                               code_commit=agent_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=False, nodes=finished_node_dirs(conn, env.run_dir))
    def resolve(res: dict) -> tuple:
        if snapshots.edit_snapshot(env.repo, node, "edit_self", attempt, res["snapshot_id"]) is None:
            return None, _unknown(res["snapshot_id"], node, "edit_self", attempt)
        return res, None

    ok, res, error = _judge("edit_self", result, body, resolve)
    commit = res["snapshot_id"] if ok else None
    if not ok and dirs["agent"].exists():
        try:
            commit = env.repo.commit_tree(dirs["agent"], agent_commit, f"{node} edit_self attempt {attempt}")
        except subprocess.CalledProcessError as exc:
            error = f"{error}\nThe agent tree could not be committed either: {exc.stderr.decode(errors='replace')}"
    if commit is not None:
        env.repo.set_ref(env.repo.attempt_ref(node, "edit_self", attempt), commit)
    return _outcome(env, "edit_self", node, attempt, dirs, result, (ok, res, error), diffs, duration, commit)


def _recipe_tools(env: PhaseEnv) -> list[str]:
    return ["video_probe", "data_ingest", "data_query", "data_fetch", "data_commit", "recipe_check",
            "hf_search", "hf_list_files", "hf_download", "job_status", "job_wait", "job_cancel", "ask", "read_knowledge",
            *sorted(b.tool for b in env.queue.backends.values())]


def smoke_contexts(env: PhaseEnv, *, conn, node: str, parent_id: str) -> dict[str, dict]:
    """The contexts this node's two phases get, marked dry_run: what the contract check feeds the
    edited code, so code that cannot digest the node's real context fails there."""
    common = dict(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id, dry_run=True)
    recipe = build_recipe_context(cfg=env.cfg, node_id=node, n_gpus=len(env.gpus), tools=_recipe_tools(env),
                                  **common)
    return scrub({"edit_self": build_edit_context(**common).model_dump(mode="json"),
                  "improve_recipe": recipe.model_dump(mode="json")},
                 env.cfg.get("isolation.blocked_names") or [], host_roots(env.cfg))


def run_improve_recipe(env: PhaseEnv, *, conn, node: str, parent_id: str, agent_commit: str, attempt: int,
                       mock_script: str | None = None,
                       dry_run: bool = False) -> PhaseOutcome:
    ctx = build_recipe_context(cfg=env.cfg, conn=conn, run_dir=env.run_dir, repo=env.repo, node_id=node,
                               parent_id=parent_id, n_gpus=len(env.gpus),
                               tools=_recipe_tools(env), dry_run=dry_run)
    # improve_recipe mounts /agent read-only: only edit_self changes code.
    dirs, result, body, diffs, duration = _run(env, phase="improve_recipe", node=node, attempt=attempt,
                                               code_commit=agent_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=True, nodes=finished_node_dirs(conn, env.run_dir))

    def resolve(res: dict) -> tuple:
        stored = snapshots.recipe_snapshot(env.run_dir, node, "improve_recipe", attempt, res["snapshot_id"])
        if stored is None:
            return None, _unknown(res["snapshot_id"], node, "improve_recipe", attempt)
        if not stored["has_result"]:
            return None, (f"result names snapshot {res['snapshot_id']!r}, which holds no data commit and recipe: "
                          "the phase's result must be a snapshot taken with data_commit, recipe and rationale")
        try:
            CommitStore(conn, BlobStore(env.run_dir, conn), ClipStore(conn)).get(stored["data_commit"])
        except KeyError:
            return None, f"snapshot {res['snapshot_id']!r} names data commit {stored['data_commit']!r}, which does not exist"
        return {"snapshot_id": res["snapshot_id"], "data_commit": stored["data_commit"], "recipe": stored["recipe"],
                "rationale": stored["rationale"], "summary": res["summary"]}, None

    return _outcome(env, "improve_recipe", node, attempt, dirs, result,
                    _judge("improve_recipe", result, body, resolve), diffs, duration, None)
