"""Run one agent phase attempt end to end (spec 7.2 steps 3 and 5)."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ar_contract.models import RESULT_MODELS
from pydantic import ValidationError

from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .context_bundle import build_edit_context, build_recipe_context, write_bundle
from .sandbox.image import ensure_image
from .sandbox.runner import Mounts, container_name, diff, run_container, snapshot


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
    return {"attempt": base, "agent": base / "agent", "workspace": base / "workspace",
            "context": base / "context", "staging": Path(run_dir) / "staging" / node / f"{phase}-{attempt}"}


def _run(env: PhaseEnv, *, phase: str, node: str, attempt: int, code_commit: str, ctx,
         mock_script: str | None, agent_readonly: bool, previous_workspace: Path | None) -> tuple:
    dirs = attempt_dirs(env.run_dir, node, phase, attempt)
    if dirs["attempt"].exists():
        shutil.rmtree(dirs["attempt"])
    env.repo.checkout(code_commit, dirs["agent"])
    if previous_workspace is not None and Path(previous_workspace).exists():
        prev_ws = Path(previous_workspace)
        shutil.copytree(prev_ws, dirs["workspace"], symlinks=True)
        # Controller ruling: the previous attempt's staging dir is named after its
        # own attempt directory (prev_ws.parent.name); MOVE it into the new
        # attempt's staging path, since a partial hf_download can be tens of GiB.
        prev_staging = Path(env.run_dir) / "staging" / node / prev_ws.parent.name
        if prev_staging.exists():
            shutil.move(str(prev_staging), str(dirs["staging"]))
    for key in ("workspace", "context", "staging"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    result_file = dirs["workspace"] / "result.json"
    if result_file.exists():
        # Controller ruling: a retry's workspace is copied from the previous
        # attempt (e.g. to keep edit_self's /workspace/edit_plan.json); that
        # attempt's own result.json must never be mistaken for this one's.
        result_file.unlink()
    (Path(env.run_dir) / "store").mkdir(exist_ok=True)
    write_bundle(ctx, dirs["context"])
    reqs = dirs["agent"] / "agent" / "requirements.txt"
    image = ensure_image(env.cfg, reqs.read_text() if reqs.exists() else "", recorder=env.recorder, node=node)
    caller = env.registry.issue(node=node, phase=phase, attempt=attempt, workspace_host=dirs["workspace"],
                                staging_host=dirs["staging"], mock_script=mock_script)
    if caller.token not in env.recorder.redact:
        # Controller ruling: TokenRegistry.issue() only redacts on the recorder it
        # was constructed with. That is usually env.recorder too, but verify
        # rather than assume -- if it isn't (as in the contract harness, Task 14),
        # a token an agent prints to stdout/stderr would otherwise leak into
        # env.recorder's own telemetry.
        env.recorder.add_redaction(caller.token)
    soft = float(env.cfg.get(f"timeouts.{phase}_s"))
    # /workspace/staging is mounted from its own host directory, so it is snapshotted separately.
    before = {"agent": snapshot(dirs["agent"], True), "workspace": snapshot(dirs["workspace"], False),
              "staging": snapshot(dirs["staging"], False)}
    env.recorder.event("phase.start", node=node, phase=phase, attempt=attempt, component="kernel",
                       payload={"code_commit": code_commit, "image": image})
    started = time.monotonic()
    try:
        result = env.runner(
            image=image, name=container_name(env.run_id, node, phase, attempt),
            mounts=Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"],
                          context=dirs["context"], store=Path(env.run_dir) / "store",
                          contract=env.cfg.repo_root / "contract", sockets=env.socket_dir,
                          agent_readonly=agent_readonly),
            command=["python", "-m", "ar_contract.run", phase],
            env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": env.default_model, "AR_NODE": node,
                 "AR_PHASE": phase, "AR_ATTEMPT": str(attempt),
                 "AR_CONTEXT_WINDOW": str(env.cfg.get("agents.context_window_tokens")),
                 "AR_COMPACT_AT": str(env.cfg.get("agents.compact_at"))},
            cpus=env.cfg.get("sandbox.cpus"), memory_gb=env.cfg.get("sandbox.memory_gb"),
            timeout_s=4 * soft,                 # hard cap (spec 14.5); liveness is Plan 4
            recorder=env.recorder, node=node, phase=phase, attempt=attempt)
    finally:
        env.registry.revoke(caller.token)
        env.queue.cancel_for_token(caller.token)     # an ended phase must not keep the GPUs
    out_file = dirs["workspace"] / "result.json"
    body = json.loads(out_file.read_text()) if out_file.exists() else None
    diffs = {"agent": diff(before["agent"], snapshot(dirs["agent"], True)),
             "workspace": diff(before["workspace"], snapshot(dirs["workspace"], False)),
             "staging": diff(before["staging"], snapshot(dirs["staging"], False))}
    return dirs, result, body, diffs, time.monotonic() - started


def _validate_result(phase: str, body: dict) -> tuple[bool, dict | None, str | None]:
    """Never trusts the container's result.json body: an untrusted `ok: true`
    result is re-validated against the phase's pydantic model before any of its
    fields are used. Returns (ok, result, error); never raises (controller
    ruling -- a malformed result, e.g. missing `data_commit`, is a failed
    attempt, never a KeyError in the kernel)."""
    if not body.get("ok"):
        return False, None, body.get("error")
    try:
        model = RESULT_MODELS[phase](**(body.get("result") or {}))
    except (TypeError, ValidationError) as exc:
        return False, None, f"the agent's result failed {RESULT_MODELS[phase].__name__} validation: {exc}"
    return True, model.model_dump(), None


def _outcome(env, phase, node, attempt, dirs, result, body, diffs, duration, commit, extra_check=None):
    if result.timed_out:
        ok, res, error = False, None, "the agent call timed out and was killed"
    elif body is None:
        ok, res, error = False, None, f"no result.json (exit code {result.exit_code}): {result.stderr[-2000:]}"
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


def run_edit_self(env: PhaseEnv, *, conn, node: str, parent_id: str, base_commit: str, attempt: int,
                  max_attempts: int, retry: dict | None, nodes_remaining: int,
                  previous_workspace: Path | None = None, mock_script: str | None = None,
                  dry_run: bool = False) -> PhaseOutcome:
    ctx = build_edit_context(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id,
                             attempt=attempt, max_attempts=max_attempts, retry=retry,
                             nodes_remaining=nodes_remaining, dry_run=dry_run)
    dirs, result, body, diffs, duration = _run(env, phase="edit_self", node=node, attempt=attempt,
                                               code_commit=base_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=False, previous_workspace=previous_workspace)
    # edit_self commits the edited code to an attempt ref whether or not the
    # attempt succeeded (spec 5.2): a failed edit is still inspectable/resumable.
    commit = env.repo.commit_tree(dirs["agent"], base_commit, f"{node} edit_self attempt {attempt}")
    env.repo.set_ref(env.repo.attempt_ref(node, "edit_self", attempt), commit)
    return _outcome(env, "edit_self", node, attempt, dirs, result, body, diffs, duration, commit)


def run_improve_recipe(env: PhaseEnv, *, conn, node: str, parent_id: str, agent_commit: str, attempt: int,
                       max_attempts: int, retry: dict | None, nodes_remaining: int,
                       previous_workspace: Path | None = None, mock_script: str | None = None,
                       dry_run: bool = False) -> PhaseOutcome:
    tools = ["video_probe", "data_ingest", "data_query", "data_commit", "recipe_check",
             "hf_search", "hf_download", "job_status", "job_wait", "job_cancel",
             *sorted(b.tool for b in env.queue.backends.values())]
    ctx = build_recipe_context(cfg=env.cfg, conn=conn, run_dir=env.run_dir, repo=env.repo, node_id=node,
                               parent_id=parent_id, attempt=attempt, max_attempts=max_attempts,
                               retry=retry, nodes_remaining=nodes_remaining, n_gpus=len(env.gpus),
                               tools=tools, dry_run=dry_run)
    # improve_recipe mounts /agent read-only: only edit_self changes code.
    dirs, result, body, diffs, duration = _run(env, phase="improve_recipe", node=node, attempt=attempt,
                                               code_commit=agent_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=True, previous_workspace=previous_workspace)

    def check_data_commit(res: dict) -> str | None:
        # `res` is already the validated RecipeResult dump, so `data_commit` is
        # guaranteed present here -- this never indexes the untrusted raw body.
        try:
            CommitStore(conn, BlobStore(env.run_dir, conn), ClipStore(conn)).get(res["data_commit"])
        except KeyError:
            return f"result names data commit {res['data_commit']!r}, which does not exist"
        return None

    return _outcome(env, "improve_recipe", node, attempt, dirs, result, body, diffs, duration, None,
                    extra_check=check_data_commit)
