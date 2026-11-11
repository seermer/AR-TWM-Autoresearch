from __future__ import annotations
import argparse, faulthandler, getpass, json, os, sys, time, traceback

from . import telemetry
from .config import KernelConfig, load_dotenv
from .control import Control, drive, exit_process, kill_recorded_groups
from .doctor import gpu_checks, report, run_checks
from .guards import check_tools_fit, choose_gpus
from .hf_upload import HfUpload, fetch, run_uploads
from .loop import NATURAL_END, Loop
from .run import (PreflightError, RunNotFound, attach_run, bootstrap_run, check_new_run, check_resume,
                  code_changes, existing_run_dir, record_envs, rescore_node)
from .run_kit import build_run_kit
from .status import format_status, run_status
from .subproc import cache_env, proc_running, proc_start_time, terminate
from .telemetry import capture, tree
from .telemetry.trace import render
from .tools.hf_tools import clear_hf_tmp
from .vcs.agents_repo import AgentsRepo

FORCE_STOP_WAIT_S = 300.0     # the kernel's force-stop cleanup kills sandboxes and GPU job groups


def main(argv: list[str] | None = None) -> int:
    os.environ.update(cache_env())
    parser = argparse.ArgumentParser(prog="ar")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show nodes of a run")
    status.add_argument("--run-id", required=True)
    status.add_argument("--json", action="store_true", help="print the raw JSON snapshot")
    tracep = sub.add_parser("trace", help="print a run's telemetry as trees of spans, or --check it")
    tracep.add_argument("--run-id", required=True)
    tracep.add_argument("--node", default=None, help="only this node's span")
    tracep.add_argument("--attempt", type=int, default=None, help="with --node: only its phase attempts numbered so")
    tracep.add_argument("--from", dest="since", type=int, default=None, metavar="SEQ", help="only records from SEQ on")
    tracep.add_argument("--depth", type=int, default=None, help="levels of spans to print")
    tracep.add_argument("--full", action="store_true", help="payloads in full, not shortened")
    tracep.add_argument("--check", action="store_true", help="list telemetry bugs instead; exit 1 if there are any")
    score = sub.add_parser("score-node", help="score a node of a run again; writes under scores/, never the run")
    score.add_argument("--run-id", required=True)
    score.add_argument("--node", required=True)
    runp = sub.add_parser("run", help="start (or --resume) the loop")
    runp.add_argument("--run-id", default=None)
    runp.add_argument("--max-nodes", type=int, default=None)
    runp.add_argument("--resume", action="store_true")
    runp.add_argument("--config", nargs="+", default=[], metavar="FILE",
                      help="YAML files of the kernel.yaml keys this run changes, applied in order, e.g. "
                           "configs/h100.yaml configs/rain_start.yaml (new runs only; a resumed run keeps its own)")
    runp.add_argument("--git-remote", default=None, metavar="URL",
                      help="push every agent branch of the run to this git repo (kept for --resume); "
                           "without it the branches stay local in runs/<run>/agents.git")
    stop = sub.add_parser("stop", help="stop a running loop")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--force", action="store_true")
    uploadp = sub.add_parser("upload", help="upload a stopped run's new and changed files to Hugging Face (HFK)")
    uploadp.add_argument("--run-id", required=True)
    fetchp = sub.add_parser("fetch", help="download a run that uploaded itself to Hugging Face (HFK), for the panel")
    fetchp.add_argument("--run-id", required=True)
    fetchp.add_argument("--delete", action="store_true",
                        help="then delete from Hugging Face every upload repo this copy holds (each checked first)")
    doctor = sub.add_parser("doctor", help="check this tree is runnable here (paths, envs, symlinks)")
    doctor.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    doctor.add_argument("--gpu", action="store_true", help="also check the cards, the driver and each env's GPU packages")
    doctor.add_argument("--config", nargs="+", default=[], metavar="FILE", help="check under these kernel.yaml overlays")
    doctor.add_argument("--run-id", default=None, help="check under this run's frozen config (before a resume)")
    args = parser.parse_args(argv)

    overlays = getattr(args, "config", [])
    if overlays and getattr(args, "resume", False):
        raise ValueError("--config applies to a new run only; a resumed run keeps the config it started with")
    cfg = KernelConfig.load(overlays=overlays)
    # Into os.environ itself, so every subprocess (e.g. WBench's VLM phase) sees
    # the keys too -- not just the kernel's own metric decisions.
    load_dotenv(cfg.repo_root / ".env", os.environ)
    # Before bootstrap_run: diagnostics must not create a run dir or rewrite versions.json.
    if args.command == "doctor":
        if args.run_id:
            if overlays:
                raise ValueError("--config and --run-id: a run is checked under its own frozen config")
            cfg = KernelConfig.for_run(existing_run_dir(cfg, args.run_id))
        return report(run_checks(cfg) + (gpu_checks(cfg) if args.gpu else []), strict=args.strict)
    if args.command == "fetch":
        if not os.environ.get("HFK", "").strip():
            raise ValueError("HFK is empty; set it in AutoResearcher/.env before `ar fetch`")
        print(fetch(cfg.runs_dir / args.run_id, os.environ["HFK"], delete=args.delete))
        return 0

    if args.command == "upload":
        return _upload(cfg, args.run_id)
    if args.command == "run":
        faulthandler.enable(sys.__stderr__, all_threads=True)   # a crash (SIGBUS, SIGSEGV) prints every thread's stack
        return _run(cfg, args)
    if args.command == "stop":
        run_dir = cfg.runs_dir / args.run_id
        if not (run_dir / "config" / "run.json").exists():
            print(f"error: no run {args.run_id!r}", file=sys.stderr)
            return 2
        telemetry.open(run_dir, cfg.dotenv_files)
        capture.install()
        root = telemetry.start("ar.stop", {"force": args.force})
        code = 1
        try:
            code = _stop(Control(run_dir), args.force)
        finally:
            root.end("ok" if code == 0 else "error", {"exit_code": code})
        return code

    # status / trace / score-node read an EXISTING run and never write into it.
    try:
        if args.command == "status":
            run_dir = existing_run_dir(cfg, args.run_id)
            d = run_status(run_dir)
            print(json.dumps(d, indent=1) if args.json else format_status(d))
            return 0
        if args.command == "trace":
            run_dir = existing_run_dir(cfg, args.run_id)
            trace = tree.load(run_dir)
            if args.check:
                findings = tree.check(trace)
                print("\n".join(findings) if findings else "no findings")
                return 1 if findings else 0
            print(render(trace, node=args.node, attempt=args.attempt, since=args.since, depth=args.depth,
                         full=args.full))
            return 0
        if args.command == "score-node":
            score, out = rescore_node(cfg, args.run_id, args.node, os.environ)
            print(json.dumps({"node": args.node, "score": score, "output": str(out)}, indent=2))
            return 0
    except (RunNotFound, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _stop(control: Control, force: bool) -> int:
    if not force:
        control.request_stop()
        if control.alive_pid() is None:
            print("no loop is running for this run; the stop request was written anyway")
        return 0
    pid = control.alive_pid()
    if pid is None:
        print("no loop is running for this run", file=sys.stderr)
        return 1
    started = proc_start_time(pid)
    terminate(pid, "ar stop --force")
    with telemetry.span("stop.wait", {"pid": pid, "limit_s": FORCE_STOP_WAIT_S}) as sp:
        deadline = time.monotonic() + FORCE_STOP_WAIT_S   # its own cleanup: sandboxes, GPU jobs
        while proc_running(pid, started) and time.monotonic() < deadline:
            time.sleep(0.5)
        sp.output = {"ended": not proc_running(pid, started)}
    if not sp.output["ended"]:
        print(f"the loop is still running (pid {pid}) {FORCE_STOP_WAIT_S:.0f} s after SIGTERM; "
              f"`kill -9 {pid}` ends it now (a resume then cleans up after it)", file=sys.stderr)
        return 1
    print("stopped")
    return 0


def _upload(cfg, run_id: str) -> int:
    run_dir = existing_run_dir(cfg, run_id)
    if (pid := Control(run_dir).alive_pid()) is not None:
        raise RuntimeError(f"the loop is still running (pid {pid}); `ar upload` is for a stopped run")
    if not os.environ.get("HFK", "").strip():
        raise ValueError("HFK is empty; set it in AutoResearcher/.env before `ar upload`")
    state_file = run_dir / "control" / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    HfUpload(run_dir, os.environ["HFK"]).upload(
        state.get("node") or "run", "manual", state.get("attempt") or 0)
    print(f"uploaded {run_dir}")
    return 0


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _run(cfg, args) -> int:
    """The start checks, all before anything is written; then the run is created or attached, claimed, and the
    writer opened."""
    run_dir = cfg.runs_dir / args.run_id if args.run_id else None
    existing = run_dir is not None and (run_dir / "config" / "run.json").exists()
    if existing and not args.resume:
        return _refuse(f"run {args.run_id!r} exists; use --resume")
    if args.resume and not existing:
        return _refuse(f"no run {args.run_id!r} to resume")
    control = Control(run_dir) if existing else None
    if control is not None and (pid := control.alive_pid()) is not None:
        return _refuse(f"a loop is already running for this run (pid {pid})")
    max_nodes = args.max_nodes if args.max_nodes is not None else control.args().get("max_nodes") if control else None
    if max_nodes is None:
        return _refuse("--max-nodes is required for a new run")
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if not os.environ.get(key, "").strip():
            raise ValueError(f"{key} is empty; set it in AutoResearcher/.env before `ar run`")
    remote = args.git_remote or (control.args().get("git_remote") if control else None)
    if remote and (error := AgentsRepo.check_remote(remote)):
        where = "the run's git remote" if args.resume else "--git-remote"
        return _refuse(f"cannot reach {where} {remote}: {error}")
    if not args.resume:
        if remote and AgentsRepo.has_run(remote, args.run_id):
            return _refuse(f"--git-remote {remote} already has a run named {args.run_id!r}; use another --run-id")
        if args.run_id and os.environ.get("HFK", "").strip():
            from huggingface_hub import HfApi
            if taken := run_uploads(HfApi(token=os.environ["HFK"]), args.run_id):
                return _refuse(f"Hugging Face already holds uploads of a run named {args.run_id!r} "
                               f"({', '.join(sorted(taken.values())[:3])}{', ...' if len(taken) > 3 else ''}); "
                               "use another --run-id")
    try:
        if args.resume:
            judge = check_resume(cfg, args.run_id, os.environ)
        else:
            new = check_new_run(cfg, args.run_id, os.environ)
    except PreflightError as exc:
        return _refuse(str(exc))
    run_cfg = KernelConfig.for_run(run_dir) if args.resume else cfg    # a new run freezes the live config
    gpus = choose_gpus(run_cfg, os.environ)
    check_tools_fit(run_cfg, gpus)
    changed = code_changes(cfg, json.loads((run_dir / "config" / "run.json").read_text())["versions"]) \
        if args.resume else []
    if changed:
        print("warning: " + _CHANGED + "; ".join(changed), file=sys.stderr)
    ctx = attach_run(cfg, args.run_id, judge, gpus) if args.resume else bootstrap_run(cfg, new, gpus)
    control = Control(ctx.run_dir)
    try:
        control.claim()        # two starts at once: before touching any run file, so one never cleans up the other
    except RuntimeError as exc:
        return _refuse(str(exc))
    telemetry.open(ctx.run_dir, cfg.dotenv_files)
    capture.install()
    w = telemetry.writer()
    root = telemetry.start("ar.run", {"cwd": os.getcwd(), "user": getpass.getuser(),
                                      "env": w.redact_env(os.environ) if w else None,
                                      "overlays": [str(p) for p in args.config], "gpus": gpus})
    try:
        code = _claimed(args, ctx, control, remote, max_nodes, changed)
    except BaseException as exc:
        root.fail(exc)
        raise
    if w is not None and w.failed:                      # the records are incomplete: never a clean exit
        code = 1
    root.end("ok" if code == 0 else "error", {"exit_code": code})
    return exit_process(code)                           # never returns: stuck worker threads must not keep it alive


_CHANGED = "the code differs from what this run started with, so later nodes are not built and scored like the earlier ones: "


def _claimed(args, ctx, control: Control, remote: str | None, max_nodes: int, changed: list[str]) -> int:
    """The run once claimed; returns the exit code."""
    if changed:
        telemetry.alert("run.code_changed", "warning", _CHANGED + "; ".join(changed), {"changes": changed})
    control.save_args(max_nodes=max_nodes, **({"git_remote": remote} if remote else {}))
    run_cfg = KernelConfig.for_run(ctx.run_dir)               # one config per run: never the live one
    record_envs(run_cfg)
    repo = AgentsRepo(ctx.run_dir / "agents.git", remote=remote, namespace=ctx.run_dir.name,
                      on_push_error=_push_failed)
    if args.resume:
        repo.push()                                     # catch up pushes a stopped kernel missed
    kill_recorded_groups(control)                       # GPU jobs a killed kernel left running
    clear_hf_tmp(ctx.run_dir)
    kit = build_run_kit(run_cfg, ctx.run_dir, ctx.gpus, os.environ, repo)
    upload = HfUpload(ctx.run_dir, os.environ["HFK"]) if os.environ.get("HFK", "").strip() else None
    loop = Loop(run_cfg, ctx, kit, repo, max_nodes=max_nodes, root_cache=run_cfg.root_cache, upload=upload)
    return drive_code(loop, kit, control)


def _push_failed(error: str) -> None:
    telemetry.alert("git.push_failed", "warning", error)
    print(f"warning: {error}", file=sys.stderr)


def drive_code(loop, kit, control: Control) -> int:
    """`drive`, as the run's exit code."""
    try:
        reason = drive(loop, kit, control)
    except BaseException as exc:                        # noqa: BLE001 -- reported, then the process ends
        telemetry.error(exc, "exit code 1")
        traceback.print_exc()
        return 1
    print(f"loop stopped: {reason}")
    return 0 if reason == NATURAL_END else 1


if __name__ == "__main__":
    sys.exit(main())
