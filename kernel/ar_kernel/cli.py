from __future__ import annotations
import argparse, json, os, shutil, signal, sys, time, traceback

from .archive.nodes import NodeStore
from .config import KernelConfig, load_dotenv, resolve_gpus
from .control import Control, drive, exit_process, kill_recorded_groups, mark_interrupted
from .doctor import report, run_checks
from .guards import alert, check_tools_fit, check_visible
from .loop import Loop
from .monitor import Monitor
from .run import PreflightError, RunNotFound, attach_run, bootstrap_run, rescore_node
from .run_kit import build_run_kit
from .status import format_status, run_status
from .subproc import cache_env, proc_running, proc_start_time
from .vcs.agents_repo import AgentsRepo

FORCE_STOP_WAIT_S = 300.0     # the kernel's force-stop cleanup kills containers and GPU job groups


def main(argv: list[str] | None = None) -> int:
    os.environ.update(cache_env())
    parser = argparse.ArgumentParser(prog="ar")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show nodes of a run")
    status.add_argument("--run-id", required=True)
    status.add_argument("--json", action="store_true", help="print the raw JSON snapshot")
    score = sub.add_parser("score-node", help="score a node of a run again; writes under scores/, never the run")
    score.add_argument("--run-id", required=True)
    score.add_argument("--node", required=True)
    runp = sub.add_parser("run", help="start (or --resume) the loop")
    runp.add_argument("--run-id", default=None)
    runp.add_argument("--max-nodes", type=int, default=None)
    runp.add_argument("--resume", action="store_true")
    runp.add_argument("--git-remote", default=None, metavar="URL",
                      help="push every agent branch of the run to this git repo (kept for --resume); "
                           "without it the branches stay local in runs/<run>/agents.git")
    stop = sub.add_parser("stop", help="stop a running loop")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--force", action="store_true")
    doctor = sub.add_parser("doctor", help="check this tree is runnable here (paths, envs, symlinks)")
    doctor.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    args = parser.parse_args(argv)

    cfg = KernelConfig.load()
    # Into os.environ itself, so every subprocess (e.g. WBench's VLM phase) sees
    # the keys too -- not just the kernel's own metric decisions.
    load_dotenv(cfg.repo_root / ".env", os.environ)
    # Before bootstrap_run: diagnostics must not create a run dir or rewrite versions.json.
    if args.command == "doctor":
        return report(run_checks(cfg), strict=args.strict)

    if args.command == "run":
        return _run(cfg, args)
    if args.command == "stop":
        run_dir = cfg.runs_dir / args.run_id
        if not (run_dir / "config" / "run.json").exists():
            print(f"error: no run {args.run_id!r}", file=sys.stderr)
            return 2
        control = Control(run_dir)
        if args.force:
            pid = control.alive_pid()
            if pid is None:
                print("no loop is running for this run", file=sys.stderr)
                return 1
            started = proc_start_time(pid)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + FORCE_STOP_WAIT_S   # its own cleanup: containers, GPU jobs
            while proc_running(pid, started) and time.monotonic() < deadline:
                time.sleep(0.5)
            if proc_running(pid, started):
                print(f"the loop is still running (pid {pid}) {FORCE_STOP_WAIT_S:.0f} s after SIGTERM; "
                      f"`kill -9 {pid}` ends it now (a resume then cleans up after it)", file=sys.stderr)
                return 1
            print("stopped")
        else:
            control.request_stop()
            if control.alive_pid() is None:
                print("no loop is running for this run; the stop request was written anyway")
        return 0

    # status / score-node read an EXISTING run and never write into it.
    try:
        if args.command == "status":
            run_dir = cfg.runs_dir / args.run_id
            if not (run_dir / "config" / "run.json").exists():
                raise RunNotFound(f"no run {args.run_id!r} under {cfg.runs_dir}")
            d = run_status(run_dir)
            print(json.dumps(d, indent=1) if args.json else format_status(d))
            return 0
        if args.command == "score-node":
            score, out = rescore_node(cfg, args.run_id, args.node, os.environ)
            print(json.dumps({"node": args.node, "score": score, "output": str(out)}, indent=2))
            return 0
    except (RunNotFound, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _run(cfg, args) -> int:
    existing = args.run_id and (cfg.runs_dir / args.run_id / "config" / "run.json").exists()
    if existing and not args.resume:
        print(f"error: run {args.run_id!r} exists; use --resume", file=sys.stderr)
        return 2
    if args.resume and not existing:
        print(f"error: no run {args.run_id!r} to resume", file=sys.stderr)
        return 2
    if not args.resume and args.max_nodes is None:           # before bootstrap_run creates the run
        print("error: --max-nodes is required for a new run", file=sys.stderr)
        return 2
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):           # before anything is created or claimed
        if not os.environ.get(key, "").strip():
            raise ValueError(f"{key} is empty; set it in AutoResearcher/.env before `ar run`")
    if not args.resume:
        check_visible(resolve_gpus(cfg, os.environ))          # the live config is what the run will freeze
        check_tools_fit(cfg, resolve_gpus(cfg, os.environ))
        if args.git_remote and (error := AgentsRepo.check_remote(args.git_remote)):
            print(f"error: cannot reach --git-remote {args.git_remote}: {error}", file=sys.stderr)
            return 2
    try:
        ctx = attach_run(cfg, args.run_id, os.environ) if args.resume else bootstrap_run(cfg, args.run_id, os.environ)
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.resume:
        try:
            status = NodeStore(ctx.conn).get("root")["status"]
        except KeyError:
            status = "missing"
        if status != "scored":                              # the run holds no information; files are kept
            print(f"error: run {args.run_id!r} cannot be resumed: its root was never scored (root: {status}). "
                  "Start a new run; this one is kept as is.", file=sys.stderr)
            return 2
    control = Control(ctx.run_dir)
    remote = args.git_remote or control.args().get("git_remote")
    if args.resume and remote and (error := AgentsRepo.check_remote(remote)):
        print(f"error: cannot reach the run's git remote {remote}: {error}", file=sys.stderr)
        return 2
    try:
        control.claim()        # before touching any run file: a second resume must not clean up a live loop
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.max_nodes is not None:
        control.save_args(max_nodes=args.max_nodes)
    if args.git_remote:
        control.save_args(git_remote=args.git_remote)
    max_nodes = control.args().get("max_nodes")
    if max_nodes is None:
        print("error: --max-nodes is required for a new run", file=sys.stderr)
        return 2
    run_cfg = KernelConfig.for_run(ctx.run_dir)
    ctx.gpus = resolve_gpus(run_cfg, os.environ)              # one config per run: never the live one
    check_visible(ctx.gpus)
    check_tools_fit(run_cfg, ctx.gpus)
    repo = AgentsRepo(ctx.run_dir / "agents.git", remote=remote, namespace=ctx.run_dir.name,
                      on_push_error=lambda error: alert(ctx.recorder, "git_push_failed", error, level="warning"))
    if args.resume:
        repo.push()                                     # catch up pushes a stopped kernel missed
    killed = kill_recorded_groups(control)              # GPU jobs a killed kernel left running
    hf_tmp = ctx.run_dir / "hf_tmp"                     # kernel-private partial downloads of a stopped kernel
    partial = sorted(p.name for p in hf_tmp.iterdir()) if hf_tmp.is_dir() else []
    for name in partial:
        shutil.rmtree(hf_tmp / name, ignore_errors=True)
    if killed or partial:
        ctx.recorder.event("run.resume_cleanup", payload={"killed_groups": killed, "hf_tmp": partial})
    mark_interrupted(ctx, "unfinished when the loop stopped (forced stop, kernel death or spent budget)")
    kit = build_run_kit(run_cfg, ctx.run_dir, ctx.gpus, ctx.recorder, os.environ)
    loop = Loop(run_cfg, ctx, kit, repo, max_nodes=max_nodes, root_cache=run_cfg.root_cache)
    monitor = Monitor(run_cfg, ctx.run_dir, ctx.recorder, ctx.gpus, kit.budget)
    try:
        reason = drive(loop, kit, control, ctx.recorder, monitor)
    except BaseException:                               # noqa: BLE001 -- reported, then the process ends
        traceback.print_exc()
        code = 1
    else:
        print(f"loop stopped: {reason}")
        code = 0
    return exit_process(code, ctx.recorder)             # never returns: stuck worker threads must not keep it alive

if __name__ == "__main__":
    sys.exit(main())
