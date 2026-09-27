from __future__ import annotations
import argparse, json, os, shutil, signal, sys
from pathlib import Path

from .archive.nodes import NodeStore
from .config import KernelConfig, load_dotenv
from .control import Control, drive, kill_recorded_groups, mark_interrupted
from .doctor import report, run_checks
from .guards import check_visible
from .loop import Loop
from .run import RunNotFound, attach_run, bootstrap_run, score_node
from .run_kit import build_run_kit
from .subproc import cache_env
from .vcs.agents_repo import AgentsRepo

def main(argv: list[str] | None = None) -> int:
    os.environ.update(cache_env())
    parser = argparse.ArgumentParser(prog="ar")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-run", help="create a run directory and record versions")
    init.add_argument("--run-id", default=None)
    status = sub.add_parser("status", help="show nodes of a run")
    status.add_argument("--run-id", required=True)
    score = sub.add_parser("score-node", help="render and score one node on the proxy")
    score.add_argument("--run-id", required=True)
    score.add_argument("--node", required=True)
    score.add_argument("--checkpoint", default=None, help="node checkpoint dir; omit for the base model")
    score.add_argument("--lora-rank", type=int, default=64)
    score.add_argument("--lora-alpha", type=int, default=64)
    runp = sub.add_parser("run", help="start (or --resume) the loop")
    runp.add_argument("--run-id", default=None)
    runp.add_argument("--max-nodes", type=int, default=None)
    runp.add_argument("--resume", action="store_true")
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

    if args.command == "init-run":
        ctx = bootstrap_run(cfg, args.run_id, os.environ)
        print(ctx.run_dir)
        return 0

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
            os.kill(pid, signal.SIGTERM)
        else:
            control.request_stop()
        return 0

    # status / score-node act on an EXISTING run: attach, never create or rewrite it.
    try:
        ctx = attach_run(cfg, args.run_id, os.environ)
    except RunNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    nodes = NodeStore(ctx.conn)
    if args.command == "status":
        for node in nodes.all():
            print(f"{node['node_id']:<12} {node['status']:<14} score={node['score']}")
        return 0

    if args.command == "score-node":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        try:
            nodes.get(args.node)
        except KeyError:
            nodes.create(args.node, None, 0)
        score, detail = score_node(cfg, ctx, args.node, checkpoint, args.lora_rank, args.lora_alpha)
        nodes.record_score(args.node, score, ctx.metric_set, detail["metrics"])
        print(json.dumps({"node": args.node, "score": score}, indent=2))
        return 0
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
    ctx = attach_run(cfg, args.run_id, os.environ) if args.resume else bootstrap_run(cfg, args.run_id, os.environ)
    control = Control(ctx.run_dir)
    if control.alive_pid() not in (None, os.getpid()):       # before touching any run file
        print(f"error: a loop is already running (pid {control.alive_pid()})", file=sys.stderr)
        return 2
    if args.max_nodes is not None:
        control.save_args(max_nodes=args.max_nodes)
    max_nodes = control.args().get("max_nodes")
    if max_nodes is None:
        print("error: --max-nodes is required for a new run", file=sys.stderr)
        return 2
    run_cfg = KernelConfig.for_run(ctx.run_dir)
    check_visible(ctx.gpus)
    repo = AgentsRepo(ctx.run_dir / "agents.git")
    killed = kill_recorded_groups(control)              # GPU jobs a killed kernel left running
    slot = ctx.run_dir / "merge_slot"                   # ~52 GB run-level transient (spec 15), not node data
    removed_slot = slot.exists()
    if removed_slot:
        shutil.rmtree(slot)
    if killed or removed_slot:
        ctx.recorder.event("run.resume_cleanup", payload={"killed_groups": killed, "merge_slot": removed_slot})
    mark_interrupted(ctx, "unfinished when the loop stopped (forced stop, kernel death or spent budget)")
    kit = build_run_kit(run_cfg, ctx.run_dir, ctx.gpus, ctx.recorder, os.environ)
    loop = Loop(run_cfg, ctx, kit, repo, max_nodes=max_nodes)
    reason = drive(loop, kit, control, ctx.recorder)
    print(f"loop stopped: {reason}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
