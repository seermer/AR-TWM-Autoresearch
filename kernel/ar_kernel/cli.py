from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

from .archive.nodes import NodeStore
from .config import KernelConfig, load_dotenv
from .doctor import report, run_checks
from .run import RunNotFound, attach_run, bootstrap_run, score_node

def main(argv: list[str] | None = None) -> int:
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

if __name__ == "__main__":
    sys.exit(main())
