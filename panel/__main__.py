"""python -m panel --run-id <run>: the run panel on every interface, port 7860 unless --port."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ar_kernel.subproc import tmp_link

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m panel", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    run_dir = REPO / "runs" / args.run_id
    if not (run_dir / "config" / "run.json").is_file():
        print(f"error: no run {args.run_id!r} under {REPO / 'runs'}", file=sys.stderr)
        return 2
    os.environ["TMPDIR"] = str(tmp_link())     # a stopped run's db copy stays inside the project
    from .server import DIST, create_app
    if not (DIST / "index.html").is_file():
        print(f"error: the panel's web app is not built ({DIST}); run scripts/setup_envs.sh panel", file=sys.stderr)
        return 2
    import uvicorn
    # an open live stream would otherwise hold Ctrl+C until every browser tab closes
    uvicorn.run(create_app(run_dir), host="0.0.0.0", port=args.port, log_level="warning", timeout_graceful_shutdown=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
