"""python -m panel --run-id <run>: the run panel behind Gradio's login, on a public share link."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_dotenv(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip("'\"")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m panel", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-share", action="store_true", help="serve locally only, no public link")
    args = parser.parse_args(argv)
    env = {**read_dotenv(REPO / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("PANEL_")}}
    user, password = env.get("PANEL_USER", ""), env.get("PANEL_PASSWORD", "")
    if not user or not password:
        print("error: set PANEL_USER and PANEL_PASSWORD in AutoResearcher/.env; the panel never runs "
              "without a login", file=sys.stderr)
        return 2
    run_dir = REPO / "runs" / args.run_id
    if not (run_dir / "config" / "run.json").is_file():
        print(f"error: no run {args.run_id!r} under {REPO / 'runs'}", file=sys.stderr)
        return 2
    # Gradio's file cache stays out of runs/ (the panel never writes there).
    os.environ.setdefault("GRADIO_TEMP_DIR", str(REPO / ".cache" / "panel"))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    from .ui import build_app
    app = build_app(run_dir)
    app.queue(default_concurrency_limit=4)
    app.launch(share=not args.no_share, server_name="127.0.0.1", server_port=args.port,
               auth=(user, password), allowed_paths=[str(run_dir)], footer_links=["gradio", "settings"],
               inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
