from __future__ import annotations
import os, subprocess
from pathlib import Path

def run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None,
               timeout: int | None = None, recorder=None, node: str = "run",
               phase: str = "-") -> subprocess.CompletedProcess:
    command = ["conda", "run", "--no-capture-output", "-n", env, *args]
    process_env = {**os.environ, **(extra_env or {})}
    if recorder is not None:
        recorder.event("subproc.start", node=node, phase=phase,
                       payload={"env": env, "args": args, "cwd": str(cwd),
                                "extra_env": extra_env or {}})
    proc = subprocess.run(command, cwd=str(cwd), env=process_env, capture_output=True,
                          text=True, timeout=timeout)
    if recorder is not None:
        recorder.event("subproc.end", node=node, phase=phase, returncode=proc.returncode,
                       payload={"stdout": proc.stdout, "stderr": proc.stderr})
    return proc


def _tail(proc, limit: int = 4000) -> str:
    """Last `limit` chars of stdout AND stderr.

    Tracebacks go to stderr, so an error message built from stdout alone hides
    the actual cause -- a render that died in a ValueError reported only the
    progress banner it had printed before failing.
    """
    parts = []
    for name in ("stdout", "stderr"):
        text = (getattr(proc, name, "") or "").strip()
        if text:
            parts.append(f"--- {name} (last {limit}) ---\n{text[-limit:]}")
    return "\n".join(parts) if parts else "(no output captured)"
