from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path

KILL_GRACE_SECONDS = 30


class SubprocTimeout(subprocess.TimeoutExpired):
    """A phase exceeded its timeout; its whole process group has been killed.

    `.output` holds whatever the job printed before it was stopped.
    """


def run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None,
               timeout: int | None = None, recorder=None, node: str = "run",
               phase: str = "-", log_path: Path | None = None) -> subprocess.CompletedProcess:
    """Run `args` inside conda env `env`.

    The job runs in its own session, so a timeout kills the WHOLE process group:
    killing only `conda run` orphaned torchrun ranks and multiprocessing workers,
    which kept holding GPUs and made the next job OOM.

    With `log_path`, stdout and stderr stream to that file as the job runs (for
    long training jobs, instead of buffering hours of output in memory); the
    returned CompletedProcess carries the file's contents as stdout.

    Every exit path records an event: subproc.end, or subproc.error on timeout
    or launch failure, including the output captured so far.
    """
    command = ["conda", "run", "--no-capture-output", "-n", env, *args]
    process_env = {**os.environ, **(extra_env or {})}
    if recorder is not None:
        recorder.event("subproc.start", node=node, phase=phase,
                       payload={"env": env, "args": args, "cwd": str(cwd),
                                "extra_env": extra_env or {},
                                "log_path": str(log_path) if log_path else None})
    sink = None
    try:
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            sink = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(command, cwd=str(cwd), env=process_env, text=True,
                                    stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        else:
            proc = subprocess.Popen(command, cwd=str(cwd), env=process_env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
    except OSError as exc:
        if sink is not None:
            sink.close()
        _error(recorder, node, phase, f"launch failed: {exc}", "", "", log_path)
        raise

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        stdout, stderr = proc.communicate()
        if sink is not None:
            sink.close()
            stdout, stderr = Path(log_path).read_text(encoding="utf-8", errors="replace"), ""
        _error(recorder, node, phase, f"timed out after {timeout}s; process group killed",
               stdout or "", stderr or "", log_path)
        raise SubprocTimeout(command, timeout, output=stdout, stderr=stderr) from None
    except BaseException:
        _kill_group(proc)          # e.g. KeyboardInterrupt: never leave the job running
        raise
    finally:
        if sink is not None and not sink.closed:
            sink.close()

    if log_path is not None:
        stdout, stderr = Path(log_path).read_text(encoding="utf-8", errors="replace"), ""
    result = subprocess.CompletedProcess(command, proc.returncode, stdout or "", stderr or "")
    if recorder is not None:
        recorder.event("subproc.end", node=node, phase=phase, returncode=proc.returncode,
                       payload={"stdout": result.stdout, "stderr": result.stderr,
                                "log_path": str(log_path) if log_path else None})
    return result


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the job's process group, then SIGKILL whatever is left."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            proc.poll()            # reap our direct child, or its zombie keeps the group "alive"
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)


def _error(recorder, node, phase, message, stdout, stderr, log_path) -> None:
    if recorder is not None:
        recorder.event("subproc.error", node=node, phase=phase,
                       payload={"message": message, "stdout": stdout[-20000:],
                                "stderr": stderr[-20000:],
                                "log_path": str(log_path) if log_path else None})


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
