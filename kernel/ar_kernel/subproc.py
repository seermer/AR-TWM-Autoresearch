from __future__ import annotations
import os, signal, socket, subprocess, threading, time
from pathlib import Path

from .config import REPO_ROOT

KILL_GRACE_SECONDS = 30
KILL_FORCE_GRACE_SECONDS = 10

CACHE_VARS = {"HF_HOME": "huggingface", "XDG_CACHE_HOME": "", "TORCH_HOME": "torch",
              "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
              "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "nv", "PIP_CACHE_DIR": "pip",
              "UV_CACHE_DIR": "uv"}


def free_port() -> int:
    """A free local TCP port (for MASTER_PORT, a vLLM server, ...)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cache_dir() -> Path:
    """Where model downloads, compile caches and package caches go: inside the project
    (AutoResearcher/.cache), or AR_CACHE_DIR. The user's rule: large files stay in the project."""
    return Path(os.environ.get("AR_CACHE_DIR") or REPO_ROOT / ".cache")


def cache_env() -> dict[str, str]:
    root = cache_dir()
    return {var: str(root / sub) if sub else str(root) for var, sub in CACHE_VARS.items()}


def project_env(name: str, root: Path | None = None) -> Path | None:
    """`<repo>/.envs/<name>` when it is a conda env, else None. Every env the project runs lives
    there (user rule, 2026-09-28: environments, data, models and code all stay in the project)."""
    path = (root or REPO_ROOT) / ".envs" / name
    return path if (path / "conda-meta").is_dir() else None


def conda_command(env: str, args: list[str]) -> list[str]:
    """`conda run` argv for `env`: a conda-prefix path when it contains "/" (repo-relative unless
    absolute), otherwise the project env `.envs/<name>`."""
    if "/" in env:
        path = Path(env)
        path = path if path.is_absolute() else REPO_ROOT / path
    else:
        path = project_env(env)
        if path is None:
            raise FileNotFoundError(f"no conda env {env!r} in {REPO_ROOT / '.envs'}")
    return ["conda", "run", "--no-capture-output", "-p", str(path), *args]


def proc_start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat: with the pid it names one process, so a recycled pid
    (after kill -9 or a reboot) is never mistaken for the process that was recorded."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def proc_running(pid: int, started: str | None) -> bool:
    """The process recorded as (pid, start time) still runs: same start time, and not a zombie
    (exited but not yet reaped by its parent)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return started is not None and fields[19] == started and fields[0] != "Z"


class SubprocTimeout(subprocess.TimeoutExpired):
    """A phase exceeded its timeout; its whole process group has been killed.

    `.output` holds whatever the job printed before it was stopped.
    """


def run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None,
               timeout: int | None = None, recorder=None, node: str = "run",
               phase: str = "-", log_path: Path | None = None,
               cancel: "threading.Event | None" = None,
               poll_s: float = 1.0,
               liveness: "object | None" = None) -> subprocess.CompletedProcess:
    """Run `args` inside conda env `env`, in its own session.

    Timeouts, `cancel`, `liveness` expiry and any exception kill the WHOLE process group:
    killing only `conda run` orphaned torchrun ranks that kept holding GPUs. With `log_path`,
    output streams to that file and comes back as stdout. The result's `.cancelled` tells a
    cancel apart from an exit code that happens to match. With `AR_PGID_DIR` set, the group
    is recorded there until the wait ends, so a resume can kill what a dead kernel left.
    Every exit path records subproc.end, subproc.cancelled or subproc.error.
    """
    command = conda_command(env, args)
    process_env = {**os.environ, **cache_env(), **(extra_env or {})}
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
        proc = subprocess.Popen(command, cwd=str(cwd), env=process_env, text=True, start_new_session=True,
                                stdout=subprocess.PIPE if sink is None else sink,
                                stderr=subprocess.PIPE if sink is None else subprocess.STDOUT)
    except OSError as exc:
        if sink is not None:
            sink.close()
        _error(recorder, node, phase, f"launch failed: {exc}", "", "", log_path)
        raise
    registry = os.environ.get("AR_PGID_DIR")
    marker = None
    if registry:
        marker = Path(registry) / str(proc.pid)          # start_new_session: pgid == pid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(proc_start_time(proc.pid) or "")

    cancelled = False
    ended = False           # the wait or the kill sequence finished: the group is gone
    try:
        stdout, stderr, cancelled = _wait(proc, timeout, cancel, poll_s, liveness)
        ended = True
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        ended = True
        stdout, stderr = proc.communicate()
        if sink is not None:
            sink.close()
            stdout, stderr = Path(log_path).read_text(encoding="utf-8", errors="replace"), ""
        message = liveness.reason if (liveness is not None and liveness.reason) else f"timed out after {timeout}s"
        _error(recorder, node, phase, f"{message}; process group killed",
               stdout or "", stderr or "", log_path)
        raise SubprocTimeout(command, timeout, output=stdout, stderr=stderr) from None
    except BaseException:
        _kill_group(proc)          # e.g. KeyboardInterrupt: never leave the job running
        ended = True
        raise
    finally:
        if sink is not None and not sink.closed:
            sink.close()
        if marker is not None and ended:     # an interrupted kill keeps its record for resume
            marker.unlink(missing_ok=True)

    if log_path is not None:
        stdout, stderr = Path(log_path).read_text(encoding="utf-8", errors="replace"), ""
    result = subprocess.CompletedProcess(command, proc.returncode, stdout or "", stderr or "")
    result.cancelled = cancelled
    if recorder is not None:
        recorder.event("subproc.cancelled" if cancelled else "subproc.end", node=node, phase=phase,
                       returncode=proc.returncode,
                       payload={"stdout": result.stdout, "stderr": result.stderr,
                                "log_path": str(log_path) if log_path else None})
    return result


def _wait(proc: subprocess.Popen, timeout: int | None, cancel: "threading.Event | None",
         poll_s: float, liveness=None) -> tuple[str, str, bool]:
    """Wait for `proc`; returns (stdout, stderr, cancelled). With `cancel` or `liveness`,
    poll every `poll_s` (a timed-out communicate() does not kill the child)."""
    if cancel is None and liveness is None:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, False
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_s)
            return stdout, stderr, False
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                _kill_group(proc)
                stdout, stderr = proc.communicate()
                return stdout, stderr, True
            if liveness is not None and liveness.expired():
                raise subprocess.TimeoutExpired(proc.args, timeout or 0)
            if deadline is not None and time.monotonic() >= deadline:
                raise


def kill_group(pgid: int, reap=None) -> bool:
    """SIGTERM process group `pgid`, then SIGKILL whatever is left. False if it was already gone.
    `reap` runs while waiting: a direct child's zombie would otherwise keep the group "alive"."""
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, KILL_FORCE_GRACE_SECONDS)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return sig != signal.SIGTERM
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if reap is not None:
                reap()
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return True
            time.sleep(0.2)
    return True


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    kill_group(pgid, reap=proc.poll)


def _error(recorder, node, phase, message, stdout, stderr, log_path) -> None:
    if recorder is not None:
        recorder.event("subproc.error", node=node, phase=phase,
                       payload={"message": message, "stdout": stdout[-20000:],
                                "stderr": stderr[-20000:],
                                "log_path": str(log_path) if log_path else None})


def output_tail(proc, limit: int = 4000) -> str:
    """Last `limit` chars of stdout AND stderr: tracebacks go to stderr, so a message built
    from stdout alone hides the actual cause."""
    parts = []
    for name in ("stdout", "stderr"):
        text = (getattr(proc, name, "") or "").strip()
        if text:
            parts.append(f"--- {name} (last {limit}) ---\n{text[-limit:]}")
    return "\n".join(parts) if parts else "(no output captured)"


def file_tail(path: Path, limit: int) -> str:
    """The last `limit` characters of a text file; "" when it cannot be read."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def meminfo_gib() -> dict[str, float]:
    """/proc/meminfo's MemTotal and MemAvailable, in GiB."""
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {k: int(fields[k].split()[0]) / 2**20 for k in ("MemTotal", "MemAvailable")}
