from __future__ import annotations
import hashlib, os, pwd, signal, socket, subprocess, threading, time
from pathlib import Path

import psutil

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
    (AutoResearcher/.cache), or AR_CACHE_DIR."""
    return Path(os.environ.get("AR_CACHE_DIR") or REPO_ROOT / ".cache")


TMP_LINK_DIR = Path("/tmp")


def tmp_link() -> Path:
    """A short symlink to <cache>/tmp, used as TMPDIR: Unix socket paths (vLLM, DataLoader workers)
    are capped at 107 bytes and the project path is too long; the files stay in the project."""
    target = cache_dir() / "tmp"
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    link = TMP_LINK_DIR / f"ar-{pwd.getpwuid(os.getuid()).pw_name}-{hashlib.sha1(str(target).encode()).hexdigest()[:8]}"
    try:
        link.symlink_to(target)
    except FileExistsError:
        pass
    if not (link.is_symlink() and link.resolve() == target):
        raise RuntimeError(f"{link} exists and is not a symlink to {target}: remove it")
    return link


def cache_env() -> dict[str, str]:
    root = cache_dir()
    env = {var: str(root / sub) if sub else str(root) for var, sub in CACHE_VARS.items()}
    env["TMPDIR"] = str(tmp_link())
    return env


def project_env(name: str, root: Path | None = None) -> Path | None:
    """`<repo>/.envs/<name>` when it is a conda env, else None. Every env the project runs lives
    there (environments, data, models and code all stay in the project)."""
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


def run_in_env(env: str, args: list[str], *, cwd: Path, extra_env: dict | None = None,
               recorder=None, node: str = "run",
               phase: str = "-", log_path: Path | None = None,
               cancel: "threading.Event | None" = None,
               poll_s: float = 1.0) -> subprocess.CompletedProcess:
    """Run `args` inside conda env `env`, in its own session.

    `cancel` and any exception kill the WHOLE process group:
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
        stdout, stderr, cancelled = _wait(proc, cancel, poll_s)
        ended = True
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


def _wait(proc: subprocess.Popen, cancel: "threading.Event | None",
          poll_s: float) -> tuple[str, str, bool]:
    """Wait for `proc`; returns (stdout, stderr, cancelled). With `cancel`, poll every `poll_s`
    (a timed-out communicate() does not kill the child)."""
    if cancel is None:
        stdout, stderr = proc.communicate()
        return stdout, stderr, False
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_s)
            return stdout, stderr, False
        except subprocess.TimeoutExpired:
            if cancel.is_set():
                _kill_group(proc)
                stdout, stderr = proc.communicate()
                return stdout, stderr, True


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


def memory_cgroup(proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")) -> Path | None:
    """The memory cgroup of this process (v1 memory controller, else v2), or None."""
    lines = [line.split(":", 2) for line in proc_cgroup.read_text().splitlines()]
    for _, controllers, path in lines:
        if "memory" in controllers.split(","):
            return root / "memory" / path.lstrip("/")
    for _, controllers, path in lines:
        if controllers == "":
            return root / path.lstrip("/")
    return None


def memory_limit(cgroup: Path | None) -> tuple[Path, int] | None:
    """(cgroup dir, bytes) of the lowest memory limit on `cgroup` and its ancestors, or None."""
    best = None
    for d in [cgroup, *cgroup.parents] if cgroup else []:
        for name in ("memory.max", "memory.limit_in_bytes"):
            value = (d / name).read_text().strip() if (d / name).is_file() else ""
            if value.isdigit() and int(value) < 2**60 and (best is None or int(value) < best[1]):
                best = (d, int(value))
    return best


def memory_events(cgroup: Path | None = None) -> dict[str, int]:
    """Counters of the cgroup holding the job's limit: `max` (usage reached the limit), `oom_kill` (processes the
    kernel killed there); {} without a limit."""
    found = memory_limit(cgroup or memory_cgroup())
    if found is None:
        return {}
    d = found[0]
    if (d / "memory.events").is_file():                                   # v2
        events = dict(line.split() for line in (d / "memory.events").read_text().splitlines())
        return {"max": int(events.get("max", 0)), "oom_kill": int(events.get("oom_kill", 0))}
    oom = dict(line.split() for line in (d / "memory.oom_control").read_text().splitlines())
    return {"max": int((d / "memory.failcnt").read_text()), "oom_kill": int(oom.get("oom_kill", 0))}


def host_memory_gib(cgroup: Path | None = None) -> dict[str, float | None]:
    """Memory of this job's cgroup (the one holding its limit) in GiB: limit, used (page cache included), anon,
    file, shmem; without a limit, the whole host (limit None; used without reclaimable cache)."""
    found = memory_limit(cgroup or memory_cgroup())
    if found is None:
        mem = psutil.virtual_memory()
        return {"limit": None, "used": (mem.total - mem.available) / 2**30, "shmem": mem.shared / 2**30}
    d, limit = found
    stat = dict(line.split() for line in (d / "memory.stat").read_text().splitlines())
    v2 = (d / "memory.current").is_file()
    used = int((d / ("memory.current" if v2 else "memory.usage_in_bytes")).read_text())
    keys = ("anon", "file", "shmem") if v2 else ("total_rss", "total_cache", "total_shmem")
    return {"limit": limit / 2**30, "used": used / 2**30,
            **{k: int(stat[s]) / 2**30 for k, s in zip(("anon", "file", "shmem"), keys) if s in stat}}
