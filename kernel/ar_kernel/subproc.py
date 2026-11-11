from __future__ import annotations
import hashlib, os, pwd, re, signal, socket, subprocess, tempfile, threading, time
from pathlib import Path

import psutil

from . import telemetry
from .config import REPO_ROOT
from .faults import KernelFault

KILL_GRACE_SECONDS = 30
KILL_FORCE_GRACE_SECONDS = 10

CACHE_VARS = {"HF_HOME": "huggingface", "XDG_CACHE_HOME": "", "TORCH_HOME": "torch",
              "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
              "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "nv", "PIP_CACHE_DIR": "pip",
              "UV_CACHE_DIR": "uv", "VLLM_CONFIG_ROOT": "vllm-config", "FLASHINFER_WORKSPACE_BASE": "flashinfer",
              "CUPY_CACHE_DIR": "cupy", "MPLCONFIGDIR": "matplotlib"}     # each defaults to the home folder or /tmp


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
    env["VLLM_NO_USAGE_STATS"] = "1"           # vLLM otherwise records and sends usage statistics
    return env


def project_env(name: str, root: Path | None = None) -> Path | None:
    """`<repo>/.envs/<name>` when it is a conda env, else None. Every env the project runs lives
    there (environments, data, models and code all stay in the project)."""
    path = (root or REPO_ROOT) / ".envs" / name
    return path if (path / "conda-meta").is_dir() else None


def env_path(env: str) -> Path:
    """`env`'s prefix, existing or not: a path when it contains "/" (repo-relative unless absolute), otherwise
    `.envs/<name>`."""
    path = Path(env) if "/" in env else Path(".envs") / env
    return path if path.is_absolute() else REPO_ROOT / path


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
               log_path: Path | None = None,
               cancel: "threading.Event | None" = None,
               poll_s: float = 1.0) -> subprocess.CompletedProcess:
    """Run in a conda environment; kill and reap its process group on cancellation, failure, or leader exit.
    Output goes to log_path or temporary files; AR_PGID_DIR records groups for resume cleanup.
    The result.cancelled flag distinguishes cancellation from exit status."""
    command = conda_command(env, args)
    process_env = {**os.environ, **cache_env(), **(extra_env or {})}
    sinks = []
    try:
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            sinks = [open(log_path, "w+b")]
        else:
            tmp = cache_dir() / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            sinks = [tempfile.TemporaryFile(dir=tmp), tempfile.TemporaryFile(dir=tmp)]
        proc = subprocess.Popen(command, cwd=str(cwd), env=process_env, start_new_session=True,
                                stdout=sinks[0], stderr=sinks[1] if len(sinks) > 1 else subprocess.STDOUT)
    except OSError:
        for sink in sinks:
            sink.close()
        raise
    with telemetry.span("process", _inputs(command, process_env, cwd, proc, log_path)) as sp:
        tail = Tail([(sink.fileno(), stream) for sink, stream in zip(sinks, ("stdout", "stderr"))])
        registry = os.environ.get("AR_PGID_DIR")
        marker = None
        cancelled = False
        ended = False           # the wait or the kill sequence finished: the group is gone
        try:
            if registry:
                marker = Path(registry) / str(proc.pid)          # start_new_session: pgid == pid
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(proc_start_time(proc.pid) or "")
            cancelled, usage = wait_leader(proc, cancel, poll_s)
            kill_group(proc.pid, "leftover")       # what the leader left running: its group outlives it
            ended = True
            tail.stop()
            outputs = []
            for sink in sinks:
                sink.seek(0)
                text = sink.read().decode("utf-8", errors="replace")
                outputs.append(text.replace("\r\n", "\n").replace("\r", "\n"))   # as text mode reads it
        except BaseException:
            if not ended:
                _kill_group(proc)      # e.g. KeyboardInterrupt: never leave the job running
                ended = True
            raise
        finally:
            tail.stop()                # before the close: its fds must not be reused under it
            for sink in sinks:
                sink.close()
            if marker is not None and ended:     # an interrupted kill keeps its record for resume
                marker.unlink(missing_ok=True)
        _ended(sp, proc.returncode, usage)

    stdout, stderr = outputs if len(outputs) == 2 else (outputs[0], "")
    result = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    result.cancelled = cancelled
    return result


def run_sandbox(argv: list[str], env: dict, *, out_path: Path, err_path: Path, inputs: dict,
                cancel: "threading.Event | None" = None, poll_s: float = 1.0, on_start=None) -> int:
    """bwrap `argv` in its own session as a `sandbox` span (`inputs` and the pid); stdout and stderr go to the files
    and their lines to log records. `cancel` or an exception SIGKILLs bwrap, whose --die-with-parent ends the rest.
    Returns the exit code (negative: the signal). A launch that fails raises KernelFault before any span."""
    sinks = []
    try:
        sinks = [open(out_path, "w+b"), open(err_path, "w+b")]
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=sinks[0], stderr=sinks[1],
                                start_new_session=True)     # a Ctrl-C to the kernel is not the agent's
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        for sink in sinks:
            sink.close()
        raise KernelFault(f"the sandbox could not start: {type(exc).__name__}: {exc}") from exc
    try:
        with telemetry.span("sandbox", {**inputs, "pid": proc.pid}) as sp:
            tail = Tail([(sinks[0].fileno(), "stdout"), (sinks[1].fileno(), "stderr")])
            try:
                if on_start is not None:
                    on_start(proc.pid)
                _, usage = wait_leader(proc, cancel, poll_s, kill=lambda: kill(proc, "cancel"))
            except BaseException:
                if proc.poll() is None:
                    kill(proc, "exception")
                    proc.wait()
                raise
            finally:
                tail.stop()
            _ended(sp, proc.returncode, usage)
    finally:
        for sink in sinks:
            sink.close()
    return proc.returncode


def _inputs(command: list[str], env: dict | None, cwd, proc: subprocess.Popen, log_path=None) -> dict:
    w = telemetry.writer()
    return {"command": command, "env": w.redact_env(env) if w and env is not None else env,
            "cwd": str(cwd) if cwd is not None else os.getcwd(), "log_path": log_path, "pid": proc.pid,
            "pgid": os.getpgid(proc.pid) if proc.returncode is None else None}


def _ended(sp, returncode: int, usage=None, **output) -> None:
    """A process span's end: the exit code or the signal, and its resource use."""
    if returncode < 0:
        sp.status, output["signal"] = "killed", signal.Signals(-returncode).name
    else:
        sp.status, output["exit_code"] = ("ok" if returncode == 0 else "error"), returncode
    if usage is not None:
        output["rusage"] = {k: getattr(usage, k) for k in dir(usage) if k.startswith("ru_")}
    sp.output = output


class Tail:
    """Each new complete line of the files becomes a `log` record under the creator's current span; the
    rest when stopped. `files`: (fd of a regular file, stream name)."""

    def __init__(self, files: list[tuple[int, str]], poll_s: float = 0.5) -> None:
        self._files = [[fd, stream, 0, b""] for fd, stream in files]
        self._poll_s, self._done = poll_s, threading.Event()
        self._thread = telemetry.thread(self._run, "ar-tail")
        self._thread.start()

    def _read(self, final: bool) -> None:
        for f in self._files:
            fd, stream, offset, partial = f
            while chunk := os.pread(fd, 1 << 20, offset):
                offset, partial = offset + len(chunk), partial + chunk
            *lines, partial = re.split(rb"\r\n|\r(?=.)|\n", partial, flags=re.S)
            if final and partial:
                lines.append(partial.rstrip(b"\r"))
                partial = b""
            for line in lines:
                telemetry.log(stream, line.decode("utf-8", errors="replace"))
            f[2], f[3] = offset, partial

    def _run(self) -> None:
        while not self._done.wait(self._poll_s):
            self._read(False)

    def stop(self) -> None:
        if not self._done.is_set():
            self._done.set()
            self._thread.join()
            self._read(True)


def run(argv: list[str], *, cwd=None, env: dict | None = None, input: str | bytes | None = None,
        timeout: float | None = None, check: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    """`subprocess.run(argv, capture_output=True, ...)` as a `process` span: the captured output goes on its
    end. A launch that fails raises before any span."""
    proc = subprocess.Popen(argv, cwd=cwd, env=env, text=text, stdin=subprocess.PIPE if input is not None else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with telemetry.span("process", _inputs(argv, env, cwd, proc)) as sp:
        try:
            stdout, stderr = proc.communicate(input, timeout)
        except subprocess.TimeoutExpired:
            kill(proc, "timeout")
            stdout, stderr = proc.communicate()
            _ended(sp, proc.returncode, stdout=_shown(stdout), stderr=_shown(stderr))
            sp.status = "timeout"
            raise subprocess.TimeoutExpired(argv, timeout, stdout, stderr) from None
        except BaseException:
            kill(proc, "exception")
            proc.wait()
            raise
        _ended(sp, proc.returncode, stdout=_shown(stdout), stderr=_shown(stderr))
    result = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def _shown(output: str | bytes):
    """Captured output as recorded: bytes that are not UTF-8 text (a tar archive) by their size."""
    if isinstance(output, bytes):
        try:
            return output.decode()
        except UnicodeDecodeError:
            return {"bytes": len(output)}
    return output


def kill(proc: subprocess.Popen, why: str) -> None:
    """SIGKILL one process (not its group), as a `process.kill` span."""
    with telemetry.span("process.kill", {"pid": proc.pid, "why": why}) as sp:
        gone = proc.poll() is not None
        if not gone:
            proc.kill()
        sp.output = {"signals": [] if gone else ["SIGKILL"], "already_gone": gone}


def terminate(pid: int, why: str) -> bool:
    """SIGTERM one process that is not our child, as a `process.kill` span. False if it was already gone."""
    with telemetry.span("process.kill", {"pid": pid, "why": why}) as sp:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            sp.output = {"signals": [], "already_gone": True}
            return False
        sp.output = {"signals": ["SIGTERM"], "already_gone": False}
        return True


def wait_leader(proc: subprocess.Popen, cancel: "threading.Event | None" = None, poll_s: float = 1.0,
                kill=None) -> tuple[bool, os.struct_rusage]:
    """Wait for `proc` itself, not its children; returns (cancelled, rusage) and sets `proc.returncode`.
    A set `cancel`, checked every `poll_s`, first runs `kill()` (default: kill the whole group).
    Reaps with os.wait4: Popen.wait would reap behind its back and lose the rusage."""
    usage = []

    def reap(block: bool = False) -> None:
        if not usage:
            pid, status, rusage = os.wait4(proc.pid, 0 if block else os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                usage.append(rusage)

    cancelled = False
    if cancel is not None:
        reap()
        while not usage and not cancel.is_set():
            time.sleep(poll_s)
            reap()
        if not usage:
            if kill is None:
                kill_group(proc.pid, "cancel", reap=reap)     # an unreaped leader would keep the group "alive"
            else:
                kill()
            cancelled = True
    reap(block=True)
    return cancelled, usage[0]


def kill_group(pgid: int, why: str, reap=None) -> bool:
    """SIGTERM process group `pgid`, then SIGKILL whatever is left, as a `process.kill` span. False if it was
    already gone. `reap` runs while waiting: a direct child's zombie would otherwise keep the group "alive"."""
    with telemetry.span("process.kill", {"pgid": pgid, "why": why}) as sp:
        sent = []
        sp.output = {"signals": sent, "already_gone": False}
        for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, KILL_FORCE_GRACE_SECONDS)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                sp.output["already_gone"] = not sent
                return bool(sent)
            sent.append(sig.name)
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
    kill_group(proc.pid, "exception", reap=proc.poll)    # start_new_session: pgid == pid, also once reaped


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
    except OSError as exc:
        telemetry.error(exc, "an empty tail")
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
