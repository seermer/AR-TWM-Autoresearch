"""Run one agent call in one bwrap sandbox."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..guards import device_minors

# Env var names whose value must never reach telemetry, no matter which Recorder
# is used (defense in depth: a Recorder that was never told about this run's
# token via add_redaction must still not leak it through the recorded sandbox settings).
_SECRET_ENV_KEYS = ("AR_TOKEN",)


@dataclass
class Mounts:
    agent: Path
    workspace: Path
    staging: Path
    context: Path
    contract: Path
    sockets: Path
    agent_readonly: bool = False
    # node id -> node dir of each finished node, mounted read-only at /nodes/<id>. Its eval/ is hidden:
    # per-case outputs would let an agent fit the proxy cases.
    nodes: dict[str, Path] = field(default_factory=dict)
    # The agent code that runs, mounted read-only at /code, when it is not the tree at /agent
    # (edit_self: the parent's code edits the tree).
    code: Path | None = None


@dataclass
class RunResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    container: str = ""


def container_name(run_id: str, node: str, phase: str, attempt: int) -> str:
    raw = f"ar-{run_id}-{node}-{phase}-{attempt}-{secrets.token_hex(3)}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def snapshot(root: Path, hash_files: bool) -> dict[str, str]:
    """path -> fingerprint. Hash small code trees; use size+mtime for workspaces
    that may hold gigabytes of video. Only regular files count (never a FIFO or
    a symlink); a file the agent made unreadable is recorded as "unreadable"."""
    out = {}
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            if hash_files:
                try:
                    with p.open("rb") as f:
                        out[rel] = hashlib.file_digest(f, "sha256").hexdigest()
                except OSError:
                    out[rel] = "unreadable"
            else:
                st = p.stat()
                out[rel] = f"{st.st_size}:{st.st_mtime_ns}"
    return out


def restore_owner_access(root: Path) -> None:
    """Give the owner rw (and x on directories) on everything under `root` again, never following
    a link. A chmod-000 file or directory an agent leaves (sandboxes run as the host uid) must not
    break the kernel's later hash, copy, commit or delete of the tree."""
    def grant(path: str, bits: int) -> None:
        try:
            if not os.path.islink(path):
                os.chmod(path, os.lstat(path).st_mode | bits)
        except OSError:
            pass

    grant(str(root), 0o700)
    for dirpath, dirnames, filenames in os.walk(root):     # top-down: a dir is fixed before it is entered
        for name in dirnames:
            grant(os.path.join(dirpath, name), 0o700)
        for name in filenames:
            grant(os.path.join(dirpath, name), 0o600)


def diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {"added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k])}


# Host files the sandbox reads: DNS, TLS certificates, and what the host tools under /usr need.
ETC_FILES = ("/etc/resolv.conf", "/etc/hosts", "/etc/ssl", "/etc/ca-certificates", "/etc/alternatives",
             "/etc/ld.so.cache", "/etc/nsswitch.conf")
GPU_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")


def bwrap_args(mounts: Mounts, command, env, env_dir: Path, shm_gb: float,
                gpu_minors=()) -> tuple[list[str], dict[str, str]]:
    """The bwrap argv and the sandbox's whole environment (passed to bwrap as its own, so the token
    never appears on a command line)."""
    environment = {"PATH": f"{env_dir}/bin:/usr/bin:/bin", "HOME": "/workspace/.home",
                   "PYTHONPATH": "/ar_contract", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                   "AR_SOCKET_DIR": "/run/ar", "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context",
                   "AR_WORKSPACE": "/workspace"}
    bwrap = shutil.which("bwrap")       # the kernel's PATH: Popen would search the sandbox's
    if bwrap is None:
        raise FileNotFoundError("bwrap is not on PATH: install bubblewrap")
    argv = [bwrap, "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup",
            "--disable-userns", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr"]
    for d in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(d):
            argv += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            argv += ["--ro-bind", d, d]
    for f in ETC_FILES:
        if os.path.exists(f):
            argv += ["--ro-bind", os.path.realpath(f), f]
    argv += ["--ro-bind", str(env_dir), str(env_dir), "--proc", "/proc", "--dev", "/dev",
             "--size", str(4 * 2**30), "--tmpfs", "/tmp", "--size", str(int(shm_gb * 2**30)), "--tmpfs", "/dev/shm"]
    if gpu_minors:
        for dev in (*GPU_DEVICES, *(f"/dev/nvidia{m}" for m in gpu_minors)):
            argv += ["--dev-bind", dev, dev]
        argv += ["--ro-bind", "/sys", "/sys"]
    argv += ["--bind" if not mounts.agent_readonly else "--ro-bind", str(mounts.agent), "/agent",
             "--bind", str(mounts.workspace), "/workspace",
             "--bind", str(mounts.staging), "/workspace/staging",     # its own mount: no hard link across
             "--ro-bind", str(mounts.context), "/context",
             "--ro-bind", str(mounts.contract), "/ar_contract",
             "--ro-bind", str(mounts.sockets), "/run/ar"]              # connect works; deleting a socket does not
    if mounts.code is not None:
        argv += ["--ro-bind", str(mounts.code), "/code"]
        environment["AR_CODE_DIR"] = "/code"
    for node, path in mounts.nodes.items():
        argv += ["--ro-bind", str(path), f"/nodes/{node}"]
        if (Path(path) / "eval").is_dir():
            argv += ["--size", "4096", "--tmpfs", f"/nodes/{node}/eval", "--remount-ro", f"/nodes/{node}/eval"]
    argv += ["--remount-ro", "/", "--chdir", "/workspace", "--", *command]     # the root folder bwrap made, too
    return argv, {**environment, **env}


def _redacted(settings: dict) -> dict:
    """Copy `settings` with the value of each secret env key replaced by a placeholder. Applied to the
    settings the runner records, independent of the Recorder's own redaction list, so a sandbox
    token never lands in telemetry even if this Recorder instance was never told about it."""
    env = {k: "<redacted>" if k in _SECRET_ENV_KEYS else v for k, v in settings["environment"].items()}
    return {**settings, "environment": env}


def _read_output(path: Path, stream: str, recorder, name: str, base: dict) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        recorder.event("sandbox.error", container=name, payload={"stderr": f"could not read the sandbox's {stream}: {exc}"},
                       **base)
        return f"could not read the sandbox's {stream}: {exc}\n"


def run_container(*, name: str, mounts: Mounts, command: list[str], env: dict, env_dir: Path, shm_gb: float,
                  recorder, node: str, phase: str, attempt: int, gpus=(), log_name: str = "sandbox") -> RunResult:
    """stdout and stderr go to <log_name>.stdout / .stderr next to the workspace folder."""
    (Path(mounts.workspace) / ".home").mkdir(parents=True, exist_ok=True)
    base = dict(node=node, phase=phase, attempt=attempt, component="sandbox")
    started = time.monotonic()
    out_path, err_path = (Path(mounts.workspace).parent / f"{log_name}.{s}" for s in ("stdout", "stderr"))
    proc = None
    try:
        try:
            argv, environment = bwrap_args(mounts, command, env, env_dir, shm_gb, device_minors(list(gpus)))
            recorder.event("sandbox.start", container=name,
                           payload={"settings": _redacted({"argv": argv, "environment": environment})}, **base)
            with open(out_path, "w") as out, open(err_path, "w") as err:
                proc = subprocess.Popen(argv, env=environment, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                        start_new_session=True)     # a Ctrl-C to the kernel is not the agent's
        except (OSError, ValueError, subprocess.SubprocessError) as exc:   # e.g. no bwrap, a card without a minor
            recorder.event("sandbox.error", payload={"stderr": f"{type(exc).__name__}: {exc}"}, container=name, **base)
            return RunResult(None, "", f"{type(exc).__name__}: {exc}", time.monotonic() - started, name)
        exit_code = proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()                   # --die-with-parent ends everything inside
            proc.wait()
        for root in (mounts.agent, mounts.workspace, mounts.staging):   # everything the agent can write
            restore_owner_access(root)
    stdout = _read_output(out_path, "stdout", recorder, name, base)
    stderr = _read_output(err_path, "stderr", recorder, name, base)
    result = RunResult(exit_code, stdout, stderr, time.monotonic() - started, name)
    recorder.event("sandbox.end", container=name, exit_code=exit_code, duration_s=result.duration_s,
                   payload={"stdout": stdout, "stderr": stderr}, **base)
    return result
