"""Run one agent call in one bwrap sandbox."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import fs, telemetry
from ..faults import KernelFault
from ..guards import device_minors
from ..subproc import project_env, run_sandbox

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
    # The /workspace of each earlier subphase of this improve_recipe phase, oldest first, mounted read-only at /earlier/<n>.
    subphases: list[Path] = field(default_factory=list)


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


def snapshot(root: Path, hash_files: bool) -> dict[str, tuple]:
    """path -> (size, fingerprint). Hash small code trees; use the mtime for workspaces
    that may hold gigabytes of video. Only regular files count (never a FIFO or
    a symlink); a file the agent made unreadable is fingerprinted "unreadable"."""
    out = {}
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            mark = st.st_mtime_ns
            if hash_files:
                try:
                    with p.open("rb") as f:
                        mark = hashlib.file_digest(f, "sha256").hexdigest()
                except OSError as exc:
                    telemetry.error(exc, 'fingerprinted "unreadable"')
                    mark = "unreadable"
            out[str(p.relative_to(root))] = (st.st_size, mark)
    return out


def restore_owner_access(root: Path) -> None:
    """Give the owner rw (and x on directories) on everything under `root` again, never following
    a link. A chmod-000 file or directory an agent leaves (sandboxes run as the host uid) must not
    break the kernel's later hash, copy, commit or delete of the tree."""
    def grant(path: str, bits: int) -> None:
        try:
            if not os.path.islink(path) and (mode := os.lstat(path).st_mode) | bits != mode:
                fs.chmod(path, stat.S_IMODE(mode | bits))
        except OSError as exc:
            telemetry.error(exc, "the path keeps its mode")

    grant(str(root), 0o700)
    for dirpath, dirnames, filenames in os.walk(root):     # top-down: a dir is fixed before it is entered
        for name in dirnames:
            grant(os.path.join(dirpath, name), 0o700)
        for name in filenames:
            grant(os.path.join(dirpath, name), 0o600)


def diff(before: dict[str, tuple], after: dict[str, tuple]) -> dict[str, list[str]]:
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
    bwrap = project_env("bwrap")
    if bwrap is None:
        raise FileNotFoundError("no .envs/bwrap: run scripts/setup_envs.sh bwrap")
    argv = [str(bwrap / "bin" / "bwrap"), "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup",
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
    for n, path in enumerate(mounts.subphases, 1):
        argv += ["--ro-bind", str(path), f"/earlier/{n}"]
    argv += ["--remount-ro", "/", "--chdir", "/workspace", "--", *command]     # the root folder bwrap made, too
    return argv, {**environment, **env}


def _read_output(path: Path, stream: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        telemetry.error(exc, f"the {stream} text says it could not be read")
        return f"could not read the sandbox's {stream}: {exc}\n"


def _manifest(path: Path) -> list[list]:
    """[relative path, size] of every file a /nodes mount shows: its eval/ is hidden."""
    out = []
    for d, dirs, files in os.walk(path):
        if d == str(path):
            dirs[:] = [x for x in dirs if x != "eval"]
        out += [[os.path.relpath(os.path.join(d, f), path), os.lstat(os.path.join(d, f)).st_size] for f in files]
    return sorted(out)


def _inputs(name: str, argv: list[str], environment: dict, mounts: Mounts, gpus) -> dict:
    if (w := telemetry.writer()) is None:
        return {}
    return {"name": name, "argv": argv, "env": w.redact_env(environment), "gpus": list(gpus),
            "nodes": {node: _manifest(path) for node, path in mounts.nodes.items()}}


def _writable(mounts: Mounts) -> dict[str, tuple[Path, bool]]:
    """Each folder the sandbox can write: sandbox path -> (host path, whether its files are hashed)."""
    return {"/workspace": (Path(mounts.workspace), False), "/workspace/staging": (Path(mounts.staging), False),
            **({} if mounts.agent_readonly else {"/agent": (Path(mounts.agent), True)})}


def files(mounts: Mounts) -> dict[str, dict]:
    """The snapshot of each folder the sandbox can write."""
    return {root: snapshot(path, hashed) for root, (path, hashed) in _writable(mounts).items()}


def record_changes(mounts: Mounts, before: dict[str, dict]) -> None:
    """Each file added, changed or removed since `before` (`files`), with its size."""
    with telemetry.span("sandbox.changes", {root: str(path) for root, (path, _) in _writable(mounts).items()}) as sp:
        after = files(mounts)
        sp.output = {root: {kind: [[p, (before if kind == "removed" else after)[root][p][0]] for p in paths]
                            for kind, paths in diff(before[root], after[root]).items()} for root in after}


def run_container(*, name: str, mounts: Mounts, command: list[str], env: dict, env_dir: Path, shm_gb: float,
                  gpus=(), log_name: str = "sandbox",
                  cancel: threading.Event | None = None, on_start=None, poll_s: float = 1.0) -> RunResult:
    """stdout and stderr go to <log_name>.stdout / .stderr next to the workspace folder. `cancel` kills the
    sandbox; `on_start` gets bwrap's pid. Raises KernelFault when the sandbox itself fails (bwrap cannot
    start or set it up)."""
    fs.mkdir(Path(mounts.workspace) / ".home")
    started = time.monotonic()
    out_path, err_path = (Path(mounts.workspace).parent / f"{log_name}.{s}" for s in ("stdout", "stderr"))
    try:
        try:
            argv, environment = bwrap_args(mounts, command, env, env_dir, shm_gb, device_minors(list(gpus)))
        except (OSError, ValueError) as exc:          # e.g. no bwrap, a card without a minor
            raise KernelFault(f"the sandbox could not start: {type(exc).__name__}: {exc}") from exc
        exit_code = run_sandbox(argv, environment, out_path=out_path, err_path=err_path,
                                inputs=_inputs(name, argv, environment, mounts, gpus), cancel=cancel,
                                poll_s=poll_s, on_start=on_start)
    finally:
        for root in (mounts.agent, mounts.workspace, mounts.staging):   # everything the agent can write
            restore_owner_access(root)
    stdout = _read_output(out_path, "stdout")
    stderr = _read_output(err_path, "stderr")
    result = RunResult(exit_code, stdout, stderr, time.monotonic() - started, name)
    if exit_code != 0 and stderr.startswith("bwrap: "):        # bwrap's own error: the command never ran
        raise KernelFault(f"the sandbox could not be set up: {stderr.splitlines()[0]}")
    return result
