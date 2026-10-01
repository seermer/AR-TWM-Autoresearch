"""Run one agent call in one container (spec 9.5)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Env var names whose value must never reach telemetry, no matter which Recorder
# is used (defense in depth: a Recorder that was never told about this run's
# token via add_redaction must still not leak it through a recorded docker argv).
_SECRET_ENV_KEYS = ("AR_TOKEN",)


@dataclass
class Mounts:
    agent: Path
    workspace: Path
    staging: Path
    context: Path
    store: Path
    contract: Path
    sockets: Path
    agent_readonly: bool = False
    # node id -> node dir of each finished node, mounted read-only at /nodes/<id>. Its eval/ is hidden:
    # per-case outputs would let an agent fit the proxy cases.
    nodes: dict[str, Path] = field(default_factory=dict)


@dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float
    stats: list[dict] = field(default_factory=list)
    container: str = ""


def container_name(run_id: str, node: str, phase: str, attempt: int) -> str:
    raw = f"ar-{run_id}-{node}-{phase}-{attempt}-{secrets.token_hex(3)}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def container_prefix(run_id: str, node: str | None = None) -> str:
    raw = f"ar-{run_id}-" + (f"{node}-" if node else "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def kill_run_containers(run_id: str, node: str | None = None) -> list[str]:
    """Force-remove this run's (or node's) containers: on force stop, and on resume for what a
    killed kernel left running (spec 14.4). Files are never touched."""
    prefix = container_prefix(run_id, node)
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
                            capture_output=True, text=True).stdout
    names = [n for n in listed.split() if n.startswith(prefix)]
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return names


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
    a link. A chmod-000 file or directory an agent leaves (containers run as the host uid) must not
    break the kernel's later hash, copy, commit or delete of the tree."""
    def grant(path: str, bits: int) -> None:
        try:
            if not os.path.islink(path):
                os.chmod(path, os.lstat(path).st_mode | bits)
        except OSError:                  # e.g. a mount point docker created as root
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


def _docker_args(image, name, mounts: Mounts, command, env, cpus, memory_gb, network="bridge") -> list[str]:
    base_env = {"HOME": "/workspace/.home", "PYTHONPATH": "/ar_contract", "AR_SOCKET_DIR": "/run/ar",
                "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context", "AR_WORKSPACE": "/workspace"}
    args = ["docker", "run", "-d", "--name", name,
            "--network", network, "--user", f"{os.getuid()}:{os.getgid()}",
            "--read-only", "--tmpfs", "/tmp:rw,size=4g",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "4096",
            "--cpus", str(cpus), "--memory", f"{memory_gb}g",
            "-v", f"{mounts.agent}:/agent:{'ro' if mounts.agent_readonly else 'rw'}",
            "-v", f"{mounts.workspace}:/workspace:rw",
            "-v", f"{mounts.staging}:/workspace/staging:rw",
            "-v", f"{mounts.context}:/context:ro",
            "-v", f"{mounts.store}:/store:ro",
            "-v", f"{mounts.contract}:/ar_contract:ro",
            "-v", f"{mounts.sockets}:/run/ar:ro"]   # connect works; deleting a socket does not
    for node, path in mounts.nodes.items():
        args += ["-v", f"{path}:/nodes/{node}:ro"]
        if (Path(path) / "eval").is_dir():
            args += ["--tmpfs", f"/nodes/{node}/eval:ro,size=4k"]
    for key, value in {**base_env, **env}.items():
        args += ["-e", f"{key}={value}"]
    return args + [image, *command]


def _redact_argv(args: list[str]) -> list[str]:
    """Copy `args`, replacing any `KEY=value` entry for a secret env key with a
    placeholder. Applied to every docker argv the runner records, independent of
    the Recorder's own redaction list, so a container token never lands in
    telemetry even if this Recorder instance was never told about it."""
    redacted = []
    for item in args:
        for key in _SECRET_ENV_KEYS:
            if item.startswith(f"{key}="):
                item = f"{key}=<redacted>"
                break
        redacted.append(item)
    return redacted


def _cpu(sample: dict) -> float:
    try:
        return float(str(sample.get("CPUPerc", "0")).rstrip("%") or 0)
    except ValueError:
        return 0.0


def run_container(*, image: str, name: str, mounts: Mounts, command: list[str], env: dict,
                  cpus: float, memory_gb: float, timeout_s: float, recorder, node: str, phase: str,
                  attempt: int, stats_every_s: float = 30.0, liveness=None,
                  poll_s: float = 5.0, network: str = "bridge") -> RunResult:
    (Path(mounts.workspace) / ".home").mkdir(parents=True, exist_ok=True)
    args = _docker_args(image, name, mounts, command, env, cpus, memory_gb, network)
    recorded_args = _redact_argv(args)
    base = dict(node=node, phase=phase, attempt=attempt, component="sandbox")
    recorder.event("sandbox.start", container=name, payload={"args": recorded_args}, **base)
    started = time.monotonic()
    stats: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(stats_every_s):
            r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
                               capture_output=True, text=True, start_new_session=True)
            if r.returncode == 0 and r.stdout.strip():
                stats.append({"t": time.monotonic() - started, **json.loads(r.stdout)})

    exit_code, timed_out, stdout, stderr, reason, waiter = None, False, "", "", None, None

    def wait():
        # Its own session: a Ctrl-C reaches the terminal's whole foreground process group, and the
        # first one is graceful (spec 14.4) -- it must not end the wait and so the container.
        return subprocess.Popen(["docker", "wait", name], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        # `docker run -d` can fail *after* it has created the container (e.g. an
        # OCI runtime error or a bad bind source): a `Created` container named
        # `name` is left behind unless the removal below also covers this path.
        launched = subprocess.run(args, capture_output=True, text=True)
        if launched.returncode != 0:
            recorder.event("sandbox.error", payload={"stderr": launched.stderr}, container=name, **base)
            stderr = launched.stderr
            return RunResult(None, False, "", stderr, time.monotonic() - started, [], name)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        if liveness is not None:
            # Container CPU is a liveness signal (spec 14.5): count samples above 1%.
            liveness.add_signal(lambda: sum(1 for s in stats if _cpu(s) > 1.0))
        waiter = wait()
        hard = time.monotonic() + timeout_s
        while True:
            try:
                out, err = waiter.communicate(timeout=poll_s)
            except subprocess.TimeoutExpired:
                out = err = None
            if out is not None:
                if out.strip().lstrip("-").isdigit():
                    exit_code = int(out.strip())
                    break
                if "No such container" in err:          # gone: nothing left to wait for
                    break
            if time.monotonic() >= hard:
                reason = f"hard cap of {timeout_s:.0f}s reached"
            elif liveness is not None:
                reason = liveness.expired()
            if reason:
                timed_out = True
                subprocess.run(["docker", "kill", name], capture_output=True)
                waiter.kill()
                waiter.communicate()
                break
            if out is not None:                         # the waiter died without an exit code: wait again
                time.sleep(poll_s)
                waiter = wait()
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        stdout, stderr = logs.stdout, logs.stderr
    finally:
        stop.set()
        if waiter is not None and waiter.poll() is None:
            waiter.kill()
            waiter.wait()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        for root in (mounts.agent, mounts.workspace, mounts.staging):   # everything the agent can write
            restore_owner_access(root)
    result = RunResult(exit_code, timed_out, stdout, stderr, time.monotonic() - started, stats, name)
    recorder.event("sandbox.end", container=name, exit_code=exit_code, timed_out=timed_out,
                   duration_s=result.duration_s,
                   payload={"stdout": stdout, "stderr": stderr, "stats": stats, "reason": reason}, **base)
    return result
