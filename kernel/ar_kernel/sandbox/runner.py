"""Run one agent call in one container."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import docker
import requests

# Env var names whose value must never reach telemetry, no matter which Recorder
# is used (defense in depth: a Recorder that was never told about this run's
# token via add_redaction must still not leak it through the recorded container settings).
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
    killed kernel left running. Files are never touched."""
    prefix = container_prefix(run_id, node)
    try:
        client = docker.from_env()
        listed = client.api.containers(all=True, filters={"name": prefix})
    except (docker.errors.DockerException, requests.RequestException):     # no docker daemon: nothing runs
        return []
    names = [n for c in listed for n in (c["Names"][0].lstrip("/"),) if n.startswith(prefix)]
    for name in names:
        _remove(client, name)
    return names


def _remove(client, name: str) -> None:
    try:
        client.api.remove_container(name, force=True)
    except docker.errors.APIError:                     # already gone
        pass


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


def _run_kwargs(image, name, mounts: Mounts, command, env, cpus, memory_gb, network="bridge",
                gpus=()) -> dict:
    """The `containers.run` settings of one agent container."""
    base_env = {"HOME": "/workspace/.home", "PYTHONPATH": "/ar_contract", "AR_SOCKET_DIR": "/run/ar",
                "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context", "AR_WORKSPACE": "/workspace"}
    volumes = [f"{mounts.agent}:/agent:{'ro' if mounts.agent_readonly else 'rw'}",
               f"{mounts.workspace}:/workspace:rw",
               f"{mounts.staging}:/workspace/staging:rw",
               f"{mounts.context}:/context:ro",
               f"{mounts.contract}:/ar_contract:ro",
               f"{mounts.sockets}:/run/ar:ro"]          # connect works; deleting a socket does not
    tmpfs = {"/tmp": "rw,size=4g"}
    base_env["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus) or "void"
    if mounts.code is not None:
        volumes.append(f"{mounts.code}:/code:ro")
        base_env["AR_CODE_DIR"] = "/code"
    for node, path in mounts.nodes.items():
        volumes.append(f"{path}:/nodes/{node}:ro")
        if (Path(path) / "eval").is_dir():
            tmpfs[f"/nodes/{node}/eval"] = "ro,size=4k"
    memory = int(memory_gb * 2**30)
    return dict(image=image, command=list(command), name=name, detach=True, network_mode=network,
                user=f"{os.getuid()}:{os.getgid()}", read_only=True, tmpfs=tmpfs, cap_drop=["ALL"],
                security_opt=["no-new-privileges"], pids_limit=4096, nano_cpus=int(cpus * 1e9),
                mem_limit=memory, shm_size=memory, volumes=volumes, environment={**base_env, **env},
                **({"runtime": "nvidia"} if gpus else {}))     # the NVIDIA runtime exposes exactly these GPUs


def _redacted(kwargs: dict) -> dict:
    """Copy `kwargs` with the value of each secret env key replaced by a placeholder. Applied to the
    settings the runner records, independent of the Recorder's own redaction list, so a container
    token never lands in telemetry even if this Recorder instance was never told about it."""
    env = {k: "<redacted>" if k in _SECRET_ENV_KEYS else v for k, v in kwargs["environment"].items()}
    return {**kwargs, "environment": env}


def _cpu_percent(s: dict) -> float:
    """CPU use as `docker stats` computes it, from one stats sample."""
    cpu, pre = s["cpu_stats"], s["precpu_stats"]
    used = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
    system = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    cores = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or ())
    return used / system * cores * 100.0 if used > 0 and system > 0 else 0.0


def run_container(*, image: str, name: str, mounts: Mounts, command: list[str], env: dict,
                  cpus: float, memory_gb: float, timeout_s: float, recorder, node: str, phase: str,
                  attempt: int, stats_every_s: float = 30.0, liveness=None,
                  poll_s: float = 5.0, network: str = "bridge", gpus=()) -> RunResult:
    (Path(mounts.workspace) / ".home").mkdir(parents=True, exist_ok=True)
    kwargs = _run_kwargs(image, name, mounts, command, env, cpus, memory_gb, network, gpus)
    base = dict(node=node, phase=phase, attempt=attempt, component="sandbox")
    recorder.event("sandbox.start", container=name, payload={"kwargs": _redacted(kwargs)}, **base)
    started = time.monotonic()
    stats: list[dict] = []
    stop = threading.Event()
    client = docker.from_env()

    def sample(container) -> None:
        while not stop.wait(stats_every_s):
            try:
                s = container.stats(stream=False)
            except (docker.errors.DockerException, requests.RequestException):
                continue
            stats.append({"t": time.monotonic() - started, "cpu_percent": _cpu_percent(s),
                          "memory_bytes": s["memory_stats"].get("usage"), "pids": s["pids_stats"].get("current")})

    exit_code, timed_out, stdout, stderr, reason = None, False, "", "", None
    try:
        # A failed launch can come *after* the container was created (e.g. an OCI runtime
        # error or a bad bind source): a `Created` container named `name` is left behind
        # unless the removal below also covers this path.
        try:
            container = client.containers.run(**kwargs)
        except (docker.errors.DockerException, requests.RequestException) as exc:
            recorder.event("sandbox.error", payload={"stderr": str(exc)}, container=name, **base)
            return RunResult(None, False, "", str(exc), time.monotonic() - started, [], name)
        threading.Thread(target=sample, args=(container,), daemon=True).start()
        if liveness is not None:
            # Container CPU is a liveness signal: count samples above 1%.
            liveness.add_signal(lambda: sum(1 for s in stats if s["cpu_percent"] > 1.0))
        hard = time.monotonic() + timeout_s
        while True:
            tick = time.monotonic()
            try:
                exit_code = container.wait(timeout=poll_s)["StatusCode"]
                break
            except docker.errors.NotFound:              # gone: nothing left to wait for
                break
            except (docker.errors.DockerException, requests.RequestException):
                # The poll tick (a read timeout), or the daemon briefly unreachable: wait again.
                time.sleep(max(0.0, poll_s - (time.monotonic() - tick)))
            if time.monotonic() >= hard:
                reason = f"hard cap of {timeout_s:.0f}s reached"
            elif liveness is not None:
                reason = liveness.expired()
            if reason:
                timed_out = True
                try:
                    container.kill()
                except docker.errors.APIError:          # it ended meanwhile
                    pass
                break
        try:
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        except docker.errors.APIError:
            pass
    finally:
        stop.set()
        _remove(client, name)
        for root in (mounts.agent, mounts.workspace, mounts.staging):   # everything the agent can write
            restore_owner_access(root)
    result = RunResult(exit_code, timed_out, stdout, stderr, time.monotonic() - started, stats, name)
    recorder.event("sandbox.end", container=name, exit_code=exit_code, timed_out=timed_out,
                   duration_s=result.duration_s,
                   payload={"stdout": stdout, "stderr": stderr, "stats": stats, "reason": reason}, **base)
    return result
