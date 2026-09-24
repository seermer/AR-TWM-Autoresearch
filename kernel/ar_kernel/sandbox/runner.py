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


def snapshot(root: Path, hash_files: bool) -> dict[str, str]:
    """path -> fingerprint. Hash small code trees; use size+mtime for workspaces
    that may hold gigabytes of video."""
    out = {}
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            if hash_files:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            else:
                st = p.stat()
                out[rel] = f"{st.st_size}:{st.st_mtime_ns}"
    return out


def diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {"added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k])}


def _docker_args(image, name, mounts: Mounts, command, env, cpus, memory_gb) -> list[str]:
    base_env = {"HOME": "/workspace/.home", "PYTHONPATH": "/ar_contract", "AR_SOCKET_DIR": "/run/ar",
                "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context", "AR_WORKSPACE": "/workspace"}
    args = ["docker", "run", "-d", "--name", name,
            "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--read-only", "--tmpfs", "/tmp:rw,size=4g",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "4096",
            "--cpus", str(cpus), "--memory", f"{memory_gb}g",
            "-v", f"{mounts.agent}:/agent:{'ro' if mounts.agent_readonly else 'rw'}",
            "-v", f"{mounts.workspace}:/workspace:rw",
            "-v", f"{mounts.staging}:/workspace/staging:rw",
            "-v", f"{mounts.context}:/context:ro",
            "-v", f"{mounts.store}:/store:ro",
            "-v", f"{mounts.contract}:/ar_contract:ro",
            "-v", f"{mounts.sockets}:/run/ar:rw"]
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


def run_container(*, image: str, name: str, mounts: Mounts, command: list[str], env: dict,
                  cpus: float, memory_gb: float, timeout_s: float, recorder, node: str, phase: str,
                  attempt: int, stats_every_s: float = 30.0) -> RunResult:
    (Path(mounts.workspace) / ".home").mkdir(parents=True, exist_ok=True)
    args = _docker_args(image, name, mounts, command, env, cpus, memory_gb)
    recorded_args = _redact_argv(args)
    base = dict(node=node, phase=phase, attempt=attempt, component="sandbox")
    recorder.event("sandbox.start", container=name, payload={"args": recorded_args}, **base)
    started = time.monotonic()
    stats: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(stats_every_s):
            r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                stats.append({"t": time.monotonic() - started, **json.loads(r.stdout)})

    exit_code, timed_out, stdout, stderr = None, False, "", ""
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
        try:
            waited = subprocess.run(["docker", "wait", name], capture_output=True, text=True,
                                    timeout=timeout_s)
            exit_code = int(waited.stdout.strip() or -1)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        stdout, stderr = logs.stdout, logs.stderr
    finally:
        stop.set()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    result = RunResult(exit_code, timed_out, stdout, stderr, time.monotonic() - started, stats, name)
    recorder.event("sandbox.end", container=name, exit_code=exit_code, timed_out=timed_out,
                   duration_s=result.duration_s,
                   payload={"stdout": stdout, "stderr": stderr, "stats": stats}, **base)
    return result
