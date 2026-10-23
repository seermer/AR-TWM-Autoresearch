"""Who is calling, and which host paths their container paths may name.

Every container gets its own token. The gateway and the tool server resolve it
to a Caller, which attributes telemetry (node, phase, attempt) and bounds every
path argument to that container's workspace. Anything else is refused.
"""
from __future__ import annotations

import os
import secrets
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

WORKSPACE = PurePosixPath("/workspace")
STAGING = WORKSPACE / "staging"


class PathError(ValueError):
    """A path argument does not name something inside the caller's workspace."""


class Fault:
    """Set once by a kernel thread when the caller's phase must stop; the loop then raises KernelFault."""

    def __init__(self, recorder=None) -> None:
        self.recorder, self.reason, self.event, self._lock = recorder, None, threading.Event(), threading.Lock()


@dataclass(frozen=True)
class Caller:
    token: str
    node: str
    phase: str
    attempt: int
    workspace_host: Path
    staging_host: Path
    mock_script: str | None = None
    scrub_names: tuple[str, ...] = ()      # replaced in everything a kernel tool gives this caller
    host_roots: tuple = ()                 # isolation.host_roots: replaced likewise
    agent_host: Path | None = None         # None in improve_recipe
    base_commit: str | None = None         # that tree's parent commit
    fault: Fault = field(default_factory=Fault, compare=False, repr=False)
    sandbox: dict = field(default_factory=dict, compare=False, repr=False)    # {"pid": bwrap's pid} once started

    def fail(self, reason: str) -> None:
        """Stop this caller's phase: its sandbox is killed and the loop raises KernelFault(reason)."""
        with self.fault._lock:
            if self.fault.event.is_set():
                return
            self.fault.reason = reason
            self.fault.event.set()
        if self.fault.recorder is not None:
            self.fault.recorder.event("phase.fault", node=self.node, phase=self.phase, attempt=self.attempt,
                                      component="kernel", payload={"reason": reason})


class TokenRegistry:
    def __init__(self, recorder, scrub_names=(), host_roots=()) -> None:
        self._recorder, self._scrub_names, self._host_roots = recorder, tuple(scrub_names), tuple(host_roots)
        self._by_token: dict[str, Caller] = {}
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._busy: dict[str, int] = {}         # token -> tool calls in flight
        self._on_revoke: list[Callable[[str], None]] = []

    def issue(self, *, node: str, phase: str, attempt: int, workspace_host: Path,
              staging_host: Path, mock_script: str | None = None, agent_host: Path | None = None,
              base_commit: str | None = None) -> Caller:
        token = "ar-" + secrets.token_urlsafe(32)
        self._recorder.add_redaction(token)
        caller = Caller(token, node, phase, attempt, Path(workspace_host), Path(staging_host),
                        mock_script, self._scrub_names, self._host_roots,
                        Path(agent_host) if agent_host is not None else None, base_commit, Fault(self._recorder))
        with self._lock:
            self._by_token[token] = caller
        return caller

    def lookup(self, token: str | None) -> Caller | None:
        with self._lock:
            return self._by_token.get(token or "")

    def begin(self, token: str | None) -> Caller | None:
        """lookup() for a tool call, which counts as in flight until end()."""
        with self._lock:
            caller = self._by_token.get(token or "")
            if caller is not None:
                self._busy[caller.token] = self._busy.get(caller.token, 0) + 1
            return caller

    def end(self, token: str) -> None:
        with self._lock:
            self._busy[token] -= 1
            if not self._busy[token]:
                del self._busy[token]
                self._idle.notify_all()

    def wait_idle(self, token: str) -> None:
        """After revoke(): until the token's tool calls in flight have returned."""
        with self._lock:
            while self._busy.get(token):
                self._idle.wait()

    def revoke(self, token: str) -> None:
        with self._lock:
            revoked = self._by_token.pop(token, None) is not None
        if revoked:
            for listener in list(self._on_revoke):
                listener(token)

    def on_revoke(self, fn: Callable[[str], None]) -> None:
        """Register a callback invoked with the token whenever revoke() actually revokes it.

        Listeners run outside any registry lock.
        """
        self._on_revoke.append(fn)


def bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    return value.strip() or None if scheme.lower() == "bearer" else None


def to_host(caller: Caller, container_path: str) -> Path:
    """Map a container path to the host, refusing anything outside the workspace.

    Resolution follows symlinks, so a link the agent planted inside /workspace
    cannot point the kernel at a host file.
    """
    if not container_path or not container_path.startswith("/"):
        raise PathError(f"{container_path!r} is not an absolute container path")
    path = PurePosixPath(container_path)
    if path == STAGING or STAGING in path.parents:
        base, root = caller.staging_host, STAGING
    elif path == WORKSPACE or WORKSPACE in path.parents:
        base, root = caller.workspace_host, WORKSPACE
    else:
        raise PathError(f"{container_path} is outside /workspace: kernel tools read only files under /workspace, "
                        "so copy the file there first")
    host = (base / path.relative_to(root)).resolve()
    if not host.is_relative_to(base.resolve()):
        raise PathError(f"{container_path} escapes its mount: kernel tools do not follow a symbolic link (or '..') "
                        "out of /workspace or /workspace/staging, so copy the file instead")
    return host


def open_file(caller: Caller, host: Path) -> int:
    """A read-only fd on `host` (a path `to_host` gave), without trusting the path between check and use: the
    agent keeps running, so any directory on it can be swapped for a link meanwhile. The file is opened with
    O_NOFOLLOW (and O_NONBLOCK, against a swapped-in FIFO), and the open file must be a regular file still
    inside the caller's mounts."""
    fd = os.open(host, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        to_container(caller, Path(os.readlink(f"/proc/self/fd/{fd}")))   # PathError if it now lies outside the mounts
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PathError(f"{to_container(caller, host)} is not a file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def to_container(caller: Caller, host_path: Path) -> str:
    host = Path(host_path).resolve()
    for base, root in ((caller.staging_host, STAGING), (caller.workspace_host, WORKSPACE)):
        if host.is_relative_to(base.resolve()):
            return str(root / host.relative_to(base.resolve()))
    raise PathError(f"{host} is not inside this caller's workspace")
