"""Who is calling, and which host paths their container paths may name.

Every container gets its own token. The gateway and the tool server resolve it
to a Caller, which attributes telemetry (node, phase, attempt) and bounds every
path argument to that container's workspace. Anything else is refused.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

WORKSPACE = PurePosixPath("/workspace")
STAGING = WORKSPACE / "staging"


class PathError(ValueError):
    """A path argument does not name something inside the caller's workspace."""


@dataclass(frozen=True)
class Caller:
    token: str
    node: str
    phase: str
    attempt: int
    workspace_host: Path
    staging_host: Path
    mock_script: str | None = None


class TokenRegistry:
    def __init__(self, recorder) -> None:
        self._recorder = recorder
        self._by_token: dict[str, Caller] = {}
        self._lock = threading.Lock()
        self._on_revoke: list[Callable[[str], None]] = []

    def issue(self, *, node: str, phase: str, attempt: int, workspace_host: Path,
              staging_host: Path, mock_script: str | None = None) -> Caller:
        token = "ar-" + secrets.token_urlsafe(32)
        self._recorder.add_redaction(token)
        caller = Caller(token, node, phase, attempt, Path(workspace_host), Path(staging_host),
                        mock_script)
        with self._lock:
            self._by_token[token] = caller
        return caller

    def lookup(self, token: str | None) -> Caller | None:
        with self._lock:
            return self._by_token.get(token or "")

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
        raise PathError(f"{container_path} is outside /workspace")
    host = (base / path.relative_to(root)).resolve()
    if not host.is_relative_to(base.resolve()):
        raise PathError(f"{container_path} escapes its mount")
    return host


def to_container(caller: Caller, host_path: Path) -> str:
    host = Path(host_path).resolve()
    for base, root in ((caller.staging_host, STAGING), (caller.workspace_host, WORKSPACE)):
        if host.is_relative_to(base.resolve()):
            return str(root / host.relative_to(base.resolve()))
    raise PathError(f"{host} is not inside this caller's workspace")
