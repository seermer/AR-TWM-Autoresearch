"""Start/stop the gateway and tool server on a run's Unix socket directory."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import uvicorn


def socket_dir_for(run_dir: Path) -> Path:
    return Path(run_dir) / "sockets"


class RunServices:
    """Sockets are bound through /proc/self/fd/<folder fd>: AF_UNIX paths are capped at 107 bytes
    and the real folder path is longer."""

    def __init__(self, socket_dir: Path) -> None:
        self.socket_dir = Path(socket_dir)
        self._fd: int | None = None
        self._servers: list[uvicorn.Server] = []
        self._threads: list[threading.Thread] = []
        self._names: list[str] = []

    def start(self, gateway_app, tools_app, ready_timeout_s: float = 20.0) -> None:
        self.socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.socket_dir, 0o700)
        self._fd = os.open(self.socket_dir, os.O_DIRECTORY)
        for name, app in (("gateway.sock", gateway_app), ("tools.sock", tools_app)):
            path = Path(f"/proc/self/fd/{self._fd}") / name
            if path.exists():
                path.unlink()
            server = uvicorn.Server(uvicorn.Config(app, uds=str(path), log_level="warning",
                                                    timeout_keep_alive=900))
            thread = threading.Thread(target=server.run, name=f"ar-{name}", daemon=True)
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)
            self._names.append(name)
        deadline = time.monotonic() + ready_timeout_s
        while not all(s.started for s in self._servers):
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"services did not start within {ready_timeout_s}s")
            time.sleep(0.05)
        for name in ("gateway.sock", "tools.sock"):
            os.chmod(name, 0o600, dir_fd=self._fd)

    def stop(self) -> None:
        for server in self._servers:
            server.should_exit = True
        for thread in self._threads:
            thread.join(timeout=10)
        stuck = [name for name, thread in zip(self._names, self._threads) if thread.is_alive()]
        self._servers.clear()
        self._threads.clear()
        self._names.clear()
        # Clean up the sockets before raising, so a stuck thread never leaves one behind.
        if self._fd is not None:
            for name in ("gateway.sock", "tools.sock"):
                try:
                    os.unlink(name, dir_fd=self._fd)
                except FileNotFoundError:
                    pass
            os.close(self._fd)
            self._fd = None
        if stuck:
            raise RuntimeError(f"service thread(s) did not stop within the join timeout: "
                               f"{', '.join(stuck)}")
