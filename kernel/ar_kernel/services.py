"""Start/stop the gateway and tool server on a run's Unix socket directory."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import uvicorn

from . import fs, telemetry


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
        self._span: telemetry.Span | None = None

    def start(self, gateway_app, tools_app, ready_timeout_s: float = 20.0) -> None:
        """Under a `services` span that `stop` ends: the servers' own log lines land in it."""
        self._span = telemetry.start("services", {"socket_dir": str(self.socket_dir)})
        with telemetry.use(self._span):
            fs.mkdir(self.socket_dir)
            fs.chmod(self.socket_dir, 0o700)
            self._fd = os.open(self.socket_dir, os.O_DIRECTORY)
            for name, app in (("gateway.sock", gateway_app), ("tools.sock", tools_app)):
                fs.remove(self.socket_dir / name)
                server = uvicorn.Server(uvicorn.Config(app, uds=f"/proc/self/fd/{self._fd}/{name}", log_level="warning",
                                                        log_config=None, timeout_keep_alive=900))   # to the root logger
                thread = telemetry.thread(server.run, f"ar-{name}")
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
                fs.chmod(self.socket_dir / name, 0o600)

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
            with telemetry.use(self._span):
                for name in ("gateway.sock", "tools.sock"):
                    fs.remove(self.socket_dir / name)
            os.close(self._fd)
            self._fd = None
        if self._span is not None:
            self._span.end("error" if stuck else "ok", {"stuck": stuck})
            self._span = None
        if stuck:
            raise RuntimeError(f"service thread(s) did not stop within the join timeout: "
                               f"{', '.join(stuck)}")
