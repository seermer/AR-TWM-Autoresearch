"""Start/stop the gateway and tool server on a run's Unix socket directory."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path

import uvicorn


def socket_dir_for(run_dir: Path) -> Path:
    """A short, private, per-run directory. AF_UNIX paths are capped at 107 bytes
    and a socket under runs/<run_id>/ is ~125 (verified fact 12)."""
    digest = hashlib.sha1(str(Path(run_dir).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"ar-{digest}"


class RunServices:
    def __init__(self, socket_dir: Path) -> None:
        self.socket_dir = Path(socket_dir)
        self._servers: list[uvicorn.Server] = []
        self._threads: list[threading.Thread] = []

    def start(self, gateway_app, tools_app, ready_timeout_s: float = 20.0) -> None:
        self.socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.socket_dir, 0o700)
        for name, app in (("gateway.sock", gateway_app), ("tools.sock", tools_app)):
            path = self.socket_dir / name
            if path.exists():
                path.unlink()
            server = uvicorn.Server(uvicorn.Config(app, uds=str(path), log_level="warning",
                                                    timeout_keep_alive=900))
            thread = threading.Thread(target=server.run, name=f"ar-{name}", daemon=True)
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)
        deadline = time.monotonic() + ready_timeout_s
        while not all(s.started for s in self._servers):
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"services did not start within {ready_timeout_s}s")
            time.sleep(0.05)
        for name in ("gateway.sock", "tools.sock"):
            os.chmod(self.socket_dir / name, 0o600)

    def stop(self) -> None:
        for server in self._servers:
            server.should_exit = True
        for thread in self._threads:
            thread.join(timeout=10)
        self._servers.clear()
        self._threads.clear()
        for name in ("gateway.sock", "tools.sock"):
            (self.socket_dir / name).unlink(missing_ok=True)
