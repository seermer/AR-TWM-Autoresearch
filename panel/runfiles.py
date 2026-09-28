"""Read-only access to one run folder: safe paths, archive.db, agents.git, text files."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

HEAD_TAIL = 256 * 1024


def loads(value: Any, default: Any = None) -> Any:
    """JSON text from an archive column, or `default` when it is missing or not JSON."""
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except ValueError:
        return default


def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


class RunFiles:
    def __init__(self, run_dir: Path | str) -> None:
        self.root = Path(run_dir).resolve()
        if not (self.root / "config" / "run.json").is_file():
            raise FileNotFoundError(f"{self.root} is not a run folder (no config/run.json)")

    def path(self, rel: str | Path) -> Path:
        """A path inside the run folder; anything that resolves outside it is refused."""
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"{rel} is outside the run folder")
        return target

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        """Read-only SQL. With no -wal file (no writer), `immutable=1` keeps SQLite from
        creating -wal/-shm side files; with a live writer, plain `mode=ro`."""
        db = self.root / "archive.db"
        if not db.is_file():
            return []
        live = (self.root / "archive.db-wal").exists()
        uri = f"{db.as_uri()}?mode=ro" + ("" if live else "&immutable=1")
        for tries in range(5):
            try:
                conn = sqlite3.connect(uri, uri=True, timeout=5.0)
                try:
                    conn.row_factory = sqlite3.Row
                    return [dict(r) for r in conn.execute(sql, args)]
                finally:
                    conn.close()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or tries == 4:
                    raise
                time.sleep(0.2)
        return []

    def git(self, *args: str) -> str | None:
        """stdout of a read-only git command on agents.git, or None when it fails."""
        repo = self.root / "agents.git"
        if not repo.is_dir():
            return None
        r = subprocess.run(["git", "--git-dir", str(repo), *args], capture_output=True,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None

    def read_text(self, rel: str | Path, limit: int = 2_000_000) -> str | None:
        """A text file; above `limit` bytes only its first and last 256 KB."""
        p = self.path(rel)
        if not p.is_file():
            return None
        size = p.stat().st_size
        if size <= limit:
            return p.read_bytes().decode("utf-8", errors="replace")
        with p.open("rb") as handle:
            head = handle.read(HEAD_TAIL)
            handle.seek(size - HEAD_TAIL)
            tail = handle.read()
        return (head.decode("utf-8", errors="replace")
                + f"\n\n[... {size - 2 * HEAD_TAIL} bytes not shown; the file is {size} bytes ...]\n\n"
                + tail.decode("utf-8", errors="replace"))

    def read_json(self, rel: str | Path) -> Any | None:
        text = self.read_text(rel, limit=50_000_000)
        return loads(text) if text is not None else None

    def host_path(self, node: str, phase: str, attempt: int, container_path: str) -> str | None:
        """The run-relative host path of a path the agent saw in its container."""
        attempt_dir = f"nodes/{node}/attempts/{phase}-{attempt}"
        mounts = (("/workspace/staging/", f"staging/{node}/{phase}-{attempt}/"),
                  ("/workspace/", f"{attempt_dir}/workspace/"), ("/agent/", f"{attempt_dir}/agent/"))
        for prefix, host in mounts:
            if container_path.startswith(prefix):
                rel = host + container_path[len(prefix):]
                try:
                    self.path(rel)
                except ValueError:
                    return None
                return rel
        return None

    def loop_pid(self) -> int | None:
        """The loop's pid if it is alive: the recorded start time must match /proc (the
        kernel's Control.alive_pid rule), so a stale pid file never reads as running."""
        try:
            pid_s, started = (self.root / "control" / "loop.pid").read_text().split()
            pid = int(pid_s)
            now = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, ValueError, IndexError):
            return None
        return pid if now == started else None
