from __future__ import annotations
import hashlib, os, sqlite3, time
from pathlib import Path

KINDS = {"video": ".mp4", "caption": ".json", "pose": ".npz"}

class BlobError(ValueError):
    """The blob cannot be stored."""

class BlobStore:
    def __init__(self, run_dir: Path, conn: sqlite3.Connection) -> None:
        self.root = Path(run_dir) / "store" / "blobs"
        self.conn = conn
        for kind in KINDS:
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def path(self, digest: str, kind: str) -> Path:
        if kind not in KINDS:
            raise BlobError(f"unknown kind {kind!r}; allowed: {sorted(KINDS)}")
        return self.root / kind / f"{digest}{KINDS[kind]}"

    def exists(self, digest: str) -> bool:
        return self.conn.execute("SELECT 1 FROM blobs WHERE digest=?", (digest,)).fetchone() is not None

    def size(self, digest: str) -> int:
        row = self.conn.execute("SELECT bytes FROM blobs WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        return int(row["bytes"])

    def put(self, path: Path, kind: str) -> str:
        # Validate kind argument explicitly at the top
        if kind not in KINDS:
            raise BlobError(f"unknown kind {kind!r}; allowed: {sorted(KINDS)}")

        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()

        # Derive the target path directly
        target = self.root / kind / f"{hexdigest}{KINDS[kind]}"
        nbytes = path.stat().st_size

        if target.exists():
            path.unlink()
        else:
            os.replace(path, target)
            target.chmod(0o444)

        self.conn.execute(
            "INSERT OR IGNORE INTO blobs (digest, kind, bytes, created_at) VALUES (?,?,?,?)",
            (hexdigest, kind, nbytes, time.time()),
        )
        return hexdigest
