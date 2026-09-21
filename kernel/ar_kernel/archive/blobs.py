from __future__ import annotations
import hashlib, os, shutil, sqlite3, time
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
        """Move `path` into the store under its SHA-256 and return the digest.

        Consumes the source file (the store keeps the only copy). Writes are
        temp-file + fsync + rename, so a crash never leaves a truncated file under
        a final content-addressed name, and an existing target is only trusted if
        its size matches -- a torn earlier write is replaced, not kept forever.
        """
        if kind not in KINDS:
            raise BlobError(f"unknown kind {kind!r}; allowed: {sorted(KINDS)}")
        path = Path(path)
        # A source inside the store is either a stored blob (put would unlink it
        # as the "duplicate source", destroying the only copy while the DB still
        # lists it) or something planted there. Neither is a candidate.
        if path.resolve().is_relative_to(self.root.resolve()):
            raise BlobError(f"{path} is inside the blob store; only staged files can be stored")

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        target = self.root / kind / f"{hexdigest}{KINDS[kind]}"
        nbytes = path.stat().st_size

        if target.exists() and target.stat().st_size == nbytes:
            path.unlink()
        else:
            self._install(path, target)

        self.conn.execute(
            "INSERT OR IGNORE INTO blobs (digest, kind, bytes, created_at) VALUES (?,?,?,?)",
            (hexdigest, kind, nbytes, time.time()),
        )
        return hexdigest

    def _install(self, path: Path, target: Path) -> None:
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        # shutil.move renames on one filesystem and copies across (EXDEV); either
        # way the bytes land under a temp name first.
        shutil.move(str(path), str(tmp))
        try:
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            tmp.chmod(0o444)
            os.replace(tmp, target)          # same directory -> atomic
        except BaseException:
            # tmp now holds the candidate's only copy: give it back rather than
            # deleting it, then surface the failure.
            if tmp.exists():
                shutil.move(str(tmp), str(path))
            raise
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
