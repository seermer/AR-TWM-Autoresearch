from __future__ import annotations
import json, re, sqlite3, time

from .db import write

# A clip's `ingested_by` is the id of the node that ingested it, or that id marked hidden from agents:
# <node>fail<k> (attempt k failed, or the node ran out of retries after it), <node>quarantine.
_HIDDEN = re.compile(r".+(fail[1-9][0-9]*|quarantine)")


def hidden(ingested_by: str) -> bool:
    return _HIDDEN.fullmatch(ingested_by) is not None


class ClipStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, record: dict) -> str | None:
        """Store a clip. A clip already stored under a hidden mark moves to this node; returns that mark,
        which is kept in `moved_from`."""
        row = self.conn.execute("SELECT ingested_by FROM clips WHERE clip_id=?", (record["clip_id"],)).fetchone()
        if row is not None and hidden(row[0]):
            write(self.conn, "UPDATE clips SET ingested_by=?, attempt=?, moved_from=? WHERE clip_id=?",
                  (record["ingested_by"], record["attempt"], row[0], record["clip_id"]))
            return row[0]
        write(
            self.conn, """INSERT OR IGNORE INTO clips (clip_id, video_digest, caption_digest, annotation_digest,
               pose_digest, metadata, warnings, derived_from, ingested_by, attempt, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (record["clip_id"], record["video_digest"], record["caption_digest"],
             record["annotation_digest"], record["pose_digest"], json.dumps(record["metadata"]),
             json.dumps(record["warnings"]), json.dumps(record["derived_from"]),
             record["ingested_by"], record["attempt"], time.time()),
        )
        return None

    def hide(self, node: str, mark: str, attempt: int | None = None) -> int:
        """Mark the clips `node` ingested (in `attempt` only, if given) `<node><mark>`; quarantine also takes
        its failed attempts' clips."""
        if attempt is not None:
            return write(self.conn, "UPDATE clips SET ingested_by=? WHERE ingested_by=? AND attempt=?",
                         (node + mark, node, attempt))
        failed = " OR ingested_by GLOB ?" if mark == "quarantine" else ""
        return write(self.conn, f"UPDATE clips SET ingested_by=? WHERE ingested_by=?{failed}",
                     (node + mark, node, *([node + "fail[1-9]*"] if failed else [])))

    def remove_ingested(self, node: str, attempt: int) -> int:
        """Undo the ingests of `node`'s `attempt`: a clip it took from a hidden mark gets that mark back, the rest
        are deleted."""
        moved = write(self.conn, "UPDATE clips SET ingested_by=moved_from, moved_from=NULL "
                      "WHERE ingested_by=? AND attempt=? AND moved_from IS NOT NULL", (node, attempt))
        return moved + write(self.conn, "DELETE FROM clips WHERE ingested_by=? AND attempt=?", (node, attempt))

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        clip = dict(row)
        for key in ("metadata", "warnings", "derived_from"):
            clip[key] = json.loads(clip[key])
        return clip

    def get(self, clip_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        return self._row(row)

    def all(self) -> list[dict]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM clips ORDER BY created_at")]
