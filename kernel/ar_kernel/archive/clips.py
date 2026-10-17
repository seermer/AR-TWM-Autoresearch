from __future__ import annotations
import json, re, sqlite3, time

# A clip's `ingested_by` is the id of the node that ingested it, or that id marked hidden from agents:
# <node>fail<k> (attempt k failed, or the node crashed in it), <node>quarantine, <node>interrupted.
_HIDDEN = re.compile(r".+(fail[1-9][0-9]*|quarantine|interrupted)")


def hidden(ingested_by: str) -> bool:
    return _HIDDEN.fullmatch(ingested_by) is not None


def hide_clips(conn, recorder, node: str, mark: str) -> None:
    n = ClipStore(conn).hide(node, mark)
    if n:
        recorder.event("clips.hidden", node=node, payload={"node": node, "ingested_by": node + mark, "clips": n})


class ClipStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, record: dict) -> str | None:
        """Store a clip. A clip already stored under a hidden mark moves to this node; returns that mark."""
        row = self.conn.execute("SELECT ingested_by FROM clips WHERE clip_id=?", (record["clip_id"],)).fetchone()
        if row is not None and hidden(row[0]):
            self.conn.execute("UPDATE clips SET ingested_by=? WHERE clip_id=?", (record["ingested_by"], record["clip_id"]))
            return row[0]
        self.conn.execute(
            """INSERT OR IGNORE INTO clips (clip_id, video_digest, caption_digest, annotation_digest,
               pose_digest, metadata, warnings, derived_from, ingested_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (record["clip_id"], record["video_digest"], record["caption_digest"],
             record["annotation_digest"], record["pose_digest"], json.dumps(record["metadata"]),
             json.dumps(record["warnings"]), json.dumps(record["derived_from"]),
             record["ingested_by"], time.time()),
        )
        return None

    def hide(self, node: str, mark: str) -> int:
        """Mark the clips `node` ingested `<node><mark>`; quarantine also takes its failed attempts' clips."""
        failed = " OR ingested_by GLOB ?" if mark == "quarantine" else ""
        return self.conn.execute(f"UPDATE clips SET ingested_by=? WHERE ingested_by=?{failed}",
                                 (node + mark, node, *([node + "fail[1-9]*"] if failed else []))).rowcount

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
