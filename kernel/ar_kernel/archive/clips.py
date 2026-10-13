from __future__ import annotations
import json, sqlite3, time

class ClipStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, record: dict) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO clips (clip_id, video_digest, caption_digest, annotation_digest,
               pose_digest, metadata, warnings, derived_from, ingested_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (record["clip_id"], record["video_digest"], record["caption_digest"],
             record["annotation_digest"], record["pose_digest"], json.dumps(record["metadata"]),
             json.dumps(record["warnings"]), json.dumps(record["derived_from"]),
             record["ingested_by"], time.time()),
        )

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
