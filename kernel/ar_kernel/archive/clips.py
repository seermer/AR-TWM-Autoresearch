from __future__ import annotations
import json, sqlite3, time

class ClipStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, record: dict) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO clips (clip_id, video_digest, caption_digest, pose_digest,
               camera_motion, metadata, formats, warnings, provenance, license, derived_from,
               ingested_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record["clip_id"], record["video_digest"], record["caption_digest"],
             record.get("pose_digest"), record["camera_motion"],
             json.dumps(record["metadata"]), json.dumps(record["formats"]),
             json.dumps(record["warnings"]), json.dumps(record["provenance"]),
             record.get("license"), json.dumps(record.get("derived_from", [])),
             record["ingested_by"], time.time()),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        clip = dict(row)
        for key in ("metadata", "formats", "warnings", "provenance", "derived_from"):
            clip[key] = json.loads(clip[key])
        return clip

    def get(self, clip_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        return self._row(row)

    def all(self) -> list[dict]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM clips ORDER BY created_at")]

    def eligible(self, fmt: str, prompt_mode: str | None = None) -> list[dict]:
        key = fmt if prompt_mode is None else f"{fmt}:{prompt_mode}"
        return [c for c in self.all() if key in c["formats"]]
