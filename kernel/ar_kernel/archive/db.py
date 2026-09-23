from __future__ import annotations
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES nodes(node_id),
  depth INTEGER NOT NULL,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,
  agent_commit TEXT, data_commit TEXT, recipe_hash TEXT,
  resolved_config_path TEXT, checkpoint_path TEXT,
  lora_rank INTEGER, lora_alpha INTEGER,
  score REAL, metric_set TEXT, metrics TEXT, subtree_value REAL,
  phase_timings TEXT, rationale_path TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL REFERENCES nodes(node_id),
  phase TEXT NOT NULL, idx INTEGER NOT NULL,
  outcome TEXT NOT NULL, detail TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_node_phase ON attempts(node_id, phase, idx);
CREATE TABLE IF NOT EXISTS blobs (
  digest TEXT PRIMARY KEY, kind TEXT NOT NULL, bytes INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
  clip_id TEXT PRIMARY KEY,
  video_digest TEXT NOT NULL, caption_digest TEXT NOT NULL, pose_digest TEXT,
  camera_motion TEXT NOT NULL, metadata TEXT NOT NULL, formats TEXT NOT NULL,
  warnings TEXT NOT NULL, provenance TEXT NOT NULL, license TEXT,
  derived_from TEXT NOT NULL, ingested_by TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS data_commits (
  commit_id TEXT PRIMARY KEY, parent_commit TEXT, node_id TEXT, attempt INTEGER,
  manifest TEXT NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL
);
"""

def open_db(run_dir: Path) -> sqlite3.Connection:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Tool calls and the loop write from different threads through separate
    # connections; wait for a lock instead of failing with "database is locked".
    conn = sqlite3.connect(run_dir / "archive.db", isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
