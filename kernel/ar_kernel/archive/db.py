from __future__ import annotations
import re, shutil, sqlite3, tempfile
from contextlib import contextmanager
from pathlib import Path

from .. import telemetry

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
  phase_timings TEXT, rationale_path TEXT,
  recipe_path TEXT, attempt_counts TEXT, error TEXT
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
  video_digest TEXT NOT NULL, caption_digest TEXT NOT NULL, annotation_digest TEXT NOT NULL,
  pose_digest TEXT NOT NULL, metadata TEXT NOT NULL, warnings TEXT NOT NULL,
  derived_from TEXT NOT NULL, ingested_by TEXT NOT NULL, attempt INTEGER NOT NULL,
  created_at REAL NOT NULL, moved_from TEXT
);
CREATE TABLE IF NOT EXISTS data_commits (
  n INTEGER PRIMARY KEY AUTOINCREMENT, commit_id TEXT UNIQUE NOT NULL, node_id TEXT, attempt INTEGER,
  status TEXT NOT NULL CHECK (status IN ('draft', 'accepted', 'obsolete', 'failed')),
  manifest TEXT NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS selection_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL, chosen TEXT NOT NULL, seed INTEGER NOT NULL,
  candidates TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL, phase TEXT NOT NULL, attempt INTEGER NOT NULL,
  role TEXT, tool TEXT, conversation_id TEXT NOT NULL, turn INTEGER NOT NULL, transcript TEXT,
  status INTEGER NOT NULL, input_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL, cost_usd REAL, mock INTEGER NOT NULL, audit_hits TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_node ON llm_calls(node_id);
CREATE TABLE IF NOT EXISTS activity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL, phase TEXT, attempt INTEGER,
  kind TEXT NOT NULL CHECK (kind IN ('tool', 'gate', 'job', 'ingest', 'snapshot', 'accepted')),
  name TEXT, ok INTEGER, detail TEXT, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS activity_node ON activity(node_id);
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


@contextmanager
def transaction(conn: sqlite3.Connection):
    """One atomic write on an autocommit connection, as a `db.transaction` span."""
    with telemetry.span("db.transaction") as sp:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            sp.output = {"outcome": "rolled back"}
            raise
        conn.execute("COMMIT")
        sp.output = {"outcome": "committed"}


_TABLE = re.compile(r"^\s*(?:INSERT(?: OR IGNORE)? INTO|UPDATE|DELETE FROM)\s+(\w+)", re.I)


def _rows(conn: sqlite3.Connection, table: str, where: str, args) -> list[dict]:
    return [dict(r) for r in conn.execute(f"SELECT rowid AS rowid_, * FROM {table} WHERE {where}", tuple(args))]


def write(conn: sqlite3.Connection, sql: str, params=()) -> int:
    """One INSERT, UPDATE or DELETE as a `db.write` span: the rows it changes before and after (with their rowid),
    enough to apply it again. An UPDATE or DELETE ends with `WHERE ...` whose placeholders come last. Returns
    the rowcount."""
    if telemetry.writer() is None:
        return conn.execute(sql, params).rowcount
    table = _TABLE.match(sql).group(1)
    params = tuple(params)
    if sql.lstrip().upper().startswith("INSERT"):
        with telemetry.span("db.write", {"table": table, "sql": sql, "params": params, "before": []}) as sp:
            cur = conn.execute(sql, params)
            sp.output = {"rowcount": cur.rowcount,
                         "after": _rows(conn, table, "rowid=?", [cur.lastrowid]) if cur.rowcount else []}
            return cur.rowcount
    where = re.split(r"\sWHERE\s", sql, flags=re.I)[-1]
    args = params[len(params) - where.count("?"):]
    before = _rows(conn, table, where, args)
    with telemetry.span("db.write", {"table": table, "sql": sql, "params": params, "before": before}) as sp:
        n = conn.execute(sql, params).rowcount
        ids = [r["rowid_"] for r in before]
        sp.output = {"rowcount": n, "after": _rows(conn, table, f"rowid IN ({', '.join('?' * len(ids))})", ids)
                     if ids else []}
        return n


def open_db_readonly(run_dir: Path, writer_alive: bool) -> sqlite3.Connection:
    """The archive of a run, opened without writing anything into the run folder.

    While the loop is writing, a plain read-only open (SQLite shares the writer's -shm). With no
    writer and no WAL content, `immutable=1`, which creates no side files. With no writer but a
    non-empty -wal (a killed writer's committed rows), an in-memory copy made outside the run."""
    db = Path(run_dir) / "archive.db"
    wal = db.with_name("archive.db-wal")
    if writer_alive:
        conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=30.0)
    elif not wal.exists() or wal.stat().st_size == 0:
        conn = sqlite3.connect(f"{db.as_uri()}?mode=ro&immutable=1", uri=True)
    else:
        conn = sqlite3.connect(":memory:")
        with tempfile.TemporaryDirectory() as tmp:
            for path in (db, wal):
                shutil.copyfile(path, Path(tmp) / path.name)
            source = sqlite3.connect(Path(tmp) / db.name)
            source.backup(conn)
            source.close()
    conn.row_factory = sqlite3.Row
    return conn
