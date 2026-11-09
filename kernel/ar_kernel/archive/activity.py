"""What agents' phases did, one row per fact: each finished LLM call (`llm_calls`), and each tool call, gate,
GPU job, ingest candidate, snapshot and accepted snapshot (`activity`). Insert only."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..faults import KernelFault
from .db import open_db, write


def _insert(run_dir: Path, table: str, row: dict) -> None:
    """Through a connection of its own: the gateway, tool and job threads have none."""
    try:
        conn = open_db(run_dir)
        try:
            write(conn, f"INSERT INTO {table} ({', '.join(row)}, created_at) "
                        f"VALUES ({', '.join('?' * len(row))}, ?)", (*row.values(), time.time()))
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        raise KernelFault(f"the archive could not record {table}: {type(exc).__name__}: {exc}") from exc


def add_call(run_dir: Path, *, node: str, phase: str, attempt: int, role: str | None, tool: str | None,
             conversation_id: str, turn: int, transcript: str | None, status: int, tokens: tuple[int, int, int],
             cost_usd: float | None, mock: bool, audit_hits: list[str]) -> None:
    """`tokens`: (uncached input, cached input, output)."""
    fresh, cached, output = tokens
    _insert(run_dir, "llm_calls", {
        "node_id": node, "phase": phase, "attempt": attempt, "role": role, "tool": tool,
        "conversation_id": conversation_id, "turn": turn, "transcript": transcript, "status": status,
        "input_tokens": fresh, "cached_tokens": cached, "output_tokens": output, "cost_usd": cost_usd,
        "mock": mock, "audit_hits": json.dumps(audit_hits)})


def add_activity(run_dir: Path, *, node: str, phase: str | None, attempt: int | None, kind: str,
                 name: str | None = None, ok: bool | None = None, detail: str | None = None) -> None:
    _insert(run_dir, "activity", {"node_id": node, "phase": phase, "attempt": attempt, "kind": kind,
                                  "name": name, "ok": ok, "detail": detail})


def calls(conn: sqlite3.Connection, node: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM llm_calls WHERE node_id=? ORDER BY id", (node,))
    return [{**dict(r), "audit_hits": json.loads(r["audit_hits"])} for r in rows]


def activity(conn: sqlite3.Connection, node: str) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM activity WHERE node_id=? ORDER BY id", (node,))]


def spend(conn: sqlite3.Connection) -> tuple[int, int, float]:
    """(calls, tokens, dollars) of the run's real calls; a call without a price adds no dollars."""
    row = conn.execute("SELECT COUNT(*), COALESCE(SUM(input_tokens + cached_tokens + output_tokens), 0), "
                       "COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE NOT mock").fetchone()
    return int(row[0]), int(row[1]), float(row[2])
