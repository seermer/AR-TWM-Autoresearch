from __future__ import annotations
import json, re, sqlite3, time
from pathlib import Path

# interrupted: unfinished when the loop stopped (forced stop, kernel death, spent budget); kept
# untouched, never resumed, never a parent, not counted toward max_nodes (user decision 2026-09-27).
STATUSES = {"running", "scored", "invalid_code", "invalid_recipe", "train_failed", "eval_failed",
            "crashed", "interrupted"}
UNFINISHED = ("running", "interrupted")
# node_id becomes a directory name and a telemetry file name.
NODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def run_rel(run_dir: Path, path: Path | str) -> str:
    """A path under the run, stored relative to it so a moved run still resolves."""
    return str(Path(path).resolve().relative_to(Path(run_dir).resolve()))

class NodeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, node_id: str, parent_id: str | None, depth: int) -> None:
        if not NODE_ID_RE.match(node_id or ""):
            raise ValueError(f"node_id {node_id!r} must match {NODE_ID_RE.pattern}")
        self.conn.execute(
            "INSERT INTO nodes (node_id, parent_id, depth, created_at, status) VALUES (?,?,?,?,'running')",
            (node_id, parent_id, depth, time.time()),
        )

    def set_status(self, node_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; allowed: {sorted(STATUSES)}")
        self._update("UPDATE nodes SET status=? WHERE node_id=?", (status, node_id), node_id)

    def record_score(self, node_id: str, score: float, metric_set: list[str],
                     metrics: dict[str, float]) -> None:
        self._update(
            "UPDATE nodes SET score=?, metric_set=?, metrics=?, status='scored' WHERE node_id=?",
            (float(score), json.dumps(list(metric_set)), json.dumps(metrics), node_id), node_id)

    def set_fields(self, node_id: str, **fields: object) -> None:
        allowed = {"agent_commit", "data_commit", "recipe_hash", "resolved_config_path",
                   "checkpoint_path", "lora_rank", "lora_alpha", "subtree_value",
                   "phase_timings", "rationale_path", "recipe_path",
                   "attempt_counts", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown node fields: {sorted(unknown)}")
        for key, value in fields.items():
            self._update(f"UPDATE nodes SET {key}=? WHERE node_id=?", (value, node_id), node_id)

    def _update(self, sql: str, params: tuple, node_id: str) -> None:
        """An UPDATE matching no row used to succeed silently, losing the write."""
        if self.conn.execute(sql, params).rowcount == 0:
            raise KeyError(f"no node {node_id!r}")

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        node = dict(row)
        node["metric_set"] = json.loads(node["metric_set"]) if node["metric_set"] else []
        node["metrics"] = json.loads(node["metrics"]) if node["metrics"] else {}
        return node

    def get(self, node_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        return self._row(row)

    def all(self) -> list[dict]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM nodes ORDER BY created_at")]

    def add_attempt(self, node_id: str, phase: str, index: int, outcome: str, detail: dict) -> None:
        self.conn.execute(
            "INSERT INTO attempts (node_id, phase, idx, outcome, detail, created_at) VALUES (?,?,?,?,?,?)",
            (node_id, phase, index, outcome, json.dumps(detail), time.time()),
        )

    def attempts(self, node_id: str, phase: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE node_id=? AND phase=? ORDER BY idx", (node_id, phase))
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]
