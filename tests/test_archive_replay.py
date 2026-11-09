"""P11: the archive database rebuilt from the committed effects in the run's records is the same database."""
import sqlite3

import pytest
from conftest import DERIVED, NAMED, staged_clip

from ar_kernel import telemetry
from ar_kernel.archive.activity import add_activity, add_call
from ar_kernel.archive.commits import CommitStore
from ar_kernel.archive.db import open_db, transaction
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.selection import select_parent
from ar_kernel.telemetry import tree

CFG = KernelConfig.load()


def _committed(trace, node) -> bool:
    """Not inside a transaction that rolled back or never ended."""
    parent = trace.spans.get(node.start.get("parent"))
    while parent is not None:
        if parent.start["name"] == "db.transaction":
            end = parent.end
            if end is None or trace.payload(end.get("output"))["outcome"] != "committed":
                return False
        parent = trace.spans.get(parent.start.get("parent"))
    return True


def replay(run_dir, target_dir) -> sqlite3.Connection:
    trace = tree.load(run_dir)
    conn = open_db(target_dir)
    writes = [n for n in trace.spans.values()
              if n.start["name"] == "db.write" and n.end is not None and n.end["status"] == "ok" and _committed(trace, n)]
    for n in sorted(writes, key=lambda n: n.end["seq"]):
        inputs, output = trace.payload(n.start["inputs"]), trace.payload(n.end["output"])
        table = inputs["table"]
        ids = {r["rowid_"] for r in inputs["before"] + output["after"]}
        conn.execute(f"DELETE FROM {table} WHERE rowid IN ({', '.join('?' * len(ids))})", tuple(ids))
        for row in output["after"]:
            row = {("rowid" if k == "rowid_" else k): v for k, v in row.items()}
            conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                         tuple(row.values()))
    return conn


def _tables(conn) -> dict:
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {t: [tuple(r) for r in conn.execute(f"SELECT rowid, * FROM {t} ORDER BY rowid")] for t in names}


def test_the_archive_rebuilt_from_its_records_equals_the_archive(tmp_path):
    run = tmp_path / "run"
    telemetry.open(run)
    telemetry.start("ar.run")
    conn = open_db(run)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.record_score("root", 0.5, ["m"], {"m": 0.5})
    assert select_parent(conn, CFG, "n1", 7) == "root"
    nodes.create("n1", "root", 1)
    nodes.set_fields("n1", agent_commit="abc", error=None)
    nodes.add_attempt("n1", "improve_recipe", 1, "ok", {"x": 1})
    ing = Ingestor(CFG, run, conn)
    ids = []
    for name, seconds in (("a", 4.0), ("b", 5.0), ("c", 4.5)):
        video, annotation, pose = staged_clip(run / "staging" / name, seconds)
        ids.append(ing.ingest([Candidate(video=video, annotation=annotation, pose=pose, derived_from=DERIVED,
                                         named=NAMED)], node_id="n1", attempt=1 if name != "c" else 2)[0].clip_id)
    commits = CommitStore(conn, ing.blobs, ing.clips)
    first, _ = commits.commit({"cam": {"weight": 1.0, "clips": ids[:2]}}, "one", node_id="n1", attempt=1)
    commits.commit({"cam": {"weight": 1.0, "clips": ids}}, "two", node_id="n1", attempt=2)
    commits.settle("n1", 1, first)
    commits.fail("n1", 2)
    commits.remove("n1", 2)
    ing.clips.remove_ingested("n1", 2)
    ing.blobs.remove_unreferenced()
    ing.clips.hide("n1", "fail1", 1)
    with pytest.raises(RuntimeError):
        with transaction(conn):                                         # rolled back: not in the archive
            nodes.set_status("n1", "train_failed")
            raise RuntimeError("abandoned")
    add_call(run, node="n1", phase="improve_recipe", attempt=1, role="plan", tool=None, conversation_id="c",
             turn=0, transcript=None, status=200, tokens=(1, 2, 3), cost_usd=0.1, mock=False, audit_hits=[])
    add_activity(run, node="n1", phase="improve_recipe", attempt=1, kind="tool", name="ingest", ok=True)
    rebuilt = replay(run, tmp_path / "rebuilt")
    original = _tables(conn)
    assert original["nodes"][1][5] == "running" and original["blobs"] and original["clips"]
    assert _tables(rebuilt) == original
    assert tree.check(tree.load(run)) == []
