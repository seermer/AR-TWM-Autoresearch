import threading

import pytest

from ar_kernel.archive.activity import activity, add_activity, add_call, calls, spend
from ar_kernel.archive.db import open_db
from ar_kernel.faults import KernelFault


def _call(tmp_path, **over):
    row = dict(node="n1", phase="improve_recipe", attempt=1, role="plan", tool=None, conversation_id="c1", turn=0,
               transcript="transcripts/improve_recipe-1/01-plan.jsonl", status=200, tokens=(10, 5, 3),
               cost_usd=0.5, mock=False, audit_hits=["wbench"])
    add_call(tmp_path, **{**row, **over})


def test_rows_round_trip(tmp_path):
    _call(tmp_path)
    add_activity(tmp_path, node="n1", phase="improve_recipe", attempt=1, kind="tool", name="hf_search", ok=False,
                 detail="403 denied")
    conn = open_db(tmp_path)
    [call] = calls(conn, "n1")
    assert call["audit_hits"] == ["wbench"] and call["input_tokens"] == 10 and call["cached_tokens"] == 5
    assert call["transcript"].endswith("01-plan.jsonl") and call["mock"] == 0 and call["created_at"] > 0
    [row] = activity(conn, "n1")
    assert (row["kind"], row["name"], row["ok"], row["detail"]) == ("tool", "hf_search", 0, "403 denied")
    assert calls(conn, "n2") == [] and activity(conn, "n2") == []


def test_spend_counts_real_calls_only(tmp_path):
    _call(tmp_path)
    _call(tmp_path, cost_usd=None)                      # no prices: tokens and the call still count
    _call(tmp_path, mock=True, cost_usd=None)
    assert spend(open_db(tmp_path)) == (2, 36, 0.5)


def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(KernelFault, match="activity"):
        add_activity(tmp_path, node="n1", phase=None, attempt=None, kind="other")


def test_a_failed_insert_is_a_kernel_fault(tmp_path):
    (tmp_path / "archive.db").mkdir()                    # the database cannot be opened
    with pytest.raises(KernelFault, match="llm_calls"):
        _call(tmp_path)


def test_inserts_from_several_threads(tmp_path):
    held = open_db(tmp_path)                     # as in a run, where the loop keeps its connection open

    def write(i):
        for j in range(5):
            add_activity(tmp_path, node="n1", phase="p", attempt=i, kind="tool", name=f"t{j}", ok=True)
            _call(tmp_path, attempt=i, turn=j)
    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(activity(held, "n1")) == 40 and len(calls(held, "n1")) == 40
