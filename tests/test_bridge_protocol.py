import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import protocol

from conftest import BRIDGES


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def job_in(tmp_path, items, groups=None, worker=None) -> dict:
    return {"work": str(tmp_path), "items": items, "groups": groups, "worker": worker, "file": str(tmp_path / "job.json")}


def test_messages_are_whole_lines_with_type_and_time_even_from_threads(tmp_path):
    m = protocol.Messages(tmp_path / "messages.jsonl")
    m.send("ready", info={"load_s": 1.5})
    threads = [threading.Thread(target=lambda n=n: [m.send("item", index=n, text="x" * 70000) for _ in range(5)])
               for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = lines(tmp_path / "messages.jsonl")
    assert got[0]["type"] == "ready" and got[0]["info"] == {"load_s": 1.5} and isinstance(got[0]["time"], float)
    assert sorted(g["index"] for g in got[1:]) == sorted(list(range(4)) * 5)


def test_a_worker_writes_its_own_messages_file(tmp_path):
    protocol.messages(job_in(tmp_path, [], worker=2)).send("job_done")
    assert lines(tmp_path / "messages-2.jsonl")[0]["type"] == "job_done"


def in_child(tmp_path, code: str) -> subprocess.CompletedProcess:
    """`code` in a fresh python with `protocol` and `job` (whose work folder is tmp_path): pytest's own capture
    replaces sys.stdout, a bridge's never does."""
    prelude = f"import os, sys, ctypes, subprocess, protocol\njob = {{'work': {str(tmp_path)!r}}}\n"
    return subprocess.run([sys.executable, "-c", prelude + code], env={**os.environ, "PYTHONPATH": str(BRIDGES)},
                          capture_output=True, text=True, check=True)


def test_item_log_takes_python_c_and_child_output_then_restores(tmp_path):
    r = in_child(tmp_path, """
with protocol.item_log(job, "7") as path:
    print("from python")
    print("to stderr", file=sys.stderr)
    os.write(1, b"raw fd write\\n")
    ctypes.CDLL(None).printf(b"from C\\n")
    subprocess.run([sys.executable, "-c", "print('from a child')"], check=True)
print("after", path)
""")
    text = (tmp_path / "logs" / "7.log").read_text()
    for line in ("from python", "to stderr", "raw fd write", "from C", "from a child"):
        assert line in text
    assert "after" not in text and r.stdout == f"after {tmp_path / 'logs' / '7.log'}\n"


def test_run_items_sends_ready_started_item_and_done(tmp_path):
    loaded = []
    job = job_in(tmp_path, [{"index": 0, "x": 1}, {"index": 1, "fail": True}, {"index": 2, "x": 3}])

    def one(model, item):
        if item.get("fail"):
            raise ValueError("bad item")
        return {"video": tmp_path / f"{item['index']}.mp4"}, {"seconds": 0.1, "model": model}

    protocol.run_items(job, lambda: loaded.append(1) or "M", one)
    got = lines(tmp_path / "messages.jsonl")
    assert loaded == [1]
    assert [g["type"] for g in got] == ["ready", "item_started", "item", "item_started", "item",
                                        "item_started", "item", "job_done"]
    ok, bad = got[2], got[4]
    assert ok["ok"] is True and ok["outputs"] == {"video": str(tmp_path / "0.mp4")}
    assert ok["info"] == {"seconds": 0.1, "model": "M"} and ok["log"] == str(tmp_path / "logs" / "0.log")
    assert got[1]["log"] == ok["log"] and ok["started"] <= ok["ended"]
    assert bad["ok"] is False and bad["error"] == "ValueError: bad item" and "Traceback" in bad["traceback"]


def test_run_items_puts_each_items_output_in_its_log(tmp_path):
    in_child(tmp_path, """
job["items"] = [{"index": 0}, {"index": 1}]
def one(model, item):
    print("working on", item["index"])
    if item["index"]:
        raise ValueError("bad")
    return {}, {}
print("loading")
protocol.run_items(job, lambda: None, one)
""")
    assert [(tmp_path / "logs" / f"{i}.log").read_text() for i in (0, 1)] == ["working on 0\n", "working on 1\n"]


def test_run_items_without_own_items_loads_nothing_and_sends_nothing(tmp_path):
    job = job_in(tmp_path, [{"index": 0}], groups=[[0], [1]], worker=1)
    protocol.run_items(job, lambda: (_ for _ in ()).throw(AssertionError("loaded")), None)
    assert not (tmp_path / "messages-1.jsonl").exists()


BRIDGE = """
import os, sys
import protocol
job = protocol.load_job()
if job["worker"] is None:
    sys.exit(protocol.spawn_workers(job))
print("worker", job["worker"], "sees", os.environ["CUDA_VISIBLE_DEVICES"])
def one(model, item):
    if item.get("crash"):
        os._exit(5)
    return {}, {"gpus": os.environ["CUDA_VISIBLE_DEVICES"]}
protocol.run_items(job, lambda: None, one)
"""


def run_bridge(tmp_path, items, groups) -> int:
    (tmp_path / "bridge.py").write_text(BRIDGE)
    (tmp_path / "job.json").write_text(json.dumps({"work": str(tmp_path), "items": items, "groups": groups}))
    return subprocess.run([sys.executable, str(tmp_path / "bridge.py"), "--job", str(tmp_path / "job.json")],
                          env={**os.environ, "PYTHONPATH": str(BRIDGES)}).returncode


def test_spawn_workers_runs_one_worker_per_group_on_its_gpus(tmp_path):
    assert run_bridge(tmp_path, [{"index": i} for i in range(3)], [[2, 3], [5]]) == 0
    w0, w1 = lines(tmp_path / "messages-0.jsonl"), lines(tmp_path / "messages-1.jsonl")
    assert [m["index"] for m in w0 if m["type"] == "item"] == [0, 2]
    assert [m["info"]["gpus"] for m in w1 if m["type"] == "item"] == ["5"]
    assert "worker 0 sees 2,3" in (tmp_path / "logs" / "worker0.log").read_text()


def test_a_failed_worker_makes_the_bridge_exit_non_zero(tmp_path):
    assert run_bridge(tmp_path, [{"index": 0}, {"index": 1, "crash": True}], [[0], [1]]) == 1
    assert [m["type"] for m in lines(tmp_path / "messages-1.jsonl")] == ["ready", "item_started"]
    assert lines(tmp_path / "messages-0.jsonl")[-1]["type"] == "job_done"
