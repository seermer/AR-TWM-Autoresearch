"""The bridge side of a GPU job (stdlib only: bridges run in their tools' envs).

A bridge is run as `python <bridge>.py --job <work>/job.json` and appends one JSON line per message to
`<work>/messages.jsonl` (a spawned worker: `messages-<rank>.jsonl`): `ready`, `item_started`, `item`,
`job_done`. Each item's output goes to `<work>/logs/<index>.log`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

_libc = ctypes.CDLL(None)


def load_job() -> dict:
    """The job file named by `--job`, with `file` (its path) and `worker` (`--worker`, else None)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--worker", type=int)
    a = ap.parse_args()
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    return {**job, "file": a.job, "worker": a.worker}


class Messages:
    def __init__(self, path: Path) -> None:
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._lock = threading.Lock()

    def send(self, type: str, **fields) -> None:
        data = (json.dumps({"type": type, "time": time.time(), **fields}, ensure_ascii=False) + "\n").encode()
        with self._lock:
            while data:
                data = data[os.write(self._fd, data):]


def messages(job: dict) -> Messages:
    name = "messages.jsonl" if job.get("worker") is None else f"messages-{job['worker']}.jsonl"
    return Messages(Path(job["work"]) / name)


def log_path(job: dict, name: str) -> Path:
    path = Path(job["work"]) / "logs" / f"{name}.log"
    path.parent.mkdir(exist_ok=True)
    return path


def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    _libc.fflush(None)


@contextmanager
def item_log(job: dict, name: str):
    """File descriptors 1 and 2 go to `logs/<name>.log` inside the block (children and C code included)."""
    path = log_path(job, name)
    _flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield path
    finally:
        _flush()
        for target, fd in zip((1, 2), saved):
            os.dup2(fd, target)
            os.close(fd)


def mine(items: list[dict], rank: int, world: int) -> list[dict]:
    return [item for item in items if item["index"] % world == rank]


def spawn_workers(job: dict) -> int:
    """This bridge again, once per `job["groups"]` entry with `--worker <rank>`, seeing only that group's GPUs
    (output to `logs/worker<rank>.log`). Returns 1 if any worker exited non-zero, else 0."""
    procs = []
    for rank, group in enumerate(job["groups"]):
        with open(log_path(job, f"worker{rank}"), "w", encoding="utf-8") as log:
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(sys.argv[0]), "--job", job["file"], "--worker", str(rank)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, group))},
                stdout=log, stderr=subprocess.STDOUT))
    return int(any([p.wait() for p in procs]))


def run_items(job: dict, load, one) -> None:
    """The worker's own items (`index % len(groups)`; all without workers): `load()` once, then
    `one(model, item) -> (outputs, info)` per item, each under `item_log`."""
    rank, world = job.get("worker") or 0, len(job.get("groups") or [None])
    items = mine(job["items"], rank, world)
    if not items:
        return
    out = messages(job)
    t0 = time.monotonic()
    model = load()
    out.send("ready", info={"load_s": round(time.monotonic() - t0, 2)})
    for item in items:
        index, log, started = item["index"], str(log_path(job, str(item["index"]))), time.time()
        out.send("item_started", index=index, log=log)
        try:
            with item_log(job, str(index)):
                outputs, info = one(model, item)
            out.send("item", index=index, ok=True, outputs={k: str(v) for k, v in outputs.items()}, info=info,
                     started=started, ended=time.time(), log=log)
        except Exception as exc:          # noqa: BLE001 -- an item error; the next item still runs
            out.send("item", index=index, ok=False, error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc(), started=started, ended=time.time(), log=log)
    out.send("job_done")
