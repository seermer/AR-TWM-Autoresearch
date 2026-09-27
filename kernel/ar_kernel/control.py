"""Run control (spec 14.4): the loop's pid file, graceful stop requests, and signal handling."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from .archive.nodes import NodeStore
from .guards import alert
from .sandbox.runner import kill_run_containers
from .subproc import proc_start_time


class ForceStop(BaseException):
    """Raised in the main thread by SIGTERM or a second SIGINT. A BaseException, so no
    `except Exception` in a phase swallows it; run_in_env and run_container clean up on it."""


class Control:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir) / "control"

    def _pid_file(self) -> Path:
        return self.dir / "loop.pid"

    @staticmethod
    def _start_time(pid: int) -> str | None:
        return proc_start_time(pid)

    def alive_pid(self) -> int | None:
        try:
            pid_s, started = self._pid_file().read_text().split()
            pid = int(pid_s)
        except (OSError, ValueError):
            return None
        return pid if self._start_time(pid) == started else None

    def claim(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        other = self.alive_pid()
        if other is not None and other != os.getpid():
            raise RuntimeError(f"a loop is already running for this run (pid {other})")
        self._pid_file().write_text(f"{os.getpid()} {self._start_time(os.getpid())}")

    def release(self) -> None:
        if self.alive_pid() == os.getpid():
            self._pid_file().unlink(missing_ok=True)

    def request_stop(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "stop").touch()

    def stop_requested(self) -> bool:
        return (self.dir / "stop").exists()

    def clear_stop(self) -> None:
        (self.dir / "stop").unlink(missing_ok=True)

    def args(self) -> dict:
        path = self.dir / "run_args.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_args(self, **kw) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run_args.json").write_text(json.dumps({**self.args(), **kw}))


def kill_recorded_groups(control: Control) -> list[int]:
    """Kill every process group the kernel started and never saw end (the subproc registry).
    A group whose leader is gone is still killed (its workers can outlive `conda run`, finding 6);
    only a live process with pid == pgid and another start time -- a recycled pid -- is left alone."""
    folder = control.dir / "pgids"
    killed = []
    for marker in sorted(folder.glob("*")) if folder.is_dir() else []:
        pgid, started = int(marker.name), marker.read_text().strip()
        current = proc_start_time(pgid)
        if started and (current is None or current == started):
            for sig, wait in ((signal.SIGTERM, 30.0), (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(pgid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                if sig == signal.SIGTERM:
                    killed.append(pgid)
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except (ProcessLookupError, PermissionError):
                        break
                    time.sleep(0.2)
        marker.unlink(missing_ok=True)
    return killed


def mark_interrupted(ctx, reason: str) -> list[str]:
    """On resume: every non-root node still `running` was cut off (forced stop, kernel death or a
    spent budget). It is never resumed and never cleaned up (user decision 2026-09-27): its status
    becomes `interrupted`, its files and archive rows stay, and only containers a killed kernel left
    running are removed. A fresh cycle then starts. A root left `running` is not marked:
    ensure_root scores it again."""
    nodes = NodeStore(ctx.conn)
    state_file = Path(ctx.run_dir) / "control" / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    marked = []
    for node in nodes.all():
        if node["status"] != "running" or node["parent_id"] is None:
            continue
        nid = node["node_id"]
        phase = state.get("phase") if state.get("node") == nid else None
        containers = kill_run_containers(Path(ctx.run_dir).name, nid)
        nodes.set_status(nid, "interrupted")
        nodes.set_fields(nid, error=f"{reason} (phase {phase})")
        ctx.recorder.event("node.interrupted", child=nid, payload={
            "node": nid, "reason": reason, "phase_reached": phase, "containers": containers})
        alert(ctx.recorder, "node_interrupted", f"{nid} was interrupted in phase {phase}; kept as is",
              level="warning", node_id=nid)
        marked.append(nid)
    return marked


def drive(loop, kit, control: Control, recorder, monitor=None) -> str:
    """Run `loop` with stop handling. Returns the exit reason."""
    control.claim()
    os.environ["AR_PGID_DIR"] = str(control.dir / "pgids")     # run_in_env records every group here
    control.clear_stop()
    state = {"presses": 0, "forced": False}      # forced: never raise again (cleanup must finish)

    def force(why: str) -> None:
        if state["forced"]:
            recorder.event("control", payload={"command": f"ignored during stop ({why})"})
            return
        state["forced"] = True
        raise ForceStop(why)

    def on_int(signum, frame):
        state["presses"] += 1
        if state["presses"] == 1:
            loop.graceful.set()
            recorder.event("control", payload={"command": "graceful stop (SIGINT)"})
        else:
            force("second Ctrl-C")

    def on_term(signum, frame):
        force("SIGTERM")

    old = {s: signal.signal(s, h) for s, h in ((signal.SIGINT, on_int), (signal.SIGTERM, on_term))}
    done = threading.Event()

    def watch_stop_file():
        while not done.wait(2.0):
            if control.stop_requested():
                loop.graceful.set()

    threading.Thread(target=watch_stop_file, daemon=True, name="ar-stop-file").start()
    reason = "unknown"
    try:
        kit.start()
        if monitor is not None:
            monitor.start()
        reason = loop.run()
    except ForceStop as exc:
        reason = f"force stop ({exc})"
        kill_run_containers(Path(loop.ctx.run_dir).name)
    except Exception as exc:
        reason = f"crashed: {type(exc).__name__}: {exc}"
        raise
    finally:
        state["forced"] = True                  # a signal from here on only logs
        done.set()
        try:
            if monitor is not None:
                monitor.stop()
        finally:
            try:
                kit.stop()
            finally:
                kill_recorded_groups(control)   # e.g. a job whose kill outlived JobQueue.shutdown's join
                for s, h in old.items():
                    signal.signal(s, h)
                control.release()
                recorder.event("run.stopped", payload={"reason": reason})
    return reason
