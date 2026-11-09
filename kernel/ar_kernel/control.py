"""Run control: the loop's pid file, graceful stop requests, and signal handling."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from . import fs, telemetry
from .subproc import kill_group, proc_start_time


class ForceStop(BaseException):
    """Raised in the main thread by SIGTERM or a second SIGINT. A BaseException, so no
    `except Exception` in a phase swallows it; run_in_env and run_container clean up on it."""


class Control:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir) / "control"
        self.pid_file = self.dir / "loop.pid"

    def alive_pid(self) -> int | None:
        try:
            pid_s, started = self.pid_file.read_text().split()
            pid = int(pid_s)
        except (OSError, ValueError):
            return None
        return pid if proc_start_time(pid) == started else None

    def claim(self) -> None:
        fs.mkdir(self.dir)
        other = self.alive_pid()
        if other is not None and other != os.getpid():
            raise RuntimeError(f"a loop is already running for this run (pid {other})")
        fs.write_text(self.pid_file, f"{os.getpid()} {proc_start_time(os.getpid())}")

    def release(self) -> None:
        if self.alive_pid() == os.getpid():
            fs.remove(self.pid_file)

    def request_stop(self) -> None:
        fs.mkdir(self.dir)
        fs.write_text(self.dir / "stop", "")

    def stop_requested(self) -> bool:
        return (self.dir / "stop").exists()

    def clear_stop(self) -> None:
        fs.remove(self.dir / "stop")

    def args(self) -> dict:
        path = self.dir / "run_args.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_args(self, **kw) -> None:
        fs.mkdir(self.dir)
        fs.write_text(self.dir / "run_args.json", json.dumps({**self.args(), **kw}))


def exit_process(code: int) -> None:
    """End the kernel process now. A tool call still running in a worker thread (an hf_download
    waiting out HF rate limits, say) would otherwise keep it alive after the stop: Python's exit
    joins those threads, and the pid file is already gone. Archive transactions are already committed."""
    stuck = [t.name for t in threading.enumerate()
             if t is not threading.main_thread() and t.is_alive() and not t.daemon]
    if stuck:
        message = f"exiting with {len(stuck)} thread(s) still running: {', '.join(stuck)}"
        telemetry.alert("run.shutdown", "warning", message, {"threads": stuck})
        print(f"warning: {message}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def kill_recorded_groups(control: Control) -> list[int]:
    """Kill every process group the kernel started and never saw end (the subproc registry).
    A group whose leader is gone is still killed (its workers can outlive `conda run`);
    only a live process with pid == pgid and another start time -- a recycled pid -- is left alone."""
    folder = control.dir / "pgids"
    killed = []
    for marker in sorted(folder.glob("*")) if folder.is_dir() else []:
        pgid, started = int(marker.name), marker.read_text().strip()
        current = proc_start_time(pgid)
        why = ("no start time was recorded" if not started else "the leader is gone; its group may live on"
               if current is None else "the leader is alive" if current == started else "the pid was reused")
        chosen = "kill" if started and current in (None, started) else "leave"
        telemetry.decision("run.leftover_group", {"pgid": pgid, "started": started, "now": current},
                           ["kill", "leave"], chosen, why)
        if chosen == "kill" and kill_group(pgid, "recorded group"):
            killed.append(pgid)
        fs.remove(marker)
    return killed


def drive(loop, kit, control: Control) -> str:
    """Run `loop` with stop handling. Returns the exit reason."""
    control.claim()
    os.environ["AR_PGID_DIR"] = str(control.dir / "pgids")     # run_in_env records every group here
    control.clear_stop()
    state = {"presses": 0, "forced": False}      # forced: never raise again (cleanup must finish)

    def on_signal(signum, frame):
        name = signal.Signals(signum).name
        if state["forced"]:
            chosen, why = "ignored", "a stop is already under way"
        elif signum == signal.SIGINT and state["presses"] == 0:
            chosen, why = "graceful", "first Ctrl-C"
        else:
            chosen, why = "force stop", "second Ctrl-C" if signum == signal.SIGINT else name
        state["presses"] += signum == signal.SIGINT
        telemetry.deferred_decision("run.signal", {"signal": name, "received": time.time()},
                                    ["graceful", "force stop", "ignored"], chosen, why)
        if chosen == "graceful":
            loop.graceful.set()
        elif chosen == "force stop":
            state["forced"] = True
            raise ForceStop(why)

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    done = threading.Event()

    def watch_stop_file():
        seen = False
        while not done.wait(2.0):
            if not seen and control.stop_requested():
                seen = True
                telemetry.decision("run.stop_file", {"path": str(control.dir / "stop")}, ["graceful"], "graceful",
                                   "the stop request file exists")
                loop.graceful.set()

    telemetry.thread(watch_stop_file, "ar-stop-file").start()
    try:
        kit.start()
        reason = loop.run()
    except ForceStop as exc:
        telemetry.error(exc, "the run's stop reason")
        reason = f"force stop ({exc})"
    finally:
        state["forced"] = True                  # a signal from here on is ignored
        done.set()
        with telemetry.span("run.shutdown") as sp:
            try:
                kit.stop()
            finally:
                sp.output = {"groups_killed": kill_recorded_groups(control)}   # e.g. a job whose kill outlived
                for s, h in old.items():                                       # JobQueue.shutdown's join
                    signal.signal(s, h)
                control.release()
    return reason
