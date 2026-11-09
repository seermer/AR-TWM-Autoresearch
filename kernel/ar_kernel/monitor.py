"""Run monitoring: GPU and host samples, checks and alerts. Alerts never stop anything."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import psutil

from . import telemetry
from .guards import smi
from .subproc import host_memory_gib, memory_events


def tree_mark(*paths) -> tuple[int, int, int]:
    """(files, bytes, newest mtime_ns) of the regular files at or under `paths`."""
    count = size = newest = 0
    for p in map(Path, paths):
        files = [p] if p.is_file() else (p.rglob("*") if p.is_dir() else [])
        for f in files:
            try:
                if f.is_symlink() or not f.is_file():
                    continue
                st = f.stat()
            except OSError:
                continue
            count, size, newest = count + 1, size + st.st_size, max(newest, st.st_mtime_ns)
    return count, size, newest


def _num(value: str) -> float | None:
    return float(value) if value.replace(".", "", 1).isdigit() else None     # "[N/A]" -> None


def gpu_usage_detailed() -> dict[int, dict] | None:
    """Per-GPU utilization, memory, power, temperature and compute pids from nvidia-smi."""
    table = smi(["--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
                  "--format=csv,noheader,nounits"])
    apps = smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"])
    if table is None or apps is None:
        return None
    out, by_uuid = {}, {}
    for line in table.splitlines():
        if not line.strip():
            continue
        index, uuid, util, mem, power, temp = (s.strip() for s in line.split(","))
        by_uuid[uuid] = int(index)
        out[int(index)] = {"util": _num(util), "memory_mib": _num(mem), "power_w": _num(power),
                           "temp_c": _num(temp), "pids": []}
    for line in apps.splitlines():
        if line.strip():
            pid, uuid = (s.strip() for s in line.split(","))
            if uuid in by_uuid:
                out[by_uuid[uuid]]["pids"].append(int(pid))
    return out


def process_memory(top: int = 8) -> list[dict]:
    """The `top` processes of this process's tree (ar, its tools, the agent sandboxes) holding the most host RAM
    (anon + shmem, GiB, from /proc/<pid>/status)."""
    me = psutil.Process()
    out = []
    for proc in [me, *me.children(recursive=True)]:
        try:
            status = Path(f"/proc/{proc.pid}/status").read_text()
            argv = proc.cmdline()
            cmd = " ".join([Path(argv[0]).name, *argv[1:]])[:160] if argv else proc.name()
        except (OSError, psutil.Error):                    # ended meanwhile
            continue
        kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in status.splitlines())
              if k in ("RssAnon", "RssFile", "RssShmem")}
        out.append({"pid": proc.pid, "cmd": cmd, **{k: round(kb.get(n, 0) / 2**20, 2) for k, n in
                    (("anon", "RssAnon"), ("file", "RssFile"), ("shmem", "RssShmem"))}})
    return sorted(out, key=lambda p: p["anon"] + p["shmem"], reverse=True)[:top]


def _descends_from(pid: int, ancestor: int) -> bool:
    if pid == ancestor:
        return True
    try:
        return ancestor in {p.pid for p in psutil.Process(pid).parents()}
    except psutil.Error:
        return False


class Monitor:
    """`idle_s()`: seconds since the agent's last gateway call, tool call or job progress (LastActivity)."""

    def __init__(self, cfg, run_dir: Path, gpus: list[int], budget, idle_s, *,
                 usage=gpu_usage_detailed, memory=host_memory_gib, events=memory_events, processes=process_memory,
                 clock=time.time) -> None:
        self.cfg, self.run_dir, self.gpus, self.budget, self.idle_s = cfg, Path(run_dir), list(gpus), budget, idle_s
        self.usage, self.memory, self.events, self.processes, self.clock = usage, memory, events, processes, clock
        self._oom_kills: int | None = None
        self._ram_high = False
        self._raised: set = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_check = 0.0
        self._errors: set[str] = set()

    def _once(self, key, name: str, level: str, message: str, details=None) -> bool:
        """The alert, unless `key` already raised one; whether it did."""
        if key in self._raised:
            return False
        self._raised.add(key)
        telemetry.alert(name, level, message, details)
        return True

    def tick(self) -> None:
        usage = self.usage()
        mem = {**self.memory(), "dev_shm": shutil.disk_usage("/dev/shm").used / 2**30}
        events = self.events()
        telemetry.sample(gpus=None if usage is None else {str(g): u for g, u in usage.items()},
                         memory_gib={k: v if v is None else round(v, 2) for k, v in mem.items()},
                         disk_free_gib=round(shutil.disk_usage(self.run_dir).free / 2**30, 1),
                         memory_events=events, processes=self.processes())
        kills = events.get("oom_kill", 0)
        if self._oom_kills is not None and kills > self._oom_kills:
            telemetry.alert("monitor.host_oom_kill", "error", f"the kernel killed {kills - self._oom_kills} "
                            "process(es) at the job's memory limit", {"memory_gib": mem})
        self._oom_kills = kills
        held = mem["anon"] + mem["shmem"] if "anon" in mem else mem["used"]     # not reclaimable page cache
        ram = float(self.cfg.get("host.ram_gib"))
        high = held > float(self.cfg.get("host.alert_fraction")) * ram
        if high and not self._ram_high:
            telemetry.alert("monitor.host_memory_high", "warning", f"the job holds {held:.0f} GiB of host RAM "
                            f"(anon + shmem), host.ram_gib is {ram:g}", {"memory_gib": mem})
        self._ram_high = high

    def _stall(self, now: float) -> None:
        state_file = self.run_dir / "control" / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        node, phase, attempt = state.get("node"), state.get("phase"), state.get("attempt")
        if not node or phase in (None, "idle"):
            return
        # Progress is agent activity, a growing training log or new eval output: a normal multi-hour
        # training makes no gateway or tool calls but does write its log.
        node_dir = self.run_dir / "nodes" / node
        train_log = node_dir / "attempts" / f"improve_recipe-{attempt}" / "train" / "train.log"
        marks = {"phase_start": state.get("since", 0), "agent_activity": now - self.idle_s(),
                 "train_log_or_eval": tree_mark(train_log, node_dir / "eval")[2] / 1e9}
        idle_min = (now - max(marks.values())) / 60
        limit = float(self.cfg.get("alerts.stall_min"))
        key = ("stall", node, phase, attempt)
        if idle_min <= limit:
            chosen, reason = "active", f"progress {idle_min:.0f} min ago, within {limit:g} min"
        elif self._once(key, "monitor.stall", "warning", f"no progress from {node} {phase} for {idle_min:.0f} min"):
            chosen, reason = "alert", f"no progress for {idle_min:.0f} min, over {limit:g} min"
        else:
            chosen, reason = "already alerted", "this phase attempt already had its stall alert"
        telemetry.decision("monitor.stall_check", {"node": node, "phase": phase, "attempt": attempt, "now": now,
                                                   "marks": marks}, ["active", "alert", "already alerted"],
                           chosen, reason)

    def check(self) -> None:
        now = self.clock()
        self._stall(now)
        free = shutil.disk_usage(self.run_dir).free / (1024 ** 3)
        if free < float(self.cfg.get("disk.alert_below_gb")):
            self._once(("disk", int(now // 3600)), "monitor.disk_low", "warning", f"{free:.0f} GB free under {self.run_dir}")
        window_min = float(self.cfg.get("alerts.gateway_error_window_min"))
        rate, calls = self.budget.error_rate(60 * window_min)
        limit = float(self.cfg.get("alerts.gateway_error_rate"))
        if calls < 5 or rate <= limit:
            chosen, reason = "quiet", f"{rate:.0%} of {calls} calls failed: alerts need 5 calls over {limit:.0%}"
        elif self._once(("gateway", int(now // 300)), "monitor.gateway_errors", "warning",
                        f"{rate:.0%} of the last {calls} LLM calls failed"):
            chosen, reason = "alert", f"{rate:.0%} of {calls} calls failed, over {limit:.0%}"
        else:
            chosen, reason = "already alerted", "this 5-minute window already had its alert"
        telemetry.decision("monitor.error_rate_check", {"window_min": window_min, "rate": rate, "calls": calls},
                           ["quiet", "alert", "already alerted"], chosen, reason)
        usage = self.usage() or {}
        me = os.getpid()
        for gpu, u in usage.items():
            if gpu in self.gpus:
                continue
            ours = [p for p in u["pids"] if _descends_from(p, me)]
            if ours:
                self._once(("outside", gpu, tuple(ours)), "monitor.gpu_outside_list", "warning",
                           f"kernel process(es) {ours} run on GPU {gpu}, outside the list {self.gpus}")

    def _guarded(self, step) -> None:
        try:
            step()
        except Exception as exc:                               # noqa: BLE001 -- never kill the run
            if repr(exc) not in self._errors:                  # the same failure every 5 s alerts once
                self._errors.add(repr(exc))
                telemetry.error(exc, "an alert; the monitor goes on")
                telemetry.alert("monitor.failed", "warning", f"a monitor step failed: {exc!r}")

    def poll(self) -> None:
        """One monitor step. A failing GPU sample never suppresses the checks (e.g. disk_low)."""
        self._guarded(self.tick)
        if time.monotonic() - self._last_check >= 60:
            self._last_check = time.monotonic()
            self._guarded(self.check)

    def start(self) -> None:
        sample_s = float(self.cfg.get("telemetry.gpu_sample_sec"))

        def loop():
            while not self._stop.wait(sample_s):
                self.poll()

        self._thread = telemetry.thread(loop, "ar-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
