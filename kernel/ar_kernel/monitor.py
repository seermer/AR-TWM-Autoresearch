"""Run monitoring: GPU and host samples, and alerts. Alerts never stop anything."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import psutil

from .guards import alert, smi
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
    def __init__(self, cfg, run_dir: Path, recorder, gpus: list[int], budget, *,
                 usage=gpu_usage_detailed, memory=host_memory_gib, events=memory_events, processes=process_memory,
                 clock=time.time) -> None:
        self.cfg, self.run_dir, self.recorder, self.gpus, self.budget = cfg, Path(run_dir), recorder, list(gpus), budget
        self.usage, self.memory, self.events, self.processes, self.clock = usage, memory, events, processes, clock
        self._oom_kills: int | None = None
        self._ram_high = False
        self._raised: set = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_check = 0.0
        self._errors: set[str] = set()

    def _once(self, key, kind: str, message: str, **payload) -> None:
        if key not in self._raised:
            self._raised.add(key)
            alert(self.recorder, kind, message, **payload)

    def tick(self) -> None:
        usage = self.usage()
        if usage is not None:
            self.recorder.event("gpu.sample", node="gpu",
                                gpus={str(g): {k: v for k, v in u.items() if k != "pids"} for g, u in usage.items()})
        mem = {**self.memory(), "dev_shm": shutil.disk_usage("/dev/shm").used / 2**30}
        self.recorder.event("host.sample", node="gpu", memory_gib={k: v if v is None else round(v, 2) for k, v in mem.items()},
                            disk_free_gib=round(shutil.disk_usage(self.run_dir).free / 2**30, 1),
                            memory_events=(events := self.events()), processes=self.processes())
        kills = events.get("oom_kill", 0)
        if self._oom_kills is not None and kills > self._oom_kills:
            alert(self.recorder, "host_oom_kill", f"the kernel killed {kills - self._oom_kills} process(es) at the "
                  f"job's memory limit", level="error", memory_gib=mem)
        self._oom_kills = kills
        held = mem["anon"] + mem["shmem"] if "anon" in mem else mem["used"]     # not reclaimable page cache
        ram = float(self.cfg.get("host.ram_gib"))
        high = held > float(self.cfg.get("host.alert_fraction")) * ram
        if high and not self._ram_high:
            alert(self.recorder, "host_memory_high", f"the job holds {held:.0f} GiB of host RAM (anon + shmem), "
                  f"host.ram_gib is {ram:g}", level="warning", memory_gib=mem)
        self._ram_high = high

    def check(self) -> None:
        now = self.clock()
        state_file = self.run_dir / "control" / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            node = state.get("node")
            if node and state.get("phase") not in (None, "idle"):
                events = self.recorder.events_path(node)
                # Progress is new events, a growing training log or new eval output: a normal
                # multi-hour training writes no events but does write its log.
                node_dir = self.run_dir / "nodes" / node
                train_log = node_dir / "attempts" / f"improve_recipe-{state.get('attempt')}" / "train" / "train.log"
                last = max(state.get("since", 0), events.stat().st_mtime if events.exists() else 0,
                           tree_mark(train_log, node_dir / "eval")[2] / 1e9)
                if now - last > 60 * float(self.cfg.get("alerts.stall_min")):
                    self._once(("stall", node, state.get("phase"), state.get("attempt")), "stall",
                               f"no events from {node} {state.get('phase')} for "
                               f"{(now - last) / 60:.0f} min", level="warning")
        free = shutil.disk_usage(self.run_dir).free / (1024 ** 3)
        if free < float(self.cfg.get("disk.alert_below_gb")):
            self._once(("disk", int(now // 3600)), "disk_low", f"{free:.0f} GB free under {self.run_dir}")
        rate, calls = self.budget.error_rate(60 * float(self.cfg.get("alerts.gateway_error_window_min")))
        if calls >= 5 and rate > float(self.cfg.get("alerts.gateway_error_rate")):
            self._once(("gateway", int(now // 300)), "gateway_errors",
                       f"{rate:.0%} of the last {calls} LLM calls failed")
        usage = self.usage() or {}
        me = os.getpid()
        for gpu, u in usage.items():
            if gpu in self.gpus:
                continue
            ours = [p for p in u["pids"] if _descends_from(p, me)]
            if ours:
                self._once(("outside", gpu, tuple(ours)), "gpu_outside_list",
                           f"kernel process(es) {ours} run on GPU {gpu}, outside the list {self.gpus}")

    def _guarded(self, step) -> None:
        try:
            step()
        except Exception as exc:                               # noqa: BLE001 -- never kill the run
            if repr(exc) not in self._errors:                  # the same failure every 5 s is logged once
                self._errors.add(repr(exc))
                self.recorder.event("monitor.error", payload={"error": repr(exc)})

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

        self._thread = threading.Thread(target=loop, daemon=True, name="ar-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
