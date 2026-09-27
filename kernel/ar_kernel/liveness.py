"""Soft timeouts with a probe window (spec 14.5). At the soft deadline the kernel watches the
phase's progress signals for one probe window: any change extends the deadline by
extension_frac x soft; none ends the phase. A hard cap, when set, ends it regardless.
GPU utilization is never a signal (verification-log finding 2)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


class Liveness:
    def __init__(self, soft_s: float, *, probe_window_s: float, extension_frac: float,
                 signals: list[Callable[[], object]], hard_s: float | None = None,
                 probe_every_s: float = 0.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.soft_s, self.probe_window_s, self.extension_frac = float(soft_s), probe_window_s, extension_frac
        # Signals can walk large trees (tree_mark over a WBench work dir), and callers poll every
        # second or so: during a probe window, evaluate them at most every probe_every_s.
        self.probe_every_s, self._last_eval = probe_every_s, float("-inf")
        self.hard_s, self.clock, self.signals = hard_s, clock, list(signals)
        self.started = clock()
        self.deadline = self.started + self.soft_s
        self.extensions = 0
        self.reason: str | None = None
        self._probe_start: float | None = None
        self._baseline = None

    @classmethod
    def from_config(cls, cfg, soft_s: float, signals: list, hard_s: float | None = None) -> "Liveness":
        return cls(soft_s, probe_window_s=60 * float(cfg.get("liveness.probe_window_min")),
                   extension_frac=float(cfg.get("liveness.extension_frac")), signals=signals, hard_s=hard_s,
                   probe_every_s=float(cfg.get("liveness.probe_every_s")))

    def add_signal(self, fn: Callable[[], object]) -> None:
        self.signals.append(fn)

    def expired(self) -> str | None:
        now = self.clock()
        if self.hard_s is not None and now - self.started >= self.hard_s:
            self.reason = f"hard cap of {self.hard_s:.0f}s reached"
            return self.reason
        if now < self.deadline:
            return None
        if (self._probe_start is not None and now - self._last_eval < self.probe_every_s
                and now - self._probe_start < self.probe_window_s):
            return None
        self._last_eval = now
        mark = tuple(fn() for fn in self.signals)
        if self._probe_start is None:
            self._probe_start, self._baseline = now, mark
            return None
        if mark != self._baseline:
            self.deadline = now + self.extension_frac * self.soft_s
            self.extensions += 1
            self._probe_start = None
            return None
        if now - self._probe_start >= self.probe_window_s:
            self.reason = (f"no sign of progress for {self.probe_window_s:.0f}s after the "
                           f"{self.soft_s:.0f}s soft timeout ({self.extensions} extensions)")
            return self.reason
        return None


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
