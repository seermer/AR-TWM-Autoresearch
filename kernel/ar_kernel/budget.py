"""LLM spend ledger and the optional run cap (user decision 2026-09-27: dollars only, none by
default). The gateway records every real call here and refuses new ones once the cap is
reached; the loop reads the same ledger to stop the run. Mock calls never count."""
from __future__ import annotations

import collections
import json
import threading
import time
from pathlib import Path

PRICE_KEYS = ("input", "cached_input", "output")


class BudgetError(ValueError):
    """The budget configuration cannot be enforced."""


def usage_tokens(usage) -> tuple[int, int, int]:
    """(uncached input, cached input, output) from a Chat Completions or Responses usage block."""
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int((details.get("cached_tokens") if isinstance(details, dict) else 0)
                 or usage.get("prompt_cache_hit_tokens") or 0)
    cached = min(cached, prompt)
    output = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return prompt - cached, cached, output


def _usd(name: str, value) -> float | None:
    """A price or cap as a float: YAML loads e.g. `2e-6` as a string, and a string would fail
    only after a paid call had been made."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BudgetError(f"{name} is not a number: {value!r}") from None
    if number < 0:
        raise BudgetError(f"{name} is negative: {value!r}")
    return number


class Budget:
    def __init__(self, *, max_usd: float | None = None, prices: dict | None = None) -> None:
        self.prices = {k: _usd(f"budget.usd_per_mtok.{k}", (prices or {}).get(k)) for k in PRICE_KEYS}
        if max_usd is not None and any(v is None for v in self.prices.values()):
            raise BudgetError("budget.max_usd needs budget.usd_per_mtok.{input,cached_input,output}")
        self.max_usd = _usd("budget.max_usd", max_usd)
        self.usd, self.tokens, self.calls = 0.0, 0, 0
        self._statuses: collections.deque = collections.deque(maxlen=10000)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(max_usd=cfg.get("budget.max_usd"),
                   prices=cfg.get("budget.usd_per_mtok") or {})

    def cost(self, usage) -> float | None:
        if any(v is None for v in self.prices.values()):
            return None
        fresh, cached, output = usage_tokens(usage)
        return (fresh * self.prices["input"] + cached * self.prices["cached_input"]
                + output * self.prices["output"]) / 1e6

    def _add(self, usage, cost: float | None) -> None:
        """Count one real call; the caller holds the lock."""
        self.tokens += sum(usage_tokens(usage))
        self.calls += 1
        self.usd += cost or 0.0

    def record(self, status: int, usage) -> float | None:
        cost = self.cost(usage)
        with self._lock:
            self._add(usage, cost)
            self._statuses.append((time.monotonic(), int(status)))
        return cost

    def exhausted(self) -> str | None:
        with self._lock:
            if self.max_usd is not None and self.usd >= self.max_usd:
                return f"run LLM budget spent: ${self.usd:.4f} of ${self.max_usd}"
        return None

    def error_rate(self, window_s: float) -> tuple[float, int]:
        cutoff = time.monotonic() - window_s
        with self._lock:
            recent = [s for t, s in self._statuses if t >= cutoff]
        if not recent:
            return 0.0, 0
        # Only upstream unavailability counts (rate limits, server and connection errors, 599 in
        # Upstream.post): a 400 caused by the agent's own request is not an outage.
        return sum(s == 429 or s >= 500 for s in recent) / len(recent), len(recent)

    def load(self, run_dir: Path) -> None:
        """Re-count the real calls already recorded for this run (resume and `ar status`). Resets
        the totals first, so calling it twice never double-counts. Statuses are not re-added: the
        error-rate window is about the live process."""
        with self._lock:
            self.usd, self.tokens, self.calls = 0.0, 0, 0
        for path in sorted((Path(run_dir) / "telemetry" / "events").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if '"llm.response"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:            # a line torn by kill -9 or a full disk
                    continue
                if event.get("type") != "llm.response" or event.get("mock"):
                    continue
                cost = self.cost(event.get("usage"))
                with self._lock:
                    self._add(event.get("usage"), cost)

    def snapshot(self) -> dict:
        with self._lock:
            return {"usd": self.usd if all(v is not None for v in self.prices.values()) else None,
                    "tokens": self.tokens, "calls": self.calls,
                    "max_usd": self.max_usd}
