"""A local `vllm serve` process on a set of GPUs, shared by the captioner and the eval judge."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from .jobs import run_cancellable

log = logging.getLogger(__name__)
RELEASE_SLACK_MIB = 512          # GPU memory counts as released within this of its pre-job level


def gpu_memory_mib(gpus: list[int]) -> dict[int, int] | None:
    """memory.used per GPU (nvidia-smi indices, i.e. PCI bus order), or None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        used = {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):    # ValueError: "[N/A]", "[Not Supported]"
        return None
    return {g: used[g] for g in gpus if g in used}


def _tail(path: Path, limit: int = 3000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-limit:] if path.exists() else ""


def wait_gpu_release(gpu_memory, gpus: list[int], before: dict | None,
                     timeout_s: float) -> tuple[dict | None, bool | None]:
    """Wait until every GPU is back within RELEASE_SLACK_MIB of its pre-job memory.

    (None, None) when GPU memory cannot be read. Shared by every GPU job backend
    (captioner and the GpuJob subclasses), not just this one.
    """
    if before is None:
        return None, None
    deadline = time.monotonic() + timeout_s
    while True:
        after = gpu_memory(gpus)
        if after is None:
            return None, None
        if all(after.get(g, 0) <= used + RELEASE_SLACK_MIB for g, used in before.items()):
            return after, True
        if time.monotonic() > deadline:
            return after, False
        time.sleep(1.0)


def parallelism(gpus: list[int], tensor_parallel: int | None = None) -> tuple[int, int]:
    """(tensor-parallel, data-parallel) sizes that use the passed GPUs: tp is the configured
    value or the largest power of two <= len(gpus) (vLLM cannot shard this model across
    other counts), dp fills what is left."""
    n = len(gpus)
    tp = tensor_parallel or 1 << (n.bit_length() - 1)
    dp = max(1, n // tp)
    if tp * dp < n:
        log.warning("vLLM uses %d of the %d passed GPUs: tensor-parallel %d cannot shard the model "
                    "across %d GPUs", tp * dp, n, tp, n)
    return tp, dp


def serve_command(c: dict, gpus: list[int], port: int, served_name: str,
                  media_dir: Path | None = None, mm_limits: dict | None = None) -> list[str]:
    """`vllm serve` for the model in config section `c` (the `captioner` block). `mm_limits`: media items
    per prompt (default: one video, no images)."""
    tp, dp = parallelism(gpus, c.get("tensor_parallel"))
    return ["vllm", "serve", c["model"], "--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", served_name,
            "--tensor-parallel-size", str(tp),
            *(["--data-parallel-size", str(dp)] if dp > 1 else []),
            "--max-model-len", str(c["max_model_len"]),
            "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
            "--max-num-seqs", str(c["max_num_seqs"]),
            "--max-num-batched-tokens", str(c["max_num_batched_tokens"]),
            "--reasoning-parser", c["reasoning_parser"],
            "--mm-encoder-tp-mode", c["mm_encoder_tp_mode"],
            *(["--speculative-config", json.dumps(c["speculative_config"])] if c.get("speculative_config") else []),
            "--limit-mm-per-prompt", json.dumps(mm_limits or {"image": 0, "video": 1}),
            "--media-io-kwargs", json.dumps(c["media_io_kwargs"]),
            *(["--allowed-local-media-path", str(media_dir)] if media_dir else []),
            *c.get("extra_args", [])]


class VllmServer:
    """Runs `command` (from `serve_command`) in conda env `env` on `gpus` and waits for /health.

    `start` returns the load time in seconds, or None when `cancel` was set first. `stop` kills the
    process group, waits for it, and returns its exit code."""

    def __init__(self, env: str, command: list[str], gpus: list[int], port: int, *, cwd: Path,
                 log_path: Path, recorder=None, node: str = "run", phase: str = "vllm",
                 poll_s: float = 2.0, label: str = "vLLM server") -> None:
        self.env, self.command, self.gpus, self.port = env, command, list(gpus), port
        self.cwd, self.log_path, self.recorder, self.node, self.phase = Path(cwd), Path(log_path), recorder, node, phase
        self.poll_s, self.label = poll_s, label
        self._stop, self._cancel, self._exit, self._thread = threading.Event(), None, [], None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _serve(self) -> None:
        try:
            self._exit.append(run_cancellable(
                self.env, self.command, cwd=self.cwd, log_path=self.log_path,
                cancel=SimpleNamespace(is_set=lambda: self._stop.is_set() or bool(self._cancel and self._cancel.is_set())),
                extra_env={"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus)),
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_HUB_OFFLINE": "1"},
                recorder=self.recorder, node=self.node, phase=self.phase))
        except Exception as exc:              # noqa: BLE001 -- reported as a startup failure
            self._exit.append(f"launch failed: {type(exc).__name__}: {exc}")

    def start(self, startup_timeout_s: float, cancel=None, on_poll=None) -> float | None:
        self._cancel = cancel
        self._thread = threading.Thread(target=self._serve, name=f"ar-vllm-{self.port}", daemon=True)
        started = time.monotonic()
        self._thread.start()
        deadline = started + startup_timeout_s
        with httpx.Client(timeout=5, trust_env=False) as http:
            while not (cancel and cancel.is_set()):
                if not self._thread.is_alive():
                    raise RuntimeError(f"the {self.label} exited during startup ({self._exit[0]}):\n"
                                       f"{_tail(self.log_path)}")
                try:
                    if http.get(f"{self.base_url}/health").status_code == 200:
                        if on_poll:
                            on_poll()
                        return time.monotonic() - started
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"the {self.label} was not ready within {startup_timeout_s:g} s:"
                                       f"\n{_tail(self.log_path)}")
                if on_poll:
                    on_poll()
                time.sleep(self.poll_s)
        return None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return self._exit[0] if self._exit else None
