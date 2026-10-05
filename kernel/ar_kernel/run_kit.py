"""Everything a run serves to its agents, built from the run's config."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .budget import Budget
from .contract.verify import ContractHarness
from .gateway.app import Upstream, create_gateway_app
from .gateway.mock import MockBook
from .gateway.store import CallStore
from .guards import alert
from .isolation import censor_names, host_roots
from .services import RunServices, socket_dir_for
from .tools.ask import Ask, register_ask_tool
from .tools.captioner import register_caption_tool
from .tools.context import TokenRegistry
from .tools.data_tools import DataTools, register_data_tools
from .tools.gpu_jobs import build_gpu_backends, register_gpu_tools
from .tools.hf_tools import HfTools, register_hf_tools
from .tools.jobs import JobQueue, register_job_tools
from .tools.server import ToolKit, build_tool_app, new_mcp
from .tools.skills import register_skill_tool
from .tools.vllm_server import FREE_WAITS_S, require_free_gpus


@dataclass
class RunKit:
    registry: TokenRegistry
    queue: JobQueue
    gpu_lock: threading.Lock
    budget: Budget
    services: RunServices
    harness: ContractHarness
    socket_dir: Path
    default_model: str
    gateway_app: object
    tools_app: object
    recorder: object

    def start(self) -> None:
        self.services.start(self.gateway_app, self.tools_app)
        self.harness.start()

    def stop(self) -> None:
        try:
            self.queue.shutdown()
        except RuntimeError as exc:           # a job may still hold a GPU; recorded, never fatal
            alert(self.recorder, "shutdown", str(exc))
        finally:
            try:
                self.services.stop()        # raises if a service thread will not join
            finally:
                self.harness.stop()


def build_run_kit(cfg, run_dir: Path, gpus: list[int], recorder, environ) -> RunKit:
    budget = Budget.from_config(cfg)
    budget.load(run_dir)
    registry = TokenRegistry(recorder, cfg.get("isolation.blocked_names") or [], host_roots(cfg))
    gpu_lock = threading.Lock()
    def require_free() -> None:
        require_free_gpus(gpus, int(cfg.get("gpus.free_below_mib")), waits=FREE_WAITS_S)
    queue = JobQueue(recorder, gpu_lock, wait_cap_s=float(cfg.get("tools.job_wait_max_s")), require_free=require_free)
    for backend in build_gpu_backends(cfg, run_dir, gpus, registry, recorder):
        queue.register(backend)
    kit, mcp = ToolKit(registry, recorder), new_mcp()
    register_data_tools(mcp, kit, DataTools(cfg, run_dir, recorder, gpus, gpu_lock, require_free))
    register_hf_tools(mcp, kit, HfTools(cfg, Path(run_dir) / "hf_tmp"))
    register_job_tools(mcp, kit, queue)
    register_gpu_tools(mcp, kit, queue)
    register_caption_tool(mcp, kit, queue)
    register_skill_tool(mcp, kit)
    model = environ["OPENAI_MODEL"]
    store = CallStore(recorder)
    upstream = Upstream.from_env(environ, timeout_s=float(cfg.get("gateway.upstream_timeout_s")),
                                 retries=int(cfg.get("gateway.upstream_retries")))
    register_ask_tool(mcp, kit, Ask(cfg, upstream, store, budget, model))
    gateway = create_gateway_app(
        registry=registry, store=store,
        allowed_models=set(cfg.get("gateway.model_allowlist") or []) | {model},
        upstream=upstream, mocks=MockBook.default(), budget=budget, censor=censor_names(cfg))
    socket_dir = socket_dir_for(run_dir)
    return RunKit(registry=registry, queue=queue, gpu_lock=gpu_lock, budget=budget,
                  services=RunServices(socket_dir), harness=ContractHarness(cfg, run_dir, recorder),
                  socket_dir=socket_dir, default_model=model, gateway_app=gateway,
                  tools_app=build_tool_app(mcp), recorder=recorder)
