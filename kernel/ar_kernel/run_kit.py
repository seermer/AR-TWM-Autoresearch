"""Everything a run serves to its agents, built from the run's config."""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import telemetry
from .agent_phase import PhaseEnv, smoke_contexts
from .archive.db import open_db
from .archive.nodes import NodeStore
from .budget import Budget
from .contract.verify import ContractHarness, verify_contract
from .gateway.app import Upstream, create_gateway_app
from .gateway.mock import MockBook
from .gateway.store import CallStore
from .isolation import censor_names, host_roots
from .monitor import Monitor
from .services import RunServices, socket_dir_for
from .telemetry import env_secrets
from .tools.ask import Ask, register_ask_tool
from .tools.captioner import register_caption_tool
from .tools.locate_events import register_locate_tool
from .tools.context import TokenRegistry
from .tools.data_tools import DataTools, register_data_tools
from .tools.gpu_jobs import build_gpu_backends, register_gpu_tools
from .tools.hf_tools import HfTools, hf_tmp, register_hf_tools
from .tools.jobs import JobQueue, register_job_tools
from .tools.server import ToolHealth, ToolKit, build_tool_app, new_mcp
from .tools.knowledge import register_knowledge_tool
from .tools.snapshot import SnapshotStore, register_snapshot_tool
from .tools.vllm_server import FREE_WAITS_S, process_tree, require_free_gpus


class LastActivity:
    """When an agent's phase last showed activity: a gateway call, a tool call or a job's progress."""

    def __init__(self) -> None:
        self._at, self._lock = time.monotonic(), threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self._at = time.monotonic()

    def idle_s(self) -> float:
        with self._lock:
            return time.monotonic() - self._at


@dataclass
class RunKit:
    env: PhaseEnv
    gpu_lock: threading.Lock
    budget: Budget
    services: RunServices
    harness: ContractHarness
    gateway_app: object
    tools_app: object
    last_activity: LastActivity
    monitor: Monitor

    def start(self) -> None:
        self.services.start(self.gateway_app, self.tools_app)
        self.harness.start()
        self.monitor.start()

    def stop(self) -> None:
        self.monitor.stop()
        try:
            self.env.queue.shutdown()
        except RuntimeError as exc:           # a job may still hold a GPU; reported, never fatal
            telemetry.error(exc, "an alert; the cleanup goes on")
            telemetry.alert("run.shutdown", "warning", str(exc))
            print(f"warning: {exc}", file=sys.stderr)
        finally:
            try:
                self.services.stop()        # raises if a service thread will not join
            finally:
                self.harness.stop()


def build_run_kit(cfg, run_dir: Path, gpus: list[int], environ, repo) -> RunKit:
    budget = Budget.from_config(cfg)
    conn = open_db(run_dir)
    try:
        budget.load(conn)
    finally:
        conn.close()
    registry = TokenRegistry(cfg.get("isolation.blocked_names") or [], host_roots(cfg))
    gpu_lock = threading.Lock()
    last = LastActivity()
    health = ToolHealth(int(cfg.get("tools.failure_alert_after")))

    def require_free(caller) -> None:
        require_free_gpus(gpus, int(cfg.get("gpus.free_below_mib")), waits=FREE_WAITS_S,
                          own=lambda: process_tree(caller.sandbox.get("pid")))
    queue = JobQueue(run_dir, gpu_lock, wait_cap_s=float(cfg.get("tools.job_wait_max_s")),
                     require_free=require_free, health=health, touch=last.touch)
    for backend in build_gpu_backends(cfg, run_dir, gpus, registry):
        queue.register(backend)
    kit, mcp = ToolKit(registry, run_dir, health, touch=last.touch), new_mcp()
    data = DataTools(cfg, run_dir, gpus, gpu_lock, require_free)
    registry.on_revoke(data.forget)
    register_data_tools(mcp, kit, data)
    model = environ["OPENAI_MODEL"]
    socket_dir = socket_dir_for(run_dir)
    harness = ContractHarness(cfg, run_dir, repo)
    env = PhaseEnv(cfg=cfg, run_dir=run_dir, run_id=Path(run_dir).name, registry=registry,
                   queue=queue, repo=repo, socket_dir=socket_dir, gpus=gpus, default_model=model)

    def verify(caller, commit: str):
        """The contract on an edit_self snapshot, with the contexts this node's phases get."""
        conn = open_db(run_dir)                  # a tool thread: not the loop's connection
        try:
            contexts = smoke_contexts(env, conn=conn, node=caller.node,
                                      parent_id=NodeStore(conn).get(caller.node)["parent_id"])
        finally:
            conn.close()
        return verify_contract(cfg=cfg, run_dir=run_dir, run_id=Path(run_dir).name, repo=repo, commit=commit,
                               harness=harness, node=caller.node, attempt=caller.attempt,
                               contexts=contexts)
    register_snapshot_tool(mcp, kit, SnapshotStore(run_dir, repo, data.usable, gate=data.snapshot_gate,
                                                   verify=verify))
    register_hf_tools(mcp, kit, HfTools(cfg, hf_tmp(run_dir)))
    register_job_tools(mcp, kit, queue)
    register_gpu_tools(mcp, kit, queue)
    register_caption_tool(mcp, kit, queue)
    register_locate_tool(mcp, kit, queue)
    register_knowledge_tool(mcp, kit)
    store = CallStore(run_dir, env_secrets(cfg.dotenv_files, environ), cfg.get("isolation.audit_patterns") or [])
    upstream = Upstream.from_env(environ, timeout_s=float(cfg.get("gateway.upstream_timeout_s")),
                                 wait_s=float(cfg.get("gateway.upstream_wait_s")))
    register_ask_tool(mcp, kit, Ask(cfg, upstream, store, budget, model))
    gateway = create_gateway_app(
        registry=registry, store=store,
        allowed_models=set(cfg.get("gateway.model_allowlist") or []) | {model},
        upstream=upstream, mocks=MockBook.default(), budget=budget, censor=censor_names(cfg), touch=last.touch)
    return RunKit(env=env, gpu_lock=gpu_lock, budget=budget, services=RunServices(socket_dir), harness=harness,
                  gateway_app=gateway, tools_app=build_tool_app(mcp), last_activity=last,
                  monitor=Monitor(cfg, run_dir, gpus, budget, last.idle_s))
