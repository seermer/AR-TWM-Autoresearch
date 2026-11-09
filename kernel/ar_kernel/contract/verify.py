"""Contract verification of a child's code."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ar_contract.static import ENTRY_POINTS, mentions, static_check

from .. import fs, telemetry
from ..agent_phase import install_requirements, read_result, requirements_listed
from ..gateway.app import create_gateway_app
from ..gateway.mock import MockBook
from ..gateway.store import CallStore
from ..sandbox.runner import Mounts, container_name, run_container
from ..services import RunServices, socket_dir_for
from ..tools.ask import MockAsk, register_ask_tool
from ..tools.captioner import TOOL as CAPTION_TOOL, register_caption_tool
from ..tools.locate_events import TOOL as LOCATE_TOOL, register_locate_tool
from ..tools.context import TokenRegistry
from ..tools.data_tools import register_data_tools
from ..tools.hf_tools import register_hf_tools
from ..tools.jobs import JobQueue, register_job_tools
from ..tools.server import ToolError, ToolKit, build_tool_app, new_mcp
from ..tools.knowledge import register_knowledge_tool
from ..tools import snapshot
from ..tools.snapshot import SnapshotStore, register_snapshot_tool
from ..vcs.agents_repo import CheckoutError

MOCK_MODEL = "mock-model"
MOCK_DATA_COMMIT = "dc1"          # the one data commit the mock data tools hold


@dataclass
class ContractStep:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ContractReport:
    ok: bool
    steps: list[ContractStep] = field(default_factory=list)

    def to_record(self) -> dict:
        failed = next((s for s in self.steps if not s.ok), None)
        return {"kind": "contract", "ok": self.ok, "failed_step": failed.name if failed else None,
                "detail": failed.detail if failed else "", "steps": [asdict(s) for s in self.steps]}


class _MockData:
    """Canned data tools for smoke runs; same method names as DataTools."""
    def probe(self, c, path):
        return {"frames": 96, "fps": 24.0, "width": 736, "height": 414, "duration": 4.0,
                "rotation": 0, "sar": 1.0, "display_aspect": 736 / 414}

    def ingest(self, c, candidates):
        return {"accepted": 0, "rejected": len(candidates), "warnings": [], "result_file": None,
                "rejected_for": [{"count": len(candidates), "example": "mock tool server: smoke run"}]}

    def query(self, c, filter):
        return {"total": 0, "clips": [], "result_file": ""}

    def fetch(self, c, clip_ids):
        return {"folder": "/workspace/staging/pool_clips", "copied": 0, "not_in_pool": []}

    def commit(self, c, datasets, message, include=()):
        return {"commit_id": MOCK_DATA_COMMIT, "datasets": {}, "new_clips": 0}

    def usable(self, commit_id):
        if commit_id != MOCK_DATA_COMMIT:
            raise ToolError(f"unknown data commit {commit_id!r}")

    def recipe_check(self, c, recipe, data_commit):
        return {"ok": True, "failures": []}


class _MockHf:
    def search(self, c, query, kind="dataset", limit=20):
        return []

    def list_files(self, c, repo, revision, pattern="*", limit=200, offset=0, kind="dataset"):
        return {"repo": repo, "revision": revision, "matching_files": 0, "files": []}

    def download(self, c, repo, revision, patterns, max_bytes=None, kind="dataset"):
        raise ToolError("mock tool server: downloads are disabled during a smoke run")


class _MockCaption:
    """caption_videos in smoke runs: a queued job like the real one, but it never starts vLLM."""
    name = tool = CAPTION_TOOL

    def run(self, job, cancel, report):
        return {"clips": {p: {"caption": "mock caption"} for p in job.args["paths"]}, "load_s": 0.0,
                "gpu_memory_mib": None, "gpu_memory_released": None}


class _MockLocate:
    """locate_events in smoke runs: a queued job like the real one, but it never starts vLLM."""
    name = tool = LOCATE_TOOL

    def run(self, job, cancel, report):
        return {"clips": {i["video"]: {"events": [{"event": e, "start_s": 0.0, "end_s": 1.0, "disagreeing_s": 0}
                                                  for e in i["events"]]} for i in job.args["items"]},
                "load_s": 0.0, "gpu_memory_mib": None, "gpu_memory_released": None}


class ContractHarness:
    def __init__(self, cfg, run_dir: Path, repo) -> None:
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.snapshots = SnapshotStore(run_dir, repo, _MockData().usable)
        self.registry = TokenRegistry()
        self.services = RunServices(socket_dir_for(Path(run_dir) / "contract-harness"))
        self.queue = JobQueue(run_dir, threading.Lock(), wait_cap_s=5.0)
        self.queue.register(_MockCaption())
        self.queue.register(_MockLocate())

    @property
    def socket_dir(self) -> Path:
        return self.services.socket_dir

    def start(self) -> None:
        kit, mcp = ToolKit(self.registry, self.run_dir), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_snapshot_tool(mcp, kit, self.snapshots)
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, self.queue)
        register_caption_tool(mcp, kit, self.queue)
        register_locate_tool(mcp, kit, self.queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_knowledge_tool(mcp, kit)
        store = CallStore(self.run_dir, audit_patterns=self.cfg.get("isolation.audit_patterns") or [])
        gateway = create_gateway_app(registry=self.registry, store=store,
                                     allowed_models={MOCK_MODEL}, upstream=None,
                                     mocks=MockBook.default())
        self.services.start(gateway, build_tool_app(mcp))

    def stop(self) -> None:
        self.services.stop()
        self.queue.shutdown()


def verify_contract(*, cfg, run_dir: Path, run_id: str, repo, commit: str, harness: ContractHarness,
                    node: str, attempt: int, contexts: dict[str, dict],
                    runner=run_container) -> ContractReport:
    """`contexts`: per entry point, the dry-run context its smoke run receives (agent_phase.smoke_contexts)."""
    report = ContractReport(ok=False)
    work = Path(run_dir) / "nodes" / node / "contract" / f"attempt-{attempt}"
    fs.rmtree(work, ignore_errors="an earlier check's folder")
    snapshot.clear(run_dir, repo, node, "contract", attempt)
    code = work / "agent"

    def finish(ok: bool) -> ContractReport:
        report.ok = ok
        return report

    def add(name: str, ok: bool, detail: str = "") -> None:
        """One step of the check, as a decision."""
        telemetry.decision("contract.step", {"step": name}, ["pass", "fail"], "pass" if ok else "fail",
                           detail or "passed")
        report.steps.append(ContractStep(name, ok, detail))

    try:
        repo.checkout(commit, code)
    except CheckoutError as exc:
        telemetry.error(exc, "the checkout step fails")
        add("checkout", False, str(exc))
        return finish(False)

    def folders(label: str) -> tuple:
        run_dir_l = work / label.replace(":", "_")
        ws, ctx_dir = run_dir_l / "workspace", run_dir_l / "context"
        staging = Path(run_dir) / "staging" / node / f"contract-{attempt}-{label.replace(':', '_')}"
        for d in (ws, ctx_dir, staging):
            fs.mkdir(d)
        return ws, ctx_dir, staging

    installed = work / "build" / "workspace" / ".home" / ".local"     # linked into every run's home
    try:
        listed = requirements_listed(code)
    except (OSError, UnicodeDecodeError) as exc:
        telemetry.error(exc, "the build step fails")
        add("build", False, f"cannot read agent/requirements.txt: {exc}")
        return finish(False)
    if listed:
        ws, ctx_dir, staging = folders("build")
        result = install_requirements(
            runner, name=container_name(run_id, node, "contract-install", attempt),
            mounts=Mounts(agent=code, workspace=ws, staging=staging, context=ctx_dir,
                          contract=cfg.repo_root / "contract", sockets=harness.socket_dir, agent_readonly=True),
            path="/agent/agent/requirements.txt", cfg=cfg)
        if result.exit_code != 0:
            add("build", False, f"agent/requirements.txt failed to install (exit code "
                                f"{result.exit_code}):\n{result.stderr[-6000:]}")
            return finish(False)
    add("build", True)

    entry = code / "agent" / "entry.py"
    try:
        entry_source = entry.read_text(encoding="utf-8") if entry.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:
        telemetry.error(exc, "the static step fails")
        problem = f"cannot read agent/entry.py as UTF-8 text: {exc}"
    else:
        problem = static_check(entry_source)
    add("static", problem is None, problem or "")
    if problem is not None:
        return finish(False)
    hits = mentions(code)
    add("mentions", not hits, "\n".join(hits))
    if hits:
        return finish(False)

    def run(label: str, command: list[str], context: dict | None) -> tuple:
        ws, ctx_dir, staging = folders(label)
        if installed.is_dir():
            fs.copytree(installed, ws / ".home" / ".local", link=True, symlinks=True)
        if context is not None:
            fs.write_text(ctx_dir / "context.json", json.dumps(context))
        edits = label == "smoke:edit_self"
        caller = harness.registry.issue(node=node, phase="contract", attempt=attempt,
                                        workspace_host=ws, staging_host=staging, mock_script="smoke",
                                        agent_host=code if edits else None, base_commit=commit if edits else None)
        try:
            result = runner(name=container_name(run_id, node, "contract", attempt),
                            mounts=Mounts(agent=code, workspace=ws, staging=staging, context=ctx_dir,
                                          contract=cfg.repo_root / "contract",
                                          sockets=harness.socket_dir, agent_readonly=True),
                            command=command, env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": MOCK_MODEL,
                                                  "AR_CONTEXT_WINDOW": str(cfg.get("agents.context_window_tokens")),
                                                  "AR_COMPACT_AT": str(cfg.get("agents.compact_at"))},
                            env_dir=cfg.repo_root / ".envs" / "agent", shm_gb=8)
        finally:
            harness.registry.revoke(caller.token)
            harness.queue.cancel_for_token(caller.token)   # as in the phase runner
        return result, ws

    fs.mkdir(Path(run_dir) / "store")
    probe = ("import sys; sys.path.insert(0, '/agent'); import agent.entry as e; "
             "assert callable(e.edit_self) and callable(e.improve_recipe)")
    result, _ = run("import", ["python", "-c", probe], None)
    if result.exit_code != 0:
        add("import", False, result.stderr[-6000:])
        return finish(False)
    add("import", True)

    for kind in ENTRY_POINTS:
        result, ws = run(f"smoke:{kind}", ["python", "-m", "ar_contract.run", kind], contexts[kind])
        name = f"smoke:{kind}"
        # Never follows a symlink or opens a FIFO the agent planted there.
        body = read_result(ws / "result.json") or {"ok": False, "error": "no result.json"}
        if result.exit_code != 0 or not body.get("ok"):
            tb = body.get("traceback")
            tb_tail = f"\n{tb[-6000:]}" if isinstance(tb, str) and tb else ""
            add(name, False, f"{body.get('error')}\n{result.stderr[-6000:]}{tb_tail}")
            return finish(False)
        add(name, True)
    return finish(True)
