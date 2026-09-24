"""Contract verification of a child's code (spec 9.4)."""
from __future__ import annotations

import ast
import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ar_contract.models import EditContext, RecipeContext

from ..gateway.app import create_gateway_app
from ..gateway.mock import MockBook
from ..gateway.store import CallStore
from ..sandbox.image import ImageBuildError, ensure_image
from ..sandbox.runner import Mounts, container_name, run_container
from ..services import RunServices, socket_dir_for
from ..tools.context import TokenRegistry
from ..tools.data_tools import register_data_tools
from ..tools.hf_tools import register_hf_tools
from ..tools.jobs import JobQueue, register_job_tools
from ..tools.server import ToolError, ToolKit, build_tool_app, new_mcp

MOCK_MODEL = "mock-model"
ENTRY_POINTS = ("edit_self", "improve_recipe")


@dataclass
class ContractStep:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ContractReport:
    ok: bool
    steps: list[ContractStep] = field(default_factory=list)
    image: str | None = None

    def to_retry(self) -> dict:
        failed = next((s for s in self.steps if not s.ok), None)
        return {"kind": "contract", "ok": self.ok, "failed_step": failed.name if failed else None,
                "detail": failed.detail if failed else "", "steps": [asdict(s) for s in self.steps]}


def static_check(entry_source: str) -> ContractStep:
    try:
        tree = ast.parse(entry_source)
    except SyntaxError as exc:
        return ContractStep("static", False, f"syntax error in agent/entry.py: {exc}")
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ENTRY_POINTS:
        fn = top.get(name)
        if fn is None:
            return ContractStep("static", False, f"agent/entry.py defines no top-level {name}")
        a = fn.args
        positional = a.posonlyargs + a.args
        if len(positional) != 1 or a.vararg or a.kwarg or a.kwonlyargs:
            return ContractStep("static", False,
                                f"{name} must take exactly one positional parameter (ctx)")
    return ContractStep("static", True)


class _MockData:
    """Canned data tools for smoke runs; same method names as DataTools."""
    def probe(self, c, path):
        return {"frames": 96, "fps": 24.0, "width": 736, "height": 414, "duration": 4.0,
                "rotation": 0, "sar": 1.0, "display_aspect": 736 / 414}

    def ingest(self, c, candidates):
        return [{"accepted": False, "clip_id": None, "formats": [], "warnings": [],
                 "reasons": ["mock tool server: smoke run"]} for _ in candidates]

    def query(self, c, filter):
        return {"total": 0, "returned": 0, "clips": []}

    def commit(self, c, parent, datasets, message):
        return {"commit_id": "0" * 64, "datasets": {}}

    def recipe_check(self, c, recipe, data_commit):
        return {"ok": True, "failures": []}


class _MockHf:
    def search(self, c, query, kind="dataset", limit=20):
        return []

    def download(self, c, repo, revision, patterns, max_bytes=None):
        raise ToolError("mock tool server: downloads are disabled during a smoke run")


class ContractHarness:
    def __init__(self, cfg, run_dir: Path, recorder) -> None:
        self.cfg, self.recorder = cfg, recorder
        self.registry = TokenRegistry(recorder)
        self.services = RunServices(socket_dir_for(Path(run_dir) / "contract-harness"))
        self.queue = JobQueue(recorder, threading.Lock(), wait_cap_s=5.0)

    @property
    def socket_dir(self) -> Path:
        return self.services.socket_dir

    def start(self) -> None:
        kit, mcp = ToolKit(self.registry, self.recorder), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, self.queue)
        gateway = create_gateway_app(registry=self.registry, store=CallStore(self.recorder),
                                     allowed_models={MOCK_MODEL}, upstream=None,
                                     mocks=MockBook.default())
        self.services.start(gateway, build_tool_app(mcp))

    def stop(self) -> None:
        self.services.stop()
        self.queue.shutdown()


def _smoke_context(kind: str) -> dict:
    common = {"nodes_remaining": 1, "attempt": 1, "max_attempts": 1, "dry_run": True}
    model = EditContext if kind == "edit_self" else RecipeContext
    return model(**common).model_dump()


def verify_contract(*, cfg, run_dir: Path, run_id: str, repo, commit: str, harness: ContractHarness,
                    recorder, node: str, attempt: int, import_timeout_s: float | None = None,
                    smoke_timeout_s: float | None = None, runner=run_container) -> ContractReport:
    import_timeout_s = import_timeout_s or float(cfg.get("timeouts.contract_import_s"))
    smoke_timeout_s = smoke_timeout_s or 4 * float(cfg.get("timeouts.contract_smoke_s"))
    report = ContractReport(ok=False)
    work = Path(run_dir) / "nodes" / node / "contract" / f"attempt-{attempt}"
    shutil.rmtree(work, ignore_errors=True)
    code = work / "agent"
    repo.checkout(commit, code)

    def finish(ok: bool) -> ContractReport:
        report.ok = ok
        recorder.event("contract.report", node=node, phase="contract", attempt=attempt,
                       component="contract", ok=ok, payload=report.to_retry())
        return report

    reqs_path = code / "agent" / "requirements.txt"
    try:
        report.image = ensure_image(cfg, reqs_path.read_text() if reqs_path.exists() else "",
                                    recorder=recorder, node=node)
        report.steps.append(ContractStep("build", True))
    except ImageBuildError as exc:
        report.steps.append(ContractStep("build", False, str(exc)))
        return finish(False)

    entry = code / "agent" / "entry.py"
    step = static_check(entry.read_text() if entry.exists() else "")
    report.steps.append(step)
    if not step.ok:
        return finish(False)

    def run(label: str, command: list[str], timeout_s: float, context: dict | None) -> tuple:
        run_dir_l = work / label.replace(":", "_")
        ws, ctx_dir = run_dir_l / "workspace", run_dir_l / "context"
        staging = Path(run_dir) / "staging" / node / f"contract-{attempt}-{label.replace(':', '_')}"
        for d in (ws, ctx_dir, staging):
            d.mkdir(parents=True, exist_ok=True)
        if context is not None:
            (ctx_dir / "context.json").write_text(json.dumps(context))
        caller = harness.registry.issue(node=node, phase="contract", attempt=attempt,
                                        workspace_host=ws, staging_host=staging, mock_script="smoke")
        # Controller ruling: TokenRegistry.issue() only redacts on harness.recorder.
        # verify_contract's own telemetry (this runner call, and finish()'s
        # contract.report event below) is recorded through the `recorder` argument,
        # which may be a different Recorder instance -- add the redaction there too,
        # or a token an agent prints to stdout/stderr would leak into that recorder's
        # telemetry.
        recorder.add_redaction(caller.token)
        try:
            result = runner(image=report.image, name=container_name(run_id, node, "contract", attempt),
                            mounts=Mounts(agent=code, workspace=ws, staging=staging, context=ctx_dir,
                                          store=Path(run_dir) / "store", contract=cfg.repo_root / "contract",
                                          sockets=harness.socket_dir, agent_readonly=True),
                            command=command, env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": MOCK_MODEL,
                                                  "AR_CONTEXT_WINDOW": str(cfg.get("agents.context_window_tokens")),
                                                  "AR_COMPACT_AT": str(cfg.get("agents.compact_at"))},
                            cpus=4, memory_gb=8, timeout_s=timeout_s, recorder=recorder,
                            node=node, phase="contract", attempt=attempt)
        finally:
            harness.registry.revoke(caller.token)
            harness.queue.cancel_for_token(caller.token)   # as in the phase runner (Task 15)
        return result, ws

    (Path(run_dir) / "store").mkdir(exist_ok=True)
    probe = ("import sys; sys.path.insert(0, '/agent'); import agent.entry as e; "
             "assert callable(e.edit_self) and callable(e.improve_recipe)")
    result, _ = run("import", ["python", "-c", probe], import_timeout_s, None)
    if result.timed_out or result.exit_code != 0:
        why = f"timed out after {import_timeout_s:.0f}s" if result.timed_out else result.stderr[-3000:]
        report.steps.append(ContractStep("import", False, why))
        return finish(False)
    report.steps.append(ContractStep("import", True))

    for kind in ENTRY_POINTS:
        result, ws = run(f"smoke:{kind}", ["python", "-m", "ar_contract.run", kind], smoke_timeout_s,
                         _smoke_context(kind))
        name = f"smoke:{kind}"
        if result.timed_out:
            report.steps.append(ContractStep(name, False, f"timed out after {smoke_timeout_s:.0f}s"))
            return finish(False)
        out = ws / "result.json"
        body = json.loads(out.read_text()) if out.exists() else {"ok": False, "error": "no result.json"}
        if result.exit_code != 0 or not body.get("ok"):
            report.steps.append(ContractStep(name, False, f"{body.get('error')}\n{result.stderr[-2000:]}"))
            return finish(False)
        report.steps.append(ContractStep(name, True))
    return finish(True)
