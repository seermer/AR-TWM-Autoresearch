"""Contract verification of a child's code."""
from __future__ import annotations

import ast
import json
import re
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agent_phase import read_result
from ..eval.score import AGENT_METRICS
from ..gateway.app import create_gateway_app
from ..gateway.mock import MockBook
from ..gateway.store import CallStore
from ..liveness import Liveness, tree_mark
from ..sandbox.image import ImageBuildError, ensure_image
from ..sandbox.runner import Mounts, container_name, run_container
from ..services import RunServices, socket_dir_for
from ..tools.ask import MockAsk, register_ask_tool
from ..tools.captioner import TOOL as CAPTION_TOOL, register_caption_tool
from ..tools.context import TokenRegistry
from ..tools.data_tools import register_data_tools
from ..tools.hf_tools import register_hf_tools
from ..tools.jobs import JobQueue, register_job_tools
from ..tools.server import ToolError, ToolKit, build_tool_app, new_mcp
from ..tools.skills import register_skill_tool
from ..vcs.agents_repo import CheckoutError

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
    except (ValueError, RecursionError, MemoryError) as exc:   # e.g. deeply nested agent source
        return ContractStep("static", False,
                            f"cannot parse agent/entry.py: {type(exc).__name__}: {exc}")
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


_NODE_FACT = re.compile(r"\b(n\d+|" + "|".join(alias for alias, _, _ in AGENT_METRICS.values()) + r")\b")


def prompt_check(code: Path) -> ContractStep:
    """A role prompt holds a mission and general behaviour: it may not name a node or a metric.
    Checked here, not in the agent's own self-test, which the agent can edit."""
    prompts = code / "agent" / "prompts"
    for path in sorted(p for p in prompts.rglob("*.md") if p.is_file() and not p.is_symlink()):
        found = sorted(set(_NODE_FACT.findall(path.read_text(encoding="utf-8", errors="replace"))))
        if found:
            return ContractStep("prompts", False,
                                f"agent/prompts/{path.relative_to(prompts)} names {', '.join(found)}. A prompt holds a role's "
                                "mission and general behaviour only: put facts about nodes and metrics in what "
                                "the role is told first (agent/briefing.py), not in its prompt")
    return ContractStep("prompts", True)


class _MockData:
    """Canned data tools for smoke runs; same method names as DataTools."""
    def probe(self, c, path):
        return {"frames": 96, "fps": 24.0, "width": 736, "height": 414, "duration": 4.0,
                "rotation": 0, "sar": 1.0, "display_aspect": 736 / 414}

    def ingest(self, c, candidates):
        return {"accepted": 0, "rejected": len(candidates), "warnings": [], "result_file": None,
                "rejected_for": [{"count": len(candidates), "example": "mock tool server: smoke run"}]}

    def query(self, c, filter):
        return {"total": 0, "returned": 0, "clips": []}

    def commit(self, c, parent, datasets, message):
        return {"commit_id": "0" * 64, "datasets": {}}

    def recipe_check(self, c, recipe, data_commit):
        return {"ok": True, "failures": []}


class _MockHf:
    def search(self, c, query, kind="dataset", limit=20):
        return []

    def list_files(self, c, repo, revision, pattern="*", limit=200, offset=0):
        return {"repo": repo, "revision": revision, "matching_files": 0, "files": []}

    def download(self, c, repo, revision, patterns, max_bytes=None):
        raise ToolError("mock tool server: downloads are disabled during a smoke run")


class _MockCaption:
    """caption_videos in smoke runs: a queued job like the real one, but it never starts vLLM."""
    name = tool = CAPTION_TOOL

    def run(self, job, cancel, report):
        return {"clips": {p: {"caption": "mock caption"} for p in job.args["paths"]}, "load_s": 0.0,
                "gpu_memory_mib": None, "gpu_memory_released": None}


class ContractHarness:
    def __init__(self, cfg, run_dir: Path, recorder) -> None:
        self.cfg, self.recorder = cfg, recorder
        self.registry = TokenRegistry(recorder)
        self.services = RunServices(socket_dir_for(Path(run_dir) / "contract-harness"))
        self.queue = JobQueue(recorder, threading.Lock(), wait_cap_s=5.0)
        self.queue.register(_MockCaption())

    @property
    def socket_dir(self) -> Path:
        return self.services.socket_dir

    def start(self) -> None:
        kit, mcp = ToolKit(self.registry, self.recorder), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, self.queue)
        register_caption_tool(mcp, kit, self.queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_skill_tool(mcp, kit)
        gateway = create_gateway_app(registry=self.registry, store=CallStore(self.recorder),
                                     allowed_models={MOCK_MODEL}, upstream=None,
                                     mocks=MockBook.default())
        self.services.start(gateway, build_tool_app(mcp))

    def stop(self) -> None:
        self.services.stop()
        self.queue.shutdown()


def verify_contract(*, cfg, run_dir: Path, run_id: str, repo, commit: str, harness: ContractHarness,
                    recorder, node: str, attempt: int, contexts: dict[str, dict],
                    import_timeout_s: float | None = None,
                    smoke_timeout_s: float | None = None, runner=run_container) -> ContractReport:
    """`contexts`: per entry point, the dry-run context its smoke run receives (agent_phase.smoke_contexts)."""
    import_timeout_s = import_timeout_s or float(cfg.get("timeouts.contract_import_s"))
    smoke_timeout_s = smoke_timeout_s or 4 * float(cfg.get("timeouts.contract_smoke_s"))
    report = ContractReport(ok=False)
    work = Path(run_dir) / "nodes" / node / "contract" / f"attempt-{attempt}"
    shutil.rmtree(work, ignore_errors=True)
    code = work / "agent"

    def finish(ok: bool) -> ContractReport:
        report.ok = ok
        recorder.event("contract.report", node=node, phase="contract", attempt=attempt,
                       component="contract", ok=ok, payload=report.to_retry())
        return report

    try:
        repo.checkout(commit, code)
    except CheckoutError as exc:
        report.steps.append(ContractStep("checkout", False, str(exc)))
        return finish(False)

    reqs_path = code / "agent" / "requirements.txt"
    try:
        reqs = reqs_path.read_text(encoding="utf-8") if reqs_path.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:      # e.g. a directory, or not UTF-8
        report.steps.append(ContractStep("build", False, f"cannot read agent/requirements.txt: {exc}"))
        return finish(False)
    try:
        report.image = ensure_image(cfg, reqs, recorder=recorder, node=node)
        report.steps.append(ContractStep("build", True))
    except ImageBuildError as exc:
        report.steps.append(ContractStep("build", False, str(exc)))
        return finish(False)

    entry = code / "agent" / "entry.py"
    try:
        entry_source = entry.read_text(encoding="utf-8") if entry.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:
        step = ContractStep("static", False, f"cannot read agent/entry.py as UTF-8 text: {exc}")
    else:
        step = static_check(entry_source)
    report.steps.append(step)
    if not step.ok:
        return finish(False)
    step = prompt_check(code)
    report.steps.append(step)
    if not step.ok:
        return finish(False)

    def run(label: str, command: list[str], timeout_s: float, context: dict | None,
           with_liveness: bool = False) -> tuple:
        run_dir_l = work / label.replace(":", "_")
        ws, ctx_dir = run_dir_l / "workspace", run_dir_l / "context"
        staging = Path(run_dir) / "staging" / node / f"contract-{attempt}-{label.replace(':', '_')}"
        for d in (ws, ctx_dir, staging):
            d.mkdir(parents=True, exist_ok=True)
        if context is not None:
            (ctx_dir / "context.json").write_text(json.dumps(context))
        caller = harness.registry.issue(node=node, phase="contract", attempt=attempt,
                                        workspace_host=ws, staging_host=staging, mock_script="smoke")
        # issue() redacted the token only on harness.recorder, which is not `recorder`.
        recorder.add_redaction(caller.token)
        # The two smoke runs get a liveness; the import probe does not.
        liveness = Liveness.from_config(cfg, float(cfg.get("timeouts.contract_smoke_s")),
                                        signals=[lambda: tree_mark(ws),
                                                 lambda: tree_mark(recorder.events_path(node))]) \
            if with_liveness else None
        try:
            result = runner(image=report.image, name=container_name(run_id, node, "contract", attempt),
                            mounts=Mounts(agent=code, workspace=ws, staging=staging, context=ctx_dir,
                                          contract=cfg.repo_root / "contract",
                                          sockets=harness.socket_dir, agent_readonly=True),
                            command=command, env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": MOCK_MODEL,
                                                  "AR_CONTEXT_WINDOW": str(cfg.get("agents.context_window_tokens")),
                                                  "AR_COMPACT_AT": str(cfg.get("agents.compact_at"))},
                            cpus=4, memory_gb=8, timeout_s=timeout_s, liveness=liveness, recorder=recorder,
                            node=node, phase="contract", attempt=attempt,
                            network="none")         # verification stays offline and deterministic
        finally:
            harness.registry.revoke(caller.token)
            harness.queue.cancel_for_token(caller.token)   # as in the phase runner
        return result, ws

    (Path(run_dir) / "store").mkdir(exist_ok=True)
    probe = ("import sys; sys.path.insert(0, '/agent'); import agent.entry as e; "
             "assert callable(e.edit_self) and callable(e.improve_recipe)")
    result, _ = run("import", ["python", "-c", probe], import_timeout_s, None)
    if result.timed_out or result.exit_code != 0:
        why = f"timed out after {import_timeout_s:.0f}s" if result.timed_out else result.stderr[-6000:]
        report.steps.append(ContractStep("import", False, why))
        return finish(False)
    report.steps.append(ContractStep("import", True))

    for kind in ENTRY_POINTS:
        result, ws = run(f"smoke:{kind}", ["python", "-m", "ar_contract.run", kind], smoke_timeout_s,
                         contexts[kind], with_liveness=True)
        name = f"smoke:{kind}"
        if result.timed_out:
            report.steps.append(ContractStep(name, False, f"timed out after {smoke_timeout_s:.0f}s"))
            return finish(False)
        # Never follows a symlink or opens a FIFO the agent planted there.
        body = read_result(ws / "result.json") or {"ok": False, "error": "no result.json"}
        if result.exit_code != 0 or not body.get("ok"):
            tb = body.get("traceback")
            tb_tail = f"\n{tb[-6000:]}" if isinstance(tb, str) and tb else ""
            report.steps.append(ContractStep(
                name, False, f"{body.get('error')}\n{result.stderr[-6000:]}{tb_tail}"))
            return finish(False)
        report.steps.append(ContractStep(name, True))
    return finish(True)
