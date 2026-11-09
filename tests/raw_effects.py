"""The kernel's raw state changes, thread hops and silent `except` clauses, found by parsing it (for
test_raw_effects.py). A site is "<file>::<function>: <call>"."""
from __future__ import annotations

import ast
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1] / "kernel" / "ar_kernel"
SKIPPED = ("bridges/", "doctor.py")                          # black-box scripts in other envs; a read-only command
EFFECTS_ALLOWED = ("subproc.py", "fs.py", "archive/", "vcs/", "telemetry/")
HOPS_ALLOWED = ("telemetry/",)

_OS = {"remove", "unlink", "replace", "rename", "chmod", "link", "kill", "killpg", "rmdir", "makedirs", "mkdir",
       "symlink", "truncate"}
_SHUTIL = {"rmtree", "copy", "copy2", "copyfile", "copytree", "copyfileobj", "move"}
_PATH = {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch", "chmod", "symlink_to", "hardlink_to"}
_ONE_ARG = {"rename", "replace"}                               # Path's; str.replace takes two
_SQL = ("INSERT", "UPDATE", "DELETE", "REPLACE")
_RECORDS = {"error", "log", "alert", "decision", "span", "deferred_decision"}


def _name(node: ast.AST) -> str:
    return ast.unparse(node)


def _write_mode(call: ast.Call, first_mode_arg: int) -> bool:
    mode = next((k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None and len(call.args) > first_mode_arg:
        mode = call.args[first_mode_arg]
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax+")


def _effect(call: ast.Call, rel: str) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        owner, attr = _name(f.value), f.attr
        if owner in ("fs", "subproc"):                           # the recording functions themselves
            return None
        if owner == "os" and attr in _OS or owner == "shutil" and attr in _SHUTIL:
            return f"{owner}.{attr}"
        if owner == "subprocess":
            return f"subprocess.{attr}"
        if owner == "tempfile" and attr == "NamedTemporaryFile" and any(
                k.arg == "delete" and isinstance(k.value, ast.Constant) and k.value.value is False for k in call.keywords):
            return "tempfile.NamedTemporaryFile(delete=False)"
        if attr in _PATH or attr in _ONE_ARG and len(call.args) == 1 and not call.keywords:
            return f".{attr}"
        if attr == "open" and _write_mode(call, 0):
            return ".open(write)"
        if attr in ("execute", "executemany") and not rel.startswith("archive/") and call.args:
            sql = call.args[0]
            text = sql.value if isinstance(sql, ast.Constant) else sql.values[0].value if isinstance(
                sql, ast.JoinedStr) and sql.values and isinstance(sql.values[0], ast.Constant) else ""
            if isinstance(text, str) and text.lstrip().upper().startswith(_SQL):
                return f"{attr}(SQL write)"
    elif isinstance(f, ast.Name) and f.id == "open" and _write_mode(call, 1):
        return "open(write)"
    return None


def _hop(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        if _name(f) == "threading.Thread":
            return "threading.Thread"
        if f.attr in ("submit", "map", "run_in_executor") and _name(f.value) != "telemetry":
            return f".{f.attr}"
    return None


def _records(handler: ast.ExceptHandler) -> bool:
    """The handler re-raises or writes a telemetry record."""
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _RECORDS \
                and _name(node.func.value) == "telemetry":
            return True
    return False


class _Scan(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel, self.stack = rel, []
        self.effects, self.hops, self.silent = set(), set(), set()

    def _site(self, what: str) -> str:
        return f"{self.rel}::{'.'.join(self.stack) or '<module>'}: {what}"

    def _scope(self, node) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_Call(self, node: ast.Call) -> None:
        if not self.rel.startswith(EFFECTS_ALLOWED) and (what := _effect(node, self.rel)):
            self.effects.add(self._site(what))
        if not self.rel.startswith(HOPS_ALLOWED) and (what := _hop(node)):
            self.hops.add(self._site(what))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if not _records(node):
            self.silent.add(self._site(f"except {_name(node.type) if node.type else ''}".rstrip()))
        self.generic_visit(node)


def scan() -> tuple[set[str], set[str], set[str]]:
    """(raw effects, raw thread hops, silent except clauses) of the kernel."""
    effects, hops, silent = set(), set(), set()
    for path in sorted(KERNEL.rglob("*.py")):
        rel = str(path.relative_to(KERNEL))
        if rel.startswith(SKIPPED):
            continue
        s = _Scan(rel)
        s.visit(ast.parse(path.read_text(), filename=str(path)))
        effects |= s.effects
        hops |= s.hops
        silent |= s.silent
    return effects, hops, silent
