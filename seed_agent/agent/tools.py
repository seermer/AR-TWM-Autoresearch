"""The agent's tools: file/shell tools bound to one root, arXiv search and reading,
the kernel tools (the MCP tool server) as LangChain tools, and submit_<x> result tools.
Captioning is the kernel's caption_videos GPU job."""
from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arxiv
from langchain_core.tools import StructuredTool, ToolException, tool
from pydantic import BaseModel

# Past a limit, show much less than the limit: an output that large is better searched than read.
READ_LIMIT, READ_SHOWN = 100_000, 20_000          # read_file: chars, head shown
OUTPUT_LIMIT, OUTPUT_SHOWN = 20_000, 6_000        # run_command: chars, tail shown


# ---- files and shell (root is /agent for edit_self, /workspace for improve_recipe) ----

def resolve_read(root: str, path: str) -> Path:
    """Reads may go anywhere in the container: an absolute path as is, a relative one under root."""
    return Path(path) if Path(path).is_absolute() else Path(root) / path


def resolve_inside(root: str, path: str) -> Path:
    """Writes go under root or under the workspace, which holds every role's scratch folder."""
    workspace = os.environ.get("AR_WORKSPACE", "/workspace")
    target = (Path(root).resolve() / path).resolve()
    if not any(target.is_relative_to(Path(base).resolve()) for base in (root, workspace)):
        raise ValueError(f"{path} is outside {root} and {workspace}")
    return target


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old text must appear exactly once, found {count} matches")
    return text.replace(old, new, 1)


_path_locks: dict[Path, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _path_locks_guard:
        return _path_locks.setdefault(path, threading.Lock())


def _write_atomic(target: Path, text: str) -> None:
    """Temp file + rename: a reader or a parallel call never sees a half-written file."""
    tmp = target.with_name(f".{target.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


_command_ids = itertools.count(1)


def _keep_output(command: str, status: str, output: str) -> Path:
    """Save a command's full output under /workspace/tool_output (never under /agent, whose
    files are committed as agent code), so the run keeps what the model only saw the tail of."""
    folder = Path(os.environ.get("AR_WORKSPACE", "/workspace")) / "tool_output"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"run_command-{time.strftime('%Y%m%d-%H%M%S')}-{next(_command_ids):04d}.log"
    log.write_text(f"$ {command}\n{status}\n{output}", encoding="utf-8")
    return log


def read_text(path: Path, offset: int = 0, limit: int | None = None) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    part = lines[offset:offset + limit if limit else None]
    text = "".join(part)
    if len(text) > READ_LIMIT:
        return (f"[Large file: {len(lines)} lines, {sum(map(len, lines))} chars. Only the first {READ_SHOWN} chars "
                f"of the requested lines are shown. Read fewer lines with offset and limit, or search the file.]\n"
                + text[:READ_SHOWN])
    if offset or limit:
        return f"[lines {offset + 1}-{offset + len(part)} of {len(lines)}]\n{text}"
    return text


def make_file_tools(root: str) -> list:
    writable = " or ".join(dict.fromkeys([root, "/workspace"]))

    @tool(description=f"Read a text file (absolute, or relative to {root}). `offset` and `limit` select lines.")
    def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
        return read_text(resolve_read(root, path), offset, limit)

    @tool(description=f"List a directory (absolute, or relative to {root}); directories end with '/'.")
    def list_dir(path: str = ".") -> list[str]:
        return sorted(p.name + ("/" if p.is_dir() else "") for p in resolve_read(root, path).iterdir())

    @tool(description=f"Create or overwrite a text file (relative to {root}, or absolute under {writable}).")
    def write_file(path: str, content: str) -> str:
        target = resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(target):
            _write_atomic(target, content)
        return f"wrote {len(content)} chars to {path}"

    @tool(description="Replace one exact occurrence of `old_string` with `new_string` in a text file. "
                      "`old_string` must appear exactly once.")
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = resolve_inside(root, path)
        with _lock_for(target):     # parallel edits of one file are applied one after the other
            try:        # strict: writing back text with replaced undecodable bytes would corrupt the file
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path} is not UTF-8 text (byte {exc.start}); "
                                 f"change it with run_command instead") from None
            _write_atomic(target, replace_once(text, old_string, new_string))
        return f"edited {path}"

    @tool(description=f"Run a shell command in {root} (ffmpeg, ffprobe, python, ...). It is killed after "
                      "`timeout_s` seconds. Returns the exit code and the output tail.")
    def run_command(command: str, timeout_s: int = 600) -> str:
        # Its own process group, so a timeout also kills what the shell started (python, ffmpeg, ...).
        proc = subprocess.Popen(["bash", "-lc", command], cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=int(timeout_s))
            status = f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            status = f"timed out after {timeout_s}s"
        output = stdout + stderr
        log = _keep_output(command, status, output)
        if len(output) <= OUTPUT_LIMIT:
            return f"{status}\n{output}"
        return (f"{status}\n[Long output: {len(output)} chars. Only the last {OUTPUT_SHOWN} are shown; the full "
                f"output is in {log}.]\n{output[-OUTPUT_SHOWN:]}")

    return [read_file, list_dir, write_file, edit_file, run_command]


# ---- arXiv ----


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "autoresearcher-agent"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


@tool(description="Search arXiv with the arXiv API query syntax, for example 'all:\"world model\" AND cat:cs.CV' or "
                  "'ti:camera AND abs:\"video generation\"'. Returns id, title, first published date and abstract.")
def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    n = min(int(max_results), 50)
    search = arxiv.Search(query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    return [{"id": r.get_short_id(), "title": " ".join(r.title.split()),
             "published": r.published.strftime("%Y-%m-%d"), "abstract": " ".join(r.summary.split())}
            for r in arxiv.Client(page_size=n, num_retries=0).results(search)]


class _PaperText(HTMLParser):
    """The text of an arXiv HTML paper: math as its LaTeX source, headings collected on the side."""
    SKIP = {"script", "style", "nav", "header", "footer", "button"}
    BLOCK = {"p", "div", "section", "li", "tr", "figcaption", "table", "br", "h1", "h2", "h3", "h4", "h5", "h6"}
    HEADINGS = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.headings: list[str] = []
        self.skip = 0
        self.heading: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP or tag == "math":
            if tag == "math" and not self.skip:
                self.handle_data(f" ${dict(attrs).get('alttext', '')}$ ")
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag in self.HEADINGS and not self.skip:
            self.heading = []

    def handle_endtag(self, tag):
        if tag in self.SKIP or tag == "math":
            self.skip = max(0, self.skip - 1)
        elif tag in self.HEADINGS and self.heading is not None:
            self.headings.append(" ".join("".join(self.heading).split()))
            self.heading = None

    def handle_data(self, data):
        if self.skip:
            return
        self.parts.append(data)
        if self.heading is not None:
            self.heading.append(data)


@tool(description="Fetch an arXiv paper's full text (from its HTML version) and save it to /workspace/papers/<id>.txt. "
                  "Returns the file path, its length and the section headings; read the parts you need from the file.")
def arxiv_read(arxiv_id: str) -> dict:
    try:
        html = _fetch(f"https://arxiv.org/html/{arxiv_id}")
    except urllib.error.HTTPError as exc:
        raise ToolException(f"arXiv has no HTML version of {arxiv_id} ({exc.code}); download "
                            f"https://arxiv.org/pdf/{arxiv_id} with run_command and extract its text") from None
    parser = _PaperText()
    parser.feed(html)
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    path = Path(os.environ.get("AR_WORKSPACE", "/workspace")) / "papers" / f"{arxiv_id.replace('/', '_')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "chars": len(text), "sections": [h for h in parser.headings if h]}


def local_tools(root: str, papers: bool) -> list:
    """File and shell tools bound to `root`; `papers` adds arXiv search and reading."""
    return [*make_file_tools(root), *([arxiv_search, arxiv_read] if papers else [])]


# ---- kernel tools over one MCP session (MCP 2.x: is_error, structured_content, input_schema) ----

def result_text(result) -> str:
    """A kernel tool result as text: structured content JSON-encoded (MCP splits a list result
    into one text block per item), else the text blocks. A failed call raises ToolException,
    which the harness reports to the model."""
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if result.is_error:
        raise ToolException(text)
    return text if result.structured_content is None else json.dumps(result.structured_content)


async def kernel_tools(session) -> list[StructuredTool]:
    tools = []
    for spec in (await session.list_tools()).tools:
        async def call(_name: str = spec.name, **kwargs: Any) -> str:
            return result_text(await session.call_tool(_name, kwargs))
        tools.append(StructuredTool(name=spec.name, description=spec.description or "",
                                    args_schema=spec.input_schema, coroutine=call))
    return tools


# ---- result tools: a role finishes by calling submit_<x>; invalid arguments come back
# to the model as a tool error (the harness), so it can correct them ----

def submit_tool(name: str, description: str, schema: type[BaseModel],
                check=None) -> tuple[StructuredTool, SimpleNamespace]:
    """`check`: an async function that raises ToolException when the submission cannot be accepted yet."""
    box = SimpleNamespace(value=None, name=name)       # box.value: the validated submission, once submitted

    async def submit(**kwargs) -> str:
        value = schema.model_validate(kwargs)
        if check is not None:
            await check(value)
        box.value = value
        return "submitted"

    # return_direct: an accepted submission ends the role's run (harness difference 4).
    return StructuredTool(name=name, description=description, args_schema=schema, coroutine=submit,
                          return_direct=True), box
