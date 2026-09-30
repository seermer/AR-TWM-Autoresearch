"""The agent's tools: file/shell tools bound to one root, arXiv search and reading, the timed-prompt
helper, the kernel tools (the MCP tool server) as LangChain tools, and submit_<x> result tools.
Captioning is the kernel's caption_videos GPU job."""
from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree
from types import SimpleNamespace
from typing import Any

from langchain_core.tools import StructuredTool, ToolException, tool
from pydantic import BaseModel

# Past a limit, show much less than the limit: an output that large is better searched than read.
READ_LIMIT, READ_SHOWN = 100_000, 20_000          # read_file: chars, head shown
OUTPUT_LIMIT, OUTPUT_SHOWN = 20_000, 6_000        # run_command: chars, tail shown
KEEP_BYTES = 1 << 20               # discarded_changes restores pre-existing files up to this size
FIRST_ROUND_END = 25 / 24          # the 25 history frames
ROUND = 32 / 24                    # one rollout round


# ---- files and shell (root is /agent for edit_self, /workspace for improve_recipe) ----

def resolve_read(root: str, path: str) -> Path:
    """Reads may go anywhere in the container: an absolute path as is, a relative one under root."""
    return Path(path) if Path(path).is_absolute() else Path(root) / path


def resolve_inside(root: str, path: str) -> Path:
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{path} is outside {root}")
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
    @tool
    def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file (absolute, or relative to the tool root). `offset` and `limit` select lines."""
        return read_text(resolve_read(root, path), offset, limit)

    @tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory (absolute, or relative to the tool root); directories end with '/'."""
        return sorted(p.name + ("/" if p.is_dir() else "") for p in resolve_read(root, path).iterdir())

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file (path relative to the tool root)."""
        target = resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(target):
            _write_atomic(target, content)
        return f"wrote {len(content)} chars to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a text file. `old` must appear exactly once."""
        target = resolve_inside(root, path)
        with _lock_for(target):     # parallel edits of one file are applied one after the other
            try:        # strict: writing back text with replaced undecodable bytes would corrupt the file
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path} is not UTF-8 text (byte {exc.start}); "
                                 f"change it with run_command instead") from None
            _write_atomic(target, replace_once(text, old, new))
        return f"edited {path}"

    @tool
    def run_command(command: str, timeout_s: int = 600) -> str:
        """Run a shell command in the tool root (ffmpeg, ffprobe, python, ...). Returns the exit code and the output tail."""
        # Its own process group, so a timeout also kills what the shell started (python, ffmpeg, ...).
        proc = subprocess.Popen(["bash", "-lc", command], cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=min(int(timeout_s), 3600))
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


def _entries(root: str, keep: set[Path]) -> tuple[set[Path], set[Path]]:
    """(directories, everything else) under root, never following a link, skipping the `keep` dirs."""
    dirs, others = set(), set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if Path(dirpath) / name not in keep]
        for name in dirnames:
            path = Path(dirpath) / name
            (others if path.is_symlink() else dirs).add(path)
        others.update(Path(dirpath) / name for name in filenames)
    return dirs, others


@contextmanager
def discarded_changes(*roots: str, keep: tuple[str, ...] = ()):
    """Undo, after the block, what it changed under roots: new files and directories are removed, and
    changed or deleted files and links come back. The `keep` directories (logs) are left alone. A
    pre-existing file over KEEP_BYTES (a video) is not copied, so it is left as the block left it."""
    skip = {Path(k) for k in keep}
    kept: dict[Path, tuple] = {}
    dirs: set[Path] = set()
    for root in roots:
        root_dirs, others = _entries(root, skip)
        dirs |= root_dirs
        for path in others:
            if path.is_symlink():
                kept[path] = ("link", os.readlink(path))
            elif path.is_file() and path.stat().st_size <= KEEP_BYTES:
                kept[path] = ("file", path.read_bytes())
            else:
                kept[path] = ("large", None)
    try:
        yield
    finally:
        for root in roots:
            root_dirs, others = _entries(root, skip)
            for path in sorted(others - set(kept)):
                path.unlink(missing_ok=True)
            for path in sorted(root_dirs - dirs, reverse=True):         # children before parents
                shutil.rmtree(path, ignore_errors=True)
        for path in sorted(dirs):
            path.mkdir(parents=True, exist_ok=True)
        for path, (kind, value) in kept.items():
            if kind == "link" and not (path.is_symlink() and os.readlink(path) == value):
                path.unlink(missing_ok=True)
                path.symlink_to(value)
            elif kind == "file" and (path.is_symlink() or not path.is_file() or path.read_bytes() != value):
                path.unlink(missing_ok=True)
                path.write_bytes(value)


# ---- arXiv ----

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "autoresearcher-agent"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


@tool
def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv with the arXiv API query syntax, for example 'all:"world model" AND cat:cs.CV' or
    'ti:camera AND abs:"video generation"'. Returns id, title, first published date and abstract."""
    url = ARXIV_API + "?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": min(int(max_results), 50), "sortBy": "relevance"})
    feed = ElementTree.fromstring(_fetch(url))
    return [{"id": entry.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1],
             "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
             "published": entry.findtext("a:published", "", ATOM)[:10],
             "abstract": " ".join(entry.findtext("a:summary", "", ATOM).split())}
            for entry in feed.findall("a:entry", ATOM)]


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


@tool
def arxiv_read(arxiv_id: str) -> dict:
    """Fetch an arXiv paper's full text (from its HTML version) and save it to /workspace/papers/<id>.txt.
    Returns the file path, its length and the section headings; read the parts you need from the file."""
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


ARXIV_TOOLS = [arxiv_search, arxiv_read]


# ---- timed prompts ----

def snap_segments(segments: list[dict], duration: float) -> list[dict]:
    """Snap internal boundaries to 25/24 + k*32/24 s (per_chunk rule), stretch the first and last segment
    to the clip's ends, close gaps, and drop any segment that collapses to zero length. Overlaps raise."""
    bounds, t = [], FIRST_ROUND_END
    while t < duration:
        bounds.append(t)
        t += ROUND
    ordered = sorted(segments, key=lambda s: s["time_range_s"][0])
    for a, b in zip(ordered, ordered[1:]):
        if b["time_range_s"][0] < a["time_range_s"][1] - 1e-9:
            raise ValueError(f"segments overlap: {a['prompt']!r} ends after {b['prompt']!r} starts")
    cuts = [0.0]
    for seg in ordered[:-1]:
        end = seg["time_range_s"][1]
        cuts.append(min(bounds, key=lambda b: abs(b - end)) if bounds else end)
    cuts.append(float(duration))
    return [{"time_range_s": [start, end], "prompt": seg["prompt"]}
            for seg, start, end in zip(ordered, cuts, cuts[1:]) if end - start > 1e-9]


@tool
def snap_timed_prompts(segments_json: str, duration: float) -> str:
    """Snap timed-prompt segment boundaries to rollout-round boundaries. Input and output: a JSON list of
    {"time_range_s": [start, end], "prompt": str}. Overlapping segments are refused, gaps are closed, and a
    segment that shrinks to nothing is dropped and named."""
    segments = json.loads(segments_json)
    snapped = snap_segments(segments, duration)
    kept = [s["prompt"] for s in snapped]
    dropped = [s["prompt"] for s in segments if s["prompt"] not in kept]
    out = json.dumps(snapped)
    return f"{out}\nDropped, shorter than a round after snapping: {json.dumps(dropped)}" if dropped else out


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

    return StructuredTool(name=name, description=description, args_schema=schema, coroutine=submit), box
