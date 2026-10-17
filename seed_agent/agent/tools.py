"""The agent's tools."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import arxiv
from langchain_core.tools import StructuredTool, ToolException, tool


# ---- files and shell ----

def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old text must appear exactly once, found {count} matches")
    return text.replace(old, new, 1)


def _write_atomic(target: Path, text: str) -> None:
    """Temp file + rename."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def make_file_tools(cwd: str) -> list:
    def resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else Path(cwd) / path

    @tool(description=f"Create or overwrite a text file (absolute, or relative to {cwd}).")
    def write_file(path: str, content: str) -> str:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, content)
        return f"wrote {len(content)} chars to {target}"

    @tool(description=f"Replace one exact occurrence of `old_string` with `new_string` in a text file (absolute, or "
                      f"relative to {cwd}). `old_string` must appear exactly once.")
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = resolve(path)
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path} is not UTF-8 text (byte {exc.start}); change it with run_command instead") from None
        _write_atomic(target, replace_once(text, old_string, new_string))
        return f"edited {target}"

    @tool(description=f"Run a shell command in {cwd} (cat, ls, grep, ffmpeg, python, ...). It is killed after "
                      "`timeout_s` seconds. Returns the exit code, then the output.")
    def run_command(command: str, timeout_s: int = 600) -> str:
        proc = subprocess.Popen(["bash", "-lc", command], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=int(timeout_s))
            status = f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            status = f"timed out after {timeout_s}s"
        return f"{status}\n{stdout + stderr}"

    return [write_file, edit_file, run_command]


# ---- arXiv ----


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "autoresearcher-agent"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


@tool(description="Search arXiv with the arXiv API query syntax, for example 'all:\"world model\" AND cat:cs.CV' or "
                  "'ti:camera AND abs:\"video generation\"'. Returns id, title, first published date and abstract.")
def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    if int(max_results) < 1:
        raise ToolException("max_results must be at least 1")
    n = min(int(max_results), 50)
    search = arxiv.Search(query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    return [{"id": r.get_short_id(), "title": " ".join(r.title.split()),
             "published": r.published.strftime("%Y-%m-%d"), "abstract": " ".join(r.summary.split())}
            for r in arxiv.Client(page_size=n, num_retries=0).results(search)]


class _PaperText(HTMLParser):
    """An arXiv HTML paper as text."""
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


def local_tools(cwd: str, papers: bool) -> list:
    """File and shell tools in `cwd`; `papers` adds arXiv search and reading."""
    return [*make_file_tools(cwd), *([arxiv_search, arxiv_read] if papers else [])]


# ---- kernel tools ----

def result_text(result) -> str:
    """A kernel tool result as text; an error raises ToolException."""
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if result.is_error:
        raise ToolException(text)
    return text if result.structured_content is None else json.dumps(result.structured_content)


async def kernel_tools(session) -> list[StructuredTool]:
    """Every kernel tool but snapshot."""
    tools = []
    for spec in (await session.list_tools()).tools:
        if spec.name == "snapshot":
            continue
        async def call(_name: str = spec.name, **kwargs: Any) -> str:
            return result_text(await session.call_tool(_name, kwargs))
        tools.append(StructuredTool(name=spec.name, description=spec.description or "",
                                    args_schema=spec.input_schema, coroutine=call, metadata={"parallel": True}))
    return tools
