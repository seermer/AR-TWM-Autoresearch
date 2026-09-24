"""hf_search / hf_download (spec 10). The kernel downloads; the container has no network."""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub.errors import HfHubHTTPError
from mcp.server.mcpserver import Context

from .context import STAGING
from .server import ToolError

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _move_into(src: Path, base: Path, rel: str) -> None:
    """Move `src` to base/rel without following any link under `base`: each directory is opened
    with O_NOFOLLOW and the move is relative to that directory's fd, so a link the agent plants
    or swaps in (it owns staging, and its container runs meanwhile) raises instead of redirecting."""
    *dirs, name = PurePosixPath(rel).parts
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in dirs:
            try:
                os.mkdir(part, dir_fd=fd)
            except FileExistsError:
                pass
            sub = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = sub
        os.replace(src, name, dst_dir_fd=fd)
    finally:
        os.close(fd)


def _license(obj) -> str | None:
    card = getattr(obj, "card_data", None) or getattr(obj, "cardData", None) or {}
    if isinstance(card, dict) and card.get("license"):
        lic = card["license"]
        return ",".join(lic) if isinstance(lic, list) else str(lic)
    for tag in getattr(obj, "tags", None) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return None


class HfTools:
    def __init__(self, cfg, private_dir: Path, api=None, snapshot=None) -> None:
        # Downloads land here first: a kernel-only dir (under run_dir) never mounted into a container.
        self.private_dir = Path(private_dir)
        if api is None or snapshot is None:
            import huggingface_hub
            api = api or huggingface_hub.HfApi()
            snapshot = snapshot or huggingface_hub.snapshot_download
        self.api, self.snapshot = api, snapshot
        self.cap = int(cfg.get("tools.hf_download_max_bytes"))

    def search(self, caller, query: str, kind: str = "dataset", limit: int = 20) -> list[dict]:
        if kind != "dataset":
            raise ToolError("only kind='dataset' is supported: models are not training data")
        return [{"id": d.id, "license": _license(d), "tags": list(d.tags or [])[:40],
                 "downloads": getattr(d, "downloads", None),
                 "last_modified": str(getattr(d, "last_modified", None))}
                for d in self.api.list_datasets(search=query, limit=min(int(limit), 100), full=True)]

    def download(self, caller, repo: str, revision: str, patterns: list[str],
                 max_bytes: int | None = None) -> dict:
        if not REPO_RE.match(repo or ""):
            raise ToolError(f"invalid dataset repo id {repo!r}")
        try:
            info = self.api.dataset_info(repo, revision=revision, files_metadata=True)
        except HfHubHTTPError as exc:
            raise ToolError(str(exc)) from None
        files = sorted(s.rfilename for s in info.siblings
                       if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))
        if not files:
            raise ToolError(f"no files match {patterns} in {repo}@{revision}")
        for f in files:                     # repo-reported names must stay inside dest (join below)
            p = PurePosixPath(f)
            if p.is_absolute() or ".." in p.parts or "\\" in f:
                raise ToolError(f"repo lists unsafe file path {f!r}; refusing to download")
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings}
        total = sum(sizes[f] for f in files)
        cap = min(self.cap, int(max_bytes)) if max_bytes else self.cap
        if total > cap:
            raise ToolError(f"{len(files)} files total {total} bytes, over the {cap}-byte cap; "
                            f"narrow the patterns")
        # The agent owns staging. Refuse early (before a large transfer) if a planted link already
        # sends the destination outside it; _move_into below is the guard that cannot be raced.
        rel, staging = f"hf/{repo.replace('/', '__')}/{info.sha}", caller.staging_host.resolve()
        if not (staging / rel).resolve().is_relative_to(staging):
            raise ToolError("download destination resolves outside staging; refusing")
        tmp = self.private_dir / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        try:
            try:
                self.snapshot(repo_id=repo, repo_type="dataset", revision=info.sha,
                              allow_patterns=files, local_dir=str(tmp))
            except HfHubHTTPError as exc:
                raise ToolError(str(exc)) from None
            for f in files:
                try:
                    _move_into(tmp / f, staging, f"{rel}/{f}")
                except OSError as exc:
                    raise ToolError(f"cannot place {f} in staging (a link or non-directory in the way?): "
                                    f"{exc}") from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {"repo": repo, "revision": info.sha, "bytes": total, "license": _license(info),
                "files": [str(STAGING / rel / f) for f in files],
                "provenance": {"kind": "hf_dataset", "repo": repo, "revision": info.sha,
                               "files": files}}


def register_hf_tools(mcp, kit, tools: HfTools) -> None:
    @mcp.tool(name="hf_search", description="Search Hugging Face datasets. Returns ids, license and tags.")
    async def hf_search(query: str, ctx: Context, kind: str = "dataset", limit: int = 20) -> list[dict]:
        return await kit.call(ctx, "hf_search", {"query": query, "kind": kind, "limit": limit},
                              lambda c: tools.search(c, query, kind, limit))

    @mcp.tool(name="hf_download", description="Download files matching glob patterns from a dataset "
              "repo into /workspace/staging/hf/. The revision is pinned to a commit SHA; the result "
              "includes a ready-made provenance record for data_ingest.")
    async def hf_download(repo: str, revision: str, patterns: list[str], ctx: Context,
                          max_bytes: int | None = None) -> dict[str, Any]:
        return await kit.call(ctx, "hf_download",
                              {"repo": repo, "revision": revision, "patterns": patterns,
                               "max_bytes": max_bytes},
                              lambda c: tools.download(c, repo, revision, patterns, max_bytes))
