"""hf_search / hf_download (spec 10). The kernel downloads; the container has no network."""
from __future__ import annotations

import fnmatch
import re
from typing import Any

from mcp.server.mcpserver import Context

from .context import to_container
from .server import ToolError

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    def __init__(self, cfg, api=None, snapshot=None) -> None:
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
        info = self.api.dataset_info(repo, revision=revision, files_metadata=True)
        files = sorted(s.rfilename for s in info.siblings
                       if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))
        if not files:
            raise ToolError(f"no files match {patterns} in {repo}@{revision}")
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings}
        total = sum(sizes[f] for f in files)
        cap = min(self.cap, int(max_bytes)) if max_bytes else self.cap
        if total > cap:
            raise ToolError(f"{len(files)} files total {total} bytes, over the {cap}-byte cap; "
                            f"narrow the patterns")
        dest = caller.staging_host / "hf" / repo.replace("/", "__") / info.sha
        self.snapshot(repo_id=repo, repo_type="dataset", revision=info.sha,
                      allow_patterns=files, local_dir=str(dest))
        return {"repo": repo, "revision": info.sha, "bytes": total, "license": _license(info),
                "files": [to_container(caller, dest / f) for f in files],
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
