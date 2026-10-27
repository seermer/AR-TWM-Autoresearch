"""The run folder on a private Hugging Face dataset repo: after every phase, the new and changed files go up
in one commit of tars (`uploads/<seq>/<part>.tar`); `fetch` rebuilds the folder from them, for the panel elsewhere."""
from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from .guards import alert
from .subproc import cache_dir

# Not needed to view the run: caches, partial downloads, raw downloads, the agents' own installs.
SKIP_DIRS = ("cache", "tmp", "hf_tmp", "sockets", "control/pgids", "staging/*/*/hf", "telemetry/files_hf",
             "nodes/*/attempts/*/workspace/.home", "nodes/*/attempts/*/dataset_cache")
# archive.db goes up as a consistent copy, never the live file and its WAL.
SKIP_FILES = ("archive.db", "archive.db-wal", "archive.db-shm", "control/loop.pid", "control/hf_upload.json",
              "control/hf_fetch.json")
MANIFEST = ".hf_manifest.json"          # every file of the folder at that upload
PART_BYTES = 10 * 2**30                 # well under Hugging Face's per-file limit


def _matches(rel: str, patterns) -> bool:
    parts = rel.split("/")
    return any(len(p.split("/")) == len(parts) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, p.split("/")))
               for p in patterns)


def run_files(run_dir: Path) -> dict[str, list[int]]:
    """Relative path -> [size, mtime_ns] of every regular file that is uploaded."""
    out = {}
    for root, dirs, files in os.walk(run_dir):
        base = Path(root).relative_to(run_dir).as_posix()
        rel = lambda name: name if base == "." else f"{base}/{name}"
        dirs[:] = [d for d in dirs if not _matches(rel(d), SKIP_DIRS)]
        for name in files:
            path = Path(root) / name
            if not _matches(rel(name), SKIP_FILES) and path.is_file() and not path.is_symlink():
                st = path.stat()
                out[rel(name)] = [st.st_size, st.st_mtime_ns]
    return out


class HfUpload:
    def __init__(self, run_dir: Path, recorder, token: str, api=None) -> None:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.run_dir, self.recorder, self.api = Path(run_dir), recorder, api
        self.state_path = self.run_dir / "control" / "hf_upload.json"

    def __call__(self, node: str, phase: str, attempt: int) -> None:
        """A failed upload is an alert, not an error: its files go up with the next one."""
        try:
            self.upload(node, phase, attempt)
        except Exception as exc:                                  # noqa: BLE001 -- reported, the run goes on
            alert(self.recorder, "hf_upload_failed", f"the upload after {node} {phase} failed: "
                  f"{type(exc).__name__}: {exc}", level="warning", node_id=node)

    def upload(self, node: str, phase: str, attempt: int) -> None:
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"seq": 0, "files": {}}
        repo = state.get("repo")
        if repo is None:
            repo = f"{self.api.whoami()['name']}/{self.run_dir.name}"
            self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        files = run_files(self.run_dir)
        changed = [rel for rel, stamp in files.items() if state["files"].get(rel) != stamp]
        seq = state["seq"] + 1
        parts, filled = [[]], 0
        for rel in changed:
            if parts[-1] and filled + files[rel][0] > PART_BYTES:
                parts.append([])
                filled = 0
            parts[-1].append(rel)
            filled += files[rel][0]
        tmp_root = cache_dir() / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            db = Path(tmp) / "archive.db"
            with contextlib.closing(sqlite3.connect(f"{(self.run_dir / 'archive.db').resolve().as_uri()}?mode=ro", uri=True)) as src, \
                    contextlib.closing(sqlite3.connect(db)) as dst:
                src.backup(dst)
            (Path(tmp) / MANIFEST).write_text(json.dumps(sorted(files)))
            ops = []
            for i, part in enumerate(parts):
                tar = Path(tmp) / f"{i:04d}.tar"
                with tarfile.open(tar, "w") as out:
                    for rel in part:
                        out.add(self.run_dir / rel, arcname=rel, recursive=False)
                    if i == len(parts) - 1:             # extracted last
                        out.add(db, arcname="archive.db")
                        out.add(Path(tmp) / MANIFEST, arcname=MANIFEST)
                ops.append(CommitOperationAdd(path_in_repo=f"uploads/{seq:05d}/{tar.name}", path_or_fileobj=str(tar)))
            size = sum(Path(op.path_or_fileobj).stat().st_size for op in ops)
            disable_progress_bars()                     # they would flood the console log
            try:
                self.api.create_commit(repo, ops, repo_type="dataset",
                                       commit_message=f"{node} {phase}{f' {attempt}' if attempt else ''}")
            finally:
                enable_progress_bars()
        tmp_state = self.state_path.with_suffix(".tmp")
        tmp_state.write_text(json.dumps({"repo": repo, "seq": seq, "files": files}))
        tmp_state.replace(self.state_path)
        self.recorder.event("hf_upload", node=node, phase=phase, attempt=attempt,
                            payload={"repo": repo, "upload": seq, "parts": len(ops), "files": len(changed), "bytes": size})


def fetch(run_dir: Path, repo: str | None, token: str, api=None, download=None) -> str:
    """Extracts the uploads not fetched yet into `run_dir`, then removes what the last upload no longer has."""
    if api is None or download is None:
        import huggingface_hub
        api = api or huggingface_hub.HfApi(token=token)
        download = download or huggingface_hub.hf_hub_download
    run_dir = Path(run_dir)
    repo = repo or f"{api.whoami()['name']}/{run_dir.name}"
    state_path = run_dir / "control" / "hf_fetch.json"
    if run_dir.exists() and not state_path.exists():
        raise FileExistsError(f"{run_dir} exists and was not made by fetch; it is never overwritten")
    last = json.loads(state_path.read_text())["last"] if state_path.exists() else ""
    names = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                   if f.startswith("uploads/") and f > last)
    if not names:
        return f"{run_dir} is up to date with {repo}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = cache_dir() / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            path = download(repo_id=repo, filename=name, repo_type="dataset", local_dir=tmp, token=token)
            with tarfile.open(path) as tar:
                tar.extractall(run_dir, filter="data")
    keep = set(json.loads((run_dir / MANIFEST).read_text()))
    (run_dir / MANIFEST).unlink()
    for rel in set(run_files(run_dir)) - keep:
        (run_dir / rel).unlink()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"repo": repo, "last": names[-1]}))
    return f"fetched {len(names)} uploads of {repo} into {run_dir}"
