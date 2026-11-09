"""The run folder on private Hugging Face dataset repos: after every phase, the new and changed files go up as one
commit of tars (`<part>.tar`) to their own repo `<user>/<run>-<seq>`; `fetch` rebuilds the folder from them, in
order, for the panel elsewhere, and can delete the repos it holds."""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from . import fs, telemetry
from .subproc import cache_dir

# Not needed to view the run: caches, partial downloads, the agents' own installs, staging, the rebuilt view.
SKIP_DIRS = ("cache", "tmp", "hf_tmp", "sockets", "control/pgids", "staging", "view",
             "nodes/*/attempts/*/workspace/.home", "nodes/*/attempts/*/dataset_cache")
# Attempt folders go up only once a sandbox mounted them (`add_mounted`).
ATTEMPT_DIRS = tuple(f"nodes/*/attempts/*/{d}" for d in ("workspace", "agent", "code", "context"))
# archive.db goes up as a consistent copy, never the live file and its WAL.
SKIP_FILES = ("archive.db", "archive.db-wal", "archive.db-shm", "control/loop.pid", "control/hf_upload.json",
              "control/hf_fetch.json")
MANIFEST = ".hf_manifest.json"          # every file of the folder at that upload
PART_BYTES = 10 * 2**30                 # well under Hugging Face's per-file limit


def _matches(rel: str, patterns) -> bool:
    parts = rel.split("/")
    return any(len(p.split("/")) == len(parts) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, p.split("/")))
               for p in patterns)


def run_files(run_dir: Path, mounted=None) -> dict[str, list[int]]:
    """Relative path -> [size, mtime_ns] of every regular file that is uploaded. `mounted`: the folders whose
    attempt folders go up; None keeps every attempt folder (a fetched copy holds only what went up)."""
    def kept(rel: str) -> bool:
        if _matches(rel, SKIP_DIRS):
            return False
        return mounted is None or not _matches(rel, ATTEMPT_DIRS) or any(
            rel == m or rel.startswith(f"{m}/") for m in mounted)
    out = {}
    for root, dirs, files in os.walk(run_dir):
        base = Path(root).relative_to(run_dir).as_posix()
        rel = lambda name: name if base == "." else f"{base}/{name}"
        dirs[:] = [d for d in dirs if kept(rel(d))]
        for name in files:
            path = Path(root) / name
            if not _matches(rel(name), SKIP_FILES) and path.is_file() and not path.is_symlink():
                st = path.stat()
                out[rel(name)] = [st.st_size, st.st_mtime_ns]
    return out


class HfUpload:
    def __init__(self, run_dir: Path, token: str, api=None) -> None:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.run_dir, self.api = Path(run_dir), api
        self.state_path = self.run_dir / "control" / "hf_upload.json"

    def __call__(self, node: str, phase: str, attempt: int) -> None:
        """A failed upload is a warning, not an error: its files go up with the next one."""
        try:
            self.upload(node, phase, attempt)
        except Exception as exc:                                  # noqa: BLE001 -- reported, the run goes on
            message = f"the upload after {node} {phase} failed: {type(exc).__name__}: {exc}"
            telemetry.alert("hf.upload_failed", "warning", message)
            print(f"warning: {message}", file=sys.stderr)

    def upload(self, node: str, phase: str, attempt: int) -> None:
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"seq": 0, "files": {}}
        seq = state["seq"] + 1
        with telemetry.span("hf.upload", {"node": node, "phase": phase, "attempt": attempt, "seq": seq}) as sp:
            sp.output = self._upload(state, seq, f"{node} {phase}{f' {attempt}' if attempt else ''}")

    def _upload(self, state: dict, seq: int, message: str) -> dict:
        user = _call("whoami", {}, self.api.whoami, lambda r: {"name": r["name"]})["name"]
        repo = upload_repo(user, self.run_dir.name, seq)
        _call("create_repo", {"repo": repo}, lambda: self.api.create_repo(repo, repo_type="dataset", private=True,
                                                                         exist_ok=True),   # a retry reuses it
          lambda r: {"url": str(r)})
        stale = _tars(self.api, repo)                   # a retry after a commit that landed unnoticed replaces it
        files = run_files(self.run_dir, state.get("mounted", []))
        changed = [rel for rel, stamp in files.items() if state["files"].get(rel) != stamp]
        parts, filled = [[]], 0
        for rel in changed:
            if parts[-1] and filled + files[rel][0] > PART_BYTES:
                parts.append([])
                filled = 0
            parts[-1].append(rel)
            filled += files[rel][0]
        tmp_root = cache_dir() / "tmp"
        fs.mkdir(tmp_root)
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
                ops.append(CommitOperationAdd(path_in_repo=tar.name, path_or_fileobj=str(tar)))
            ops += [CommitOperationDelete(path_in_repo=p) for p in stale if p not in {op.path_in_repo for op in ops}]
            tars = {op.path_in_repo: os.path.getsize(op.path_or_fileobj) for op in ops
                    if isinstance(op, CommitOperationAdd)}
            disable_progress_bars()                     # they would flood the console log
            try:
                _call("create_commit", {"repo": repo, "message": message, "add": tars,
                                        "delete": [op.path_in_repo for op in ops if isinstance(op, CommitOperationDelete)]},
                      lambda: self.api.create_commit(repo, ops, repo_type="dataset", commit_message=message), str)
            finally:
                enable_progress_bars()
        fs.write_text(self.state_path, json.dumps({**state, "seq": seq, "files": files}), atomic=True, sync=True)
        return {"repo": repo, "files": [[rel, files[rel][0]] for rel in changed]}


def add_mounted(run_dir: Path, folders) -> None:
    """Folders a sandbox mounts read-only, kept in the upload state for the next upload to send."""
    path = Path(run_dir) / "control" / "hf_upload.json"
    state = json.loads(path.read_text()) if path.exists() else {"seq": 0, "files": {}}
    mounted = sorted({*state.get("mounted", []), *(Path(f).relative_to(run_dir).as_posix() for f in folders)})
    if mounted != state.get("mounted", []):
        fs.mkdir(path.parent)
        fs.write_text(path, json.dumps({**state, "mounted": mounted}), atomic=True)


def _call(call: str, inputs: dict, fn, output):
    """One Hugging Face API call as a span; `output` makes what it returned recordable."""
    with telemetry.span("hf.call", {"call": call, **inputs}) as sp:
        result = fn()
        sp.output = output(result)
        return result


def upload_repo(user: str, run: str, seq: int) -> str:
    return f"{user}/{run}-{seq:05d}"


def run_uploads(api, run: str) -> dict[int, str]:
    """Upload number -> repo of every upload of `run` on Hugging Face."""
    user = api.whoami()["name"]
    name = re.compile(re.escape(f"{user}/{run}-") + r"(\d{5})")
    return {int(m[1]): d.id for d in api.list_datasets(author=user, search=f"{run}-") if (m := name.fullmatch(d.id))}


def _hashes(path: Path) -> dict[str, str]:
    """sha256 (what Hugging Face reports for an LFS file) and the git blob id (for a file kept in git)."""
    sha256, git = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            sha256.update(chunk)
            git.update(chunk)
    return {"sha256": sha256.hexdigest(), "git": git.hexdigest()}


def _tars(api, repo: str) -> dict:
    from huggingface_hub.hf_api import RepoFile
    return _call("list_repo_tree", {"repo": repo},
                 lambda: {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
                          if isinstance(f, RepoFile) and f.path != ".gitattributes"},
                 lambda tars: {p: f.size for p, f in tars.items()})


def _same(there: dict, applied: dict) -> bool:
    return set(there) == set(applied) and all(
        (f.lfs.sha256 == applied[p]["sha256"]) if f.lfs else (f.blob_id == applied[p]["git"]) for p, f in there.items())


def fetch(run_dir: Path, token: str, delete: bool = False, api=None, download=None) -> str:
    """Extracts, in order, the uploads not fetched yet into `run_dir` and removes what the last one no longer has.
    An upload already applied is applied again when its repo changed since (a retry). With `delete`, then deletes
    every upload repo this copy holds, after checking each against what it applied."""
    if api is None or download is None:
        import huggingface_hub
        api = api or huggingface_hub.HfApi(token=token)
        download = download or huggingface_hub.hf_hub_download
    run_dir = Path(run_dir)
    run = run_dir.name
    state_path = run_dir / "control" / "hf_fetch.json"
    if run_dir.exists() and not state_path.exists():
        raise FileExistsError(f"{run_dir} exists and was not made by fetch; it is never overwritten")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"last": 0, "applied": {}}
    if "applied" not in state:
        raise ValueError(f"{run_dir} was fetched from a single run repo (an older `ar fetch`); fetch into a new folder")
    remote = run_uploads(api, run)
    fetched = reapplied = 0
    tmp_root = cache_dir() / "tmp"
    for seq in sorted(remote):
        there = _tars(api, remote[seq])
        if seq <= state["last"]:
            applied = state["applied"].get(str(seq))
            if applied is None:
                raise RuntimeError(f"{remote[seq]}: {run_dir} has no record of applying it")
            if _same(there, applied):
                continue
        elif seq != state["last"] + 1:
            raise RuntimeError(f"upload {state['last'] + 1} of {run} is not on Hugging Face (the next there is {seq}): "
                               f"fetching past it would leave {run_dir} incomplete")
        tars = sorted(there)
        if not tars:                                # its commit has not landed yet: the next fetch takes it
            break
        fs.mkdir(run_dir)
        fs.mkdir(tmp_root)
        applied = {}
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            for tar_name in tars:
                path = Path(download(repo_id=remote[seq], filename=tar_name, repo_type="dataset", local_dir=tmp,
                                     token=token))
                applied[tar_name] = _hashes(path)
                with tarfile.open(path) as tar:
                    tar.extractall(run_dir, filter="data")
        keep = set(json.loads((run_dir / MANIFEST).read_text()))
        fs.remove(run_dir / MANIFEST)
        for rel in set(run_files(run_dir)) - keep:
            fs.remove(run_dir / rel)
        if seq <= state["last"]:
            reapplied += 1
        else:
            fetched += 1
        state = {"last": max(seq, state["last"]), "applied": {**state["applied"], str(seq): applied}}
        fs.mkdir(state_path.parent)
        fs.write_text(state_path, json.dumps(state))
    report = (f"fetched {fetched} uploads of {run} into {run_dir}" if fetched or reapplied else f"{run_dir} is up to date") \
        + (f" ({reapplied} applied again: changed on Hugging Face since)" if reapplied else "")
    if not delete:
        return report
    held = {seq: repo for seq, repo in remote.items() if seq <= state["last"]}
    for seq, repo in sorted(held.items()):          # every repo is checked before any is deleted
        applied = state["applied"].get(str(seq))
        if applied is None:
            raise RuntimeError(f"{repo}: {run_dir} has no record of applying it; nothing was deleted")
        if not _same(_tars(api, repo), applied):
            raise RuntimeError(f"{repo} differs from what {run_dir} applied; nothing was deleted")
    for repo in held.values():
        api.delete_repo(repo, repo_type="dataset")
    return f"{report}; deleted {len(held)} upload repos from Hugging Face"
