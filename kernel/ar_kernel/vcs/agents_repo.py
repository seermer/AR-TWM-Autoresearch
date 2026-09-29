"""agents.git: one bare repo per run holding every agent version and attempt (spec 5.2)."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

_IDENT = ["-c", "user.name=AutoResearcher kernel", "-c", "user.email=kernel@autoresearcher.local"]


class CheckoutError(Exception):
    """A committed tree cannot be extracted safely, e.g. an agent committed a symlink that leaves the tree."""


def _remote_env() -> dict:
    """A push never waits for a password or a host-key prompt: without a loaded key it fails."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0",
            "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh") + " -o BatchMode=yes"}


class AgentsRepo:
    def __init__(self, path: Path, remote: str | None = None, namespace: str = "",
                 on_push_error: Callable[[str], None] | None = None) -> None:
        """With `remote`, every ref update is pushed there, under branches named `<namespace>/...`
        so several runs can share one repo. A failed push goes to `on_push_error`, never raises."""
        self.path = Path(path)
        self.remote, self.namespace, self.on_push_error = remote, namespace, on_push_error

    def _git(self, *args: str, work_tree: Path | None = None, index: Path | None = None,
             input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ}
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        cmd = ["git", *_IDENT, "--git-dir", str(self.path)]
        if work_tree is not None:
            cmd += ["--work-tree", str(work_tree)]
        return subprocess.run(cmd + list(args), capture_output=True, env=env, input=input, check=check)

    def init(self, seed_dir: Path) -> str:
        subprocess.run(["git", "init", "--bare", "-q", str(self.path)], check=True)
        root = self.commit_tree(seed_dir, None, "seed agent")
        self.set_ref(self.branch_ref("root"), root)
        return root

    def commit_tree(self, src: Path, parent: str | None, message: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._git("add", "-A", ".", work_tree=src, index=index)
            tree = self._git("write-tree", work_tree=src, index=index).stdout.decode().strip()
        args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
        return self._git(*args).stdout.decode().strip()

    def set_ref(self, ref: str, commit: str) -> None:
        self._git("update-ref", ref, commit)
        self.push()

    @staticmethod
    def check_remote(url: str) -> str | None:
        """None if `url` is a reachable git repo, else why not."""
        try:
            r = subprocess.run(["git", "ls-remote", "--heads", url], capture_output=True, text=True,
                               env=_remote_env(), timeout=60)
        except subprocess.TimeoutExpired:
            return f"git ls-remote {url} timed out"
        return None if r.returncode == 0 else r.stderr.strip() or f"git ls-remote exited {r.returncode}"

    def push(self) -> None:
        """Push every node branch and attempt ref. Refs are only ever added, so each push also
        catches up any earlier push that failed."""
        if not self.remote:
            return
        specs = [f"refs/heads/node/*:refs/heads/{self.namespace}/node/*",
                 f"refs/attempts/*:refs/heads/{self.namespace}/attempts/*"]
        cmd = ["git", "--git-dir", str(self.path), "push", "--quiet", self.remote, *specs]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=_remote_env(), timeout=120)
            error = None if r.returncode == 0 else r.stderr.strip() or f"git push exited {r.returncode}"
        except subprocess.TimeoutExpired:
            error = "git push timed out after 120 s"
        if error and self.on_push_error:
            self.on_push_error(f"push to {self.remote} failed: {error}")

    @staticmethod
    def attempt_ref(node: str, phase: str, attempt: int) -> str:
        return f"refs/attempts/{node}/{phase}-{attempt}"

    @staticmethod
    def branch_ref(node: str) -> str:
        return f"refs/heads/node/{node}"

    def resolve(self, ref: str) -> str | None:
        r = self._git("rev-parse", "--verify", "--quiet", ref + "^{commit}", check=False)
        return r.stdout.decode().strip() or None if r.returncode == 0 else None

    def checkout(self, commit: str, dest: Path) -> None:
        dest = Path(dest)
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"{dest} is not empty")
        dest.mkdir(parents=True, exist_ok=True)
        archive = self._git("archive", "--format=tar", commit).stdout
        try:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(dest, filter="data")
        except tarfile.TarError as exc:
            raise CheckoutError(f"cannot check out {commit[:12]}: {exc}") from exc

    def read_file(self, commit: str, path: str) -> str | None:
        r = self._git("show", f"{commit}:{path}", check=False)
        return r.stdout.decode() if r.returncode == 0 else None

    def diff_stats(self, a: str, b: str) -> list[dict]:
        out = []
        for line in self._git("diff", "--numstat", a, b).stdout.decode().splitlines():
            added, removed, path = line.split("\t", 2)
            out.append({"path": path, "added": int(added) if added != "-" else 0,
                        "removed": int(removed) if removed != "-" else 0})
        return out
