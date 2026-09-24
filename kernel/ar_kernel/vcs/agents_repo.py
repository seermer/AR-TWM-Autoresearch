"""agents.git: one bare repo per run holding every agent version and attempt (spec 5.2)."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

_IDENT = ["-c", "user.name=AutoResearcher kernel", "-c", "user.email=kernel@autoresearcher.local"]


class AgentsRepo:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

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
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(dest, filter="data")

    def read_file(self, commit: str, path: str) -> str | None:
        r = self._git("show", f"{commit}:{path}", check=False)
        return r.stdout.decode() if r.returncode == 0 else None

    def diff(self, a: str, b: str) -> str:
        return self._git("diff", a, b).stdout.decode()

    def diff_stats(self, a: str, b: str) -> list[dict]:
        out = []
        for line in self._git("diff", "--numstat", a, b).stdout.decode().splitlines():
            added, removed, path = line.split("\t", 2)
            out.append({"path": path, "added": int(added) if added != "-" else 0,
                        "removed": int(removed) if removed != "-" else 0})
        return out
