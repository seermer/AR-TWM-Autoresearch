from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

class GpuPolicyError(ValueError):
    """The GPU list is unusable for this run."""


def _merged(base: dict, over: dict) -> dict:
    return {**base, **{key: _merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict)
                       else value for key, value in over.items()}}

@dataclass(frozen=True)
class KernelConfig:
    raw: dict
    repo_root: Path
    overlay: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None, overlay: Path | None = None) -> "KernelConfig":
        """Load kernel.yaml. With an explicit path the repo root is derived from
        it (<root>/configs/kernel.yaml), so a config can be loaded for a checkout
        other than the one this module lives in; relative paths inside it then
        resolve against that checkout rather than this one. `overlay` is a YAML file
        holding only the keys that differ; they replace kernel.yaml's."""
        if path is None:
            path, repo_root = REPO_ROOT / "configs" / "kernel.yaml", REPO_ROOT
        else:
            path = Path(path).resolve()
            repo_root = path.parent.parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if overlay is not None:
            overlay = Path(overlay).resolve()
            raw = _merged(raw, yaml.safe_load(overlay.read_text(encoding="utf-8")))
        return cls(raw=raw, repo_root=repo_root, overlay=overlay)

    @classmethod
    def for_run(cls, run_dir: Path) -> "KernelConfig":
        """The run's frozen kernel.yaml (snapshotted at run start) under its frozen overlay, with THIS
        checkout as the repo root. load(path) would take runs/<id> as the root and break every sibling path."""
        raw = yaml.safe_load((Path(run_dir) / "config" / "kernel.yaml").read_text(encoding="utf-8"))
        overlay = Path(run_dir) / "config" / OVERLAY_SNAPSHOT
        if overlay.exists():
            raw = _merged(raw, yaml.safe_load(overlay.read_text(encoding="utf-8")))
        return cls(raw=raw, repo_root=REPO_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _path(self, dotted: str) -> Path:
        value = Path(self.get(dotted))
        return value if value.is_absolute() else (self.repo_root / value).resolve()

    @property
    def worldmodel(self) -> Path:
        return self._path("paths.worldmodel")

    @property
    def wbench(self) -> Path:
        return self._path("paths.wbench")

    @property
    def eval_data(self) -> Path:
        """The scored benchmark: a folder of cases/, images/ and masks/."""
        return self._path("eval.data")

    @property
    def eval_cases(self) -> Path:
        return self._path("eval.cases")

    @property
    def runs_dir(self) -> Path:
        return self._path("paths.runs_dir")

    @property
    def root_cache(self) -> Path:
        return self._path("paths.root_cache")

    @property
    def scores_dir(self) -> Path:
        return self._path("paths.scores_dir")

SNAPSHOT_FILES = ("kernel.yaml", "base_recipe.yaml")
OVERLAY_SNAPSHOT = "overlay.yaml"


def run_config_path(cfg: "KernelConfig", run_dir: Path, name: str) -> Path:
    """The run's frozen copy of configs/<name> once the run exists, else the live
    file. Reading the live repo config from inside a run let a mid-run edit or
    `git pull` change what an in-flight run does."""
    snap = Path(run_dir) / "config" / name
    return snap if snap.exists() else cfg.repo_root / "configs" / name


def load_dotenv(path: Path, env: "dict | os._Environ") -> list[str]:
    """Copy KEY=VALUE pairs from `path` into `env` where `env` lacks them.

    Shell variables always win and empty values are skipped, so the
    placeholder .env shipped in the repo enables nothing. Returns the keys set.
    """
    path = Path(path)
    if not path.is_file():
        return []
    applied = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not key.isidentifier() or not value or env.get(key):
            continue
        env[key] = value
        applied.append(key)
    return applied


def resolve_gpus(cfg: KernelConfig, env: Mapping[str, str]) -> list[int]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "").strip() or str(cfg.get("gpus.default"))
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    gpus: list[int] = []
    for entry in entries:
        if not entry.isdigit():
            raise GpuPolicyError(f"CUDA_VISIBLE_DEVICES entry {entry!r} is not an integer")
        gpus.append(int(entry))
    if len(set(gpus)) != len(gpus):
        raise GpuPolicyError(f"CUDA_VISIBLE_DEVICES lists a duplicate GPU: {raw!r}")
    minimum = int(cfg.get("gpus.min_count"))
    if len(gpus) < minimum:
        raise GpuPolicyError(f"need at least {minimum} GPUs, got {len(gpus)}: {raw!r}")
    return gpus
