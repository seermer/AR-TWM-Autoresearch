from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]

class GpuPolicyError(ValueError):
    """The GPU list is unusable for this run."""


def merged(base: dict, over: dict) -> dict:
    return {**base, **{key: merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict)
                       else value for key, value in over.items()}}

@dataclass(frozen=True)
class KernelConfig:
    raw: dict
    repo_root: Path
    overlay: dict | None = None

    @classmethod
    def load(cls, path: Path | None = None, overlays: list[Path] = ()) -> "KernelConfig":
        """Load kernel.yaml. With an explicit path the repo root is derived from
        it (<root>/configs/kernel.yaml), so a config can be loaded for a checkout
        other than the one this module lives in; relative paths inside it then
        resolve against that checkout rather than this one. `overlays` are YAML files
        holding only the keys that differ; each replaces the keys of those before it."""
        if path is None:
            path, repo_root = REPO_ROOT / "configs" / "kernel.yaml", REPO_ROOT
        else:
            path = Path(path).resolve()
            repo_root = path.parent.parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        overlay = None
        for file in overlays:
            overlay = merged(overlay or {}, yaml.safe_load(Path(file).read_text(encoding="utf-8")))
        if overlay is not None:
            raw = merged(raw, overlay)
        return cls(raw=raw, repo_root=repo_root, overlay=overlay)

    @classmethod
    def for_run(cls, run_dir: Path) -> "KernelConfig":
        """The run's frozen kernel.yaml (snapshotted at run start) under its frozen overlay, with THIS
        checkout as the repo root. load(path) would take runs/<id> as the root and break every sibling path."""
        raw = yaml.safe_load((Path(run_dir) / "config" / "kernel.yaml").read_text(encoding="utf-8"))
        overlay = Path(run_dir) / "config" / OVERLAY_SNAPSHOT
        if overlay.exists():
            raw = merged(raw, yaml.safe_load(overlay.read_text(encoding="utf-8")))
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

    def render_config(self) -> dict:
        """WorldModel's configs/wbench_full.yaml under `eval.render_overrides`: what the eval and
        rollout_alayaworld render with."""
        base = yaml.safe_load((self.worldmodel / "configs" / "wbench_full.yaml").read_text(encoding="utf-8"))
        return merged(base, self.get("eval.render_overrides") or {})

    @property
    def base_recipe(self) -> Path:
        return self._path("train.base_recipe")

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


def live_config_path(cfg: "KernelConfig", name: str) -> Path:
    """configs/<name>; the base recipe is the file `train.base_recipe` names."""
    return cfg.base_recipe if name == "base_recipe.yaml" else cfg.repo_root / "configs" / name


def run_config_path(cfg: "KernelConfig", run_dir: Path, name: str) -> Path:
    """The run's frozen copy of configs/<name> once the run exists, else the live
    file. Reading the live repo config from inside a run let a mid-run edit or
    `git pull` change what an in-flight run does."""
    snap = Path(run_dir) / "config" / name
    return snap if snap.exists() else live_config_path(cfg, name)


def load_dotenv(path: Path, env: "dict | os._Environ") -> list[str]:
    """Copy KEY=VALUE pairs from `path` into `env` where `env` lacks them.

    Shell variables always win and empty values are skipped, so the
    placeholder .env shipped in the repo enables nothing. Returns the keys set.
    """
    applied = []
    for key, value in dotenv_values(path, interpolate=False).items():
        if not key.isidentifier() or not value or env.get(key):
            continue
        env[key] = value
        applied.append(key)
    return applied
