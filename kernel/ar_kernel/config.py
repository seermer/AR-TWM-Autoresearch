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


def tool_blocks(cfg) -> dict:
    return {"annotate": cfg.get("annotate") or {}, "images": cfg.get("images") or {},
            **{f"generators.{key}": block for key, block in (cfg.get("generators") or {}).items()}}


def dig(tree: dict, dotted: str, default: Any = None) -> Any:
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass(frozen=True)
class KernelConfig:
    raw: dict
    repo_root: Path
    overlay: dict | None = None

    @classmethod
    def load(cls, path: Path | None = None, overlays: list[Path] = ()) -> "KernelConfig":
        """Load base YAML with ordered overlays; resolve relative paths from its checkout."""
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
        """Read the effective run snapshot with paths relative to this checkout."""
        raw = yaml.safe_load((Path(run_dir) / "config" / "kernel.yaml").read_text(encoding="utf-8"))
        return cls(raw=raw, repo_root=REPO_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        return dig(self.raw, dotted, default)

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
    def dotenv_files(self) -> list[Path]:
        return [self.repo_root / ".env", self.worldmodel / ".env", self.wbench / ".env"]

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
        return self.repo_root / "runs"

    @property
    def root_cache(self) -> Path:
        return self._path("paths.root_cache")

    @property
    def scores_dir(self) -> Path:
        return self._path("paths.scores_dir")

def live_config_path(cfg: "KernelConfig", name: str) -> Path:
    """configs/<name>; the base recipe is the file `train.base_recipe` names."""
    return cfg.base_recipe if name == "base_recipe.yaml" else cfg.repo_root / "configs" / name


def run_config_path(cfg: "KernelConfig", run_dir: Path, name: str) -> Path:
    """Return a run's frozen config file when present, otherwise its live configuration source."""
    snap = Path(run_dir) / "config" / name
    return snap if snap.exists() else live_config_path(cfg, name)


def load_dotenv(path: Path, env: "dict | os._Environ") -> list[str]:
    """Fill unset environment keys from nonempty dotenv values; return the keys set."""
    applied = []
    for key, value in dotenv_values(path, interpolate=False).items():
        if not key.isidentifier() or not value or env.get(key):
            continue
        env[key] = value
        applied.append(key)
    return applied
