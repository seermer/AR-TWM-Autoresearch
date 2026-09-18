from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

class GpuPolicyError(ValueError):
    """The GPU list is unusable for this run."""

@dataclass(frozen=True)
class KernelConfig:
    raw: dict
    repo_root: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "KernelConfig":
        path = path or REPO_ROOT / "configs" / "kernel.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    def project_root(self) -> Path:
        return self.repo_root.parent

    @property
    def worldmodel(self) -> Path:
        return self._path("paths.worldmodel")

    @property
    def wbench(self) -> Path:
        return self._path("paths.wbench")

    @property
    def runs_dir(self) -> Path:
        return self._path("paths.runs_dir")

def resolve_gpus(cfg: KernelConfig, env: Mapping[str, str]) -> list[int]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "").strip() or str(cfg.get("gpus.default"))
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    gpus: list[int] = []
    for entry in entries:
        if not entry.isdigit():
            raise GpuPolicyError(f"CUDA_VISIBLE_DEVICES entry {entry!r} is not an integer")
        gpus.append(int(entry))
    minimum = int(cfg.get("gpus.min_count"))
    if len(gpus) < minimum:
        raise GpuPolicyError(f"need at least {minimum} GPUs, got {len(gpus)}: {raw!r}")
    return gpus
