from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import KernelConfig
from ..subproc import run_in_env, _tail
from .recipe import TUNABLE_KEYS, steps_per_epoch, write_resolved_config

@dataclass
class GateResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    resolved_path: Path | None = None
    view_roots: dict[str, Path] = field(default_factory=dict)

class Gate:
    def __init__(self, cfg: KernelConfig, commits, recorder) -> None:
        self.cfg = cfg
        self.commits = commits
        self.recorder = recorder
        self.base_recipe = cfg.repo_root / "configs" / "base_recipe.yaml"

    def check(self, recipe: dict, commit_id: str, parent_commit: str | None, node_id: str,
              node_dir: Path, run_dir: Path, gpus: list[int]) -> GateResult:
        failures: list[str] = []
        for key in recipe:
            if key not in TUNABLE_KEYS:
                failures.append(f"{key} is not tunable; allowed: {sorted(TUNABLE_KEYS)}")

        manifest = self.commits.manifest(commit_id)
        base = yaml.safe_load(self.base_recipe.read_text(encoding="utf-8"))
        height = recipe.get("sample.height", base["sample"]["height"])
        width = recipe.get("sample.width", base["sample"]["width"])
        if [height, width] not in [list(p) for p in self.cfg.get("train.resolution_allowlist")]:
            failures.append(f"resolution {height}x{width} is not in train.resolution_allowlist")
        rank = recipe.get("lora.rank", base["lora"]["rank"])
        alpha = recipe.get("lora.alpha", base["lora"]["alpha"])
        if [rank, alpha] not in [list(p) for p in self.cfg.get("train.lora_allowlist")]:
            failures.append(f"lora rank/alpha {rank}/{alpha} is not in train.lora_allowlist")

        if parent_commit is not None and commit_id != parent_commit:
            if self.commits.manifest(commit_id) == self.commits.manifest(parent_commit):
                failures.append(
                    "the data commit's manifest is identical to the parent's; every node must change data")
        elif parent_commit is not None:
            failures.append(
                "the data commit is identical to the parent's; every node must change data")

        n_gpus = len(gpus)
        for name, entry in manifest["datasets"].items():
            if entry["weight"] > 0 and 0 < len(entry["clips"]) < n_gpus:
                failures.append(f"dataset {name} has {len(entry['clips'])} clips, fewer than {n_gpus} GPUs")

        grad_accum = int(recipe.get("optimizer.grad_accum_steps", base["optimizer"]["grad_accum_steps"]))
        max_steps = int(recipe.get("optimizer.max_steps", base["optimizer"]["max_steps"]))
        epochs = int(recipe.get("optimizer.epochs", base["optimizer"]["epochs"]))
        windows, per_epoch = steps_per_epoch(manifest, n_gpus, grad_accum)
        if per_epoch < 1:
            failures.append(
                f"steps_per_epoch is 0 (epoch_windows={windows}, n_gpus={n_gpus}, "
                f"grad_accum_steps={grad_accum}); need epoch_windows >= n_gpus * grad_accum_steps")
        elif epochs * per_epoch < max_steps:
            failures.append(
                f"optimizer.epochs={epochs} x steps_per_epoch={per_epoch} < max_steps={max_steps} "
                f"(epoch_windows={windows}); training would stop early")

        if failures:
            self.recorder.event("gate.failed", node=node_id, phase="gate",
                                payload={"failures": failures, "recipe": recipe})
            return GateResult(ok=False, failures=failures)

        roots = self.commits.materialize(commit_id, Path(node_dir) / "view")
        resolved = write_resolved_config(self.cfg, self.base_recipe, recipe, roots, manifest,
                                         node_id, Path(node_dir), Path(run_dir))
        gpu_list = ",".join(str(g) for g in gpus)
        checks = [
            ("check_dataset", ["python", "scripts/tools/check_dataset.py", "--config",
                               str(resolved), "--max-messages", "0"], {}),
            ("precache_dry_run", ["python", "scripts/tools/precache_train_text_embeds.py",
                                  "--config", str(resolved), "--dry-run"], {}),
        ]
        for label, args, extra in checks:
            proc = run_in_env("alayaworld", args, cwd=self.cfg.worldmodel,
                              extra_env={"CUDA_VISIBLE_DEVICES": gpu_list, **extra},
                              timeout=3600, recorder=self.recorder, node=node_id, phase="gate")
            if proc.returncode != 0:
                failures.append(f"{label} failed (rc={proc.returncode}): {_tail(proc, 2000)}")

        describe = run_in_env(
            "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
            cwd=self.cfg.worldmodel,
            extra_env={"CONFIG_PATH": str(resolved), "CUDA_VISIBLE_DEVICES": gpu_list,
                       "DESCRIBE": "1"},
            timeout=3600, recorder=self.recorder, node=node_id, phase="gate")
        if describe.returncode != 0:
            failures.append(f"describe failed (rc={describe.returncode}): {describe.stdout[-2000:]}")

        ok = not failures
        self.recorder.event("gate.passed" if ok else "gate.failed", node=node_id, phase="gate",
                            payload={"failures": failures, "recipe": recipe,
                                     "resolved": str(resolved)})
        return GateResult(ok=ok, failures=failures, resolved_path=resolved if ok else None,
                          view_roots=roots)
