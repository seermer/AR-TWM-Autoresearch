from __future__ import annotations
import json
from pathlib import Path

import yaml

from ..archive.clips import hidden
from ..config import KernelConfig, run_config_path
from ..subproc import output_tail, run_in_env
from .recipe import TUNABLE_KEYS, steps_per_epoch, validate_recipe_values, write_resolved_config

FINDINGS = "===AR_FINDINGS==="      # a WorldModel check script's findings line: the marker, then a JSON list


def findings(label: str, proc) -> list[str]:
    """What a check script found wrong. A failure without its findings line is a crash: it raises."""
    if proc.returncode == 0:
        return []
    for line in proc.stdout.splitlines():
        if line.startswith(FINDINGS):
            return [f"{label}: {f}" for f in json.loads(line[len(FINDINGS):])]
    raise RuntimeError(f"{label} crashed (rc={proc.returncode}): {output_tail(proc, 2000)}")


def resolve_config(cfg: KernelConfig, commits, run_dir: Path, recipe: dict, data_commit: str, node_id: str,
                   out_dir: Path) -> Path:
    """The training config of `recipe` over the run's base recipe, on the data commit's view in `out_dir`."""
    roots = commits.materialize(data_commit, Path(out_dir) / "view")
    return write_resolved_config(run_config_path(cfg, run_dir, "base_recipe.yaml"), recipe, roots,
                                 commits.manifest(data_commit), node_id, Path(out_dir), Path(run_dir))


class Gate:
    def __init__(self, cfg: KernelConfig, commits, recorder) -> None:
        self.cfg = cfg
        self.commits = commits
        self.recorder = recorder

    def check(self, recipe: dict, commit_id: str, parent_commit: str | None, node_id: str,
              node_dir: Path, run_dir: Path, gpus: list[int]) -> list[str]:
        """The recipe's failures; none when it passes."""
        failures = [f"{key} is not tunable; allowed: {sorted(TUNABLE_KEYS)}"
                    for key in recipe if key not in TUNABLE_KEYS]
        value_failures = validate_recipe_values(recipe)
        if value_failures:
            # Everything below does arithmetic on these values; stop before it can raise.
            return self._failed(node_id, recipe, failures + value_failures)

        manifest = self.commits.manifest(commit_id)
        base_recipe = run_config_path(self.cfg, run_dir, "base_recipe.yaml")
        base = yaml.safe_load(base_recipe.read_text(encoding="utf-8"))
        rank = recipe.get("lora.rank", base["lora"]["rank"])
        alpha = recipe.get("lora.alpha", base["lora"]["alpha"])
        if [rank, alpha] not in [list(p) for p in self.cfg.get("train.lora_allowlist")]:
            failures.append(f"lora rank/alpha {rank}/{alpha} is not in train.lora_allowlist")

        if commit_id == parent_commit:
            failures.append("the data commit is identical to the parent's; every node must change data")
        elif parent_commit is not None and manifest == self.commits.manifest(parent_commit):
            failures.append("the data commit's manifest is identical to the parent's; every node must change data")

        n_gpus = len(gpus)
        for name, entry in manifest["datasets"].items():
            if entry["weight"] > 0 and 0 < len(entry["clips"]) < n_gpus:
                failures.append(f"dataset {name} has {len(entry['clips'])} clips, fewer than {n_gpus} GPUs")
            gone = sum(hidden(self.commits.clips.get(c)["ingested_by"]) for c in entry["clips"])
            if gone:
                failures.append(f"dataset {name} has {gone} clips that are not in the pool")

        grad_accum = int(recipe.get("optimizer.grad_accum_steps", base["optimizer"]["grad_accum_steps"]))
        max_steps = int(recipe.get("optimizer.max_steps", base["optimizer"]["max_steps"]))
        epochs = int(recipe.get("optimizer.epochs", base["optimizer"]["epochs"]))
        draws, per_epoch = steps_per_epoch(manifest, n_gpus, grad_accum)
        if per_epoch < 1:
            failures.append(
                f"too few clips for one optimizer step: one pass over the data draws {draws} clips, and a step "
                f"needs {n_gpus * grad_accum} ({n_gpus} GPUs x optimizer.grad_accum_steps={grad_accum})")
        elif epochs * per_epoch < max_steps:
            failures.append(
                f"too few clips for the step count: one pass over the data draws {draws} clips, which is "
                f"{per_epoch} optimizer steps, so optimizer.epochs={epochs} passes give {epochs * per_epoch} "
                f"steps, fewer than optimizer.max_steps={max_steps}")

        if failures:
            return self._failed(node_id, recipe, failures)

        resolved = resolve_config(self.cfg, self.commits, run_dir, recipe, commit_id, node_id, node_dir)
        gpu_list = ",".join(str(g) for g in gpus)
        checks = [
            ("check_dataset", ["python", "scripts/tools/check_dataset.py", "--config",
                               str(resolved), "--max-messages", "0"]),
            ("precache_dry_run", ["python", "scripts/tools/precache_train_text_embeds.py",
                                  "--config", str(resolved), "--dry-run"]),
        ]
        for label, args in checks:
            proc = run_in_env("alayaworld", args, cwd=self.cfg.worldmodel,
                              extra_env={"CUDA_VISIBLE_DEVICES": gpu_list},
                              recorder=self.recorder, node=node_id, phase="gate")
            failures += findings(label, proc)

        describe = run_in_env(
            "alayaworld", ["bash", "scripts/finetune/lowcompute_4x4090.sh"],
            cwd=self.cfg.worldmodel,
            extra_env={"CONFIG_PATH": str(resolved), "CUDA_VISIBLE_DEVICES": gpu_list,
                       "DESCRIBE": "1"},
            recorder=self.recorder, node=node_id, phase="gate")
        failures += list(dict.fromkeys(findings("describe", describe)))      # every rank prints its own

        self.recorder.event("gate.failed" if failures else "gate.passed", node=node_id, phase="gate",
                            payload={"failures": failures, "recipe": recipe,
                                     "resolved": str(resolved)})
        return failures

    def _failed(self, node_id: str, recipe: dict, failures: list[str]) -> list[str]:
        self.recorder.event("gate.failed", node=node_id, phase="gate",
                            payload={"failures": failures, "recipe": recipe})
        return failures
