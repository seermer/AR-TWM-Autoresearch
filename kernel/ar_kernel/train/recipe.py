from __future__ import annotations
import math
from pathlib import Path
import yaml

from .. import fs

# Rules define numeric validity; agents choose training duration and scale.
_INT, _FLOAT = "int", "float"
RECIPE_RULES = {
    "optimizer.max_steps": (_INT, 1, False),
    "optimizer.epochs": (_INT, 1, False),
    "optimizer.grad_accum_steps": (_INT, 1, False),
    "optimizer.warmup_steps": (_INT, 0, False),
    "optimizer.lr": (_FLOAT, 0.0, True),
    "optimizer.weight_decay": (_FLOAT, 0.0, False),
    "optimizer.max_grad_norm": (_FLOAT, 0.0, True),
    "lora.rank": (_INT, 1, False),
    "lora.alpha": (_INT, 1, False),
}
TUNABLE_KEYS = frozenset(RECIPE_RULES)


def validate_recipe_values(recipe: dict) -> list[str]:
    """Report malformed tunable values; the gate checks unknown keys separately."""
    failures = []
    for key, value in recipe.items():
        rule = RECIPE_RULES.get(key)
        if rule is None:
            continue
        kind, low, exclusive = rule
        if isinstance(value, bool):
            failures.append(f"{key} must be a number, got a boolean {value!r}")
            continue
        if kind == _INT and not isinstance(value, int):
            failures.append(f"{key} must be an integer, got {value!r}")
            continue
        if kind == _FLOAT and not isinstance(value, (int, float)):
            failures.append(f"{key} must be a number, got {value!r}")
            continue
        if kind == _FLOAT and not math.isfinite(value):
            failures.append(f"{key} must be finite, got {value!r}")
            continue
        if exclusive and value <= low:
            failures.append(f"{key} must be > 0, got {value!r}")
        elif value < low:
            failures.append(f"{key} must be >= {low}, got {value!r}")
    return failures


def _assign(tree: dict, dotted: str, value) -> None:
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value

def steps_per_epoch(manifest: dict, n_gpus: int, grad_accum: int) -> tuple[int, int]:
    enabled = {n: d for n, d in manifest["datasets"].items() if d["weight"] > 0 and d["clips"]}
    total_weight = sum(d["weight"] for d in enabled.values())
    epoch_windows = max(math.ceil(len(d["clips"]) / (d["weight"] / total_weight))
                        for d in enabled.values())
    per_rank = epoch_windows // n_gpus
    return epoch_windows, per_rank // grad_accum

def build_resolved_config(base_recipe: Path, recipe: dict, roots: dict[str, Path], manifest: dict,
                          node_id: str, node_dir: Path, run_dir: Path) -> dict:
    resolved = yaml.safe_load(Path(base_recipe).read_text(encoding="utf-8"))
    for dotted, value in recipe.items():
        _assign(resolved, dotted, value)

    datasets = {}
    for name, entry in manifest["datasets"].items():
        if entry["weight"] <= 0 or not entry["clips"]:
            continue
        datasets[name] = {"root": str(roots[name]), "format": "video_timed_prompts_camera",
                          "weight": entry["weight"]}
    resolved["data"]["sources"] = {}
    resolved["data"]["datasets"] = datasets

    train_dir = Path(node_dir) / "train"
    resolved["run"]["name"] = f"node_{node_id}"
    resolved["run"]["output_dir"] = str(train_dir / "outputs")
    resolved["run"]["log_dir"] = str(train_dir / "logs")
    resolved["runtime"]["text_embed_cache_dir"] = str(Path(run_dir) / "cache" / "text_embed")
    resolved["optimizer"]["checkpoint_steps"] = int(resolved["optimizer"]["max_steps"])
    resolved["optimizer"]["max_checkpoints"] = 1
    resolved["validation"]["enabled"] = False
    first = next(iter(datasets))
    for mode in resolved["validation"].get("modes", {}).values():
        mode.setdefault("dataset", {})["source"] = first
    return resolved

def lora_of(resolved: Path) -> tuple[int, int]:
    lora = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))["lora"]
    return int(lora["rank"]), int(lora["alpha"])

def write_resolved_config(base_recipe: Path, recipe: dict, roots: dict[str, Path], manifest: dict,
                          node_id: str, node_dir: Path, run_dir: Path) -> Path:
    resolved = build_resolved_config(base_recipe, recipe, roots, manifest, node_id, node_dir, run_dir)
    target = Path(node_dir) / "train_config.yaml"
    fs.mkdir(target.parent)
    fs.write_text(target, yaml.safe_dump(resolved, sort_keys=True))
    return target
