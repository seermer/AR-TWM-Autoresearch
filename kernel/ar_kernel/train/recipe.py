from __future__ import annotations
import math
from pathlib import Path
import yaml

TUNABLE_KEYS = frozenset({
    "data.overall_caption_prob",
    "optimizer.lr", "optimizer.weight_decay", "optimizer.max_grad_norm",
    "optimizer.warmup_steps", "optimizer.max_steps", "optimizer.epochs",
    "optimizer.grad_accum_steps",
    "sample.height", "sample.width",
    "lora.rank", "lora.alpha",
})

# Validity only -- type, sign, finiteness, range of a probability. These are NOT
# tuning limits: how long to train and at what learning rate is the agent's
# decision. They exist so that a malformed value is a gate failure the agent can
# retry, rather than an exception that crashes the node (spec 14.2).
_INT, _FLOAT = "int", "float"
RECIPE_RULES = {
    "optimizer.max_steps": (_INT, 1, None),
    "optimizer.epochs": (_INT, 1, None),
    "optimizer.grad_accum_steps": (_INT, 1, None),
    "optimizer.warmup_steps": (_INT, 0, None),
    "optimizer.lr": (_FLOAT, 0.0, None),            # strictly positive, see below
    "optimizer.weight_decay": (_FLOAT, 0.0, None),
    "optimizer.max_grad_norm": (_FLOAT, 0.0, None),  # strictly positive
    "data.overall_caption_prob": (_FLOAT, 0.0, 1.0),
    "sample.height": (_INT, 1, None),
    "sample.width": (_INT, 1, None),
    "lora.rank": (_INT, 1, None),
    "lora.alpha": (_INT, 1, None),
}
_STRICTLY_POSITIVE = {"optimizer.lr", "optimizer.max_grad_norm"}


def validate_recipe_values(recipe: dict) -> list[str]:
    """Return one failure message per malformed tunable value (unknown keys are
    reported separately by the gate's allowlist check)."""
    failures = []
    for key, value in recipe.items():
        rule = RECIPE_RULES.get(key)
        if rule is None:
            continue
        kind, low, high = rule
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
        if key in _STRICTLY_POSITIVE and value <= 0:
            failures.append(f"{key} must be > 0, got {value!r}")
        elif value < low:
            failures.append(f"{key} must be >= {low}, got {value!r}")
        elif high is not None and value > high:
            failures.append(f"{key} must be <= {high}, got {value!r}")
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
        spec = {"root": str(roots[name]), "format": entry["format"], "weight": entry["weight"]}
        if entry["prompt_mode"] is not None:
            spec["prompt_mode"] = entry["prompt_mode"]
        datasets[name] = spec
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
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    return target
