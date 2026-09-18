from __future__ import annotations
import copy, math
from pathlib import Path
import yaml

from ..config import KernelConfig

TUNABLE_KEYS = frozenset({
    "data.overall_caption_prob",
    "optimizer.lr", "optimizer.weight_decay", "optimizer.max_grad_norm",
    "optimizer.warmup_steps", "optimizer.max_steps", "optimizer.epochs",
    "optimizer.grad_accum_steps",
    "sample.height", "sample.width",
    "lora.rank", "lora.alpha",
})

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

def build_resolved_config(cfg: KernelConfig, base_recipe: Path, recipe: dict,
                          roots: dict[str, Path], manifest: dict, node_id: str,
                          node_dir: Path, run_dir: Path) -> dict:
    resolved = yaml.safe_load(Path(base_recipe).read_text(encoding="utf-8"))
    resolved = copy.deepcopy(resolved)
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
    for mode in resolved.get("validation", {}).get("modes", {}).values():
        mode.setdefault("dataset", {})["source"] = first
    return resolved

def write_resolved_config(cfg: KernelConfig, base_recipe: Path, recipe: dict,
                          roots: dict[str, Path], manifest: dict, node_id: str,
                          node_dir: Path, run_dir: Path) -> Path:
    resolved = build_resolved_config(cfg, base_recipe, recipe, roots, manifest, node_id,
                                     node_dir, run_dir)
    target = Path(node_dir) / "train_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    return target
