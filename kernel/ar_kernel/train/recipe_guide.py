"""What the agent is told about the tunable keys: base values from the run's own snapshot
and what each key does."""
from __future__ import annotations

from omegaconf import OmegaConf

from .recipe import TUNABLE_KEYS

_MEANING = {
    "optimizer.lr": "Peak learning rate of the LoRA weights (after warmup).",
    "optimizer.weight_decay": "Weight decay on the LoRA weights.",
    "optimizer.max_grad_norm": "Gradient clipping threshold.",
    "optimizer.warmup_steps": "Steps of linear learning-rate warmup.",
    "optimizer.max_steps": "Optimizer steps in total.",
    "optimizer.epochs": "Upper bound on passes over the data.",
    "optimizer.grad_accum_steps": "Micro-batches per optimizer step (the effective batch per GPU).",
    "lora.rank": "LoRA rank; must pair with lora.alpha from the allowlist.",
    "lora.alpha": "LoRA scaling; must pair with lora.rank from the allowlist.",
}
def recipe_guide(base_recipe: dict) -> dict:
    assert set(_MEANING) == TUNABLE_KEYS, "every tunable key needs guide text"
    tree = OmegaConf.create(base_recipe)
    return {key: {"base": OmegaConf.select(tree, key), "meaning": meaning} for key, meaning in sorted(_MEANING.items())}
