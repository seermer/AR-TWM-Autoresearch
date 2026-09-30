"""What the agent is told about the tunable keys: base values from the run's own snapshot
and what each key does."""
from __future__ import annotations

from .recipe import TUNABLE_KEYS

_MEANING = {
    "optimizer.lr": "Peak learning rate of the LoRA weights (after warmup).",
    "optimizer.weight_decay": "Weight decay on the LoRA weights.",
    "optimizer.max_grad_norm": "Gradient clipping threshold.",
    "optimizer.warmup_steps": "Steps of linear learning-rate warmup.",
    "optimizer.max_steps": "Optimizer steps in total.",
    "optimizer.epochs": "Upper bound on passes over the data.",
    "optimizer.grad_accum_steps": "Micro-batches per optimizer step (the effective batch per GPU).",
    "data.overall_caption_prob": ("Probability of using the clip's overall caption instead of its segment prompts. "
                                  "Not in the base recipe file: the trainer default applies. Only matters for "
                                  "timed-prompt datasets."),
    "sample.height": "Training frame height; must pair with sample.width from the allowlist.",
    "sample.width": "Training frame width; must pair with sample.height from the allowlist.",
    "lora.rank": "LoRA rank; must pair with lora.alpha from the allowlist.",
    "lora.alpha": "LoRA scaling; must pair with lora.rank from the allowlist.",
}
def _dig(recipe: dict, dotted: str):
    node = recipe
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def recipe_guide(base_recipe: dict) -> dict:
    assert set(_MEANING) == TUNABLE_KEYS, "every tunable key needs guide text"
    return {key: {"base": _dig(base_recipe, key), "meaning": meaning} for key, meaning in sorted(_MEANING.items())}
