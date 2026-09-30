"""What the recipe writer is told about the tunable keys: base values from the run's own snapshot
and what each key does; plus how the parent's training went."""
from __future__ import annotations

import re
from pathlib import Path

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
_STEP = re.compile(r"\[Train\] step=(\d+) .*?sigma=([\d.eE+-]+) loss=([\d.eE+-]+) grad=([\d.eE+-]+)")
# The loss depends mostly on the sampled noise level, so its trend is only readable within a sigma bin.
_SIGMA_BINS = {"sigma<0.3": (0.0, 0.3), "0.3<=sigma<0.6": (0.3, 0.6), "sigma>=0.6": (0.6, float("inf"))}


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


def train_summary(train_log: Path) -> dict:
    """Mean loss per sigma bin in the first and second half of training, and the mean gradient norm."""
    try:
        text = Path(train_log).read_text(errors="replace")
    except OSError:
        return {}
    rows = [(float(m[2]), float(m[3]), float(m[4])) for m in map(_STEP.search, text.splitlines()) if m]
    if not rows:
        return {}
    half = len(rows) // 2
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"steps": len(rows),
            "loss_by_sigma": {name: {"first_half": mean([r[1] for r in rows[:half] if lo <= r[0] < hi]),
                                     "second_half": mean([r[1] for r in rows[half:] if lo <= r[0] < hi])}
                              for name, (lo, hi) in _SIGMA_BINS.items()},
            "mean_grad_norm": mean([r[2] for r in rows])}
