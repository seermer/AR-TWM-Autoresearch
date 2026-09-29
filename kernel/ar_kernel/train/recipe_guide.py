"""What the recipe writer is told about the tunable keys: base values from the run's own snapshot,
what each key does, what is worth trying and what it costs; plus how the parent's training went."""
from __future__ import annotations

import re
from pathlib import Path

from .recipe import TUNABLE_KEYS

# key -> (meaning, what is worth trying, cost)
_TEXT = {
    "optimizer.lr": ("Peak learning rate of the LoRA weights (after warmup).",
                     "Lower (2e-5) when the new data is close to the model's own outputs or the loss is noisy; "
                     "higher (1e-4) only with many steps: at few steps a high rate overshoots.",
                     "None on time."),
    "optimizer.weight_decay": ("Weight decay on the LoRA weights.",
                               "Rarely worth changing; larger values pull the update toward zero.", "None."),
    "optimizer.max_grad_norm": ("Gradient clipping threshold.",
                                "Lower it if the parent's `mean_grad_norm` shows spikes; the base value rarely clips.",
                                "None."),
    "optimizer.warmup_steps": ("Steps of linear learning-rate warmup.",
                               "Keep it near a sixth of max_steps.", "None."),
    "optimizer.max_steps": ("Optimizer steps in total.",
                            "Scale with the amount of new data: more windows justify more steps. The base value "
                            "is the reference budget; a much smaller value trains far less than the base run.",
                            "Time grows linearly with max_steps x grad_accum_steps."),
    "optimizer.epochs": ("Upper bound on passes over the data.",
                         "Set it so epochs x steps_per_epoch >= max_steps (the gate checks this).", "None."),
    "optimizer.grad_accum_steps": ("Micro-batches per optimizer step (the effective batch per GPU).",
                                   "Larger values smooth the update; steps_per_epoch shrinks accordingly.",
                                   "Time grows linearly with it."),
    "data.overall_caption_prob": ("Probability of using the clip's overall caption instead of its segment prompts.",
                                  "Not in the base recipe file: the trainer default applies. Only matters for timed-prompt datasets.", "None."),
    "sample.height": ("Training frame height; must pair with sample.width from the allowlist.",
                      "A smaller allowed size is faster and uses less memory.", "Tokens grow with height x width."),
    "sample.width": ("Training frame width; must pair with sample.height from the allowlist.", "As above.", "As above."),
    "lora.rank": ("LoRA rank; must pair with lora.alpha from the allowlist.",
                  "Higher rank fits more new content; lower rank stays closer to the base model.",
                  "Small memory increase."),
    "lora.alpha": ("LoRA scaling; must pair with lora.rank from the allowlist.", "Follow the allowed pair.", "None."),
}
_STEP = re.compile(r"\[Train\] step=(\d+) .*?loss=([\d.eE+-]+) grad=([\d.eE+-]+) .*?time=([\d.]+)s")


def _dig(recipe: dict, dotted: str):
    node = recipe
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def recipe_guide(base_recipe: dict) -> dict:
    assert set(_TEXT) == TUNABLE_KEYS, "every tunable key needs guide text"
    return {key: {"base": _dig(base_recipe, key), "meaning": meaning, "try": tip, "cost": cost}
            for key, (meaning, tip, cost) in sorted(_TEXT.items())}


def train_summary(train_log: Path) -> dict:
    """Loss (first vs last quarter), mean gradient norm and seconds per step from a train.log."""
    try:
        text = Path(train_log).read_text(errors="replace")
    except OSError:
        return {}
    rows = [(float(m[2]), float(m[3]), float(m[4])) for m in map(_STEP.search, text.splitlines()) if m]
    if not rows:
        return {}
    quarter = max(1, len(rows) // 4)
    mean = lambda xs: sum(xs) / len(xs)
    return {"steps": len(rows),
            "loss_first_quarter": round(mean([r[0] for r in rows[:quarter]]), 4),
            "loss_last_quarter": round(mean([r[0] for r in rows[-quarter:]]), 4),
            "mean_grad_norm": round(mean([r[1] for r in rows]), 4),
            "sec_per_step": round(mean([r[2] for r in rows]), 1)}
