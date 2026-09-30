---
name: recipe_constraints_checked_by_kernel
description: Use when writing a training recipe: the allowlists and step constraints recipe_check enforces.
---

# Recipe constraints the kernel checks

- `sample.height` and `sample.width` are a pair from `resolution_allowlist`; `lora.rank` and `lora.alpha` are a pair from `lora_allowlist`.
- `steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps)` must be at least 1, and `optimizer.epochs * steps_per_epoch` at least `optimizer.max_steps`.
- `epoch_windows` is the largest, over the enabled datasets of the commit, of `ceil(clips / (weight / total_weight))`: the draws one epoch needs so that each dataset is seen once at its sampling weight.
