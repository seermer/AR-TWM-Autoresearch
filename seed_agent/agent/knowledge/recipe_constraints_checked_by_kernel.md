---
name: recipe_constraints_checked_by_kernel
description: Use when choosing optimizer.epochs, optimizer.max_steps, optimizer.grad_accum_steps or dataset weights: the step constraints recipe_check enforces.
---

# Recipe constraints the kernel checks

- `steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps)` must be at least 1, and `optimizer.epochs * steps_per_epoch` at least `optimizer.max_steps`.
- `epoch_windows` is the largest, over the enabled datasets of the commit, of `ceil(clips / (weight / total_weight))`: the draws one epoch needs so that each dataset is seen once at its sampling weight.
