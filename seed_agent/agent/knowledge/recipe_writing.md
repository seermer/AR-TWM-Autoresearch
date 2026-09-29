## Constraints the kernel checks

- `sample.height` and `sample.width` must be a pair from `resolution_allowlist`, and `lora.rank` and `lora.alpha` a pair from `lora_allowlist`.
- `steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps)` must be at least 1, and `optimizer.epochs * steps_per_epoch` must be at least `optimizer.max_steps`.
- Training time grows with `optimizer.max_steps`. A run over the wall-time limit fails.

## Where to look

- Scale `optimizer.max_steps` and `optimizer.grad_accum_steps` to the number of windows the data gives and to the time budget.
- Set `optimizer.lr` by how far the new data is from the model's own outputs.
- `lora.rank` sets how much the adapter can change.
- Compare `parent_train` (loss trend, gradient norm, seconds per step) with the parent's recipe before choosing.
