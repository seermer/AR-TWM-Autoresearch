# Recipe constraints the kernel checks

- `sample.height` and `sample.width` are a pair from `resolution_allowlist`; `lora.rank` and `lora.alpha` are a pair from `lora_allowlist`.
- `steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps)` must be at least 1, and `optimizer.epochs * steps_per_epoch` at least `optimizer.max_steps`.
