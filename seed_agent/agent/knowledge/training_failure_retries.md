---
name: training_failure_retries
description: Use when an improve_recipe attempt is a retry after a failed training run (retry.kind is train).
---

# Training failures

- A precache or training failure comes back as the next improve_recipe attempt's `retry.json` with `retry.kind == "train"` and the tail of the training log. That includes a run that wrote a checkpoint but failed anyway (for example NaN or inf loss): such a checkpoint is never scored.
