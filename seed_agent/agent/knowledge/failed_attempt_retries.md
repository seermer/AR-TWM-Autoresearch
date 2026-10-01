---
name: failed_attempt_retries
description: Use when an improve_recipe attempt is a retry (`retry` is set): what each kind of failure means and what the retry carries.
---

# Retries after a failed attempt

- `retry.kind == "improve_recipe"`: the attempt itself failed (no valid submission); `retry.error` says why.
- `retry.kind == "gate"`: the kernel's pre-training check refused the submission; `retry.failures` lists why.
- `retry.kind == "train"`: precache or training failed; `retry.failure`, `retry.detail` and `retry.log_tail` (the last 20,000 characters of the training log) say how. That includes a run that wrote a checkpoint but failed anyway (for example NaN or inf loss): such a checkpoint is never scored.
- Gate and training retries also carry the submitted `data_commit`, `recipe` and `rationale`. The data commit is still valid and can be submitted again with a different recipe.
- The failed attempt's plans, with the engineer's reports, follow the failure in the Retry section and are in `/workspace/plans.json`.
