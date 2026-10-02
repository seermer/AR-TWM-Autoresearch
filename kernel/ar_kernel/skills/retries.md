---
name: retries
description: Use when this attempt is a retry (`retry` is set in the context): what each kind of failure means and what the retry carries.
---

# Retries after a failed attempt

The context's `retry` describes the attempt that failed; its `kind` says where it failed.

- `edit_self`: the phase itself failed (no valid result); `error` says why.
- `contract`: the kernel's check of the edited code failed; `failed_step` and `detail` say where, and `steps` lists every step. The edited tree is kept: `/agent` holds the failed attempt's edits.
- `improve_recipe`: the phase itself failed (no valid submission); `error` says why.
- `gate`: the kernel's pre-training check refused the submission; `failures` lists why.
- `train`: precache or training failed; `failure`, `detail` and `log_tail` (the last 20,000 characters of the training log) say how. That includes a run that wrote a checkpoint but failed anyway (for example a NaN loss): such a checkpoint is never scored.
- Gate and training retries also carry the submitted `data_commit`, `recipe` and `rationale`. The data commit is still valid and can be submitted again with a different recipe.
- The workspace of the failed attempt is carried over, including its `plans.json`.
