You write the node's training recipe: values for tunable keys only (listed in the context
with their types and bounds). Everything else comes from the base recipe. Hyperparameter-only
changes are not allowed: the recipe exists to fit the data you built.

Checks the kernel runs, so get them right the first time:
- sample.height/sample.width must be an allowed resolution pair; lora.rank/lora.alpha an
  allowed pair.
- steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps) must
  be >= 1, and optimizer.epochs * steps_per_epoch >= optimizer.max_steps.
- Training time grows with max_steps; a run over the wall-time limit fails.

The context has `base_recipe` (the defaults), `recipe_guide` (what each tunable does, its base
value, what is worth trying and what it costs) and `parent_train` (how the parent's training
went). Do not just copy the parent's recipe: choose the settings that fit the data you built and
say in the rationale which ones you changed from the base and why, or why you kept them. Good
candidates: `optimizer.max_steps` and `grad_accum_steps` scaled to the number of windows and the
time budget, `optimizer.lr` for how far the new data is from the model's own outputs, and
`lora.rank`. Recipe changes alone never count: the data commit must differ from the parent's.

If you are given failures from a previous check, fix exactly those. Call submit_recipe with
the recipe (tunable key -> value) and a short rationale tying the recipe to the data.
