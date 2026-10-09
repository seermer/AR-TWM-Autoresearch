# Aspect rule and tool sizes (agreed 2026-10-09, implemented 2026-10-09)

Status: implemented. Decided while implementing: `rollout_alayaworld`'s subject mask must also be 960x544; a
frame image is checked again on the staged copy the job uses (an item error there); no GPU run was made, so
the H3 times and memory at 1376x768, 1344x768 and 1024x576 with 29 steps are not measured.

## Why

- Ingest accepts any resolution but only clips within 2% of 16:9, because training stretches every
  frame to the recipe's size without cropping (`WorldModel/fastvideo/dataset/t2v_datasets.py`,
  `_process_frame`; `WorldModel/docs/TRAINING.md`, "Aspect ratio close to 16:9").
- The 2% rule stays. What is wrong is how off-aspect material is handled:
  - the `data_formats` skill gives one blind centre-crop command;
  - the ingest rejection says only "is not within 2% of 16:9";
  - `rollout_h3` and `rollout_ltx25` centre-crop keyframes silently, and `rollout_alayaworld`
    stretches its first frame silently.
- The default sizes of `generate_images` (1280x720) and `rollout_h3` (960x544) are not the sizes the
  benchmark's first frames use.

## Facts checked

- Benchmark first-frame sizes (289 cases): 1376x768 (186), 1536x1024 (72), 2752x1536 (14),
  1344x768 (9), 4096x3072 (3), 1024x571 (2), 1331x768 (1), 1536x864 (1), 3072x4096 (1).
- The evaluation and `rollout_alayaworld` always render 544x960 (`WorldModel/configs/wbench_full.yaml`);
  a node's training size does not change that. The evaluation stretches each first frame to it.
- The recipe sets the training size (`sample.height`, `sample.width`, a pair from
  `train.resolution_allowlist`); it is not inferred from the data.
- The allowlist was chosen for GPU memory (`WorldModel/docs/LOWCOMPUTE.md`): 416x736 is the
  low-compute default, 352x608 the out-of-memory fallback, 544x960 was added 2026-10-06. 352x608 is
  2.8% off 16:9.
- MiniMax H3 needs height and width in multiples of 32.
- A rejected ingest row already names its clip: `index` and `video` (the staged path), for every check.
- H3 at 1376x768 was measured once, at 4 steps: 16 s per step, 14 GiB per card. About 8 minutes per
  clip at 29 steps is a projection.

## The rule

Applied in this order by `generate_images` and every enabled rollout tool:

1. The requested size must be within 2% of 16:9.
2. It must be on the tool's own list.
3. A keyframe or first-frame image must pass check 1 and be exactly the size the job renders. It is
   then used as it is: no crop and no resize in the bridge.
4. A failure refuses the item at submit (the existing `refuse` path): nothing is submitted, and the
   refused-items file names each item's index and the check it failed.

## Per tool

| tool | sizes | change |
|---|---|---|
| `generate_images` | multiples of 16 in 256..1920, 2% rule first; default 1376x768 | new default, new check |
| `rollout_h3` | 1376x768 (default), 1344x768, 1024x576, 960x544 | size list, `height`/`width` arguments, keyframe check, no crop |
| `rollout_ltx25` | 1024x576 | keyframe check, no crop |
| `rollout_alayaworld` | 960x544 | first frame must be 960x544; no stretch |
| `rollout_wan22` (switched off) | - | untouched; a config note that it does not meet the rule |

1536x1024 (3:2) is offered by no tool. The three benchmark sizes above 1920 on a side are too large.
1024x576 stands in for 1024x571, and 1344x768 for 1331x768.

## Elsewhere

- `data_formats` skill, "Converting a video": remove the centre-crop command. Say: prefer material
  that is already 16:9 and generate at the clip size; if cropping, choose the window from the frames,
  not the centre; drop a clip rather than lose the subject or most of the picture (judgment, no
  number); never pad with bars, never stretch, never relabel the pixel aspect; crop first, then
  annotate the camera. Mention that ffmpeg or Python can do the crop. No commands or code examples.
- Ingest: same rule as today. The rejection reason states the clip's aspect and how far it is from
  16:9 and points to the skill; the `data_ingest` description gets the same pointer. No size list for
  ingest. Row identification stays as it is.
- Training: remove `[352, 608]` from `train.resolution_allowlist` (one config line, one test).
  Not checked: whether an existing node's recipe uses it.
- Agent-facing text: update the descriptions, argument hints and skills of these tools (including the
  `generate_images` skill's "keep the default 1280x720"). None of it may name the benchmark.

## Order of work

1. Text and checks: skill, ingest reason, the rule in every tool, allowlist, `generate_images`
   default, with tests.
2. `rollout_h3` size list and `height`/`width` arguments. Its real smoke test generates keyframes at
   960x544 today; make it render at the new default.
3. GPU runs on GPUs 0-3 only: H3 at 1376x768 with 29 steps for a measured time, then fit tests at
   1344x768 and 1024x576. A size that does not fit is left off the list and reported.
4. Full test suite, the review, a report.

## State of the tree when this was written

- Committed and pushed: `e3147d4`, `rollout_h3` through LightX2V.
- Uncommitted, the user's own edits: `skills/rollout_alayaworld.md`, `tools/rollouts.py`, one
  sentence of the `rollout_h3` description in `tools/h3.py`.
- Uncommitted, engine benchmark leftovers: four comment lines in `scripts/setup_envs.sh`,
  `envs/gen-sglang.*`, `envs/gen-vllm-omni.*`.
