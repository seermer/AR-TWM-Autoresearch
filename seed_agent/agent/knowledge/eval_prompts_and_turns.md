---
name: eval_prompts_and_turns
description: Use when you need to know how the WBench eval drives AlayaWorld (turns, rounds, the prompt of each turn, camera actions) or how its results are aggregated into metrics, dimensions, case groups and the node score.
---

# How the eval runs a case and how its results are aggregated

## Running a case

- A case is a first frame plus a list of turns. Each turn is 3 rounds of 32 frames: 96 frames, 4 s at 24 fps. The published video drops the last 7 frames, so its last turn has 89.
- A turn is a navigation action, an event edit, a subject action or a perspective switch. The four kinds are scored by different metrics (`navigation_trajectory`, `event_edit_adherence`, `subject_action_adherence`, `perspective_switch_adherence`).
- One prompt is used for all 3 rounds of a turn. It is the character description, then the environment description, then the instruction text of every event edit and subject action so far, each appended as written (turn 1 ends "... Climb.", turn 2 "... Climb. Descend."). The instruction of a turn is in the prompt from the first frame of that turn.
- A perspective switch adds one sentence for its own turn ("The view switches from first-person to third-person ...", or "The camera cuts to a different third-person viewpoint: ...") and a holding sentence on later turns ("The view remains ..."). Only the latest switch is stated.
- The prompt carries no camera text. Navigation is given as camera poses: per 8 frames, 0.16 scene units of translation or 6 degrees of rotation. An event, subject-action or perspective turn repeats the previous navigation action; a case with no navigation turn moves forward (W) throughout.
- The model is rendered with a 4-step student adapter plus the node's fine-tune, from the same released weights at every node.

## Aggregating the results

- Metric: each case gets a value between 0 and 1 for every metric that applies to it. A metric's value is the mean over the cases it applies to, so metrics rest on different numbers of cases.
- Score: the score is a weighted mean of the 22 metric means: `event_edit_adherence`, `subject_action_adherence`, `perspective_switch_adherence` and `causal_fidelity` weigh 4.5 each, every other metric weighs 1. Those four are half the score.
- Dimension: the plain mean of its metrics' values. The five dimensions are not weighted and are not part of the score.
  - `quality`: aesthetic_quality, imaging_quality, temporal_flickering, dynamic_degree, motion_smoothness, hpsv3_quality.
  - `consistency`: background_consistency, segment_continuity, perspective_consistency, subject_consistency, geometric_consistency, photometric_consistency, spatial_consistency, gated_spatial_consistency.
  - `interaction`: navigation_trajectory, event_edit_adherence, subject_action_adherence, perspective_switch_adherence.
  - `setting`: scene_adherence, subject_adherence.
  - `physical`: visual_plausibility, causal_fidelity.
- Case group: the same five dimensions computed over one group of cases alone: the cases with a given interaction type, perspective or scene category. A case with turns of several types is in several interaction-type groups. A group's `interaction` holds only the metrics that apply to its cases, so for the perspective-switch group it is `perspective_switch_adherence`.
- Where they are: the first message has the score, dimension and metric tables and the groups by interaction type and perspective. `/context/context.json` has all of it for each lineage node and sibling under `aggregates` (`metrics`, `dimensions`, `strata`), including the scene-category groups.
