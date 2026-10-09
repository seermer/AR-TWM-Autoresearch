---
name: rollout_alayaworld
tool: rollout_alayaworld
description: Use when planning or writing rollout_alayaworld items: what the tool takes and gives, and the text the model is given in each round.
---

# The text the model is given in each round

- The tool writes one prompt per turn from the item's fields, and the model renders every round of that turn with it:
  - the scene in prose: `character_prompt`, then `scene_prompt`;
  - then every `event` and `subject_action` given so far, in turn order: an instruction stays in the text of every later turn;
  - then, once a `viewpoint_change` was given, a sentence the tool writes from it: on that turn that the view switches (with the text after the code's colon, if any), on later turns that the view remains so. A newer change replaces it.
- A turn's `action` (W, A, S, D, left, right, up, down, stop) is not text: it drives the camera control directly.
- `viewpoint` decides how those moves are carried out: in `third_person` the camera orbits the subject, in `first_person` it turns and moves as the viewer's own eyes.
- So write `scene_prompt` and `character_prompt` as descriptions, and each `event` or `subject_action` as a sentence that reads naturally after the scene text, not as a command to the model.
- The published caption holds one segment per round with the prompt that round was asked to render (equal neighbours merged). It says what was asked for, not what the clip shows: check each round against the frames and caption the clip again where they differ (caption_videos).
- The clips are on the per_chunk round grid, and the prompt changes at each turn that adds an instruction and on the turn after a viewpoint change: they fit `video_timed_prompts_camera` with `prompt_mode: per_chunk`.
