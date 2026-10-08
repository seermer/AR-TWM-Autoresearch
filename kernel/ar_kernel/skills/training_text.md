---
name: training_text
description: Use when writing captions or timed prompt segments, or choosing a prompt_mode: what text the model is trained with.
---

# What text the model is trained with

- A prompt is the model's whole text condition for the frames it covers. It describes what is on screen during those frames; it is not an instruction relative to earlier text.
- Write it as prose: who and what is there, and what happens.
- `video_caption_camera` and `video_caption_static`: every training window of a clip uses the clip's `caption`.
- `video_timed_prompts_camera` with `prompt_mode: per_chunk`:
  - Windows start only on round boundaries, one per round.
  - A window is trained with the prompt active at the middle of its 32 target frames.
  - When the prompt changes at a round boundary, that round's window has its history under the old prompt and its target under the new one. This is what teaches the model to follow a prompt that changes during a rollout.
  - The clip's `caption` is never used.
- `video_timed_prompts_camera` with `prompt_mode: segment`:
  - Each draw picks one segment and places the window at a random position inside it.
  - The window is trained with that segment's prompt, or with the clip's `caption` with probability `data.overall_caption_prob` (0.1 unless the recipe sets it).
  - A window never spans two prompts, so this mode does not teach a change of prompt.
  - Segments shorter than 2.375 s are never drawn.
- Choose `per_chunk` when the point is to react to a prompt that changes mid-rollout (an event, an action, a new viewpoint). Choose `segment` for long, steady stretches described part by part: it gives more distinct windows per clip.
