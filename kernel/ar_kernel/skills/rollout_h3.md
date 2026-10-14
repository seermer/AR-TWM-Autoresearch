---
name: rollout_h3
tool: rollout_h3
description: Use when planning or writing rollout_h3 items: what the tool takes and gives, and what its clips still need before training.
---

# How the model prompt is built from an item

- The tool writes one prompt for the whole clip, as one continuous shot: `scene_prompt`, then `character_prompt`, then the first turn's perspective, then every turn at the time it starts.
- A turn is written as its `event`, then the camera move of its `action` in its `perspective`.
- So write `scene_prompt` and `character_prompt` as descriptions of what is there, and each `event` as one sentence about what happens. Do not describe the camera: the action does.
- The clip is split into turns of equal length. Each turn lasts at least 2.4 s, so the clip's `seconds` bounds how many turns it holds.
- The published annotation holds your texts and, for each turn, the time it was asked to start. The model follows those times loosely: find when each event happens (locate_events), set the turns' starts to match, and leave out an event the clip does not show.
