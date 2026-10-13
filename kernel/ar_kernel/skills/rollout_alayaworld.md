---
name: rollout_alayaworld
tool: rollout_alayaworld
description: Use when planning or writing rollout_alayaworld items: what the tool takes and gives, and the text the model is given in each turn.
---

# The text the model is given in each turn

- The tool writes one text per turn from the item's fields: `scene_prompt`, then `character_prompt`, then the turn's `event`. Nothing is carried over from earlier turns.
- On a turn whose `perspective` differs from the turn before it, the tool adds the sentence that says the view switches. That turn's `action` must be `stop`.
- A turn's `action` is not text: it drives the camera control directly.
- The first turn's `perspective` decides how the actions are carried out for the whole clip: in `third_person` the camera orbits the subject, in `first_person` it turns and moves as the viewer's own eyes.
- So write `scene_prompt` and `character_prompt` as descriptions of what is there, and each `event` as one sentence that reads naturally after them, not as a command to the model.
- The published annotation holds your texts and, for each turn, the time it starts in the clip. It says what was asked for, not what the clip shows: an event can show up early, late, weakly or not at all. Find when each event happens (locate_events), set the turns' starts to match, and leave out an event the clip does not show.
