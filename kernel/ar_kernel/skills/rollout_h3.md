---
name: rollout_h3
tool: rollout_h3
description: Use when planning or writing rollout_h3 items: what the tool takes and gives, and what its clips still need before training.
---

# How the model prompt is built from an item

- The tool writes one prompt for the whole clip: `scene_prompt` as a sentence, then "The whole video is one continuous shot with smooth motion and no cuts.", then every turn as "At MM:SS.mmm, <the turn's prompt>." with that turn's start time.
- So write a turn's `prompt` as words that read on after "At 00:03.708,": what happens from then on, the camera included. The first turn gets "At 00:00.000,".
- Turn starts: the first turn starts at frame 0. The clip holds `(frames - 25) // 32` rounds, every turn gets the same whole number of them (n), and turn k (counting from 0) starts at frame `25 + 32 * n * k`. The last turn runs to the end of the clip. A 243-frame clip has 6 rounds: 2 turns start at frames 0 and 121, 3 turns at 0, 89 and 153.
