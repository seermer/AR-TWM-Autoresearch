---
name: rollout_wan22
tool: rollout_wan22
description: Use when planning or writing rollout_wan22 items: what the tool takes and gives, and what its clips still need before training.
---

# What the published clip carries

- The caption published with a clip is the item's `prompt`, word for word. It says what was asked for, not what was rendered: check the clip against it, and caption the clip again where they differ (caption_videos).
- One prompt covers the whole clip, so a clip has no segments. For a clip that changes partway, render a continuation from the clip's last frame and write the segments yourself.
- Frame counts of 57, 89 and 121 are allowed here and are also 25 + 32k, so such a clip ends on a training round boundary.
