---
name: rollout_ltx25
tool: rollout_ltx25
description: Use when planning or writing rollout_ltx25 items: what the tool takes and gives, and what its clips still need before training.
---

# What the published clip carries

- A result gives the clip and the item's `prompt`, word for word. The prompt says what was asked for, not what was rendered: check the clip against it.
- The clip has no annotation and no pose. Bring it to 960x544, run annotate_camera on the result, and write the annotation yourself (skill `data_formats`).
- One prompt covers the whole clip. For a clip in which something happens partway, find when it happens (locate_events) and start a turn there.
