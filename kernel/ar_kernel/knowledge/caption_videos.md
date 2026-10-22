---
name: caption_videos
tool: caption_videos
description: Use when clips need a scene description, or when many clips must be checked by a model that sees the whole clip: what caption_videos takes and gives.
---

# What the model sees, and what else it can do

- The model is shown the clip sampled at 2 frames per second, with frames scaled down. Something that lasts well under a second, or a small detail, can be missed.
- It answers `prompt` once per clip. The prompt decides the style: ask for the text the model should be trained with (knowledge file `training_text`), for example a description of the scene for `scene_prompt`.
- `prompt` can be any question about a whole clip, asked of many clips at once: whether an event happens, whether a text matches the clip, how the camera moves. Use it for checks over a full set; it does not count against the images of `ask`.
- It sees one file at a time and gives no times: locate_events says when an event happens.
- A cut is a jump between two neighboring frames. Frames sampled this far apart cannot show one: annotate_camera reports cuts.
