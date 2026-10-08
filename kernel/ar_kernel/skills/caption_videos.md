---
name: caption_videos
tool: caption_videos
description: Use when clips need captions, or when many clips must be checked by a model that sees the whole clip: what caption_videos takes and gives.
---

# What the model sees, and what else it can do

- The model is shown the clip sampled at 2 frames per second, with frames scaled down. Something that lasts well under a second, or a small detail, can be missed.
- It answers `prompt` once per clip. The prompt decides the style: ask for the text the model should be trained with (skill `training_text`).
- `prompt` can be any question about a whole clip, asked of many clips at once: whether an event happens, whether a saved caption matches, how the camera moves. Use it for checks over a full set; it does not count against the images of `ask`.
- It sees one file at a time and gives no times. For a segment's prompt, cut that part of the clip into its own file and caption the file.
