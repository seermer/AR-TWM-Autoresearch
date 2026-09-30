---
name: captioning_with_caption_videos
description: Use when captioning clips with the caption_videos tool or saving caption files.
---

# Captioning with caption_videos

- Each video goes to the kernel's local video model, which sees the whole clip. A job loads the model once (minutes), then takes seconds per clip, so send every clip of a round in one call and then job_wait.
- The prompt is yours. Ask for what the format needs, for example "Write one factual caption (1-3 sentences) describing the scene and how the camera moves."
- The result maps each path to `{"caption": ...}` or `{"error": ...}`. Save `{"caption": "<text>"}` as `captions/<id>.json` under `/workspace/staging/`: data_ingest only takes staged files. Retry an error, or write that caption another way.
