---
name: annotate_camera
tool: annotate_camera
description: Use when a clip needs a camera pose, or when its camera motion must be measured: what annotate_camera takes and gives.
---

# Using the poses

- A pose file covers exactly the frames of the mp4 that was passed. Annotate the file you will ingest; if you trim a clip after annotating it, slice its poses to the same frames (skill `data_formats`).
- A cut inside a clip makes the poses after it meaningless: find cuts first (skill `clip_quality`).
- The poses also answer whether the camera moves at all: near-zero total translation and rotation means a fixed camera, which is ingested as `static` with no pose (skill `clip_quality` has the checks).
- rollout_alayaworld's `commanded_camera` is the path its moves asked for, not what the video shows: annotate the clip instead of ingesting that file as its pose.
