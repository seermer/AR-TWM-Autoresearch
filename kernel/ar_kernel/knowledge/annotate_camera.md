---
name: annotate_camera
tool: annotate_camera
description: Use when a clip needs a camera pose, or when its camera motion must be measured: what annotate_camera takes and gives.
---

# Using the poses

- A pose file covers exactly the frames of the mp4 that was passed. Annotate the file you will ingest, at 960x544; if you trim a clip after annotating it, slice its poses to the same frames (knowledge file `data_formats`).
- `cut` in a result says where the clip cuts to another shot: the picture and the measured camera both jump there. data_ingest rejects such a clip. Trim the clip to one shot and annotate that file again.
- rollout_alayaworld's `commanded_camera` is the path its actions asked for, not what the video shows: annotate the clip instead of ingesting that file as its pose.
