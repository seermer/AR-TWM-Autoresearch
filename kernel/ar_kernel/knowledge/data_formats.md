---
name: data_formats
description: Use when preparing a clip, its annotation or its camera poses for data_ingest: what the kernel accepts, and how to convert to it.
---

# Data formats

## What data_ingest takes

A candidate is a video, an annotation, a pose file and `derived_from`.

- Video: .mp4, exactly 960x544, fps >= 24 (higher is subsampled), at least 2.4 s long, no display rotation (re-encode with rotation applied), and one continuous shot: a clip with a cut is rejected.
- Annotation: a JSON file. An example with two turns:

  ```json
  {"scene_prompt": "A dry savanna at noon, an acacia tree on the left, low yellow grass.",
   "character_prompt": "A hiker in a red jacket.",
   "turns": [{"start_s": 0.0, "perspective": "first_person"},
             {"start_s": 4.0, "event": "...", "perspective": "first_person"}]}
  ```

  - `scene_prompt` is required. `character_prompt` is the subject, when the clip has one.
  - `turns`: the clip, turn by turn. `start_s` is when the turn starts, in seconds of the clip. The first is 0 and each later one is larger than the one before. A turn ends where the next starts; the last ends with the clip.
  - `event`: one sentence, or left out. At most one per turn.
  - `perspective`: `first_person` or `third_person`. Required on every turn.
  - The first turn lasts at least 2.4 s, every later turn at least 1.4 s.
  - Knowledge file `training_text` says what text the model is trained with.
- Poses: an npz with cam_c2w [N,4,4] with N = the mp4's frame count, camera-to-world, OpenCV convention, finite, bottom row [0,0,0,1], orthonormal rotation with det +1. Optional intrinsics [3,3] or [N,3,3] in pixels. annotate_camera writes this file. Every clip needs one.
- `derived_from`: `{"kind": ..., "name": ...}`, what the clip was made from, for a person to read. `kind` is `model`, `dataset` or `synthetic_method`; `name` is the model, the dataset or the method. A rollout tool's result carries it filled in; change it when you made something else of the clip.
- Each dataset of a commit needs at least as many clips as training GPUs.
- Clips are immutable: a re-annotated, cropped or trimmed clip is a new clip.

## Converting a video

- Probe: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation -of json in.mp4`
- Display aspect = width * SAR / height, with width and height swapped for a 90 or 270 degree rotation.
- 960x544 is close to 16:9. Bring a clip of another shape to it by cropping, then scale the crop to 960x544; ffmpeg or a Python script can do this. Never pad with bars, and never change only the aspect label of the file.
- Choose the crop from the frames, so that the subject and the action stay inside it for the whole clip. The center is often the wrong crop.
- Drop a clip when the crop would lose the subject or most of the picture.
- Crop or resize first, then annotate the camera: a pose measured on another size or crop does not fit the clip you ingest.
- Re-encoding applies any display rotation. Encode to 8-bit 4:2:0 so that 10-bit or 4:4:4 sources stay decodable.
- Keep the source frame rate if it is at least 24 fps. Never raise it by duplicating frames.
- Trimming changes the frame count, so slice the pose array to the same frames, and move the turns' starts.

## Camera poses

- The pose npz holds `cam_c2w` with shape [N, 4, 4]: camera-to-world, OpenCV convention (x right, y down, z forward).
- From world-to-camera matrices, invert them.
- From OpenGL convention (y up, z backward): `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`.
