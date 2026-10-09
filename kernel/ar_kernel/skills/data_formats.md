---
name: data_formats
description: Use when preparing clips, captions, timed prompt segments or camera poses for data_ingest: the only formats the kernel accepts, and how to convert to them.
---

# Data formats

## The standard formats (the only ones accepted)

- video_caption_camera: video + caption + per-frame camera poses.
- video_timed_prompts_camera: as above + caption "segments": [{"time_range_s": [start, end), "prompt"}].
  per_chunk mode: every boundary inside the clip on a round boundary 25/24 + k*32/24 s (within half a frame).
  segment mode: segments shorter than 2.375 s are never trained.
- video_caption_static: video + caption; truly fixed camera; no poses (identity is used).
- Training window: 57 frames at 24 fps (25 history + 32 target). Rollout rounds are 32 frames.
- Video: .mp4, fps >= 24 (higher is subsampled), duration >= 2.375 s, DISPLAY aspect within 2% of 16:9,
  no display rotation (re-encode with rotation applied). Frames are resized, never cropped.
- Caption: non-empty "caption" string.
- Poses: cam_c2w [N,4,4] with N = the mp4's frame count, camera-to-world, OpenCV convention, finite,
  bottom row [0,0,0,1], orthonormal rotation with det +1. Optional intrinsics [3,3] or [N,3,3] in pixels.
- Each enabled dataset needs at least as many clips as training GPUs.
- Clips are immutable: a re-captioned, cropped or trimmed clip is a new clip with derived_from set.

## Converting a video

- Probe: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation -of json in.mp4`
- Display aspect = width * SAR / height, with width and height swapped for a 90 or 270 degree rotation.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -pix_fmt yuv420p -crf 18 -an out.mp4`. Re-encoding also applies any rotation, and `-pix_fmt yuv420p` keeps 10-bit or 4:4:4 sources decodable.
- Keep the source frame rate if it is at least 24 fps. Never raise it by duplicating frames.
- Trimming changes the frame count, so slice the pose array to the same frames.

## Camera poses

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4]: camera-to-world, OpenCV convention (x right, y down, z forward).
- From world-to-camera matrices, invert them.
- From OpenGL convention (y up, z backward): `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`.
- A clip is `static` only if a measurement shows the camera does not move (skill `clip_quality`), never because of its prompt or file name.

## Timed prompt segments

- Segment boundaries fall on round boundaries: 25/24 s, then every 32/24 s.
- A clip of 25 + 32k frames (57, 89, 121, 153, ...) ends on a round boundary.
- Caption each segment from its own part of the clip.
- A text-or-image-to-video tool given a clip's last frame as its first frame continues that clip; the continuation starts on that same frame, so a 97-frame continuation adds 96 new frames (3 rounds).
