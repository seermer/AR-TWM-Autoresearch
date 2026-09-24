## Converting clips to the standard formats

- Probe first: `ffprobe -v error -select_streams v:0 -show_entries
  stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation
  -of json in.mp4`. Display aspect = width * SAR / height, with width and height swapped for
  a 90 or 270 degree rotation.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf
  "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -crf 18 -an out.mp4`.
  Re-encoding also applies any display rotation.
- Frame rate: keep the source rate if it is >= 24 fps; never raise it by duplicating frames.
- Trimming changes the frame count: slice the pose array to the same frames (N poses for
  N frames).

## Camera poses

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4]: camera-to-world, OpenCV convention
  (x right, y down, z forward).
- From world-to-camera matrices, invert them. From OpenGL convention (y up, z backward),
  flip the y and z axes of the camera frame: `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`.

## Timed prompts

- Segment boundaries must fall on round boundaries: 25/24 s, then every 32/24 s. Use
  snap_timed_prompts after writing the segments.

## Captions

- caption_clip sends video frames to the agent model, so it needs a vision-capable model; with
  a text-only model every call returns a tool error, so write captions another way.
