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

- caption_videos sends each video file to the kernel's local video model (it sees the whole
  clip, not frames). One job loads the model once (minutes), then takes seconds per clip:
  batch every clip of a round into one call, then job_wait. The prompt is yours; ask for what
  the format needs, e.g. "Write one factual caption (1-3 sentences) describing the scene and
  how the camera moves."
- The result maps each path to {"caption": ...} or {"error": ...}. Save the caption as
  `{"caption": "<text>"}` in captions/<id>.json.
- For timed prompts, caption each segment's trimmed clip and snap the boundaries with
  snap_timed_prompts.

## Training failures

- A precache or training failure of any kind comes back as the next `improve_recipe`
  attempt's `retry.json` with `retry.kind == "train"`, plus a tail of the training log. This
  includes a run that wrote a checkpoint but still failed (e.g. NaN/inf loss) — that
  checkpoint is never scored. *(2026-09-27, Plan 4 as built.)*

## Generated clips

- GPU tools: rollout_alayaworld (WBench-style cases), rollout_wan22 and rollout_ltx25 (text or
  first-frame to video), generate_images (first frames for AlayaWorld cases and image-to-video
  items), annotate_camera (poses for a video). Each is a slow GPU job: batch many items per call,
  then job_wait.
- Each result item carries a ready `candidate` (video, caption, provenance) for data_ingest.
- No clip comes back with a pose. Run annotate_camera on candidate.video, then ingest it with
  that pose as 'moving'.
- Never label a clip 'static' from its prompt: the models ignore "camera steady". Run
  annotate_camera, or check the frames, first.
- AlayaWorld follows translation actions reliably, rotation (turns, orbits) only weakly. Its
  captions have one segment per round, so its clips are eligible for
  video_timed_prompts_camera:per_chunk.

## Known interface facts

*(2026-09-29, from the acceptance run's tool errors.)*

- `data_ingest` moves each staged file into the archive; copy first if you still need it.
- Candidate paths are absolute container paths (`/workspace/staging/...`), and `caption` is the path of a
  caption JSON file, not the caption text.
- `prompt_mode` in `data_commit` is only for `video_timed_prompts_camera`; omit it for other formats.
- `annotate_camera` takes at most 64 items per job; split larger batches.
- `recipe_check` takes a flat `{key: value}` map of tunable keys, with no wrapper such as `rules`.
- `unzip`, `curl`, `wget`, `git` and `7z` are in the container image; without them, `python -m zipfile` works.
- `hf_search` needs every word of the query in the dataset id or tags; check `accessible` from
  `hf_list_files` before downloading a gated repo.
- Sources that worked: TartanAirVideos zips, the `Kunho/RealEstate10K-videos` mirror clips (when
  `accessible` is true), Wan clips through `rollout_wan22`.
