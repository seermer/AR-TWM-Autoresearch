## Converting clips to the standard formats

- Probe first: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation -of json in.mp4`
- Display aspect is width * SAR / height, with width and height swapped for a 90 or 270 degree rotation. It must be 16:9 within 2%, with no display rotation left in the file.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -crf 18 -an out.mp4`
- Re-encoding also applies any display rotation.
- Keep the source frame rate if it is at least 24 fps. Never raise it by duplicating frames.
- Trimming changes the frame count, so slice the pose array to the same frames (N poses for N frames).

## Camera poses

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4], N equal to the mp4 frame count: camera-to-world, OpenCV convention (x right, y down, z forward).
- From world-to-camera matrices, invert them.
- From OpenGL convention (y up, z backward), flip the y and z axes of the camera frame: `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`
- Never label a clip `static` from its prompt or filename. Check it with the methods under "Checking clip quality".

## Timed prompts

- Segment boundaries fall on round boundaries: 25/24 s, then every 32/24 s. Write the segments, then call snap_timed_prompts.
- For timed prompts, caption each segment's trimmed clip.

## Captions

- caption_videos sends each video file to the kernel's local video model, which sees the whole clip and not frames. One job loads the model once (minutes), then takes seconds per clip, so batch every clip of a round into one call and then job_wait.
- The prompt is yours. Ask for what the format needs, for example "Write one factual caption (1-3 sentences) describing the scene and how the camera moves."
- The result maps each path to `{"caption": ...}` or `{"error": ...}`. Save the caption as `{"caption": "<text>"}` in `captions/<id>.json`. A clip whose entry is an error needs another try or a caption written another way.

## Checking clip quality

You cannot view images, so measure. The container has internet access and `pip install --user` works, so install whatever helps (for example `opencv-python-headless`, `scenedetect`, or a small CLIP or aesthetic model) and pick the method that fits the defect.

- Frozen or static video: `ffmpeg -i in.mp4 -vf freezedetect=n=-60dB:d=1 -f null -` lists frozen spans. Or take the mean absolute difference between consecutive downscaled frames and compare it with the moving clips you already trust.
- Black or flat frames: `ffmpeg -i in.mp4 -vf blackdetect=d=0.2 -f null -`
- Blur: the variance of the Laplacian per frame (`cv2.Laplacian(gray, cv2.CV_64F).var()`), compared across the batch. Drop the lowest tail rather than using one absolute threshold.
- Cuts and scene changes inside a clip: `scenedetect` or a histogram difference between neighbouring frames. A clip that contains a cut breaks the pose and the caption.
- Camera motion against the pose: the total translation and rotation in `cam_c2w` should be small for a static clip and clearly non-zero for a moving one. Dense optical flow (`cv2.calcOpticalFlowFarneback`) between frames should agree on the direction.
- Caption against video: ask caption_videos a narrow question about the clip (for example the camera motion or the setting) and compare the answer with the caption you saved.
- Sample a few clips from each source before converting all of it.

## Generated clips

- GPU tools: rollout_alayaworld (WBench-style cases), rollout_wan22 and rollout_ltx25 (text or first-frame to video), generate_images (first frames for AlayaWorld cases and image-to-video items), and annotate_camera (poses for a video). Each is a slow GPU job: batch many items per call, then job_wait.
- annotate_camera takes at most 64 items per job; split larger batches.
- Each result item carries a ready `candidate` (video, caption, provenance) for data_ingest.
- No clip comes back with a pose. Run annotate_camera on `candidate.video`, then ingest the clip with that pose as `moving`.
- Video models ignore "camera steady", so run annotate_camera or the checks above before calling a generated clip `static`.
- AlayaWorld follows translation actions reliably and rotation (turns, orbits) only weakly. Its captions have one segment per round, so its clips are eligible for `video_timed_prompts_camera:per_chunk`.

## Tool interface facts

- hf_download lands files under `/workspace/staging/hf/` and returns a ready provenance record. For a clip you derive (cropped, trimmed, re-captioned) use `{"kind": "derived", "from": [clip_ids], "transform": "<what you did>"}` and set `derived_from`.
- hf_search needs every word of the query in the dataset id or tags. hf_list_files shows `accessible`; do not download a repo where it is false.
- data_ingest takes candidates under `/workspace/staging/` and moves each staged file into the archive, so copy first if you still need it. Candidate paths are absolute container paths, and `caption` is the path of a caption JSON file, not the caption text.
- `prompt_mode` in data_commit applies only to `video_timed_prompts_camera`; omit it for other formats.
- job_wait waits at most 300 s per call; call it again until the job is done.
- recipe_check takes a flat `{key: value}` map of tunable keys, with no wrapper such as `rules`.
- `unzip`, `curl`, `wget`, `git` and `7z` are in the container image. Without them, `python -m zipfile` works.

## Training failures

- A precache or training failure of any kind comes back as the next improve_recipe attempt's `retry.json` with `retry.kind == "train"`, plus a tail of the training log. This includes a run that wrote a checkpoint but still failed (for example NaN or inf loss): that checkpoint is never scored.
