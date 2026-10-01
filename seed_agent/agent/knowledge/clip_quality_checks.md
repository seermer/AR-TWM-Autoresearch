---
name: clip_quality_checks
description: Use when checking clips for defects before ingesting them: frozen, black, blurry or cut video, camera motion that disagrees with the pose, captions that disagree with the video.
---

# Checking clip quality

Install what helps with `pip install --user` (for example `scenedetect`, or a small CLIP or aesthetic model).

- Frozen video: `ffmpeg -i in.mp4 -vf freezedetect=n=-60dB:d=1 -f null -` lists frozen spans. Or compare the mean absolute difference of consecutive downscaled frames with clips you already trust.
- Black or flat frames: `ffmpeg -i in.mp4 -vf blackdetect=d=0.2 -f null -`
- Blur: the variance of the Laplacian per frame (`cv2.Laplacian(gray, cv2.CV_64F).var()`), compared across the batch. Drop the lowest tail rather than using one absolute threshold.
- Cuts inside a clip: `scenedetect`, or a histogram difference between neighbouring frames. Poses estimated across a cut are not valid, and one caption cannot describe both shots. data_ingest adds a warning to a clip whose pose jumps between two frames (a step 15 times the clip's median step, or a turn of 20 degrees), naming the frames.
- Camera motion: the total translation and rotation in `cam_c2w` is near zero for a static clip and clearly non-zero for a moving one. Dense optical flow (`cv2.calcOpticalFlowFarneback`) should agree on the direction.
- Caption against video: ask caption_videos a narrow question about the clip (the camera motion, the setting) and compare the answer with the saved caption.
- Sample a few clips from a source before converting all of it.
