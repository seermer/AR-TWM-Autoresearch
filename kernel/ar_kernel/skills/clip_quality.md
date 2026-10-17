---
name: clip_quality
description: Use when checking clips for defects before ingesting them: frozen, black, blurry or cut video, camera motion that disagrees with the pose, texts that disagree with the video.
---

# Checking clip quality

Install what helps with `pip install --user` (for example `scenedetect`, or a small CLIP or aesthetic model).

- Frozen video: `ffmpeg -i in.mp4 -vf freezedetect=n=-60dB:d=1 -f null -` lists frozen spans. Or compare the mean absolute difference of consecutive downscaled frames with clips you already trust.
- Black or flat frames: `ffmpeg -i in.mp4 -vf blackdetect=d=0.2 -f null -`
- Blur: the variance of the Laplacian per frame (`cv2.Laplacian(gray, cv2.CV_64F).var()`), compared across the batch. Drop the lowest tail rather than using one absolute threshold.
- Cuts inside a clip: a cut is a jump between two neighboring frames, so it cannot be judged from a sheet of sampled frames or from frames a second apart. annotate_camera reports a cut in its result (`cut`), where the picture and the measured camera both jump, and data_ingest rejects a clip that has one. Poses measured across a cut are not valid.
- Camera motion: the total translation and rotation in `cam_c2w` are near zero for a camera that stays put and clearly non-zero for one that moves. Dense optical flow (`cv2.calcOpticalFlowFarneback`) should agree on the direction.
- The turns of a generated clip may not be reliable (what a turn asked for can show up early, late, weakly or not at all), so find when each event happens (locate_events) before using its turn as a label.
- Text against video: ask the captioning tool a narrow question about the clip (the camera motion, the setting) and compare the answer with the text you have.
- Sample a few clips from a source before converting all of it.
