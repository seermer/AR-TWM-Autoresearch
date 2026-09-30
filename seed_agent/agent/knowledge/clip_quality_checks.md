# Checking clip quality

You cannot view images, so measure. The container has internet access and `pip install --user` works, so install what helps (for example `opencv-python-headless`, `scenedetect`, or a small CLIP or aesthetic model).

- Frozen video: `ffmpeg -i in.mp4 -vf freezedetect=n=-60dB:d=1 -f null -` lists frozen spans. Or compare the mean absolute difference of consecutive downscaled frames with clips you already trust.
- Black or flat frames: `ffmpeg -i in.mp4 -vf blackdetect=d=0.2 -f null -`
- Blur: the variance of the Laplacian per frame (`cv2.Laplacian(gray, cv2.CV_64F).var()`), compared across the batch. Drop the lowest tail rather than using one absolute threshold.
- Cuts inside a clip: `scenedetect`, or a histogram difference between neighbouring frames. A cut breaks the pose and the caption.
- Camera motion: the total translation and rotation in `cam_c2w` is near zero for a static clip and clearly non-zero for a moving one. Dense optical flow (`cv2.calcOpticalFlowFarneback`) should agree on the direction.
- Caption against video: ask caption_videos a narrow question about the clip (the camera motion, the setting) and compare the answer with the saved caption.
- Sample a few clips from a source before converting all of it.
