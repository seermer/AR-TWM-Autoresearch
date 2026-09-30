# Camera pose files

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4], N equal to the mp4 frame count: camera-to-world, OpenCV convention (x right, y down, z forward).
- From world-to-camera matrices, invert them.
- From OpenGL convention (y up, z backward): `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`
- A clip is `static` only if a measurement shows the camera does not move (see clip_quality_checks.md), never because of its prompt or filename.
