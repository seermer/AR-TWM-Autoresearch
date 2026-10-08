---
name: generate_images
tool: generate_images
description: Use when a rollout needs a first or last frame that does not exist yet: what generate_images takes and gives.
---

# Using an image as a frame

- An image is not training data by itself: it becomes the first frame of a rollout_alayaworld item, or a keyframe of rollout_h3 or rollout_ltx25.
- rollout_h3 and rollout_ltx25 center-crop a keyframe to their clip size, which is close to 16:9. Keep the default 1280x720 so that nothing at the edges is cut off.
- The clip starts from this frame, so write the image prompt to show what the scene text of its item describes, from the viewpoint the item names.
