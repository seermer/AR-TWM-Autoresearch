---
name: generate_images
tool: generate_images
description: Use when a rollout needs a first or last frame that does not exist yet: what generate_images takes and gives.
---

# Using an image as a frame

- An image is not training data by itself: it becomes the first frame of a rollout_alayaworld item, or a keyframe of rollout_h3 or rollout_ltx25.
- A rollout uses the image as it is, with no crop and no resize, and refuses one of another size. Give generate_images the width and height that rollout will render: each rollout tool's description lists its sizes.
- The clip starts from this frame, so write the image prompt to show what the scene text of its item describes, from the viewpoint the item names.
