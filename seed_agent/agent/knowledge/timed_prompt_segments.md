---
name: timed_prompt_segments
description: Use when writing timed prompts (caption segments) for video_timed_prompts_camera clips.
---

# Timed prompts

- Segment boundaries fall on round boundaries: 25/24 s, then every 32/24 s. Write the segments, then call snap_timed_prompts.
- Caption each segment from its own trimmed part of the clip.
- A clip of 25 + 32k frames (57, 89, 121, 153, ...) ends on a round boundary.
- rollout_wan22 and rollout_ltx25 accept a first frame. Given a clip's last frame, they continue that clip; the continuation starts on that same frame, so a 97-frame continuation adds 96 new frames (3 rounds).
