---
name: generating_clips_with_gpu_tools
description: Use when generating clips, first frames or poses with the GPU tools (rollout_alayaworld, rollout_wan22, rollout_ltx25, generate_images, annotate_camera).
---

# Generating clips with the GPU tools

- rollout_alayaworld renders WBench-style cases with the released model or, given `node`, with a scored node's fine-tune. rollout_wan22 and rollout_ltx25 turn text or a first frame into video, generate_images makes first frames, and annotate_camera estimates poses for a video.
- Each is a slow GPU job: batch many items per call, then job_wait. Each job takes at most 100 items. annotate_camera takes clips of at most 1200 frames; trim longer clips first. Its poses are estimated from the video alone, with frame 0 as the identity.
- Each rollout result item carries a ready `candidate` (video, caption, provenance) for data_ingest. generate_images returns images and annotate_camera returns poses, with no candidate.
- No generated clip comes with a pose. Run annotate_camera on `candidate.video` and ingest the clip with that pose as `moving`.
- Video models ignore "camera steady": measure before calling a generated clip `static`.
- AlayaWorld follows translation actions reliably and rotation (turns, orbits) only weakly. Its captions have one segment per round, so its clips fit `video_timed_prompts_camera` in per_chunk mode.
