# Generating clips with the GPU tools

- rollout_alayaworld renders WBench-style cases, rollout_wan22 and rollout_ltx25 turn text or a first frame into video, generate_images makes first frames, and annotate_camera estimates poses for a video.
- Each is a slow GPU job: batch many items per call, then job_wait. annotate_camera takes at most 64 items per job.
- Each result item carries a ready `candidate` (video, caption, provenance) for data_ingest.
- No generated clip comes with a pose. Run annotate_camera on `candidate.video` and ingest the clip with that pose as `moving`.
- Video models ignore "camera steady": measure before calling a generated clip `static`.
- AlayaWorld follows translation actions reliably and rotation (turns, orbits) only weakly. Its captions have one segment per round, so its clips fit `video_timed_prompts_camera` in per_chunk mode.
