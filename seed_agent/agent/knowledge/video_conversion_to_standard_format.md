---
name: video_conversion_to_standard_format
description: Use when converting a downloaded or generated video to the standard format: probing, aspect ratio, cropping, rotation, frame rate, trimming.
---

# Converting a video to the standard format

- Probe: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation -of json in.mp4`
- Display aspect = width * SAR / height, with width and height swapped for a 90 or 270 degree rotation. It must be 16:9 within 2%, with no rotation left in the file.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -pix_fmt yuv420p -crf 18 -an out.mp4`. Re-encoding also applies any rotation, and `-pix_fmt yuv420p` keeps 10-bit or 4:4:4 sources decodable.
- Write the files you will ingest under `/workspace/staging/`: data_ingest only takes staged files.
- Keep the source frame rate if it is at least 24 fps. Never raise it by duplicating frames.
- Trimming changes the frame count, so slice the pose array to the same frames.
