from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_ASPECT = 16 / 9

@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int
    duration: float

def probe_video(path: Path) -> VideoInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    frames = int(stream["nb_read_frames"])
    duration = float(stream.get("duration") or (frames / fps if fps else 0.0))
    return VideoInfo(frames=frames, fps=fps, width=int(stream["width"]),
                     height=int(stream["height"]), duration=duration)

def aspect_ok(info: VideoInfo, tolerance: float) -> bool:
    return abs((info.width / info.height) - TARGET_ASPECT) <= TARGET_ASPECT * tolerance
