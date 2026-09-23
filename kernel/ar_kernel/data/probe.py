from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_ASPECT = 16 / 9

@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int                 # coded (stored) pixels
    height: int
    duration: float
    rotation: int = 0          # display rotation in degrees: 0, 90, 180 or 270
    sar: float = 1.0           # sample (pixel) aspect ratio

    @property
    def display_aspect(self) -> float:
        """Aspect as shown: coded width x pixel aspect, swapped for 90/270 rotation."""
        width, height = self.width * self.sar, float(self.height)
        if self.rotation in (90, 270):
            width, height = height, width
        return width / height


def _ratio(text: str | None) -> float:
    if not text or ":" not in text:
        return 1.0
    num, den = (float(x) for x in text.split(":"))
    return num / den if num > 0 and den > 0 else 1.0     # "0:1" / "N/A" mean square pixels


def _rotation(stream: dict) -> int:
    raw = 0.0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            raw = float(side["rotation"])
    if not raw:
        raw = float((stream.get("tags") or {}).get("rotate") or 0)   # pre-6.0 muxers
    return int(round(raw)) % 360


def probe_video(path: Path) -> VideoInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries",
         "stream=nb_read_frames,avg_frame_rate,width,height,duration,sample_aspect_ratio"
         ":stream_tags=rotate:stream_side_data=rotation",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    frames = int(stream["nb_read_frames"])
    duration = float(stream.get("duration") or (frames / fps if fps else 0.0))
    return VideoInfo(frames=frames, fps=fps, width=int(stream["width"]),
                     height=int(stream["height"]), duration=duration,
                     rotation=_rotation(stream), sar=_ratio(stream.get("sample_aspect_ratio")))


def aspect_ok(info: VideoInfo, tolerance: float) -> bool:
    return abs(info.display_aspect - TARGET_ASPECT) <= TARGET_ASPECT * tolerance
