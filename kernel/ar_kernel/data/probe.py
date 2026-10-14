from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av

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


def _upright(rotation: int) -> list[str]:
    """The filters the ffmpeg CLI puts first to show a rotated video upright."""
    return {90: ["transpose=cclock"], 180: ["hflip", "vflip"], 270: ["transpose=clock"]}.get(rotation % 360, [])


def video_frames(path: Path, chain: list[str], start: float = 0.0) -> Iterator[av.VideoFrame]:
    """The first video stream's frames, upright as the ffmpeg CLI shows them, through the filter `chain`;
    with `start`, only those at or after `start` seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start:
            container.seek(int(start / stream.time_base), stream=stream)        # the keyframe before `start`
        graph = None
        for frame in container.decode(stream):
            if start and frame.time < start:
                continue
            if graph is None:
                graph = av.filter.Graph()
                nodes = [graph.add_buffer(template=stream)]
                for spec in _upright(frame.rotation) + chain:
                    name, _, args = spec.partition("=")
                    nodes.append(graph.add(name, args or None))
                graph.link_nodes(*nodes, graph.add("buffersink")).configure()
            graph.push(frame)
            yield graph.pull()


def probe_video(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count, rotation = 0, 0
        for frame in container.decode(stream):
            rotation = rotation if count else frame.rotation % 360
            count += 1
        fps = float(stream.average_rate or 0)
        duration = (round(float(stream.duration * stream.time_base), 6) if stream.duration is not None
                    else count / fps if fps else 0.0)
        return VideoInfo(frames=count, fps=fps, width=stream.width, height=stream.height, duration=duration,
                         rotation=rotation, sar=float(stream.sample_aspect_ratio or 1))


def aspect_ok(info: VideoInfo, tolerance: float) -> bool:
    return abs(info.display_aspect - TARGET_ASPECT) <= TARGET_ASPECT * tolerance
