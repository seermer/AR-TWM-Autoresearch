"""rollout_h3: clips rendered by MiniMax H3 (through LightX2V, in its own env) from the agent's
scene, subject and turns.

The kernel writes the whole prompt: one continuous shot in MiniMax's prompt format, the scene and
subject, the first turn's perspective, then each turn announced by the time it starts ("About 4.0
seconds in, in the same continuous shot, ...") with its event, the camera sentence of its action and,
when the perspective changes, the switch sentence. The clip is split into equal turns. MiniMax's guide
documents a timestamp only at a cut ("[Shot 2] At ..."); that keeps the timing but changes the scene,
so the in-shot form is our own and its timing is loose. The clip is published with its annotation
(the planned start of each turn) and carries no pose.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..data.annotation import FIRST_TURN_FRAMES, FPS, SWITCH
from ..data.probe import probe_video
from ..faults import KernelFault
from ..subproc import meminfo_gib
from .gpu_jobs import IMAGE_EXTS, GpuJob
from .rollouts import SIZE, check_published, check_turn_item, checked_repo, item_annotation
from .server import ToolError

H3_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "h3_generate.py"
MIN_FRAMES = 124
ITEM_KEYS = ("scene_prompt", "character_prompt", "seed", "image", "seconds", "turns")
NO_CUTS = "The whole video is one continuous shot with smooth motion and no cuts."
FIRST = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
SOUNDSCAPE, MUSIC = "Natural ambient sound of the scene.", "N/A"
VIEW = {"first_person": "First-person view.", "third_person": "Third-person view from behind the subject."}
# The camera sentence of each action. The evaluation's own (WBench src/models/text/prompt_builder.py),
# plus the three it lacks: `stop` in both perspectives and third-person S+A.
CAMERA = {
    "first_person": {
        "W": "The camera pushes forward.",
        "S": "The camera pulls back.",
        "A": "The camera moves to the left.",
        "D": "The camera moves to the right.",
        "left": "The camera pans to the left.",
        "right": "The camera pans to the right.",
        "up": "The camera tilts up.",
        "down": "The camera tilts down.",
        "W+A": "The camera moves diagonally forward-left.",
        "W+D": "The camera moves diagonally forward-right.",
        "S+A": "The camera moves diagonally backward-left.",
        "S+D": "The camera moves diagonally backward-right.",
        "W+left": "The camera pushes forward while panning to the left.",
        "W+right": "The camera pushes forward while panning to the right.",
        "W+up": "The camera pushes forward while tilting up.",
        "W+down": "The camera pushes forward while tilting down.",
        "stop": "The camera holds still.",
    },
    "third_person": {
        "W": "The subject moves forward. The camera follows, maintaining distance and framing.",
        "S": "The subject moves backward. The camera follows, maintaining distance and framing.",
        "A": "The subject moves to the left. The camera tracks left, keeping the subject centered.",
        "D": "The subject moves to the right. The camera tracks right, keeping the subject centered.",
        "left": "The camera orbits counterclockwise around the subject. The subject stays still.",
        "right": "The camera orbits clockwise around the subject. The subject stays still.",
        "up": "The camera cranes upward while tilting down to keep the subject centered.",
        "down": "The camera cranes downward while tilting up to keep the subject centered.",
        "W+A": "The subject moves diagonally forward-left. The camera follows along the same diagonal.",
        "W+D": "The subject moves diagonally forward-right. The camera follows along the same diagonal.",
        "S+A": "The subject moves backward-left. The camera retreats along the same diagonal.",
        "S+D": "The subject moves backward-right. The camera retreats along the same diagonal.",
        "W+left": "The subject moves forward. The camera arcs counterclockwise while following.",
        "W+right": "The subject moves forward. The camera arcs clockwise while following.",
        "W+up": "The subject moves forward. The camera cranes upward while following.",
        "W+down": "The subject moves forward. The camera cranes downward while following.",
        "stop": "The subject stays still. The camera holds its position.",
    },
}


def clip_frames(seconds, default: int, maximum: int) -> int:
    """The frame count H3 can render (17n+5) nearest to `seconds`; `default` when no length is given."""
    if seconds is None:
        return default
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ToolError(f"seconds must be a number: got {seconds!r}")
    frames = 17 * round((seconds * FPS - 5) / 17) + 5
    if not MIN_FRAMES <= frames <= maximum:
        raise ToolError(f"seconds must be between {MIN_FRAMES / FPS:.1f} and {maximum / FPS:.1f}: got {seconds}")
    return frames


def turn_starts(frames: int, n_turns: int) -> list[int]:
    """The first frame of each turn: the clip in equal turns, each boundary on a whole frame."""
    if frames // n_turns < FIRST_TURN_FRAMES:
        raise ToolError(f"a clip of {frames / FPS:.1f} seconds holds at most {frames // FIRST_TURN_FRAMES} "
                        f"turns: got {n_turns}")
    return [round(frames * k / n_turns) for k in range(n_turns)]


def _line(text: str) -> str:
    return " ".join(text.split())


def _sentence(text: str) -> str:
    text = _line(text).rstrip(",;: ")
    return text if text.rstrip("\"')”’").endswith((".", "!", "?", "。", "！", "？", "…")) else text + "."


def build_prompt(item: dict, starts: list[int]) -> str:
    """The item as one continuous shot in MiniMax's base prompt format (alignment line, three fields)."""
    turns = item["turns"]
    body = " ".join(["[Shot 1]", _sentence(item["scene_prompt"]),
                     *([_sentence(item["character_prompt"])] if item.get("character_prompt", "").strip() else []),
                     VIEW[turns[0]["perspective"]], NO_CUTS])
    previous = None
    for turn, start in zip(turns, starts):
        perspective = turn["perspective"]
        parts = [*([_sentence(turn["event"])] if turn.get("event") else []), CAMERA[perspective][turn["action"]],
                 *([SWITCH[perspective]] if previous and perspective != previous else [])]
        when = "From the start," if start == 0 else f"About {start / FPS:.1f} seconds in, in the same continuous shot,"
        body += f" {when} {' '.join(parts)}"
        previous = perspective
    line = f"{FIRST}\n\n" if item.get("image") else ""
    return (f"{line}integrated_multimodal_description: {body}\n\n"
            f"overall_soundscape: {SOUNDSCAPE}\n\nnon_diegetic_music: {MUSIC}")


def launcher(ranks: int) -> list[str]:
    return ["python", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}"]


class H3Backend(GpuJob):
    """rollout_h3: one torchrun worker with a rank on every GPU of the run (sequence parallel)."""
    name = tool = "rollout_h3"
    kind = "rollout"
    generator = "minimax-h3"
    config_key = "generators.h3"        # max_items
    file_keys = frame_keys = ("image",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_frames, self.max_frames = self.block["frames"]
        self.description = (
            "MiniMax H3 is the most powerful video generation model of the rollout tools: use this tool for a new "
            "training clip unless the clip needs a feature that only another rollout tool has. Render training clips "
            "with MiniMax H3 from a scene, a subject and a list of turns, as one continuous shot. A GPU job: "
            "returns {job_id} at once; collect with job_wait. Each item is the scene and subject text, a seed and "
            "a list of turns, and may have `image`, a first frame of exactly "
            f"{SIZE[0]}x{SIZE[1]} (it is used as it is), and `seconds`, the clip's length: "
            f"{MIN_FRAMES / FPS:.1f} to {self.max_frames / FPS:.1f}, default {self.default_frames / FPS:.1f}. The "
            "clip gets the nearest length the model can render, which the result reports. A turn has a navigation "
            "action, a perspective, and may have one event: one sentence about something that happens during the "
            "turn. The kernel writes the whole model prompt from these: the scene and the subject, then each turn "
            "at the time it starts, with its event and the camera move of its action. The clip is split into "
            f"turns of equal length, each at least {FIRST_TURN_FRAMES / FPS:.1f} seconds. Start `scene_prompt` with "
            "the visual style, e.g. 'Live-action,'. Sound is not yours to describe. Each result item gives a `candidate`: `video` "
            f"({SIZE[0]}x{SIZE[1]}, 24 fps, silent), `annotation` (a file: your scene and subject text and the "
            "turns, each with the time it was asked to start) and `derived_from`. It has no pose: run "
            "annotate_camera on candidate.video, then data_ingest the video, the annotation and that pose. An "
            "event may land a second or more earlier or later than its turn's start, or not appear at all: find "
            "when it happens (locate_events) and set the turn's start in the annotation to match what the clip "
            "shows. Metadata, not labels: `h3_prompt`.")

    def frame_size(self, args):
        return SIZE

    def check_item(self, item):
        if item.get("image") is not None and not str(item["image"]).lower().endswith(IMAGE_EXTS):
            raise ToolError(f"image must be an image file ({', '.join(IMAGE_EXTS)}): got {item['image']!r}")
        check_turn_item(item, ITEM_KEYS)
        turn_starts(clip_frames(item.get("seconds"), self.default_frames, self.max_frames), len(item["turns"]))

    def _layout(self, item) -> tuple[int, list[int]]:
        frames = clip_frames(item.get("seconds"), self.default_frames, self.max_frames)
        return frames, turn_starts(frames, len(item["turns"]))

    def produce(self, job, items, work, out, cancel, report):
        requests = {}
        for item in items:
            frames, starts = self._layout(item)
            requests[str(item["index"])] = {"prompt": build_prompt(item, starts), "frames": frames}
        (work / "requests.json").write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
        checked_repo(self.cfg, self.config_key, "LightX2V")
        avail, rss, reserve = meminfo_gib()["MemAvailable"], self.block["peak_rss_gib"], self.block["host_reserve_gib"]
        if avail - reserve < rss:
            raise KernelFault(f"not enough free host RAM for rollout_h3: MemAvailable {avail:.0f} GiB - "
                               f"host_reserve_gib {reserve:g} < peak_rss_gib {rss:g}; free host memory and retry")
        config = json.loads((self.cfg.repo_root / self.block["config"]).read_text(encoding="utf-8"))
        cache = self.cfg.repo_root / config["adaln_cache_dir"]
        if not cache.is_dir():
            raise KernelFault(f"no AdaLN cache at {cache}: build it once (README, model weights)")
        config["adaln_cache_dir"] = str(cache)
        config["attn_type"] = self.block["attn_type"]
        config["parallel"]["seq_p_size"] = len(self.gpus)
        (work / "lightx2v.json").write_text(json.dumps(config), encoding="utf-8")
        return self.run_workers(self.block["env"], lambda r, world: [
            *launcher(len(self.gpus)), str(H3_BRIDGE), "--items", str(work / "items.json"),
            "--requests", str(work / "requests.json"), "--out", str(out),
            "--weights", str(self.cfg.repo_root / self.block["weights"]), "--config", str(work / "lightx2v.json"),
            "--height", str(SIZE[1]), "--width", str(SIZE[0])],
            [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "DTYPE": "BF16",
                       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})

    def finish(self, job, item, out):
        frames, starts = self._layout(item)
        silent = out / f"{item['index']}.silent.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{item['index']}.mp4"), "-an",
                        "-c:v", "copy", str(silent)], check=True)
        info = probe_video(silent)
        if (info.width, info.height) != SIZE:
            raise ValueError(f"rendered clip is {info.width}x{info.height}, not {SIZE[0]}x{SIZE[1]}")
        check_published(info)
        if info.frames != frames:
            raise ValueError(f"rendered clip has {info.frames} frames, not {frames}")
        return {"video": silent, "annotation": self.write_annotation(out, item, item_annotation(item, starts)),
                "frames": frames, "seconds": frames / FPS, "h3_prompt": build_prompt(item, starts)}
