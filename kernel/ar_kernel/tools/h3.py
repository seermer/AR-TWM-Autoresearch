"""rollout_h3: clips rendered by MiniMax H3 (diffusers' MiniMaxH3ModularPipeline, in its own env)
from the agent's per-turn prompts.

The agent gives a scene and a list of turns. The kernel puts the turn boundaries on the training
round grid (frame 25 + 32j), writes one continuous shot in MiniMax's prompt format with an in-shot
timestamp at each boundary ("At 00:03.708, ..."), and publishes the clip with one caption segment
per turn. A cut per turn ("[Shot 2] At ...") keeps the timing but changes the scene, so it is not
used. The clip carries no pose.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..data.probe import probe_video
from .gpu_jobs import GpuJob, check_item_seed, check_keyframes, is_int
from .rollouts import FFMPEG_TIMEOUT_S, check_published
from .server import ToolError

H3_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "h3_generate.py"
MIN_FRAMES = 124
FPS = 24
HISTORY = 25                # the first round boundary
ROUND = 32
MIN_TURN_ROUNDS = 2         # 2.67 s: segments under 2.375 s are never trained, and H3's timing has ~0.5 s of slack
SOUND = "overall_soundscape: Natural ambient sound of the scene.\n\nnon_diegetic_music: N/A"
FIRST = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
BOTH = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
        "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {end}-second mark of "
        "the target video.")
LAST = ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "{end}-second mark of the target video.")


def turn_starts(frames: int, n_turns: int) -> list[int]:
    """The first frame of each turn: frame 0, then round boundaries, each turn the same number of rounds."""
    rounds = (frames - HISTORY) // ROUND
    per = rounds // n_turns
    if per < MIN_TURN_ROUNDS:
        raise ToolError(f"a {frames}-frame clip holds at most {rounds // MIN_TURN_ROUNDS} turns "
                        f"(a turn lasts at least {MIN_TURN_ROUNDS} rounds of {ROUND} frames): got {n_turns}")
    return [0] + [HISTORY + ROUND * per * k for k in range(1, n_turns)]


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _stamp(frame: int) -> str:
    seconds = frame / FPS
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def build_prompt(scene: str, turns: list[str], starts: list[int], frames: int, keyframes: list[int]) -> str:
    """The item as one continuous shot in MiniMax's base prompt format (alignment line, three fields)."""
    body = f"[Shot 1] {_sentence(scene)} {_sentence(turns[0])}"
    for text, start in zip(turns[1:], starts[1:]):
        body += f" At {_stamp(start)}, {_sentence(text)}"
    end = f"{frames / FPS:.2f}"
    line = {(0,): FIRST, (-1, 0): BOTH, (-1,): LAST}.get(tuple(sorted(keyframes)), "").format(end=end)
    return (f"{line}\n\n" if line else "") + f"integrated_multimodal_description: {body}\n\n{SOUND}"


def build_caption(scene: str, turns: list[str], starts: list[int], frames: int) -> dict:
    """The whole clip in `caption`; one segment per turn, in seconds, tiling [0, frames / 24]."""
    ends = [*starts[1:], frames]
    caption = " Then ".join(_sentence(t) for t in turns)
    return {"caption": f"{_sentence(scene)} {caption}",
            "segments": [{"time_range_s": [s / FPS, e / FPS], "prompt": f"{_sentence(scene)} {_sentence(t)}"}
                         for t, s, e in zip(turns, starts, ends)]}


class H3Backend(GpuJob):
    """rollout_h3: one worker with every GPU of the run (the int8 transformer is spread across them)."""
    name = tool = "rollout_h3"
    kind = "rollout"
    generator = "minimax-h3"
    config_key = "generators.h3"        # timeout_s, max_items
    takes_keyframes = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        default, maximum = self.block["frames"]
        h, w = self.block["resolution"]
        self.description = (
            "Render training clips with MiniMax H3 from a scene and a list of turns: one continuous shot in "
            "which each turn's text takes effect at that turn's start. A GPU job: returns {job_id} at once; "
            f"collect with job_wait. Slow: about 20 minutes per {maximum}-frame clip. `frames`: 17n+5, "
            f"{MIN_FRAMES} <= frames <= {maximum}, default {default} (one value for the job). Item: "
            "{'scene_prompt': str, 'turns': [{'prompt': str}], 'keyframes'?: [{'image': file under /workspace, "
            "'frame': 0 or -1}], 'seed': int}. Turns start on training round boundaries (frame 25, then every "
            f"32 frames) and each lasts at least {MIN_TURN_ROUNDS} rounds, so a {maximum}-frame clip holds at most "
            f"{((maximum - HISTORY) // ROUND) // MIN_TURN_ROUNDS} turns. Write a turn as what happens, camera "
            "included, in phrases like: the camera pushes in / pulls out, pans left / right, trucks left / right, "
            "tilts up / down, pedestals up / down, arcs around the subject, tracks the subject, holds a static "
            "shot; add 'with small / large amplitude' or 'at slow / fast speed' when it matters. A keyframe at "
            f"frame 0 is the first frame and one at -1 the last; images are center-cropped to {w}x{h}. A last "
            "keyframe must be a view the shot can reach from the first: an unrelated image is reached by a cut "
            "in the final frames. Each "
            f"result item gives a `candidate` for data_ingest (a {w}x{h}, 24 fps, silent mp4, a caption with one "
            "segment per turn, provenance); it carries no pose/camera_motion -- add one (annotate_camera then "
            "'moving', or 'static') before ingesting. Timing is approximate (about half a second): check the "
            "frames before trusting a segment boundary. Metadata, not labels: `turn_segments` and `h3_prompt`.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.setdefault("frames", default)
        if not (is_int(frames) and MIN_FRAMES <= frames <= maximum and (frames - 5) % 17 == 0):
            raise ToolError(f"frames must be 17n+5 with {MIN_FRAMES} <= frames <= {maximum}: got {frames!r}")

    def check_item(self, item):
        if "image" in item:
            raise ToolError("image is not a field of this tool: use keyframes [{'image': path, 'frame': 0}]")
        if not isinstance(item.get("scene_prompt"), str) or not item["scene_prompt"].strip():
            raise ToolError("scene_prompt must be a non-empty string")
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ToolError("turns must be a non-empty list")
        for t, turn in enumerate(turns, 1):
            if not (isinstance(turn, dict) and set(turn) == {"prompt"} and isinstance(turn["prompt"], str)
                    and turn["prompt"].strip()):
                raise ToolError(f"turn {t} must be {{'prompt': non-empty text}}: got {turn!r}")
        check_keyframes(item)
        if any(k["frame"] not in (0, -1) for k in item.get("keyframes") or []):
            raise ToolError("a keyframe's frame must be frame 0 (the first frame) or -1 (the last)")
        check_item_seed(item)

    def check_item_for(self, item, args):
        turn_starts(args["frames"], len(item["turns"]))

    def _layout(self, job, item):
        turns = [t["prompt"] for t in item["turns"]]
        return turns, turn_starts(job.args["frames"], len(turns))

    def produce(self, job, items, work, out, cancel, report):
        frames = job.args["frames"]
        prompts = {}
        for item in items:
            turns, starts = self._layout(job, item)
            prompts[str(item["index"])] = build_prompt(item["scene_prompt"], turns, starts, frames,
                                                       [k["frame"] for k in item["keyframes"]])
        (work / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        h, w = self.block["resolution"]
        return self.run_workers(self.block["env"], lambda r, world: [
            "python", str(H3_BRIDGE), "--items", str(work / "items.json"), "--prompts", str(work / "prompts.json"),
            "--out", str(out), "--rank", str(r), "--world", str(world),
            "--weights", str(self.cfg.repo_root / self.block["weights"]), "--frames", str(frames),
            "--height", str(h), "--width", str(w), "--gpu0-reserve-gib", str(self.block["gpu0_reserve_gib"])],
            [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"})

    def finish(self, job, item, out):
        frames = job.args["frames"]
        h, w = self.block["resolution"]
        silent = out / f"{item['index']}.silent.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{item['index']}.mp4"), "-an",
                        "-c:v", "copy", str(silent)], check=True, timeout=FFMPEG_TIMEOUT_S)
        info = probe_video(silent)
        if (info.width, info.height) != (w, h):
            raise ValueError(f"rendered clip is {info.width}x{info.height}, not {w}x{h}")
        check_published(info)
        if info.frames != frames:
            raise ValueError(f"rendered clip has {info.frames} frames, not {frames}")
        turns, starts = self._layout(job, item)
        caption = self.write_caption(out, item, build_caption(item["scene_prompt"], turns, starts, frames))
        segments = [{"turn_index": k, "prompt": t, "frame_start": s, "frame_end_exclusive": e}
                    for k, (t, s, e) in enumerate(zip(turns, starts, [*starts[1:], frames]))]
        return {"video": silent, "caption": caption, "frames": info.frames, "turn_segments": segments,
                "h3_prompt": build_prompt(item["scene_prompt"], turns, starts, frames,
                                          [k["frame"] for k in item.get("keyframes") or []])}
