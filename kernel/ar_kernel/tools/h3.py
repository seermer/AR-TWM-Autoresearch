"""rollout_h3: clips rendered by MiniMax H3 (through LightX2V, in its own env) from the agent's
per-turn prompts.

The agent gives a scene and a list of turns. The kernel puts the turn boundaries on the training
round grid (frame 25 + 32j), writes one continuous shot in MiniMax's prompt format with an in-shot
timestamp at each turn's start ("At 00:03.708, ..."), and publishes the clip with a caption of
the scene only (the turns and their planned frame ranges are metadata). MiniMax's guide documents a timestamp only at a cut ("[Shot 2] At ..."); that
keeps the timing but changes the scene, so the in-shot form is our own and its timing is loose.
The clip carries no pose.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..data.probe import probe_video
from ..subproc import meminfo_gib
from .gpu_jobs import GpuJob, check_item_seed, check_keyframes, check_listed_size, is_int, sizes_text
from .rollouts import FFMPEG_TIMEOUT_S, check_published, checked_repo
from .server import ToolError

H3_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "h3_generate.py"
MIN_FRAMES = 124
FPS = 24
HISTORY = 25                # the first round boundary
ROUND = 32
MIN_TURN_ROUNDS = 2         # 2.67 s: segments under 2.375 s are never trained, and H3's timing is loose
NO_CUTS = "The whole video is one continuous shot with smooth motion and no cuts."
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


def _line(text: str) -> str:
    return " ".join(text.split())


def _says_something(text) -> bool:
    return isinstance(text, str) and any(c.isalnum() for c in text)


def _sentence(text: str) -> str:
    text = _line(text).rstrip(",;: ")
    return text if text.rstrip("\"')”’").endswith((".", "!", "?", "。", "！", "？", "…")) else text + "."


def _stamp(frame: int) -> str:
    seconds = frame / FPS
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def build_prompt(scene: str, turns: list[str], starts: list[int], frames: int, keyframes: list[int],
                 soundscape: str, music: str) -> str:
    """The item as one continuous shot in MiniMax's base prompt format (alignment line, three fields)."""
    body = f"[Shot 1] {_sentence(scene)} {NO_CUTS}"
    for text, start in zip(turns, starts):
        body += f" At {_stamp(start)}, {_sentence(text)}"
    end = f"{frames / FPS:.2f}"
    line = {(0,): FIRST, (-1, 0): BOTH, (-1,): LAST}.get(tuple(sorted(keyframes)), "").format(end=end)
    return ((f"{line}\n\n" if line else "") + f"integrated_multimodal_description: {body}\n\n"
            f"overall_soundscape: {_line(soundscape)}\n\nnon_diegetic_music: {_line(music)}")


def launcher(ranks: int) -> list[str]:
    return ["python", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}"]


class H3Backend(GpuJob):
    """rollout_h3: one torchrun worker with a rank on every GPU of the run (sequence parallel)."""
    name = tool = "rollout_h3"
    kind = "rollout"
    generator = "minimax-h3"
    config_key = "generators.h3"        # timeout_s, max_items
    takes_keyframes = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        default, maximum = self.block["frames"]
        h, w = self.block["resolutions"][0]
        self.description = (
            "Render training clips with MiniMax H3 from a scene and a list of turns: one continuous shot with "
            "no cuts, in which each turn's text is asked for at that turn's start. A GPU job: returns {job_id} at once; "
            f"collect with job_wait. A job loads for several minutes, then takes about 4 minutes per {maximum}-frame "
            "clip at the smallest size and about 8 at the largest. Params (one value per job): `frames` (17n+5, "
            f"{MIN_FRAMES} <= frames <= {maximum}, default {default}), `width`/`height` (one of "
            f"{sizes_text(self.block['resolutions'])}; default {w}x{h}). Item: "
            "{'scene_prompt': str, 'turns': [{'prompt': str}], 'overall_soundscape': str, 'non_diegetic_music': "
            "str, 'keyframes'?: [{'image': file under /workspace, 'frame': 0 or -1}], 'seed': int}. Start "
            "`scene_prompt` with the visual style, e.g. 'Live-action,'. `overall_soundscape`: 1-4 sentences on "
            "the ambient and action sounds; `non_diegetic_music`: the background score, or 'N/A'. The kernel "
            "builds the model prompt from these. Turns start on training round boundaries (frame 25, then every "
            f"32 frames) and each lasts at least {MIN_TURN_ROUNDS} rounds, so a {maximum}-frame clip holds at most "
            f"{((maximum - HISTORY) // ROUND) // MIN_TURN_ROUNDS} turns. Write a turn as what happens, camera "
            "included, in phrases like: the camera pushes in / pulls out, pans left / right, trucks left / right, "
            "tilts up / down, pedestals up / down, arcs around the subject, tracks the subject, holds a static "
            "shot; add 'with small / large amplitude' or 'at slow / fast speed' when it matters. A keyframe at "
            "frame 0 is the first frame and one at -1 the last; a keyframe image must be exactly the job's width x "
            "height: it is used as it is. A last "
            "keyframe must be a view the shot can reach from the first: an unrelated image is reached by a cut "
            "in the final frames. Each "
            "result item gives a `candidate` for data_ingest (a 24 fps, silent mp4, a caption holding "
            "the scene only, provenance); it carries no pose/camera_motion -- add one (annotate_camera then "
            "'moving', or 'static') before ingesting. An event may land a second or more earlier or later "
            "than its turn, or not appear at all. The published caption therefore has no per-turn segments: "
            "check the frames, then write the segments you ingest to match what the clip shows. Metadata, not "
            "labels: `turn_segments` (each turn's prompt and its planned frame range) and `h3_prompt`.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.setdefault("frames", default)
        if not (is_int(frames) and MIN_FRAMES <= frames <= maximum and (frames - 5) % 17 == 0):
            raise ToolError(f"frames must be 17n+5 with {MIN_FRAMES} <= frames <= {maximum}: got {frames!r}")
        check_listed_size(self.cfg, args, self.block["resolutions"])

    def check_item(self, item):
        if "image" in item:
            raise ToolError("image is not a field of this tool: use keyframes [{'image': path, 'frame': 0}]")
        if not _says_something(item.get("scene_prompt")):
            raise ToolError("scene_prompt must be a non-empty string")
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ToolError("turns must be a non-empty list")
        for t, turn in enumerate(turns, 1):
            if not (isinstance(turn, dict) and set(turn) == {"prompt"} and _says_something(turn["prompt"])):
                raise ToolError(f"turn {t} must be {{'prompt': non-empty text}}: got {turn!r}")
        for key in ("overall_soundscape", "non_diegetic_music"):
            if not _says_something(item.get(key)):
                raise ToolError(f"{key} must be a non-empty string")
        check_keyframes(item)
        if any(k["frame"] not in (0, -1) for k in item.get("keyframes") or []):
            raise ToolError("a keyframe's frame must be frame 0 (the first frame) or -1 (the last)")
        check_item_seed(item)

    def check_item_for(self, item, args):
        turn_starts(args["frames"], len(item["turns"]))

    def _layout(self, job, item):
        turns = [t["prompt"] for t in item["turns"]]
        return turns, turn_starts(job.args["frames"], len(turns))

    def _prompt(self, job, item) -> str:
        turns, starts = self._layout(job, item)
        return build_prompt(item["scene_prompt"], turns, starts, job.args["frames"],
                            [k["frame"] for k in item.get("keyframes") or []],
                            item["overall_soundscape"], item["non_diegetic_music"])

    def produce(self, job, items, work, out, cancel, report):
        frames = job.args["frames"]
        prompts = {str(item["index"]): self._prompt(job, item) for item in items}
        (work / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        h, w = job.args["height"], job.args["width"]
        checked_repo(self.cfg, self.config_key, "LightX2V")
        avail, rss, reserve = meminfo_gib()["MemAvailable"], self.block["peak_rss_gib"], self.block["host_reserve_gib"]
        if avail - reserve < rss:
            raise RuntimeError(f"not enough free host RAM for rollout_h3: MemAvailable {avail:.0f} GiB - "
                               f"host_reserve_gib {reserve:g} < peak_rss_gib {rss:g}; free host memory and retry")
        config = json.loads((self.cfg.repo_root / self.block["config"]).read_text(encoding="utf-8"))
        cache = self.cfg.repo_root / config["adaln_cache_dir"]
        if not cache.is_dir():
            raise RuntimeError(f"no AdaLN cache at {cache}: build it once (README, model weights)")
        config["adaln_cache_dir"] = str(cache)
        config["parallel"]["seq_p_size"] = len(self.gpus)
        (work / "lightx2v.json").write_text(json.dumps(config), encoding="utf-8")
        return self.run_workers(self.block["env"], lambda r, world: [
            *launcher(len(self.gpus)), str(H3_BRIDGE), "--items", str(work / "items.json"),
            "--prompts", str(work / "prompts.json"), "--out", str(out),
            "--weights", str(self.cfg.repo_root / self.block["weights"]), "--config", str(work / "lightx2v.json"),
            "--frames", str(frames), "--height", str(h), "--width", str(w)],
            [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "DTYPE": "BF16",
                       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})

    def finish(self, job, item, out):
        frames = job.args["frames"]
        h, w = job.args["height"], job.args["width"]
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
        caption = self.write_caption(out, item, {"caption": _sentence(item["scene_prompt"])})
        segments = [{"turn_index": k, "prompt": t, "frame_start": s, "frame_end_exclusive": e}
                    for k, (t, s, e) in enumerate(zip(turns, starts, [*starts[1:], frames]))]
        return {"video": silent, "caption": caption, "frames": info.frames, "turn_segments": segments,
                "h3_prompt": self._prompt(job, item)}
