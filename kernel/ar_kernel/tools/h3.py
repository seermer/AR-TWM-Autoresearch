"""rollout_h3: clips rendered by MiniMax H3 (diffusers' MiniMaxH3ModularPipeline, in its own env)
from the agent's per-turn prompts.

The agent gives a scene and a list of turns. The kernel puts the turn boundaries on the training
round grid (frame 25 + 32j), writes one continuous shot in MiniMax's prompt format with an in-shot
timestamp at each boundary ("At 00:03.708, ..."), and publishes the clip with one caption segment
per turn. A cut per turn ("[Shot 2] At ...") keeps the timing but changes the scene, so it is not
used. The clip carries no pose.
"""
from __future__ import annotations

from .server import ToolError

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
