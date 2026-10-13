"""A clip's annotation: its scene, its subject and its turns. The texts of a rollout's turns and of a
training clip's labels are assembled here, the same way."""
from __future__ import annotations

import math

FPS = 24
FIRST_TURN_FRAMES, LATER_TURN_FRAMES = 57, 32      # what one turn needs to be trained on
FIRST_TURN_S, LATER_TURN_S = 2.4, 1.4              # the same, as agents are told
PERSPECTIVES = ("first_person", "third_person")
DERIVED_KINDS = ("model", "dataset", "synthetic_method")
SWITCH = {"third_person": "The view switches from first-person to third-person, revealing the character from behind.",
          "first_person": "The view switches back to first-person, returning to the character's own eyes."}


def turn_text(scene: str, character: str | None, event: str | None = None, switch_to: str | None = None) -> str:
    """One turn's text: scene, subject, the turn's event, then the switch sentence."""
    parts = (scene, character, event, SWITCH[switch_to] if switch_to else None)
    return " ".join(p.strip() for p in parts if p and p.strip())


def turn_texts(scene: str, character: str | None, turns: list[dict]) -> list[str]:
    """The text of every turn. A turn whose perspective differs from the one before it switches."""
    out, previous = [], None
    for turn in turns:
        perspective = turn.get("perspective") or previous
        out.append(turn_text(scene, character, turn.get("event"),
                             perspective if previous and perspective != previous else None))
        previous = perspective
    return out


def check_annotation(annotation) -> list[str]:
    """What is wrong with an annotation's shape; empty when it is well formed."""
    if not isinstance(annotation, dict):
        return ["the annotation must be a JSON object"]
    problems = [f"unknown field {key!r}" for key in annotation if key not in ("scene_prompt", "character_prompt", "turns")]
    if not isinstance(annotation.get("scene_prompt"), str) or not annotation["scene_prompt"].strip():
        problems.append("scene_prompt is required: a non-empty string")
    if not isinstance(annotation.get("character_prompt", ""), str):
        problems.append("character_prompt must be a string")
    turns = annotation.get("turns")
    if not isinstance(turns, list) or not turns or not all(isinstance(t, dict) for t in turns):
        return problems + ["turns is required: a non-empty list of objects"]
    previous = None
    for n, turn in enumerate(turns, start=1):
        problems += [f"turn {n}: unknown field {key!r}" for key in turn if key not in ("start_s", "event", "perspective")]
        start = turn.get("start_s")
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start):
            problems.append(f"turn {n}: start_s is required: a number of seconds")
        elif previous is None and start != 0:
            problems.append(f"turn 1 must start at 0, not at {start}")
        elif previous is not None and start <= previous:
            problems.append(f"turn {n} starts at {start} s, not after turn {n - 1} ({previous} s)")
        else:
            previous = start
        if turn.get("event") is not None and (not isinstance(turn["event"], str) or not turn["event"].strip()):
            problems.append(f"turn {n}: event must be one sentence, or absent")
        if "perspective" in turn or len(turns) > 1:
            if turn.get("perspective") not in PERSPECTIVES:
                problems.append(f"turn {n}: perspective must be one of {', '.join(PERSPECTIVES)}")
    return problems


def turn_frames(annotation: dict, frames: int) -> list[tuple[int, int]]:
    """(first frame, end frame) of every turn at 24 fps, rounded as the training loader rounds."""
    starts = [round(t["start_s"] * FPS) for t in annotation["turns"]]
    return [(lo, min(hi, frames)) for lo, hi in zip(starts, [*starts[1:], frames])]


def short_turns(annotation: dict, frames: int) -> list[str]:
    """One message per turn too short to be trained on, in seconds."""
    out = []
    for n, (lo, hi) in enumerate(turn_frames(annotation, frames), start=1):
        if hi - lo < (FIRST_TURN_FRAMES if n == 1 else LATER_TURN_FRAMES):
            out.append(f"turn {n} lasts {max(hi - lo, 0) / FPS:.2f} s; " + (
                f"the first turn must last at least {FIRST_TURN_S} s" if n == 1 else
                f"a turn after the first must last at least {LATER_TURN_S} s"))
    return out


def caption_file(annotation: dict, duration_s: float) -> dict:
    """WorldModel's caption file for an annotated clip: one timed prompt per turn."""
    scene, character, turns = annotation["scene_prompt"], annotation.get("character_prompt"), annotation["turns"]
    starts = [float(t["start_s"]) for t in turns]
    return {"caption": turn_text(scene, character),
            "segments": [{"time_range_s": [lo, hi], "prompt": text}
                         for lo, hi, text in zip(starts, [*starts[1:], duration_s], turn_texts(scene, character, turns))]}
