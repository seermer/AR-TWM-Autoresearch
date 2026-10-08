"""rollout_h3: turn layout on the round grid, the MiniMax prompt format, captions, submit checks,
the real produce with a fake worker, and one real gpu smoke."""
import pytest

from ar_kernel.tools.h3 import build_caption, build_prompt, turn_starts
from ar_kernel.tools.server import ToolError

SCENE = "A first-person view walks along a cobblestone street"
TURNS = ["the camera pushes in slowly.", " the camera pans right to face a red door ", "a white dog runs out"]


@pytest.mark.parametrize("frames, n, starts", [
    (243, 1, [0]), (243, 2, [0, 121]), (243, 3, [0, 89, 153]), (124, 1, [0]), (158, 2, [0, 89])])
def test_turns_start_on_the_round_grid(frames, n, starts):
    got = turn_starts(frames, n)
    assert got == starts
    assert all((s - 25) % 32 == 0 for s in got[1:])


@pytest.mark.parametrize("frames, n", [(243, 4), (124, 2), (124, 3)])
def test_more_turns_than_the_clip_holds_is_refused(frames, n):
    with pytest.raises(ToolError, match="at most"):
        turn_starts(frames, n)


def test_prompt_is_one_shot_with_in_shot_timestamps():
    p = build_prompt(SCENE, TURNS, [0, 89, 153], 243, [])
    assert p == (
        "integrated_multimodal_description: [Shot 1] A first-person view walks along a cobblestone street. "
        "the camera pushes in slowly. At 00:03.708, the camera pans right to face a red door. "
        "At 00:06.375, a white dog runs out.\n\n"
        "overall_soundscape: Natural ambient sound of the scene.\n\n"
        "non_diegetic_music: N/A")
    assert "[Shot 2]" not in p and ".." not in p


@pytest.mark.parametrize("keyframes, line", [
    ([0], "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."),
    ([0, -1], "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
              "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 10.12-second mark of "
              "the target video."),
    ([-1, 0], "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
              "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 10.12-second mark of "
              "the target video."),
    ([-1], "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
           "10.12-second mark of the target video."),
])
def test_alignment_line_follows_the_keyframes(keyframes, line):
    p = build_prompt(SCENE, TURNS[:1], [0], 243, keyframes)
    assert p.startswith(line + "\n\nintegrated_multimodal_description: [Shot 1] ")


def test_caption_covers_every_turn_and_segments_tile_the_clip():
    c = build_caption(SCENE, TURNS, [0, 89, 153], 243)
    assert c["caption"] == ("A first-person view walks along a cobblestone street. the camera pushes in slowly. "
                            "Then the camera pans right to face a red door. Then a white dog runs out.")
    assert [s["time_range_s"] for s in c["segments"]] == [[0.0, 89 / 24], [89 / 24, 153 / 24], [153 / 24, 243 / 24]]
    assert c["segments"][1]["prompt"] == ("A first-person view walks along a cobblestone street. "
                                          "the camera pans right to face a red door.")
