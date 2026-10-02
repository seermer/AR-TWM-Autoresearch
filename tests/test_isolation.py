import json

from ar_kernel.config import KernelConfig
from ar_kernel.isolation import blocked, copies_held_out, scrub

CFG = KernelConfig.load()
CASE = json.loads((CFG.wbench / "data" / "cases" / "case_196.json").read_text())
SENTENCE = "Torch sconces on both walls cast flickering orange light."      # nine words of case 196's scene text


def test_blocked_names_match_any_case_and_position():
    assert blocked(CFG, "meituan-longcat/WBench") and blocked(CFG, "someone/my-wbench-mirror")
    assert not blocked(CFG, "org/walks") and not blocked(CFG, "") and not blocked(CFG, None)


def test_a_copied_run_of_eight_words_is_caught_with_a_prefix_a_suffix_or_a_split():
    assert SENTENCE in CASE["environment_prompt"]
    assert copies_held_out(CFG, f"A clip. {SENTENCE} Then more.")
    assert copies_held_out(CFG, "torch sconces, on both walls: cast flickering orange light")     # case, punctuation
    assert copies_held_out(CFG, "Torch sconces on both walls. Cast flickering orange light.")    # split in two
    assert copies_held_out(CFG, "unrelated", CASE["interactions"][0]["prompt"])


def test_short_phrases_and_paraphrases_pass():
    assert not copies_held_out(CFG, "First-person viewer.")
    assert not copies_held_out(CFG, "Wall torches throw a wavering orange glow along a stone castle corridor.")
    assert not copies_held_out(CFG, None, "", 3)


def test_scrub_replaces_names_in_nested_values():
    out = scrub({"log": ["[run_WBench] 4 items", 3], "ok": True}, ["wbench"])
    assert out == {"log": ["[run_render] 4 items", 3], "ok": True}
