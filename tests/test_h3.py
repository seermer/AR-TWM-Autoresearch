"""rollout_h3: turn layout on the round grid, the MiniMax prompt format, submit checks,
the real produce with a fake worker, and one real gpu smoke."""
import copy
import json
import subprocess
import threading
from pathlib import Path

import pytest
from PIL import Image

from conftest import job_result
from ar_kernel.config import KernelConfig
from ar_kernel.data.probe import probe_video
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import h3
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.h3 import H3Backend, build_prompt, turn_starts
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

SCENE = "A first-person view walks along a cobblestone street"
SOUND, MUSIC = "Footsteps on stone and a light wind.", "N/A"
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
    p = build_prompt(SCENE, TURNS, [0, 89, 153], 243, [], SOUND, MUSIC)
    assert p == (
        "integrated_multimodal_description: [Shot 1] A first-person view walks along a cobblestone street. "
        "The whole video is one continuous shot with smooth motion and no cuts. "
        "At 00:00.000, the camera pushes in slowly. At 00:03.708, the camera pans right to face a red door. "
        "At 00:06.375, a white dog runs out.\n\n"
        "overall_soundscape: Footsteps on stone and a light wind.\n\n"
        "non_diegetic_music: N/A")
    assert "[Shot 2]" not in p and ".." not in p


@pytest.mark.parametrize("text, clean", [
    ("the camera pans right,", "the camera pans right."), ("a dog runs out ;", "a dog runs out."),
    ("she waves!", "she waves!"), ('he says "go."', 'he says "go."'), ("镜头向右平移。", "镜头向右平移。"),
    ("a long\n\nturn  with   gaps", "a long turn with gaps.")])
def test_a_turn_is_one_clean_sentence(text, clean):
    p = build_prompt(SCENE, [text], [0], 243, [], "wind\n\nnon_diegetic_music: drums", MUSIC)
    assert f"At 00:00.000, {clean}\n\noverall_soundscape: wind non_diegetic_music: drums\n\n" in p
    assert p.count("\n\n") == 2                          # the three fields, and nothing an item wrote


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
    p = build_prompt(SCENE, TURNS[:1], [0], 243, keyframes, SOUND, MUSIC)
    assert p.startswith(line + "\n\nintegrated_multimodal_description: [Shot 1] ")


def timed_caption(candidate: dict) -> dict:
    """The caption an agent writes from `turn_segments` once the frames confirm them."""
    return {"caption": SCENE, "segments": [
        {"time_range_s": [t["frame_start"] / 24, t["frame_end_exclusive"] / 24], "prompt": f"{SCENE}. {t['prompt']}"}
        for t in candidate["turn_segments"]]}


REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def h3_cfg(env="autoresearcher", **over):
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["h3"].update({"env": env, "enabled": True, **over})
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


def item(**over):
    return {"scene_prompt": SCENE, "turns": [{"prompt": t} for t in TURNS], "overall_soundscape": SOUND,
            "non_diegetic_music": MUSIC, "seed": 7, **over}


@pytest.fixture
def h3_env(tmp_path, monkeypatch):
    """The real H3Backend.produce/run_workers, with h3_generate.py swapped for the fake worker."""
    monkeypatch.setattr(h3, "H3_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(ws / "first.png")
    Image.new("RGB", (640, 360), (30, 20, 10)).save(ws / "last.png")
    q.register(H3Backend(h3_cfg(), tmp_path / "run", [0, 1, 4, 5], reg, rec, gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_h3"].submit(q, caller, {"items": items, **params})


@pytest.mark.parametrize("bad, params, match", [
    (item(scene_prompt=" "), {}, "scene_prompt"),
    (item(scene_prompt=" , "), {}, "scene_prompt"),
    (item(turns=[{"prompt": ";"}]), {}, "turn 1"),
    (item(turns=[{"prompt": "..."}]), {}, "turn 1"),
    (item(scene_prompt="?!"), {}, "scene_prompt"),
    (item(non_diegetic_music="-"), {}, "non_diegetic_music"),
    (item(turns=[]), {}, "turns"),
    (item(turns=[{"prompt": ""}]), {}, "turn 1"),
    (item(turns=[{"prompt": "walk", "action": "W"}]), {}, "turn 1"),
    (item(turns=[{"prompt": "a"}] * 4), {}, "at most 3 turns"),
    (item(), {"frames": 124}, "at most 1 turns"),
    (item(seed="x"), {}, "seed"),
    (item(overall_soundscape=" "), {}, "overall_soundscape"),
    ({k: v for k, v in item().items() if k != "non_diegetic_music"}, {}, "non_diegetic_music"),
    (item(keyframes=[{"image": "first.png", "frame": 5}]), {}, "frame 0 .* or -1"),
    (item(keyframes=[{"image": "first.png", "frame": 0}, {"image": "last.png", "frame": 0}]), {}, "repeat"),
    (item(image="first.png"), {}, "image"),
    (item(), {"frames": 240}, r"17n\+5"), (item(), {"frames": 107}, "124"), (item(), {"frames": 260}, "243"),
])
def test_h3_submit_refuses(h3_env, bad, params, match):
    q, caller, _ = h3_env
    with pytest.raises(ToolError, match=match):
        submit(q, caller, [bad], **params)


def test_h3_job_over_the_configured_cap_is_refused(h3_env):
    q, caller, _ = h3_env
    cap = REAL.get("generators.h3.max_items")
    assert cap == 30
    with pytest.raises(ToolError, match="at most 30 items per job"):
        submit(q, caller, [item()] * (cap + 1))


def test_h3_produces_a_silent_clip_with_a_scene_caption_and_turn_metadata(h3_env):
    q, caller, staging = h3_env
    items = [item(), item(turns=[{"prompt": "the camera holds still"}],
                          keyframes=[{"image": "first.png", "frame": 0}, {"image": "last.png", "frame": -1}])]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    by = {i["index"]: i for i in job_result(out)["items"]}
    job = out["id"]
    c = by[0]["candidate"]
    assert c["provenance"]["generator"] == "minimax-h3" and c["provenance"]["seed"] == 7
    assert "pose" not in c and "camera_motion" not in c and c["frames"] == 243
    info = probe_video(staging / "rollouts" / job / "0.mp4")
    assert (info.width, info.height, info.frames, round(info.fps)) == (960, 544, 243, 24)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(staging / "rollouts" / job / "0.mp4")], capture_output=True, text=True).stdout.split()
    assert streams == ["video"]
    caption = json.loads((staging / "rollouts" / job / "0.json").read_text())
    assert caption == {"caption": SCENE + "."}             # the scene only: turns are the agent's to confirm
    assert [t["prompt"] for t in c["turn_segments"]] == TURNS
    assert c["h3_prompt"] == by[0]["worker"]["prompt"] == h3.build_prompt(SCENE, TURNS, [0, 89, 153], 243, [], SOUND, MUSIC)
    assert [(t["frame_start"], t["frame_end_exclusive"]) for t in c["turn_segments"]] == [(0, 89), (89, 153), (153, 243)]
    w = by[1]["worker"]
    assert w["prompt"].startswith("How the reference pictures align") and [k["frame"] for k in w["keyframes"]] == [0, -1]
    assert w["gpus"] == "0,1,4,5" and w["rank"] == 0             # one worker with every GPU
    assert w["weights"] == str(REAL.repo_root / REAL.get("generators.h3.weights")) and w["gpu0_reserve_gib"] == 16


def test_h3_finish_refuses_a_render_of_the_wrong_size_or_length(tmp_path):
    from types import SimpleNamespace
    rec = Recorder(tmp_path / "run")
    b = H3Backend(h3_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    out = tmp_path / "out"
    out.mkdir()
    staged = {"index": 0, **item()}
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=1024, height=576)
    with pytest.raises(ValueError, match="960x544"):
        b.finish(SimpleNamespace(args={"frames": 243}), staged, out)
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=960, height=544)
    with pytest.raises(ValueError, match="48 frames, not 243"):
        b.finish(SimpleNamespace(args={"frames": 243}), staged, out)


def test_h3_finished_candidate_passes_the_real_ingestor_as_per_chunk(tmp_path):
    from types import SimpleNamespace
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from tests.conftest import write_poses
    run_dir = tmp_path / "run"
    out = run_dir / "staging" / "rollouts"
    out.mkdir(parents=True)
    make_mp4(out / "0.mp4", seconds=243 / 24, fps=24, width=960, height=544)
    rec = Recorder(run_dir)
    res = H3Backend(h3_cfg(), run_dir, [0, 1, 2, 3], TokenRegistry(rec), rec).finish(
        SimpleNamespace(args={"frames": 243}), {"index": 0, **item()}, out)
    pose = write_poses(out / "vigeo.npz", n_frames=res["frames"], width=960, height=544)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    assert json.loads(Path(res["caption"]).read_text()) == {"caption": SCENE + "."}
    timed = out / "timed.json"
    timed.write_text(json.dumps(timed_caption(res)))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), caption=timed, pose=pose,
                                camera_motion="moving", provenance={"kind": "rollout", "generator": "minimax-h3",
                                "job_id": "j1", "inputs_hash": "x", "seed": 7})], node_id="n1")
    assert r.accepted, r.reasons
    assert "video_timed_prompts_camera:per_chunk" in r.formats       # the planned boundaries are on the round grid


def test_h3_is_registered_only_when_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((True, True), (False, False)):
        names = [b.name for b in build_gpu_backends(h3_cfg(enabled=enabled), tmp_path / "run", [0, 1, 2, 3],
                                                     TokenRegistry(rec), rec)]
        assert ("rollout_h3" in names) is present


def _bridge():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h3_generate", h3.H3_BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # torch and diffusers are imported inside main(), not here
    return mod


def test_bridge_spreads_the_blocks_by_memory_and_keeps_small_layers_on_gpu_0():
    b = _bridge()
    dmap = b.block_device_map([23.6, 23.6, 23.6, 23.6], 16)
    blocks = [dmap[f"transformer_blocks.{i}"] for i in range(50)]
    assert [blocks.count(d) for d in range(4)] == [5, 15, 15, 15]        # the split the spike measured
    assert blocks == sorted(blocks)                                       # contiguous: one hop per card
    assert all(dmap[name] == 0 for name in b.SMALL)
    two = b.block_device_map([48.0, 48.0], 16)
    assert [list(two.values()).count(d) for d in (0, 1)][1] == 30
    assert set(b.block_device_map([80.0], 16).values()) == {0}


def test_bridge_refuses_cards_that_cannot_hold_the_text_encoder():
    b = _bridge()
    b.require_memory([23.6] * 4)
    with pytest.raises(b.TooSmall, match="about 70 GiB"):            # too small: the kernel stops the run on it
        b.require_memory([23.6, 23.6])
    oom = b.out_of_memory("the transformer", [23.6] * 4)             # a busy card, not the machine: a job failure
    assert not isinstance(oom, b.TooSmall) and str(oom) == (
        "MiniMax H3 ran out of GPU memory loading the transformer on 4 card(s) of 24, 24, 24, 24 GiB; "
        "another process may be holding GPU memory")


@pytest.mark.parametrize("size", [(1280, 720), (800, 1088), (640, 360), (960, 544)])
def test_bridge_crops_any_keyframe_to_the_canvas(size):
    img = _bridge().fit(Image.new("RGB", size, (5, 6, 7)), (960, 544))
    assert img.size == (960, 544) and img.mode == "RGB"


@pytest.mark.gpu
def test_real_h3_rollout(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_h3.py -m gpu -s --basetemp=.cache/pytest/gpu   (~70 min)

    A text-only single-turn item, a three-turn item with first and last keyframes and a two-turn
    item with only a last keyframe (Z-Image frames), through the real tool path; each candidate
    gets a ViGeo pose and, with segments written from its `turn_segments`, must ingest as
    video_timed_prompts_camera:per_chunk."""
    import os
    import shutil
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from ar_kernel.tools.annotate import AnnotateBackend
    from ar_kernel.tools.images import ImageBackend
    from tests.test_rollouts import _wait

    gpus = [int(g) for g in os.environ.get("AR_TEST_GPUS", "0,1,2,3").split(",")]
    cfg = h3_cfg(env=REAL.get("generators.h3.env"))
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    for b in (ImageBackend, AnnotateBackend, H3Backend):
        q.register(b(cfg, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    host = lambda p: staging / Path(p).relative_to("/workspace/staging")
    try:
        img = _wait(q, caller, q.backends["generate_images"].submit(q, caller, {"width": 960, "height": 544, "items": [
            {"prompt": "a quiet harbor at dawn seen from the quay, fishing boats, photorealistic", "seed": 3},
            {"prompt": "the same harbor seen from the end of the pier looking back at the town, photorealistic",
             "seed": 4}]})["job_id"])
        assert img["state"] == "done", img.get("error")
        first, last = (i["image"] for i in job_result(img)["items"])
        audio = {"overall_soundscape": "A light wind and quiet natural ambience.", "non_diegetic_music": "N/A"}
        items = [
            {"scene_prompt": "Live-action, a first-person view on a forest trail in autumn, tall pines, low sun",
             "turns": [{"prompt": "the camera pushes in at slow speed along the trail"}], "seed": 11, **audio},
            {"scene_prompt": "Live-action, a first-person view on a quay in a quiet harbor at dawn",
             "turns": [{"prompt": "the camera pushes in at slow speed along the quay"},
                       {"prompt": "the camera pans right with large amplitude toward the fishing boats"},
                       {"prompt": "a seagull lands on the nearest boat"}],
             "keyframes": [{"image": first, "frame": 0}, {"image": last, "frame": -1}], "seed": 12, **audio},
            {"scene_prompt": "Live-action, a first-person view on a quay in a quiet harbor at dawn",
             "turns": [{"prompt": "the camera trucks left at slow speed along the water's edge"},
                       {"prompt": "the camera pans left and settles on the white fishing boat moored at the quay"}],
             "keyframes": [{"image": first, "frame": -1}], "seed": 13, **audio}]
        out = _wait(q, caller, submit(q, caller, items)["job_id"])
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
        assert all("candidate" in by[i] for i in range(3)), by
        cands = [by[i]["candidate"] for i in range(3)]
        print("h3 worker:", [by[i]["worker"] for i in range(3)])
        ann = _wait(q, caller, q.backends["annotate_camera"].submit(
            q, caller, {"items": [{"video": c["video"]} for c in cands]})["job_id"])
        assert ann["state"] == "done", ann.get("error")
    finally:
        q.shutdown()
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    for i, c in enumerate(cands):
        a = job_result(ann)["items"][i]
        assert "error" not in a, a
        stage = staging / "ingest" / str(i)
        stage.mkdir(parents=True)
        shutil.copy(host(c["video"]), stage / "v.mp4")
        assert set(json.loads(host(c["caption"]).read_text())) == {"caption"}
        (stage / "c.json").write_text(json.dumps(timed_caption(c)))
        shutil.copy(host(a["pose"]), stage / "p.npz")
        [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=stage / "p.npz",
                                      camera_motion="moving", provenance=c["provenance"], license=c["license"])],
                           node_id="gpu")
        assert res.accepted, res.reasons
        assert "video_timed_prompts_camera:per_chunk" in res.formats, res.formats
