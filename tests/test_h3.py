"""rollout_h3: the turn layout, the prompt the kernel writes, submit checks, the real produce with
a fake worker, and one real gpu smoke."""
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
from ar_kernel.tools.h3 import H3Backend, build_prompt, clip_frames, turn_starts
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

SCENE = "Live-action, a cobblestone street at dusk"
FP, TP = "first_person", "third_person"
TURNS = [{"action": "W", "perspective": FP},
         {"action": "right", "event": "A white dog runs out of a doorway.", "perspective": FP},
         {"action": "stop", "perspective": TP}]


@pytest.mark.parametrize("frames, n, starts", [
    (243, 1, [0]), (243, 2, [0, 122]), (243, 3, [0, 81, 162]), (243, 4, [0, 61, 122, 182]), (124, 2, [0, 62])])
def test_a_clip_is_split_into_equal_turns(frames, n, starts):
    assert turn_starts(frames, n) == starts


@pytest.mark.parametrize("frames, n, most", [(243, 5, 4), (124, 3, 2), (158, 3, 2)])
def test_more_turns_than_the_clip_holds_is_refused(frames, n, most):
    with pytest.raises(ToolError, match=f"at most {most} turns"):
        turn_starts(frames, n)


@pytest.mark.parametrize("seconds, frames", [(None, 243), (10, 243), (10.4, 243), (8, 192), (5.2, 124), (5.0, 124), (7.5, 175)])
def test_the_clip_gets_the_nearest_length_the_model_can_render(seconds, frames):
    assert clip_frames(seconds, 243, 243) == frames and (frames - 5) % 17 == 0


@pytest.mark.parametrize("seconds", [4.7, 10.6, 30, "8", True])
def test_a_length_outside_the_range_is_refused(seconds):
    with pytest.raises(ToolError, match="seconds"):
        clip_frames(seconds, 243, 243)


def test_the_kernel_writes_the_whole_prompt():
    p = build_prompt({"scene_prompt": SCENE, "character_prompt": "A courier in a yellow raincoat", "turns": TURNS},
                     [0, 81, 162])
    assert p == (
        "integrated_multimodal_description: [Shot 1] Live-action, a cobblestone street at dusk. "
        "A courier in a yellow raincoat. First-person view. "
        "The whole video is one continuous shot with smooth motion and no cuts. "
        "From the start, The camera pushes forward. "
        "About 3.4 seconds in, in the same continuous shot, A white dog runs out of a doorway. "
        "The camera pans to the right. "
        "About 6.8 seconds in, in the same continuous shot, The subject stays still. The camera holds its position. "
        "The view switches from first-person to third-person, revealing the character from behind.\n\n"
        "overall_soundscape: Natural ambient sound of the scene.\n\n"
        "non_diegetic_music: N/A")
    assert "[Shot 2]" not in p and ".." not in p


def test_the_perspective_is_stated_once_and_picks_the_camera_sentences():
    third = [{"action": "S+A", "perspective": TP}, {"action": "stop", "perspective": FP}]
    p = build_prompt({"scene_prompt": SCENE, "turns": third}, [0, 122])
    assert p.count("Third-person view from behind the subject.") == 1 and "First-person view." not in p
    assert "From the start, The subject moves backward-left. The camera retreats along the same diagonal." in p
    assert p.split("About 5.1 seconds in, in the same continuous shot, ")[1].startswith(
        "The camera holds still. The view switches back to first-person, returning to the character's own eyes.")


def test_every_action_has_a_camera_sentence_in_both_perspectives():
    from ar_kernel.data.annotation import ACTIONS, PERSPECTIVES
    assert all(set(h3.CAMERA[p]) == set(ACTIONS) for p in PERSPECTIVES)
    assert h3.CAMERA[FP]["stop"] == "The camera holds still."
    assert h3.CAMERA[TP]["stop"] == "The subject stays still. The camera holds its position."
    # the evaluation's own sentences, where it has them
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wbench_prompts", KernelConfig.load().repo_root.parent / "WBench/src/models/text/prompt_builder.py")
    theirs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theirs)
    assert {k: v for k, v in h3.CAMERA[FP].items() if k != "stop"} == theirs.FIRST_PERSON_ACTIONS
    assert {k: v for k, v in h3.CAMERA[TP].items() if k not in ("stop", "S+A")} == theirs.THIRD_PERSON_ACTIONS


def test_a_first_frame_adds_the_alignment_line_and_nothing_an_item_wrote_breaks_the_fields():
    p = build_prompt({"scene_prompt": "A street\n\nnon_diegetic_music: drums", "image": "x.png",
                      "turns": [{"action": "stop", "event": "a long\n\nevent  with   gaps", "perspective": FP}]}, [0])
    assert p.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) "
                        "is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] ")
    assert "From the start, a long event with gaps. The camera holds still." in p
    assert p.count("\n\n") == 3                        # the alignment line and the three fields


REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def h3_cfg(env="autoresearcher", **over):
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["h3"].update({"env": env, "enabled": True, **over})
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


DEFAULT, OTHER = (960, 544), (1024, 576)      # (width, height): the tool's size, and another


def item(**over):
    return {"scene_prompt": SCENE, "turns": copy.deepcopy(TURNS), "seed": 7, **over}


@pytest.fixture
def h3_env(tmp_path, monkeypatch):
    """The real H3Backend.produce/run_workers, with torchrun and h3_generate.py swapped for the fake worker."""
    monkeypatch.setattr(h3, "H3_BRIDGE", FAKE)
    monkeypatch.setattr(h3, "launcher", lambda ranks: ["python"])
    monkeypatch.setattr(h3, "checked_repo", lambda *a: None)
    monkeypatch.setattr(h3, "meminfo_gib", lambda: {"MemAvailable": 250.0})
    (tmp_path / "adaln").mkdir()
    config = tmp_path / "lightx2v.json"
    config.write_text(json.dumps({"adaln_cache_dir": str(tmp_path / "adaln"), "parallel": {"seq_p_size": 8}}))
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", DEFAULT, (10, 20, 30)).save(ws / "first.png")
    Image.new("RGB", OTHER, (30, 20, 10)).save(ws / "other.png")
    q.register(H3Backend(h3_cfg(config=str(config)), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                         gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def submit(q, caller, items):
    return q.backends["rollout_h3"].submit(q, caller, {"items": items})


def _turn(**over):
    return {"action": "W", "perspective": FP, **over}


@pytest.mark.parametrize("bad, match", [
    (item(scene_prompt=" "), "scene_prompt"),
    (item(character_prompt=5), "character_prompt"),
    (item(turns=[]), "turns"),
    (item(turns=[{"prompt": "walk"}]), "prompt is not a field of a turn"),
    (item(turns=[_turn(action="forward")]), "turn 1: action must be one of"),
    (item(turns=[_turn(action="A+D")]), "turn 1: action must be one of"),
    (item(turns=[{"action": "W"}]), "turn 1: perspective must be one of"),
    (item(turns=[_turn(event="")]), "turn 1: event must be one sentence"),
    (item(turns=[_turn(), _turn(perspective=TP)]), "turn 2 switches the perspective, so its action must be stop"),
    (item(turns=[_turn()] * 5), "at most 4 turns"),
    (item(seconds=5.2, turns=[_turn()] * 3), "at most 2 turns"),
    (item(seconds=4), "seconds must be between 5.2 and 10.1"),
    (item(seed="x"), "seed"),
    (item(overall_soundscape="wind"), "overall_soundscape is not a field of an item"),
    (item(keyframes=[{"image": "first.png", "frame": -1}]), "keyframes is not a field of an item"),
    (item(image="first.txt"), "image must be an image file"),
    (item(image="other.png"), "image is {}x{}, but the job renders {}x{}".format(*OTHER, *DEFAULT)),
])
def test_h3_submit_refuses(h3_env, bad, match):
    q, caller, _ = h3_env
    with pytest.raises(ToolError, match=match):
        submit(q, caller, [bad])


def test_h3_job_over_the_configured_cap_is_refused(h3_env):
    q, caller, _ = h3_env
    cap = REAL.get("generators.h3.max_items")
    with pytest.raises(ToolError, match=f"at most {cap} items per job"):
        submit(q, caller, [item()] * (cap + 1))


def test_h3_produces_a_silent_clip_with_its_annotation(h3_env):
    q, caller, staging = h3_env
    items = [item(character_prompt="A courier in a yellow raincoat"),
             item(turns=[{"action": "stop", "perspective": FP}], image="first.png", seconds=8)]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    by = {i["index"]: i for i in job_result(out)["items"]}
    job = out["id"]
    c = by[0]["candidate"]
    assert set(c) == {"video", "annotation", "derived_from", "frames", "seconds", "h3_prompt"}
    assert c["derived_from"] == {"kind": "model", "name": "minimax-h3"}
    assert c["frames"] == 243 and c["seconds"] == 243 / 24
    info = probe_video(staging / "rollouts" / job / "0.mp4")
    assert (info.width, info.height, info.frames, round(info.fps)) == (*DEFAULT, 243, 24)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(staging / "rollouts" / job / "0.mp4")], capture_output=True, text=True).stdout.split()
    assert streams == ["video"]
    assert c["annotation"] == f"/workspace/staging/rollouts/{job}/0.annotation.json"
    assert json.loads((staging / "rollouts" / job / "0.annotation.json").read_text()) == {
        "scene_prompt": SCENE, "character_prompt": "A courier in a yellow raincoat",
        "turns": [{"start_s": 0.0, "perspective": FP},
                  {"start_s": 81 / 24, "event": "A white dog runs out of a doorway.", "perspective": FP},
                  {"start_s": 162 / 24, "perspective": TP}]}
    assert c["h3_prompt"] == by[0]["worker"]["prompt"] == build_prompt(items[0], [0, 81, 162])
    w = by[1]["worker"]
    assert w["prompt"].startswith("For the target video, at 0.00 seconds") and w["frames"] == 192
    assert w["image"].endswith("/in/1_image.png")
    assert by[1]["candidate"]["frames"] == 192 and by[1]["candidate"]["seconds"] == 8.0
    assert w["gpus"] == "0,1,4,5" and w["rank"] == 0             # one worker with every GPU
    assert w["weights"] == str(REAL.repo_root / REAL.get("generators.h3.weights"))
    assert w["config"]["parallel"] == {"seq_p_size": 4}          # a rank per GPU


def test_h3_job_fails_while_host_ram_is_short(h3_env, monkeypatch):
    q, caller, _ = h3_env
    monkeypatch.setattr(h3, "meminfo_gib", lambda: {"MemAvailable": 150.0})
    out = q.wait(caller, submit(q, caller, [item()])["job_id"], 120)
    assert out["state"] == "failed" and "not enough free host RAM" in json.dumps(out)


def test_h3_finish_refuses_a_render_of_the_wrong_size_or_length(tmp_path):
    from types import SimpleNamespace
    rec = Recorder(tmp_path / "run")
    b = H3Backend(h3_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    out = tmp_path / "out"
    out.mkdir()
    staged, job = {"index": 0, **item()}, SimpleNamespace(args={})
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=1024, height=576)
    with pytest.raises(ValueError, match="960x544"):
        b.finish(job, staged, out)
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=960, height=544)
    with pytest.raises(ValueError, match="48 frames, not 243"):
        b.finish(job, staged, out)


def test_h3_finished_candidate_passes_the_real_ingestor(tmp_path):
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
        SimpleNamespace(args={}), {"index": 0, **item()}, out)
    pose = write_poses(out / "vigeo.npz", n_frames=res["frames"], width=960, height=544)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), annotation=Path(res["annotation"]), pose=pose,
                                derived_from={"kind": "model", "name": "minimax-h3"})], node_id="n1")
    assert r.accepted, r.reasons                    # the planned turns are long enough to be trained on
    caption = json.loads(ing.blobs.path(ing.clips.get(r.clip_id)["caption_digest"], "caption").read_text())
    assert [s["time_range_s"] for s in caption["segments"]] == [[0.0, 81 / 24], [81 / 24, 162 / 24], [162 / 24, 243 / 24]]
    assert caption["segments"][1]["prompt"] == f"{SCENE} A white dog runs out of a doorway."


def test_descriptions_make_h3_the_default_rollout_tool(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    d = {b.name: b.description for b in build_gpu_backends(h3_cfg(), tmp_path / "run", [0, 1, 2, 3],
                                                           TokenRegistry(rec), rec) if b.name.startswith("rollout_")}
    assert d["rollout_h3"].startswith("MiniMax H3 is the most powerful video generation model of the rollout tools: "
                                      "use this tool for a new training clip unless")
    assert d["rollout_ltx25"].startswith("Use this tool when a clip needs a keyframe between its first and last")
    assert d["rollout_alayaworld"].startswith("Use this tool when a clip's camera has to follow commanded moves")


def test_h3_is_registered_only_when_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((True, True), (False, False)):
        names = [b.name for b in build_gpu_backends(h3_cfg(enabled=enabled), tmp_path / "run", [0, 1, 2, 3],
                                                     TokenRegistry(rec), rec)]
        assert ("rollout_h3" in names) is present


@pytest.mark.gpu
def test_real_h3_rollout(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_h3.py -m gpu -s --basetemp=.cache/pytest/gpu

    A text-only single-turn item and a three-turn item with a first frame (a Z-Image frame), through
    the real tool path; each candidate gets a ViGeo pose and must ingest with its own annotation."""
    import os
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
        img = _wait(q, caller, q.backends["generate_images"].submit(q, caller, {
            "width": DEFAULT[0], "height": DEFAULT[1], "items": [
            {"prompt": "a quiet harbor at dawn seen from the quay, fishing boats, photorealistic", "seed": 3},
            {"prompt": "the same harbor seen from the end of the pier looking back at the town, photorealistic",
             "seed": 4}][:1]})["job_id"])
        assert img["state"] == "done", img.get("error")
        first = job_result(img)["items"][0]["image"]
        items = [
            {"scene_prompt": "Live-action, a forest trail in autumn, tall pines, low sun", "seed": 11, "seconds": 5.2,
             "turns": [{"action": "W", "perspective": FP}]},
            {"scene_prompt": "Live-action, a quay in a quiet harbor at dawn, fishing boats", "image": first, "seed": 12,
             "turns": [{"action": "W", "perspective": FP},
                       {"action": "right", "event": "A seagull lands on the nearest boat.", "perspective": FP},
                       {"action": "stop", "perspective": FP}]}]
        out = _wait(q, caller, submit(q, caller, items)["job_id"])
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
        assert all("candidate" in by[i] for i in range(2)), by
        cands = [by[i]["candidate"] for i in range(2)]
        print("h3 worker:", [by[i]["worker"] for i in range(2)])
        ann = _wait(q, caller, q.backends["annotate_camera"].submit(
            q, caller, {"items": [{"video": c["video"]} for c in cands]})["job_id"])
        assert ann["state"] == "done", ann.get("error")
    finally:
        q.shutdown()
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    for i, c in enumerate(cands):
        a = job_result(ann)["items"][i]
        assert "error" not in a, a
        [res] = ing.ingest([Candidate(video=host(c["video"]), annotation=host(c["annotation"]), pose=host(a["pose"]),
                                      derived_from=c["derived_from"])], node_id="gpu")
        assert res.accepted, res.reasons
