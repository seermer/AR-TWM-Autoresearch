"""rollout_alayaworld: case/config writers, submit checks, finish (CPU), the real produce with a
fake worker, plus one real AlayaWorld gpu smoke per variant."""
import copy
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ar_kernel.config import KernelConfig
from ar_kernel.subproc import run_in_env
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import rollouts
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.rollouts import AlayaWorldBackend, case_json, render_config
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def small_cfg(env="autoresearcher", enabled=("dmd4", "ar30")):
    raw = copy.deepcopy(REAL.raw)
    a = raw["generators"]["alayaworld"]
    a["env"] = env
    a["variants"] = {v: {"enabled": v in enabled} for v in ("dmd4", "ar30")}
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


def first_person(**over):
    return {"image": "frame.png", "perspective": "first_person",
            "environment_prompt": "A wet city street at night with neon signs.",
            "character_prompt": "", "perspective_prompt": "First-person view at eye level.",
            "turns": [{"action": "W"}, {"action": "left", "event_edit": "It starts to rain heavily."}], **over}


def third_person(**over):
    return {"image": "frame.png", "perspective": "third_person", "subject_mask": "mask.png",
            "environment_prompt": "A mountain trail above a valley.",
            "character_prompt": "A hiker in a red jacket.",
            "perspective_prompt": "Third-person view from behind the hiker.",
            "turns": [{"action": "W+left", "subject_action": "The hiker raises a hand and waves."},
                      {"action": "stop", "perspective_switch": "tp_to_fp"}], **over}


# ---- case writer: WorldModel's own loader reads it back ----

_LOAD = r"""
import json, sys
from alaya.data.wbench import WBenchNaviDataset
ds = WBenchNaviDataset(root=sys.argv[1], width=960, height=544, frames=1,
                       case_ids=sys.argv[2].split(","), include_non_navigation=True)
out = {}
for i in range(len(ds)):
    m = ds[i]["metadata"]
    out[m["wbench_case_id"]] = {"schedule": m["wbench_prompt_schedule"], "actions": m["wbench_turn_actions"],
                                "perspective": m["wbench_perspective"], "mask": m["wbench_subject_mask"],
                                "turns": ds.full_turn_counts[i]}
print("===AR===" + json.dumps(out) + "===AR===")
"""


def test_case_files_load_with_the_worldmodel_loader(tmp_path):
    data = tmp_path / "data"
    for sub in ("cases", "images", "masks"):
        (data / sub).mkdir(parents=True)
    Image.new("RGB", (640, 360), (90, 120, 60)).save(data / "images" / "case_3.png")
    Image.new("RGB", (640, 360), (90, 120, 60)).save(data / "images" / "case_4.png")
    Image.new("L", (640, 360), 255).save(data / "masks" / "case_4_mask.png")
    (data / "cases" / "case_3.json").write_text(json.dumps(case_json(3, first_person(), "images/case_3.png", None)))
    (data / "cases" / "case_4.json").write_text(json.dumps(
        case_json(4, third_person(), "images/case_4.png", "masks/case_4_mask.png")))
    proc = run_in_env("alayaworld", ["python", "-c", _LOAD, str(data), "3,4"], cwd=REAL.worldmodel)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.split("===AR===")[1])

    fp = got["3"]
    base = "A wet city street at night with neon signs."
    assert fp["schedule"] == [base, f"{base} It starts to rain heavily."]
    assert fp["actions"] == ["W", "left"] and fp["perspective"] == "first_person" and fp["turns"] == 2
    assert fp["mask"] == ""

    tp = got["4"]
    scene = "A hiker in a red jacket. A mountain trail above a valley. The hiker raises a hand and waves."
    assert tp["schedule"][0] == scene
    assert tp["schedule"][1].startswith(scene + " The view switches from third-person to first-person")
    assert tp["actions"] == ["W+left", "stop"] and tp["perspective"] == "third_person"
    assert tp["mask"] == str(data / "masks" / "case_4_mask.png")


def test_case_json_uses_the_wbench_schema():
    case = case_json(7, third_person(), "images/case_7.jpg", "masks/case_7_mask.png")
    assert case["id"] == "7" and case["metric_list"] == []
    assert case["settings"]["initial_image"] == "images/case_7.jpg"
    assert case["settings"]["subject_mask"] == "masks/case_7_mask.png"
    assert case["settings"]["perspective"] == "third_person"
    assert case["interactions"] == [
        {"type": "navigation", "action": "W+left", "turn": 1},
        {"type": "subject_action", "action": "The hiker raises a hand and waves.", "turn": 1},
        {"type": "navigation", "action": "stop", "turn": 2},
        {"type": "perspective_switch", "action": "tp_to_fp", "turn": 2}]


# ---- render config ----

def _flat(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def test_render_config_has_the_listed_fields(tmp_path):
    c = render_config(REAL, variant="dmd4", rounds_per_turn=1, seed=7, indices=[0, 2], work=tmp_path,
                      text_cache=tmp_path / "te")
    mode = c["validation"]["modes"]["wbench"]
    assert mode["dataset"]["root"] == str(tmp_path / "data")
    assert mode["dataset"]["case_ids"] == ["0", "2"]
    assert mode["dataset"]["include_non_navigation"] is True
    assert mode["wbench_output_dir"] == str(tmp_path / "videos")
    assert mode["wbench_chunks_per_turn"] == 1
    assert c["run"]["output_dir"] == str(tmp_path / "rollout") and c["run"]["log_dir"] == str(tmp_path / "logs")
    assert c["run"]["seed"] == 7
    assert c["validation"]["per_sample_seed"] is True and c["validation"]["save_joystick"] is False
    assert c["runtime"]["text_embed_cache_dir"] == str(tmp_path / "te")
    for key, value in c["paths"].items():
        if isinstance(value, str):
            assert Path(value).is_absolute() and value.startswith(str(REAL.worldmodel)), key
    assert c["paths"]["dmd_resume"] == str(REAL.worldmodel / "weights/alaya-world-dmd")
    # everything else is the eval's own wbench_full.yaml
    source = yaml.safe_load((REAL.worldmodel / "configs" / "wbench_full.yaml").read_text())
    assert c["sample"] == source["sample"] and c["spatial_memory"] == source["spatial_memory"]
    assert mode["memory_start_round"] == source["validation"]["modes"]["wbench"]["memory_start_round"]


def test_ar30_differs_from_dmd4_in_exactly_the_four_keys(tmp_path):
    kw = dict(rounds_per_turn=3, seed=1, indices=[0], work=tmp_path, text_cache=tmp_path / "te")
    dmd4, ar30 = (_flat(render_config(REAL, variant=v, **kw)) for v in ("dmd4", "ar30"))
    assert dmd4.keys() == ar30.keys()
    diff = {k: ar30[k] for k in dmd4 if dmd4[k] != ar30[k]}
    assert diff == {("paths", "dmd_resume"): None, ("validation", "sampling_steps"): 30,
                    ("validation", "scheduler"): "shift", ("validation", "cfg_scale"): 3.0}


# ---- submit checks ----

@pytest.fixture
def env(tmp_path, monkeypatch):
    """The real AlayaWorldBackend: produce launches the fake worker in place of the prompt
    precache and run_wbench.py (wbench mode: it writes the measured output layout)."""
    monkeypatch.setattr(rollouts, "PRECACHE", [str(FAKE), "--precache"])
    monkeypatch.setattr(rollouts, "RUN_WBENCH", [str(FAKE)])
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", (640, 360), (90, 120, 60)).save(ws / "frame.png")
    mask = Image.new("L", (640, 360), 0)
    mask.paste(255, (250, 100, 390, 340))
    mask.save(ws / "mask.png")
    (ws / "mask.txt").write_text("not an image")
    q.register(AlayaWorldBackend(small_cfg(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                                 gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging, tmp_path / "run"
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_alayaworld"].submit(q, caller, {"items": items, **params})


@pytest.mark.parametrize("item, params, match", [
    (first_person(turns=[{"action": "jump"}]), {}, "unknown action"),
    (first_person(turns=[{"action": "W+fly"}]), {}, "unknown action"),
    (first_person(turns=[{"action": ""}]), {}, "unknown action"),
    (first_person(turns=[]), {}, "turns"),
    (first_person(turns=[{"action": "W"}] * 10), {}, "at most 9 turns"),
    (first_person(), {"rounds_per_turn": 0}, "rounds_per_turn"),
    (first_person(), {"rounds_per_turn": 4}, "rounds_per_turn"),
    (third_person(subject_mask="mask.txt"), {}, "subject_mask"),
    (first_person(perspective="top_down"), {}, "perspective"),
    (first_person(environment_prompt=""), {}, "environment_prompt"),
    (first_person(image="frame.txt"), {}, "image"),
    (first_person(turns=[{"action": "W", "camera": "fly"}]), {}, "camera"),
    (first_person(), {"variant": "dmd8"}, "variant"),
    (first_person(), {"seed": "x"}, "seed"),
])
def test_submit_refuses(env, item, params, match):
    q, caller, _, _ = env
    with pytest.raises(ToolError, match=match):
        submit(q, caller, [item], **params)


def test_submit_refuses_a_disabled_variant(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AlayaWorldBackend(small_cfg(enabled=("dmd4",)), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    with pytest.raises(ToolError, match="variant"):
        b.check_args({"items": [first_person()], "variant": "ar30"})


def test_submit_accepts_every_wbench_action_and_fills_defaults(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AlayaWorldBackend(small_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    actions = ["W", "S", "A", "D", "left", "right", "up", "down", "stop", "W+left", "S+D", "W+down", "w"]
    args = {"items": [first_person(turns=[{"action": a}]) for a in actions]}
    b.check_args(args)
    assert (args["variant"], args["rounds_per_turn"], args["seed"]) == ("dmd4", 3, 42)


def test_limits_come_from_the_generator_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AlayaWorldBackend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    a = REAL.get("generators.alayaworld")
    assert (b.max_items, b.timeout_s, b.max_turns) == (a["max_items"], a["timeout_s"], a["max_turns"])


# ---- finish: trim to the per_chunk grid, poses, segments ----

def _render_output(out, i, *, rounds, cpt, prompts, first=-7):
    """What produce leaves in out/ for one case. Measured layout (Step 2): round r = mp4 frames
    [first + 32r, first + 32r + 32) with first = -7, and nominal turn_segments (as WorldModel
    writes them, ignoring `first`)."""
    frames = first + rounds * 32
    make_mp4(out / f"{i}.mp4", seconds=frames / 24, fps=24, width=960, height=544)
    c2w = np.tile(np.eye(4), (frames, 1, 1))
    c2w[:, 0, 3] = np.arange(frames) * 0.02
    np.savez(out / f"{i}.camera.npz", cam_c2w=c2w.astype(np.float32))
    segs = [{"turn_index": t, "action": "W", "chunk_start": t * cpt, "chunk_end_exclusive": (t + 1) * cpt,
             "chunk_count": cpt, "frame_start": t * cpt * 32, "frame_end_exclusive": min(frames, (t + 1) * cpt * 32),
             "frame_count": min(frames, (t + 1) * cpt * 32) - t * cpt * 32} for t in range(rounds // cpt)]
    (out / f"{i}.sidecar.json").write_text(json.dumps({
        "actions": ["W"] * (rounds // cpt), "turn_segments": segs, "output_rounds": rounds, "num_frames": frames,
        "prompt_schedule": [{"round": r, "turn": r // cpt, "prompt": prompts[r // cpt]} for r in range(rounds)]}))
    (out / f"{i}.json").write_text(json.dumps({"ok": True}))
    return frames


def _job(**args):
    return type("Job", (), {"id": "j1", "node": "n1", "args": {"variant": "dmd4", "seed": 42, **args}})()


def _backend(tmp_path):
    rec = Recorder(tmp_path / "run")
    return AlayaWorldBackend(small_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)


def _check_boundaries(segments, frames):
    assert segments[0]["time_range_s"][0] == 0.0
    assert abs(segments[-1]["time_range_s"][1] * 24 - frames) < 1e-6
    for a, b in zip(segments, segments[1:]):
        t = a["time_range_s"][1]
        assert t == b["time_range_s"][0]
        k = round((t * 24 - 25) / 32)
        assert abs(t * 24 - (25 + 32 * k)) <= 0.5, t


@pytest.mark.parametrize("cpt, prompts, first, trim", [
    (3, ["walk", "walk and rain"], -7, 0),      # the measured layout: already on the grid
    (1, ["a", "b", "c"], -7, 0),
    (3, ["walk", "walk and rain"], 0, 7),       # a layout starting on a round boundary: trimmed
])
def test_finish_puts_round_boundaries_on_the_grid(tmp_path, cpt, prompts, first, trim):
    out = tmp_path / "out"
    out.mkdir()
    rounds = len(prompts) * cpt
    frames = _render_output(out, 0, rounds=rounds, cpt=cpt, prompts=prompts, first=first)
    res = _backend(tmp_path).finish(_job(rounds_per_turn=cpt), {"index": 0}, out)
    assert res["trim"] == trim and res["frames"] == frames - trim
    from ar_kernel.data.probe import probe_video
    assert probe_video(res["video"]).frames == frames - trim
    with np.load(res["pose"]) as z:
        c2w = z["cam_c2w"]
    assert len(c2w) == frames - trim and np.allclose(c2w[0], np.eye(4))
    assert np.allclose(c2w[:, 0, 3], np.arange(frames - trim) * 0.02, atol=1e-5)   # saved[trim:], re-based
    cap = json.loads(Path(res["caption"]).read_text())
    assert cap["caption"] == prompts[0]
    assert [s["prompt"] for s in cap["segments"]] == prompts                     # one per turn, merged
    _check_boundaries(cap["segments"], frames - trim)
    assert res["camera_motion"] == "moving" and res["actions"] == ["W"] * len(prompts)
    # turn_segments come back in published-clip frames, on the same grid
    starts = [t["frame_start"] for t in res["turn_segments"]]
    assert starts[0] == 0 and all((f - 25) % 32 == 0 for f in starts[1:])
    assert res["turn_segments"][-1]["frame_end_exclusive"] == frames - trim


def test_finish_refuses_a_frame_count_off_the_round_layout(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _render_output(out, 0, rounds=3, cpt=3, prompts=["a"])
    side = json.loads((out / "0.sidecar.json").read_text())
    (out / "0.sidecar.json").write_text(json.dumps({**side, "output_rounds": 5}))
    with pytest.raises(ValueError, match="5 rounds"):
        _backend(tmp_path).finish(_job(rounds_per_turn=3), {"index": 0}, out)


def test_finish_merges_equal_prompts_and_keeps_changes_on_round_boundaries(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    frames = _render_output(out, 0, rounds=4, cpt=1, prompts=["a", "a", "b", "a"])
    res = _backend(tmp_path).finish(_job(rounds_per_turn=1), {"index": 0}, out)
    segs = json.loads(Path(res["caption"]).read_text())["segments"]
    assert [s["prompt"] for s in segs] == ["a", "b", "a"]
    assert [round(s["time_range_s"][0] * 24) for s in segs] == [0, 57, 89]
    _check_boundaries(segs, frames)


def test_finish_refuses_a_camera_path_of_the_wrong_length(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _render_output(out, 0, rounds=3, cpt=3, prompts=["a"])
    np.savez(out / "0.camera.npz", cam_c2w=np.tile(np.eye(4), (95, 1, 1)))
    with pytest.raises(ValueError, match="95 frames"):
        _backend(tmp_path).finish(_job(rounds_per_turn=3), {"index": 0}, out)


def test_finished_candidate_passes_the_real_ingestor_as_per_chunk(tmp_path):
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    run_dir = tmp_path / "run"
    out = run_dir / "staging" / "rollouts"          # ingest only takes files staged under the run
    out.mkdir(parents=True)
    _render_output(out, 0, rounds=6, cpt=3, prompts=["A forest trail.", "A forest trail. Rain starts."])
    res = _backend(tmp_path).finish(_job(rounds_per_turn=3), {"index": 0}, out)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), caption=Path(res["caption"]), pose=Path(res["pose"]),
                                camera_motion="moving", provenance={"kind": "rollout", "generator": "alayaworld-dmd4",
                                "job_id": "j1", "inputs_hash": "x", "seed": 42})], node_id="n1")
    assert r.accepted, r.reasons
    assert "video_timed_prompts_camera:per_chunk" in r.formats


# ---- the real produce/run_workers/_collect with the fake worker ----

def run_job(q, caller, items, **params):
    out = q.wait(caller, submit(q, caller, items, **params)["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in out["result"]["items"]}


def test_produce_renders_and_publishes_candidates(env):
    q, caller, staging, run_dir = env
    job, by = run_job(q, caller, [first_person(), third_person()], seed=5)
    for i, item in by.items():
        c = item["candidate"]
        assert c["video"] == f"/workspace/staging/rollouts/{job}/{i}.mp4"
        assert c["caption"] == f"/workspace/staging/rollouts/{job}/{i}.json"
        assert c["pose"] == f"/workspace/staging/rollouts/{job}/{i}.npz"
        assert c["camera_motion"] == "moving" and c["license"] == REAL.get("generators.alayaworld.license")
        assert c["provenance"]["generator"] == "alayaworld-dmd4" and c["provenance"]["seed"] == 5
        assert c["frames"] == 2 * 3 * 32 - 7
        with np.load(staging / "rollouts" / job / f"{i}.npz") as z:
            assert len(z["cam_c2w"]) == c["frames"]
        cap = json.loads((staging / "rollouts" / job / f"{i}.json").read_text())
        assert len(cap["segments"]) == 2
    assert by[0]["candidate"]["actions"] == ["W", "left"]
    assert by[1]["candidate"]["actions"] == ["W+left", "stop"]
    assert len(by[1]["candidate"]["turn_segments"]) == 2
    work = run_dir / "jobs" / job
    wb = json.loads((work / "fake_wbench.json").read_text())
    argv = wb["argv"]
    assert argv[argv.index("--gpus") + 1] == "0,1,4,5" and argv[argv.index("--cases") + 1] == "0,1"
    assert int(argv[argv.index("--master-port") + 1]) > 0 and wb["cwd"] == str(REAL.worldmodel)
    pre = json.loads((work / "fake_precache.json").read_text())
    assert pre["gpus"] == "0,1" and pre["gemma"] and "--device-map" in pre["argv"]
    cfg = yaml.safe_load((work / "render_config.yaml").read_text())
    assert cfg["run"]["seed"] == 5 and cfg["runtime"]["text_embed_cache_dir"] == str(run_dir / "cache" / "text_embed")
    assert not (work / "data").exists() and not (work / "videos").exists()


def test_rounds_per_turn_1_gives_a_segment_per_round(env):
    q, caller, staging, _ = env
    job, by = run_job(q, caller, [third_person()], rounds_per_turn=1)
    assert by[0]["candidate"]["frames"] == 2 * 32 - 7
    cap = json.loads((staging / "rollouts" / job / "0.json").read_text())
    assert len(cap["segments"]) == 2 and cap["segments"][1]["time_range_s"][0] * 24 == pytest.approx(25)


def test_a_case_without_video_and_a_bad_image_fail_alone(env):
    q, caller, staging, _ = env
    (staging.parent / "ws" / "broken.png").write_bytes(b"not a png")
    _, by = run_job(q, caller, [first_person(environment_prompt="NO_VIDEO here"), first_person(image="broken.png"),
                                first_person()])
    assert by[0]["error"] == "no video rendered"
    assert by[1]["error"].startswith("input:")
    assert "candidate" in by[2]


def test_ar30_job_is_named_after_its_variant(env):
    q, caller, _, run_dir = env
    job, by = run_job(q, caller, [first_person(turns=[{"action": "W"}])], variant="ar30")
    assert by[0]["candidate"]["provenance"]["generator"] == "alayaworld-ar30"
    cfg = yaml.safe_load((run_dir / "jobs" / job / "render_config.yaml").read_text())
    assert cfg["validation"]["sampling_steps"] == 30 and cfg["paths"]["dmd_resume"] is None


def test_build_gpu_backends_includes_alayaworld_only_with_an_enabled_variant(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in (((), False), (("dmd4",), True), (("ar30",), True)):
        names = [b.name for b in build_gpu_backends(small_cfg(enabled=enabled), tmp_path / "run", [0, 1, 2, 3],
                                                    TokenRegistry(rec), rec)]
        assert ("rollout_alayaworld" in names) is present


# ---- real AlayaWorld gpu smoke, per variant (spec 16.3 item 5) ----

HIKER = ("Photorealistic wide shot, a hiker in a bright red jacket and grey backpack walking away from the camera "
         "along a narrow dirt trail on a green mountain ridge, seen from behind, full body in the center of the "
         "frame, valley and distant peaks under a clear afternoon sky, natural light")


def _hiker_mask(path):
    """Hand-drawn around the hiker of generate_images(HIKER, seed 3) at 1280x720 (Task 6 report)."""
    from PIL import ImageDraw
    m = Image.new("L", (1280, 720), 0)
    d = ImageDraw.Draw(m)
    d.ellipse((646, 260, 697, 305), fill=255)
    d.polygon([(603, 300), (737, 300), (745, 468), (715, 472), (708, 662), (642, 662), (630, 478), (598, 468)],
              fill=255)
    m.save(path)


def _wait(q, caller, job_id):
    out = q.wait(caller, job_id, 3600)
    while out["state"] not in ("done", "failed", "cancelled"):
        out = q.wait(caller, job_id, 3600)
    return out


def _peak_sampler(gpus):
    """Peak nvidia-smi memory per GPU while the job runs (the worker's own peak is not visible
    through run_wbench)."""
    import subprocess
    import time
    peak, stop = {g: 0 for g in gpus}, threading.Event()

    def loop():
        while not stop.is_set():
            res = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            for line in res.stdout.splitlines():
                i, mib = (int(x) for x in line.split(","))
                if i in peak:
                    peak[i] = max(peak[i], mib)
            time.sleep(2)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return peak, stop


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["dmd4", "ar30"])
def test_real_alayaworld_rollout(tmp_path, variant):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_rollouts.py -m gpu -s --basetemp=.cache/pytest/gpu

    First person: frame 0 of an example clip. Third person: a generate_images (Z-Image) frame with a
    hand-drawn mask (the tools chain). Both published candidates must ingest as per_chunk, and
    ViGeo run on the published videos must agree with the published poses."""
    import os
    import shutil
    import subprocess
    import time
    from ar_kernel.archive.db import open_db
    from ar_kernel.config import resolve_gpus
    from ar_kernel.data.ingest import Candidate, Ingestor
    from ar_kernel.tools.annotate import AnnotateBackend
    from ar_kernel.tools.images import ImageBackend
    from tests.test_annotate import _rel0, _relative_rotation_errors, _umeyama_ate

    gpus = resolve_gpus(REAL, {"CUDA_VISIBLE_DEVICES": os.environ.get("AR_TEST_GPUS", "0,1,2,3")})
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["alayaworld"]["variants"][variant] = {"enabled": True}
    cfg = KernelConfig(raw=raw, repo_root=REAL.repo_root)
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                    str(REAL.worldmodel / "data/examples/video_caption_camera/videos/clip_0001.mp4"),
                    "-frames:v", "1", str(ws / "street.png")], check=True)
    _hiker_mask(ws / "hiker_mask.png")
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    for b in (ImageBackend, AlayaWorldBackend, AnnotateBackend):
        q.register(b(cfg, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    host = lambda p: staging / Path(p).relative_to("/workspace/staging")
    try:
        img = _wait(q, caller, q.backends["generate_images"].submit(
            q, caller, {"items": [{"prompt": HIKER, "seed": 3}]})["job_id"])
        assert img["state"] == "done", img.get("error")
        hiker = img["result"]["items"][0]["image"]
        items = [
            {"image": "street.png", "perspective": "first_person",
             "environment_prompt": "A rainy city street lined with parked cars and trees, wet asphalt reflecting "
                                   "the grey daylight, apartment blocks in the distance.",
             "perspective_prompt": "First-person view at eye level from the sidewalk.",
             "turns": [{"action": "W", "event_edit": "A red umbrella blows across the street in the wind."},
                       {"action": "left", "subject_action": "A cyclist in a yellow raincoat rides past."}]},
            {"image": hiker, "subject_mask": "hiker_mask.png", "perspective": "third_person",
             "environment_prompt": "A narrow dirt trail along a green mountain ridge above a wide valley, "
                                   "distant peaks under a clear afternoon sky.",
             "character_prompt": "A hiker in a bright red jacket and grey backpack.",
             "perspective_prompt": "Third-person view from behind the hiker.",
             "turns": [{"action": "W", "subject_action": "The hiker raises the right arm and waves."},
                       {"action": "right", "event_edit": "Low clouds roll over the ridge."}]},
        ]
        peak, stop = _peak_sampler(gpus)
        t0 = time.monotonic()
        try:
            out = _wait(q, caller, q.backends["rollout_alayaworld"].submit(
                q, caller, {"items": items, "variant": variant, "seed": 42})["job_id"])
        finally:
            stop.set()
        wall = time.monotonic() - t0
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in out["result"]["items"]}
        assert all("candidate" in by[i] for i in (0, 1)), by
        cands = [by[i]["candidate"] for i in (0, 1)]
        ann = _wait(q, caller, q.backends["annotate_camera"].submit(
            q, caller, {"items": [{"video": c["video"]} for c in cands]})["job_id"])
        assert ann["state"] == "done", ann.get("error")
    finally:
        q.shutdown()
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    rows, failures = [], []
    for i, c in enumerate(cands):
        with np.load(host(c["pose"])) as z:
            pub = _rel0(z["cam_c2w"])
        a = ann["result"]["items"][i]
        assert "error" not in a, a
        with np.load(host(a["pose"])) as z:
            est = _rel0(z["cam_c2w"])
        rot = float(np.median(_relative_rotation_errors(est, pub)))
        path = float(np.linalg.norm(np.diff(pub[:, :3, 3], axis=0), axis=1).sum())
        ate = _umeyama_ate(est[:, :3, 3], pub[:, :3, 3]) / path if path > 0 else None
        # final heading about the vertical, published vs ViGeo: the per-step rotation gate cannot
        # see a turn the video only half makes (0.75 deg/frame commanded is under its 1 deg bound)
        heading = lambda m: round(float(np.degrees(np.arctan2(m[-1, 0, 2], m[-1, 2, 2]))), 1)
        # copy: ingest moves the files into the archive
        stage = staging / "ingest" / str(i)
        stage.mkdir(parents=True)
        for role in ("video", "caption", "pose"):
            shutil.copy(host(c[role]), stage / Path(c[role]).name)
        [res] = ing.ingest([Candidate(video=stage / Path(c["video"]).name, caption=stage / Path(c["caption"]).name,
                                      pose=stage / Path(c["pose"]).name, camera_motion=c["camera_motion"],
                                      provenance=c["provenance"], license=c["license"])], node_id="gpu")
        caption = json.loads(host(c["caption"]).read_text())
        rows.append({"item": i, "frames": c["frames"], "trim": c["trim"], "rot_err_deg": round(rot, 4),
                     "ate_frac": ate and round(ate, 4), "path": round(path, 3),
                     "heading_deg_published_vs_vigeo": [heading(pub), heading(est)],
                     "segments": [[round(t * 24, 2) for t in s["time_range_s"]] for s in caption["segments"]],
                     "ingest_formats": res.formats, "generator": c["provenance"]["generator"]})
        if not (res.accepted and "video_timed_prompts_camera:per_chunk" in res.formats):
            failures.append(f"{i}: ingest {res.reasons}")
        if rot >= 1.0:
            failures.append(f"{i}: rotation error {rot:.3f} deg >= 1")
        if ate is not None and ate >= 0.10:
            failures.append(f"{i}: ATE {ate:.1%} of path >= 10%")
    print(json.dumps({"variant": variant, "gpus": gpus, "wall_s": round(wall, 1), "peak_mib": peak,
                      "gpu_memory_mib": out["result"]["gpu_memory_mib"], "rows": rows}, indent=1))
    assert out["result"]["gpu_memory_released"] is True
    assert not failures, failures
