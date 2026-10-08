"""rollout_alayaworld: case/config writers, submit checks, finish (CPU), the real produce with a
fake worker, plus one real AlayaWorld gpu smoke. Also rollout_wan22 (Wan2.2 TI2V-5B):
submit checks, finish (crop/probe, CPU), the real produce with a fake worker, plus one real Wan
gpu smoke. And rollout_ltx25 (LTX-2.5 distilled/dev): submit checks, the host-RAM worker rule,
finish (audio strip/probe, CPU), the real produce with a fake worker, plus one real gpu smoke
per enabled variant."""
from conftest import job_result
import copy
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.data.probe import probe_video
from ar_kernel.subproc import run_in_env
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import rollouts
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.rollouts import AlayaWorldBackend, Ltx25Backend, Wan22Backend, case_json, render_config
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def small_cfg(env="autoresearcher", enabled=True):
    raw = copy.deepcopy(REAL.raw)
    a = raw["generators"]["alayaworld"]
    a["env"] = env
    a["enabled"] = enabled
    a.pop("variants", None)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


def first_person(**over):
    return {"image": "frame.png", "viewpoint": "first_person",
            "scene_prompt": "A wet city street at night with neon signs.",
            "character_prompt": "", "viewpoint_prompt": "First-person view at eye level.",
            "turns": [{"action": "W"}, {"action": "left", "event": "It starts to rain heavily."}], **over}


def third_person(**over):
    return {"image": "frame.png", "viewpoint": "third_person", "subject_mask": "mask.png",
            "scene_prompt": "A mountain trail above a valley.",
            "character_prompt": "A hiker in a red jacket.",
            "viewpoint_prompt": "Third-person view from behind the hiker.",
            "turns": [{"action": "W+left", "subject_action": "The hiker raises a hand and waves."},
                      {"action": "stop", "viewpoint_change": "tp_to_fp"}], **over}


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
    c = render_config(REAL, rounds_per_turn=1, seed=7, indices=[0, 2], work=tmp_path,
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
    # everything else is the eval's own wbench_full.yaml
    source = yaml.safe_load((REAL.worldmodel / "configs" / "wbench_full.yaml").read_text())
    assert c["sample"] == source["sample"] and c["spatial_memory"] == source["spatial_memory"]
    assert mode["memory_start_round"] == source["validation"]["modes"]["wbench"]["memory_start_round"]


def test_render_config_uses_the_ar_teacher(tmp_path):
    c = render_config(REAL, rounds_per_turn=3, seed=1, indices=[0], work=tmp_path, text_cache=tmp_path / "te")
    assert c["paths"]["dmd_resume"] is None
    v = c["validation"]
    assert (v["sampling_steps"], v["scheduler"], v["cfg_scale"]) == (30, "shift", 3.0)


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
    (first_person(viewpoint="top_down"), {}, "viewpoint"),
    (first_person(scene_prompt=""), {}, "scene_prompt"),
    (first_person(image="frame.txt"), {}, "image"),
    (first_person(turns=[{"action": "W", "camera": "fly"}]), {}, "camera"),
    (first_person(), {"seed": "x"}, "seed"),
])
def test_submit_refuses(env, item, params, match):
    q, caller, _, _ = env
    with pytest.raises(ToolError, match=match):
        submit(q, caller, [item], **params)


def test_submit_accepts_every_action_and_fills_defaults(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AlayaWorldBackend(small_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    actions = ["W", "S", "A", "D", "left", "right", "up", "down", "stop", "W+left", "S+D", "W+down", "w"]
    args = {"items": [first_person(turns=[{"action": a}]) for a in actions]}
    b.check_args(args)
    assert (args["rounds_per_turn"], args["seed"]) == (3, 42) and "variant" not in args


def test_limits_come_from_the_generator_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = AlayaWorldBackend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    a = REAL.get("generators.alayaworld")
    assert (b.timeout_s, b.max_turns) == (a["timeout_s"], a["max_turns"])


# ---- finish: trim to the per_chunk grid, poses, segments ----

def _render_output(out, i, *, rounds, cpt, prompts, first=-7):
    """What produce leaves in out/ for one case. Measured layout: round r = mp4 frames
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
    return type("Job", (), {"id": "j1", "node": "n1", "args": {"seed": 42, **args}})()


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
    assert "pose" not in res and "camera_motion" not in res      # commanded path is metadata only
    with np.load(res["commanded_camera"]) as z:
        c2w = z["cam_c2w"]
    assert len(c2w) == frames - trim and np.allclose(c2w[0], np.eye(4))
    assert np.allclose(c2w[:, 0, 3], np.arange(frames - trim) * 0.02, atol=1e-5)   # saved[trim:], re-based
    cap = json.loads(Path(res["caption"]).read_text())
    assert cap["caption"] == prompts[0]
    assert [s["prompt"] for s in cap["segments"]] == prompts                     # one per turn, merged
    _check_boundaries(cap["segments"], frames - trim)
    assert res["actions"] == ["W"] * len(prompts)
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
    # the agent's next step: annotate_camera gives a ViGeo-shaped pose, one per published frame
    from tests.conftest import write_poses
    pose = write_poses(out / "vigeo.npz", n_frames=res["frames"], width=960, height=544)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), caption=Path(res["caption"]), pose=pose,
                                camera_motion="moving", provenance={"kind": "rollout", "generator": "alayaworld-ar30",
                                "job_id": "j1", "inputs_hash": "x", "seed": 42})], node_id="n1")
    assert r.accepted, r.reasons
    assert "video_timed_prompts_camera:per_chunk" in r.formats


# ---- the real produce/run_workers/_collect with the fake worker ----

def run_job(q, caller, items, **params):
    out = q.wait(caller, submit(q, caller, items, **params)["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in job_result(out)["items"]}


def test_produce_renders_and_publishes_candidates(env):
    q, caller, staging, run_dir = env
    job, by = run_job(q, caller, [first_person(), third_person()], seed=5)
    for i, item in by.items():
        c = item["candidate"]
        assert c["video"] == f"/workspace/staging/rollouts/{job}/{i}.mp4"
        assert c["caption"] == f"/workspace/staging/rollouts/{job}/{i}.json"
        assert "pose" not in c and "camera_motion" not in c
        assert c["commanded_camera"] == f"/workspace/staging/rollouts/{job}/{i}.commanded_camera.npz"
        assert c["license"] == REAL.get("generators.alayaworld.license")
        assert c["provenance"]["generator"] == "alayaworld-ar30" and c["provenance"]["seed"] == 5
        assert c["frames"] == 2 * 3 * 32 - 7
        with np.load(staging / "rollouts" / job / f"{i}.commanded_camera.npz") as z:
            assert len(z["cam_c2w"]) == c["frames"]
        assert not (staging / "rollouts" / job / f"{i}.npz").exists()
        cap = json.loads((staging / "rollouts" / job / f"{i}.json").read_text())
        assert len(cap["segments"]) == 2
    assert by[0]["candidate"]["actions"] == ["W", "left"]
    assert by[1]["candidate"]["actions"] == ["W+left", "stop"]
    assert len(by[1]["candidate"]["turn_segments"]) == 2
    assert q.status(caller, job)["progress"] == {"done": 2, "total": 2}
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


def test_precache_is_charged_to_the_job_timeout(env):
    """A hung prompt precache is killed at the job's timeout_s, not left running forever."""
    import time
    q, caller, _, run_dir = env
    q.backends["rollout_alayaworld"].timeout_s = 2
    t0 = time.monotonic()
    job = submit(q, caller, [first_person(scene_prompt="PRECACHE_HANG street")])["job_id"]
    out = q.wait(caller, job, 60)
    assert out["state"] == "failed" and "timed out" in out["error"], out
    assert time.monotonic() - t0 < 30
    assert not (run_dir / "jobs" / job / "fake_wbench.json").exists()        # never reached the render


def test_rounds_per_turn_1_gives_a_segment_per_round(env):
    q, caller, staging, _ = env
    job, by = run_job(q, caller, [third_person()], rounds_per_turn=1)
    assert by[0]["candidate"]["frames"] == 2 * 32 - 7
    cap = json.loads((staging / "rollouts" / job / "0.json").read_text())
    assert len(cap["segments"]) == 2 and cap["segments"][1]["time_range_s"][0] * 24 == pytest.approx(25)


def test_a_case_without_video_and_a_bad_image_fail_alone(env):
    q, caller, staging, _ = env
    (staging.parent / "ws" / "broken.png").write_bytes(b"not a png")
    _, by = run_job(q, caller, [first_person(scene_prompt="NO_VIDEO here"), first_person(image="broken.png"),
                                first_person()])
    assert by[0]["error"] == "no video rendered"
    assert by[1]["error"].startswith("input:")
    assert "Error" in by[1]["error"].split(":")[1]     # the exception type is named
    assert "candidate" in by[2]


def test_a_non_oserror_from_pil_fails_only_its_own_item(env, monkeypatch):
    """PIL's decompression-bomb guard raises DecompressionBombError -- a plain Exception, not an
    OSError -- on an oversize image. produce() must catch that too and fail only that item,
    rather than letting it escape and fail the whole job (the other items still render)."""
    q, caller, staging, _ = env
    Image.new("RGB", (100, 100), (1, 2, 3)).save(staging.parent / "ws" / "tiny.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20000)   # tiny.png (10k px) stays under 2x this;
    _, by = run_job(q, caller, [first_person(image="tiny.png"), first_person(), first_person(image="tiny.png")])
    assert "candidate" in by[0] and "candidate" in by[2]
    assert by[1]["error"].startswith("input:") and "Error" in by[1]["error"].split(":")[1]     # the exception type is named
    assert "decompression bomb" in by[1]["error"].lower()


def test_a_job_renders_with_the_ar_teacher(env):
    q, caller, _, run_dir = env
    job, by = run_job(q, caller, [first_person(turns=[{"action": "W"}])])
    assert by[0]["candidate"]["provenance"]["generator"] == "alayaworld-ar30"
    cfg = yaml.safe_load((run_dir / "jobs" / job / "render_config.yaml").read_text())
    assert cfg["validation"]["sampling_steps"] == 30 and cfg["paths"]["dmd_resume"] is None


# ---- a node's fine-tune ----

def test_render_config_with_a_node_adds_its_lora_and_history_encoder(tmp_path):
    kw = dict(rounds_per_turn=3, seed=1, indices=[0], work=tmp_path, text_cache=tmp_path / "te")
    node = dict(node_lora=tmp_path / "lora", node_rank=32, history_encoder=tmp_path / "ckpt" / "history_encoder.pt")
    plain, tuned = (_flat(render_config(REAL, **kw, **extra)) for extra in ({}, node))
    assert {k: tuned[k] for k in plain if plain[k] != tuned[k]} == {
        ("paths", "dmd_resume"): str(tmp_path / "lora"),
        ("paths", "history_encoder"): str(tmp_path / "ckpt" / "history_encoder.pt"),
        ("lora", "rank"): 32, ("lora", "alpha"): 32}


def _scored_node(run_dir, node_id="n2", rank=32):
    """A scored node with a checkpoint in the run's archive; returns the checkpoint dir."""
    checkpoint = run_dir / "nodes" / node_id / "checkpoint-300"
    checkpoint.mkdir(parents=True)
    conn = open_db(run_dir)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.record_score("root", 0.5, [], {})
    nodes.create("failed", "root", 1)
    nodes.create(node_id, "root", 1)
    nodes.set_fields(node_id, checkpoint_path=f"nodes/{node_id}/checkpoint-300", lora_rank=rank)
    nodes.record_score(node_id, 0.6, [], {})
    conn.close()
    return checkpoint


@pytest.mark.parametrize("node", ["n9", "root", "failed"])
def test_submit_refuses_a_node_without_a_fine_tune(env, node):
    q, caller, _, run_dir = env
    _scored_node(run_dir)
    with pytest.raises(ToolError, match="node"):
        submit(q, caller, [first_person()], node=node)


def test_a_node_job_renders_with_the_nodes_fine_tune(env):
    q, caller, _, run_dir = env
    checkpoint = _scored_node(run_dir)
    job, by = run_job(q, caller, [first_person(turns=[{"action": "W"}])], node="n2")
    assert by[0]["candidate"]["provenance"]["generator"] == "alayaworld-ar30@n2"
    cfg = yaml.safe_load((run_dir / "jobs" / job / "render_config.yaml").read_text())
    assert cfg["paths"]["dmd_resume"] == str(checkpoint) and cfg["lora"]["rank"] == 32
    assert cfg["paths"]["history_encoder"] == str(checkpoint / "history_encoder.pt")


def test_build_gpu_backends_includes_alayaworld_only_when_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((False, False), (True, True)):
        names = [b.name for b in build_gpu_backends(small_cfg(enabled=enabled), tmp_path / "run", [0, 1, 2, 3],
                                                    TokenRegistry(rec), rec)]
        assert ("rollout_alayaworld" in names) is present


# ---- real AlayaWorld gpu smoke ----

HIKER = ("Photorealistic wide shot, a hiker in a bright red jacket and grey backpack walking away from the camera "
         "along a narrow dirt trail on a green mountain ridge, seen from behind, full body in the center of the "
         "frame, valley and distant peaks under a clear afternoon sky, natural light")


def _hiker_mask(path):
    """Hand-drawn around the hiker of generate_images(HIKER, seed 3) at 1280x720."""
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
def test_real_alayaworld_rollout(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_rollouts.py -m gpu -s --basetemp=.cache/pytest/gpu

    First person: frame 0 of an example clip. Third person: a generate_images (Z-Image) frame with a
    hand-drawn mask (the tools chain). The candidates carry no pose; each must ingest as per_chunk
    with the pose annotate_camera (ViGeo) gives it. Agreement with the commanded camera path is
    printed as a diagnostic only."""
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
    raw["generators"]["alayaworld"]["enabled"] = True
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
        hiker = job_result(img)["items"][0]["image"]
        items = [
            {"image": "street.png", "viewpoint": "first_person",
             "scene_prompt": "A rainy city street lined with parked cars and trees, wet asphalt reflecting "
                                   "the grey daylight, apartment blocks in the distance.",
             "viewpoint_prompt": "First-person view at eye level from the sidewalk.",
             "turns": [{"action": "W", "event": "A red umbrella blows across the street in the wind."},
                       {"action": "left", "subject_action": "A cyclist in a yellow raincoat rides past."}]},
            {"image": hiker, "subject_mask": "hiker_mask.png", "viewpoint": "third_person",
             "scene_prompt": "A narrow dirt trail along a green mountain ridge above a wide valley, "
                                   "distant peaks under a clear afternoon sky.",
             "character_prompt": "A hiker in a bright red jacket and grey backpack.",
             "viewpoint_prompt": "Third-person view from behind the hiker.",
             "turns": [{"action": "W", "subject_action": "The hiker raises the right arm and waves."},
                       {"action": "right", "event": "Low clouds roll over the ridge."}]},
        ]
        peak, stop = _peak_sampler(gpus)
        t0 = time.monotonic()
        try:
            out = _wait(q, caller, q.backends["rollout_alayaworld"].submit(
                q, caller, {"items": items, "seed": 42})["job_id"])
        finally:
            stop.set()
        wall = time.monotonic() - t0
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
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
        assert "pose" not in c and "camera_motion" not in c, c
        a = job_result(ann)["items"][i]
        assert "error" not in a, a
        with np.load(host(a["pose"])) as z:
            vigeo = z["cam_c2w"]
        with np.load(host(c["commanded_camera"])) as z:
            commanded = z["cam_c2w"]
        if len(vigeo) != c["frames"]:
            failures.append(f"{i}: ViGeo pose has {len(vigeo)} frames, the clip {c['frames']}")
        if len(commanded) != c["frames"]:
            failures.append(f"{i}: commanded camera has {len(commanded)} frames, the clip {c['frames']}")
        # Diagnostics only (user decision 2026-09-26): how far the video strays from the commanded path.
        est, cmd = _rel0(vigeo), _rel0(commanded)
        rot = float(np.median(_relative_rotation_errors(est, cmd)))
        path = float(np.linalg.norm(np.diff(cmd[:, :3, 3], axis=0), axis=1).sum())
        ate = _umeyama_ate(est[:, :3, 3], cmd[:, :3, 3]) / path if path > 0 else None
        heading = lambda m: round(float(np.degrees(np.arctan2(m[-1, 0, 2], m[-1, 2, 2]))), 1)
        # the agent's route: ViGeo's pose + camera_motion 'moving' (copies: ingest moves files)
        stage = staging / "ingest" / str(i)
        stage.mkdir(parents=True)
        shutil.copy(host(c["video"]), stage / "v.mp4")
        shutil.copy(host(c["caption"]), stage / "c.json")
        shutil.copy(host(a["pose"]), stage / "p.npz")
        [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=stage / "p.npz",
                                      camera_motion="moving", provenance=c["provenance"], license=c["license"])],
                           node_id="gpu")
        caption = json.loads(host(c["caption"]).read_text())
        rows.append({"item": i, "frames": c["frames"], "trim": c["trim"],
                     "diag_vs_commanded": {"rot_err_deg": round(rot, 4), "ate_frac": ate and round(ate, 4),
                                           "heading_deg_commanded_vs_vigeo": [heading(cmd), heading(est)]},
                     "segments": [[round(t * 24, 2) for t in s["time_range_s"]] for s in caption["segments"]],
                     "ingest_formats": res.formats, "ingest_reasons": res.reasons,
                     "generator": c["provenance"]["generator"]})
        if not (res.accepted and "video_timed_prompts_camera:per_chunk" in res.formats):
            failures.append(f"{i}: ingest {res.reasons}")
    print(json.dumps({"gpus": gpus, "wall_s": round(wall, 1), "peak_mib": peak,
                      "gpu_memory_mib": job_result(out)["gpu_memory_mib"], "rows": rows}, indent=1))
    assert job_result(out)["gpu_memory_released"] is True
    assert not failures, failures


@pytest.mark.gpu
def test_real_alayaworld_rollout_of_a_node(tmp_path):
    """AR_TEST_CHECKPOINT=<a trained node's checkpoint dir> AR_TEST_RANK=<its LoRA rank> AR_TEST_GPUS=0,1,2,3
    pytest tests/test_rollouts.py -m gpu -s -k of_a_node --basetemp=.cache/pytest/gpu

    The same case and seed rendered by the released model and by the node: both publish a candidate,
    and the node's clip differs from the released model's (its fine-tune is applied)."""
    import os
    import subprocess
    from ar_kernel.config import resolve_gpus

    if not os.environ.get("AR_TEST_CHECKPOINT"):
        pytest.skip("set AR_TEST_CHECKPOINT and AR_TEST_RANK")
    gpus = resolve_gpus(REAL, {"CUDA_VISIBLE_DEVICES": os.environ.get("AR_TEST_GPUS", "0,1,2,3")})
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["alayaworld"]["enabled"] = True
    cfg = KernelConfig(raw=raw, repo_root=REAL.repo_root)
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    _scored_node(run_dir, rank=int(os.environ["AR_TEST_RANK"])).rmdir()
    (run_dir / "nodes" / "n2" / "checkpoint-300").symlink_to(Path(os.environ["AR_TEST_CHECKPOINT"]).resolve())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                    str(REAL.worldmodel / "data/examples/video_caption_camera/videos/clip_0001.mp4"),
                    "-frames:v", "1", str(ws / "street.png")], check=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    q.register(AlayaWorldBackend(cfg, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    item = {"image": "street.png", "viewpoint": "first_person",
            "scene_prompt": "A rainy city street lined with parked cars and trees.",
            "turns": [{"action": "W"}]}
    frames = {}
    try:
        for node in (None, "n2"):
            args = {"items": [dict(item)], "seed": 42, "rounds_per_turn": 1}
            out = _wait(q, caller, q.backends["rollout_alayaworld"].submit(
                q, caller, args if node is None else {**args, "node": node})["job_id"])
            assert out["state"] == "done", out.get("error")
            [got] = job_result(out)["items"]
            assert "candidate" in got, got
            assert got["candidate"]["provenance"]["generator"] == "alayaworld-ar30" + (f"@{node}" if node else "")
            assert job_result(out)["gpu_memory_released"] is True
            assert not (run_dir / "jobs" / out["id"] / "eval").exists()
            video = staging / Path(got["candidate"]["video"]).relative_to("/workspace/staging")
            frames[node] = np.frombuffer(subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", "scale=160:90", "-f", "rawvideo",
                 "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout, np.uint8).astype(np.float32)
    finally:
        q.shutdown()
    assert frames[None].shape == frames["n2"].shape
    diff = float(np.abs(frames[None] - frames["n2"]).mean())
    print(json.dumps({"mean_abs_diff": round(diff, 3)}))
    assert diff > 0.5

# ---- Wan22Backend (rollout_wan22) ----

def small_wan_cfg(env="autoresearcher", enabled=True, **over):
    raw = copy.deepcopy(REAL.raw)
    w = raw["generators"]["wan22"]
    w["env"] = env
    w["enabled"] = enabled
    w.update(over)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


@pytest.fixture
def wan_env(tmp_path, monkeypatch):
    """The real Wan22Backend.produce/run_workers, with wan22_generate.py swapped for the fake
    worker (wan mode: it writes a synthetic 1280x704, --frames-frame mp4 and echoes the argv)."""
    monkeypatch.setattr(rollouts, "WAN22_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(ws / "frame.png")
    q.register(Wan22Backend(small_wan_cfg(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                            gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def run_wan(q, caller, items, **params):
    out = q.wait(caller, q.backends["rollout_wan22"].submit(q, caller, {"items": items, **params})["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in job_result(out)["items"]}


@pytest.mark.parametrize("frames, match", [
    (120, r"4k\+1"), (1, "1 < frames"), (2, r"4k\+1"), (10000, "1 < frames"), ("x", "1 < frames")])
def test_wan_bad_frames_are_refused_at_submit(wan_env, frames, match):
    q, caller, _ = wan_env
    with pytest.raises(ToolError, match=match):
        q.backends["rollout_wan22"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}], "frames": frames})


def test_wan_missing_prompt_is_refused_at_submit(wan_env):
    q, caller, _ = wan_env
    with pytest.raises(ToolError, match="prompt"):
        q.backends["rollout_wan22"].submit(q, caller, {"items": [{"seed": 1}]})


def test_wan_missing_seed_is_refused_at_submit(wan_env):
    q, caller, _ = wan_env
    with pytest.raises(ToolError, match="seed"):
        q.backends["rollout_wan22"].submit(q, caller, {"items": [{"prompt": "p"}]})


def test_wan_image_must_be_a_path(wan_env):
    q, caller, _ = wan_env
    with pytest.raises(ToolError, match="image"):
        q.backends["rollout_wan22"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1, "image": 3}]})


def test_wan_limits_come_from_the_generator_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = Wan22Backend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    w = REAL.get("generators.wan22")
    assert b.timeout_s == w["timeout_s"]


def test_wan_produces_and_publishes_a_cropped_candidate_with_no_pose_or_camera_motion(wan_env):
    q, caller, staging = wan_env
    prompts = ["a walk in the woods", "a hiker on a ridge"]
    job, by = run_wan(q, caller, [{"prompt": prompts[0], "seed": 1},
                                  {"prompt": prompts[1], "image": "frame.png", "seed": 2}])
    for i, item in by.items():
        c = item["candidate"]
        assert c["video"] == f"/workspace/staging/rollouts/{job}/{i}.mp4"
        assert c["caption"] == f"/workspace/staging/rollouts/{job}/{i}.json"
        assert "pose" not in c and "camera_motion" not in c
        assert c["provenance"]["generator"] == "wan2.2-ti2v-5b" and c["provenance"]["seed"] == i + 1
        assert c["license"] == REAL.get("generators.wan22.license")
        info = probe_video(staging / "rollouts" / job / f"{i}.mp4")
        assert (info.width, info.height) == (1248, 704) and round(info.fps) == 24
        cap = json.loads((staging / "rollouts" / job / f"{i}.json").read_text())
        assert cap == {"caption": prompts[i]}
    assert by[1]["worker"]["image"].endswith("_image.png")     # the staged copy, not the workspace path
    assert by[0]["worker"]["image"] is None


def test_wan_frames_default_and_max_come_from_config(wan_env):
    q, caller, staging = wan_env
    default, maximum = REAL.get("generators.wan22.frames")
    job, by = run_wan(q, caller, [{"prompt": "p", "seed": 1}])
    assert by[0]["worker"]["frames"] == default
    job, by = run_wan(q, caller, [{"prompt": "p", "seed": 1}], frames=maximum)
    assert by[0]["worker"]["frames"] == maximum


def test_wan_workers_get_one_gpu_each_from_the_configured_repo_and_weights(wan_env):
    q, caller, staging = wan_env
    _, by = run_wan(q, caller, [{"prompt": "p", "seed": i} for i in range(5)])
    for i, item in by.items():
        assert item["worker"]["gpus"] == str([0, 1, 4, 5][i % 4]) and item["worker"]["rank"] == i % 4
        assert item["worker"]["repo"] == str(REAL.repo_root / REAL.get("generators.wan22.repo"))
        assert item["worker"]["ckpt_dir"] == str(REAL.repo_root / REAL.get("generators.wan22.weights"))
        assert item["worker"]["offload_model"] is True and item["worker"]["t5_cpu"] is True


def test_wan_extra_args_can_disable_offload_and_t5_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(rollouts, "WAN22_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    cfg = small_wan_cfg(extra_args={"offload_model": False, "t5_cpu": False})
    q.register(Wan22Backend(cfg, tmp_path / "run", [0, 1, 2, 3], reg, rec, gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    try:
        _, by = run_wan(q, caller, [{"prompt": "p", "seed": 1}])
    finally:
        q.shutdown()
    assert by[0]["worker"]["offload_model"] is False and by[0]["worker"]["t5_cpu"] is False


def test_wan_build_gpu_backends_includes_it_only_when_a_variant_is_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((True, True), (False, False)):
        names = [b.name for b in build_gpu_backends(small_wan_cfg(enabled=enabled), tmp_path / "run",
                                                     [0, 1, 2, 3], TokenRegistry(rec), rec)]
        assert ("rollout_wan22" in names) is present


@pytest.mark.parametrize("pin", ["0" * 40, None])
def test_wan_refuses_to_run_off_the_pinned_commit(tmp_path, monkeypatch, pin):
    """The backend's own git check, not the worker's: produce() must raise before any worker
    launches, so a silently-updated clone (or a config with no pin) fails the whole job."""
    monkeypatch.setattr(rollouts, "WAN22_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=60)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    cfg = small_wan_cfg(repo=".", commit=pin)     # repo_root itself is a real git repo
    q.register(Wan22Backend(cfg, tmp_path / "run", [0, 1, 2, 3], reg, rec, gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    try:
        job_id = q.backends["rollout_wan22"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}]})["job_id"]
        out = q.wait(caller, job_id, 60)
    finally:
        q.shutdown()
    assert out["state"] == "failed" and "commit" in out["error"] and str(pin) in out["error"]


def test_wan_missing_clone_fails_the_job_with_a_clear_message(tmp_path, monkeypatch):
    """An uncloned third_party/Wan2.2 must say which path and where the setup is documented, not
    surface a bare CalledProcessError."""
    monkeypatch.setattr(rollouts, "WAN22_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=60)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    missing = tmp_path / "no_such_clone"
    cfg = small_wan_cfg(repo=str(missing))
    q.register(Wan22Backend(cfg, tmp_path / "run", [0, 1, 2, 3], reg, rec, gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    try:
        job_id = q.backends["rollout_wan22"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}]})["job_id"]
        out = q.wait(caller, job_id, 60)
    finally:
        q.shutdown()
    assert out["state"] == "failed"
    assert str(missing) in out["error"] and "into that folder" in out["error"]
    assert "CalledProcessError" not in out["error"]


def _wan_backend(tmp_path, **over):
    rec = Recorder(tmp_path / "run")
    return Wan22Backend(small_wan_cfg(**over), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)


def test_finish_refuses_a_non_24fps_render(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=30, width=1280, height=704)
    with pytest.raises(ValueError, match="fps"):
        _wan_backend(tmp_path).finish(None, {"index": 0, "prompt": "p"}, out)


def test_finish_refuses_a_render_off_the_1280x704_working_size(tmp_path):
    """A far-from-16:9 'image' makes Wan render at a different resolution (measured: WanTI2V.i2v
    keeps the image's own aspect, not 1280x704) -- finish() must turn that into a clear item error,
    not a broken or silently-wrong crop."""
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=800, height=1088)
    with pytest.raises(ValueError, match="1280x704"):
        _wan_backend(tmp_path).finish(None, {"index": 0, "prompt": "p"}, out)


def test_finish_crops_to_1248x704_and_writes_the_prompt_caption(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=1280, height=704)
    res = _wan_backend(tmp_path).finish(None, {"index": 0, "prompt": "hello there"}, out)
    info = probe_video(res["video"])
    assert (info.width, info.height) == (1248, 704) and round(info.fps) == 24
    assert res["frames"] == info.frames
    assert json.loads(Path(res["caption"]).read_text()) == {"caption": "hello there"}


# ---- real Wan 2.2 TI2V-5B gpu smoke ----

@pytest.mark.gpu
def test_real_wan22_rollout(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_rollouts.py -m gpu -k wan22 -s --basetemp=.cache/pytest/gpu

    2 T2V + 2 I2V items, one worker per GPU in AR_TEST_GPUS (frames = the configured default). The I2V
    item's image is a generate_images (Z-Image) frame -- the tools chain. Each candidate then
    ingests as video_caption_static (camera_motion: static), and one of them, after
    annotate_camera, ingests as video_caption_camera (camera_motion: moving)."""
    import os
    import shutil
    import time
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from ar_kernel.tools.annotate import AnnotateBackend
    from ar_kernel.tools.images import ImageBackend

    gpus = [int(g) for g in os.environ.get("AR_TEST_GPUS", "0,1,2,3").split(",")]   # one worker per GPU
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["wan22"]["enabled"] = True
    cfg = KernelConfig(raw=raw, repo_root=REAL.repo_root)
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    for b in (ImageBackend, Wan22Backend, AnnotateBackend):
        q.register(b(cfg, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    host = lambda p: staging / Path(p).relative_to("/workspace/staging")
    try:
        img = _wait(q, caller, q.backends["generate_images"].submit(
            q, caller, {"items": [{"prompt": "a red barn in an open field, photorealistic", "seed": 9},
                                  {"prompt": "a cup of coffee on a wooden table, photorealistic", "seed": 10}]})["job_id"])
        assert img["state"] == "done", img.get("error")
        frames = [i["image"] for i in job_result(img)["items"]]
        items = [
            {"prompt": "A slow walk through a sunlit forest path, camera steady.", "seed": 1},
            {"prompt": "Ocean waves gently rolling onto a quiet beach at sunset.", "seed": 2},
            {"prompt": "The red barn under a slowly moving cloud, camera steady.", "image": frames[0], "seed": 3},
            {"prompt": "A cup of coffee steaming on a wooden table, camera steady.", "image": frames[1], "seed": 4},
        ]
        t0 = time.monotonic()
        out = _wait(q, caller, q.backends["rollout_wan22"].submit(q, caller, {"items": items})["job_id"])
        wall = time.monotonic() - t0
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
        print(json.dumps({"wan_wall_s": round(wall, 1), "items": by}, indent=1))
        assert all("candidate" in by[i] for i in range(4)), by
        cands = [by[i]["candidate"] for i in range(4)]
        ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
        static_rows, failures = [], []
        for i, c in enumerate(cands):
            stage = staging / "ingest" / f"static{i}"
            stage.mkdir(parents=True)
            shutil.copy(host(c["video"]), stage / "v.mp4")
            shutil.copy(host(c["caption"]), stage / "c.json")
            [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=None,
                                          camera_motion="static", provenance=c["provenance"],
                                          license=c["license"])], node_id="gpu")
            static_rows.append({"item": i, "frames": c["frames"], "worker": by[i]["worker"],
                                "accepted": res.accepted, "formats": res.formats, "reasons": res.reasons})
            if not (res.accepted and "video_caption_static" in res.formats):
                failures.append(f"static {i}: {res.reasons}")
        # one candidate through annotate_camera -> moving -> video_caption_camera
        ann = _wait(q, caller, q.backends["annotate_camera"].submit(
            q, caller, {"items": [{"video": cands[0]["video"]}]})["job_id"])
        assert ann["state"] == "done", ann.get("error")
        a = job_result(ann)["items"][0]
        assert "error" not in a, a
        stage = staging / "ingest" / "moving0"
        stage.mkdir(parents=True)
        shutil.copy(host(cands[0]["video"]), stage / "v.mp4")
        shutil.copy(host(cands[0]["caption"]), stage / "c.json")
        shutil.copy(host(a["pose"]), stage / "p.npz")
        [mres] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=stage / "p.npz",
                                       camera_motion="moving", provenance=cands[0]["provenance"],
                                       license=cands[0]["license"])], node_id="gpu")
        if not (mres.accepted and "video_caption_camera" in mres.formats):
            failures.append(f"moving 0: {mres.reasons}")
        print(json.dumps({"gpus": gpus, "wall_s": round(wall, 1), "gpu_memory_mib": job_result(out)["gpu_memory_mib"],
                          "static_rows": static_rows, "moving_formats": mres.formats}, indent=1))
    finally:
        q.shutdown()
    assert job_result(out)["gpu_memory_released"] is True
    assert not failures, failures


@pytest.mark.parametrize("size", [(1280, 720), (800, 1088), (640, 360), (1280, 704)])
def test_wan_bridge_fits_any_first_frame_to_1280x704(size):
    """WanTI2V.i2v keeps the input image's own aspect (best_output_size): a 1280x720 frame would
    render at 1248x704 and a portrait one at 800x1088. The bridge center-crops/resizes every first
    frame to exactly 1280x704 first, so Wan's own resize is the identity and every clip is 1280x704."""
    from ar_kernel.bridges.wan22_generate import fit_first_frame
    img = Image.new("RGB", size, (200, 10, 10))
    img.paste((10, 200, 10), (size[0] // 2 - 4, size[1] // 2 - 4, size[0] // 2 + 4, size[1] // 2 + 4))
    fitted = fit_first_frame(img)
    assert fitted.size == (1280, 704)
    assert fitted.getpixel((640, 352))[1] > 150       # centered: the center marker stays at the center


# ---- Ltx25Backend (rollout_ltx25) ----

def _head():
    import subprocess
    return subprocess.run(["git", "-C", str(REAL.repo_root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def small_ltx_cfg(env="autoresearcher", enabled=("distilled", "dev"), **over):
    """The real ltx25 block with the given variants enabled, pinned to
    repo_root's own HEAD (a real git repo, so the CPU tests need no LTX-2 clone), with the RAM
    tests' host_reserve_gib of 60."""
    raw = copy.deepcopy(REAL.raw)
    b = raw["generators"]["ltx25"]
    b.update(env=env, repo=".", commit=_head(), host_reserve_gib=60)
    b["variants"] = list(enabled)
    b.update(over)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


GIB = {"MemTotal": 251.0, "MemAvailable": 240.0}


@pytest.fixture
def ltx_env(tmp_path, monkeypatch):
    """The real Ltx25Backend.produce/run_workers, with ltx25_generate.py swapped for the fake
    worker (ltx mode) and /proc/meminfo for a 251 GiB host with 240 GiB available."""
    monkeypatch.setattr(rollouts, "LTX25_BRIDGE", FAKE)
    monkeypatch.setattr(rollouts, "meminfo_gib", lambda: dict(GIB))
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(ws / "frame.png")

    def make(gpus=(0, 1, 4, 5), **over):
        q.register(Ltx25Backend(small_ltx_cfg(**over), tmp_path / "run", list(gpus), reg, rec,
                                gpu_memory=lambda g: {i: 100 for i in g}))
        return q
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield make, caller, staging
    q.shutdown()


def run_ltx(q, caller, items, **params):
    out = q.wait(caller, q.backends["rollout_ltx25"].submit(q, caller, {"items": items, **params})["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in job_result(out)["items"]}


@pytest.mark.parametrize("params, match", [
    ({"frames": 120}, r"8k\+1"), ({"frames": 1}, "1 < frames"), ({"frames": 125}, r"8k\+1"),
    ({"frames": 100001}, "1 < frames"), ({"frames": "x"}, "1 < frames"),
    ({"height": 720, "width": 1280}, "resolutions"), ({"width": 960}, "resolutions"),
    ({"height": 576.0}, "ints"), ({"width": True}, "ints"), ({"variant": "pro"}, "variant")])
def test_ltx_bad_params_are_refused_at_submit(ltx_env, params, match):
    make, caller, _ = ltx_env
    q = make()
    with pytest.raises(ToolError, match=match):
        q.backends["rollout_ltx25"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}], **params})


@pytest.mark.parametrize("item, match", [
    ({"seed": 1}, "prompt"), ({"prompt": " ", "seed": 1}, "prompt"), ({"prompt": "p"}, "seed"),
    ({"prompt": "p", "seed": True}, "seed"),
    ({"prompt": "p", "seed": 1, "keyframes": [{"image": 3, "frame": 0}]}, "each keyframe"),
    ({"prompt": "p", "seed": 1, "image": "frame.png"}, "image")])
def test_ltx_bad_items_are_refused_at_submit(ltx_env, item, match):
    make, caller, _ = ltx_env
    q = make()
    with pytest.raises(ToolError, match=match):
        q.backends["rollout_ltx25"].submit(q, caller, {"items": [item]})


def test_ltx_a_disabled_variant_is_refused(ltx_env):
    make, caller, _ = ltx_env
    q = make(enabled=("distilled",))
    with pytest.raises(ToolError, match="variant"):
        q.backends["rollout_ltx25"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}], "variant": "dev"})


def test_ltx_limits_come_from_the_generator_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = Ltx25Backend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    blk = REAL.get("generators.ltx25")
    assert b.timeout_s == blk["timeout_s"]


def test_ltx_produces_and_publishes_a_silent_24fps_candidate_with_no_pose_or_camera_motion(ltx_env):
    make, caller, staging = ltx_env
    q = make()
    prompts = ["a walk in the woods", "a hiker on a ridge"]
    job, by = run_ltx(q, caller, [{"prompt": prompts[0], "seed": 1}, {"prompt": prompts[1], "seed": 2, "keyframes": [
        {"image": "frame.png", "frame": 0}, {"image": "frame.png", "frame": -1}]}])
    (h, w), default = REAL.get("generators.ltx25.resolutions")[0], REAL.get("generators.ltx25.frames")[0]
    for i, item in by.items():
        c = item["candidate"]
        assert c["video"] == f"/workspace/staging/rollouts/{job}/{i}.mp4"
        assert c["caption"] == f"/workspace/staging/rollouts/{job}/{i}.json"
        assert "pose" not in c and "camera_motion" not in c
        assert c["provenance"]["generator"] == "ltx-2.5-distilled" and c["provenance"]["seed"] == i + 1
        assert c["license"] == "LTX-2 Community License"
        path = staging / "rollouts" / job / f"{i}.mp4"
        info = probe_video(path)
        assert (info.width, info.height, info.frames) == (w, h, default) and round(info.fps) == 24
        import subprocess
        streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                                  str(path)], capture_output=True, text=True, check=True).stdout.split()
        assert streams == ["video"]                      # the fake's audio track is stripped
        assert json.loads((staging / "rollouts" / job / f"{i}.json").read_text()) == {"caption": prompts[i]}
        wk = item["worker"]
        assert wk["weights"] == str(REAL.repo_root / REAL.get("generators.ltx25.weights"))
        assert (wk["variant"], wk["quantization"], wk["offload"]) == ("distilled", "fp8-cast", "cpu")
    frames1 = by[1]["worker"]["keyframes"]
    assert [k["frame"] for k in frames1] == [0, -1]
    assert all("_keyframe" in k["image"] for k in frames1) and by[0]["worker"]["keyframes"] == []


@pytest.mark.parametrize("frame, ok", [(0, True), (48, True), (-1, True), (49, False), (1000, False)])
def test_ltx_keyframe_frames_must_be_inside_the_clip(ltx_env, frame, ok):
    make, caller, _ = ltx_env
    q = make()
    args = {"items": [{"prompt": "p", "seed": 1, "keyframes": [{"image": "frame.png", "frame": frame}]}], "frames": 49}
    if ok:
        assert q.backends["rollout_ltx25"].submit(q, caller, args)["job_id"]
    else:
        with pytest.raises(ToolError, match="outside the clip"):
            q.backends["rollout_ltx25"].submit(q, caller, args)


def test_ltx_last_frame_given_twice_is_a_repeat(ltx_env):
    make, caller, _ = ltx_env
    q = make()
    frames = [{"image": "frame.png", "frame": -1}, {"image": "frame.png", "frame": 48}]
    with pytest.raises(ToolError, match="repeat"):
        q.backends["rollout_ltx25"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1, "keyframes": frames}],
                                                       "frames": 49})


def test_ltx_dev_job_is_named_after_its_variant(ltx_env):
    make, caller, _ = ltx_env
    q = make()
    _, by = run_ltx(q, caller, [{"prompt": "p", "seed": 1}], variant="dev", frames=9)
    assert by[0]["candidate"]["provenance"]["generator"] == "ltx-2.5-dev"
    assert by[0]["worker"]["variant"] == "dev" and by[0]["worker"]["frames"] == 9


@pytest.mark.parametrize("gpus, rss, cap, workers", [
    ((0, 1, 4, 5), 60, None, 3),        # floor((251 - 60) / 60) = 3 < 4 GPUs
    ((0, 1, 4, 5), 40, None, 4),        # RAM allows 4: one per GPU
    ((0, 1, 4, 5), 40, 2, 2),           # the config's `workers` caps it
    ((0, 1), 10, None, 2)])
def test_ltx_worker_count_follows_host_ram(ltx_env, gpus, rss, cap, workers):
    make, caller, _ = ltx_env
    q = make(gpus=gpus, workers=cap, peak_rss_gib=rss)
    _, by = run_ltx(q, caller, [{"prompt": "p", "seed": i} for i in range(5)])
    for i, item in by.items():
        assert item["worker"]["rank"] == i % workers and item["worker"]["gpus"] == str(gpus[i % workers])


def test_ltx_ram_shortfall_fails_the_job_before_any_worker(ltx_env, monkeypatch):
    """Not even one worker fits in MemAvailable - host_reserve_gib: the job fails with a clear
    message instead of letting the host OOM killer pick a victim."""
    make, caller, _ = ltx_env
    monkeypatch.setattr(rollouts, "meminfo_gib", lambda: {"MemTotal": 251.0, "MemAvailable": 90.0})
    q = make(peak_rss_gib=40)
    job_id = q.backends["rollout_ltx25"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}]})["job_id"]
    out = q.wait(caller, job_id, 60)
    assert out["state"] == "failed"
    assert "MemAvailable 90 GiB" in out["error"] and "peak_rss_gib 40" in out["error"]


def test_ltx_workers_shrink_to_what_mem_available_holds(ltx_env, monkeypatch):
    """4 GPUs, but MemAvailable - reserve fits 3 workers: the job runs on 3, not refused."""
    make, caller, _ = ltx_env
    monkeypatch.setattr(rollouts, "meminfo_gib", lambda: {"MemTotal": 251.0, "MemAvailable": 185.0})
    q = make(peak_rss_gib=40)
    _, by = run_ltx(q, caller, [{"prompt": "p", "seed": i} for i in range(6)])
    assert {item["worker"]["gpus"] for item in by.values()} == {"0", "1", "4"}
    assert all(item["worker"]["rank"] == i % 3 for i, item in by.items())


def test_ltx_a_one_item_job_charges_one_worker(ltx_env, monkeypatch):
    """RAM is charged only for workers that get an item: 1 item runs with room for just 1 worker."""
    make, caller, _ = ltx_env
    monkeypatch.setattr(rollouts, "meminfo_gib", lambda: {"MemTotal": 251.0, "MemAvailable": 105.0})
    q = make(peak_rss_gib=40)
    _, by = run_ltx(q, caller, [{"prompt": "p", "seed": 1}])
    assert by[0]["worker"]["gpus"] == "0" and by[0]["worker"]["rank"] == 0


def test_ltx_meminfo_reads_proc():
    m = rollouts.meminfo_gib()
    assert 0 < m["MemAvailable"] <= m["MemTotal"]


def test_ltx_missing_clone_fails_the_job_with_a_clear_message(ltx_env, tmp_path):
    make, caller, _ = ltx_env
    missing = tmp_path / "no_such_clone"
    q = make(repo=str(missing))
    out = q.wait(caller, q.backends["rollout_ltx25"].submit(
        q, caller, {"items": [{"prompt": "p", "seed": 1}]})["job_id"], 60)
    assert out["state"] == "failed"
    assert str(missing) in out["error"] and "into that folder" in out["error"]


def test_ltx_refuses_to_run_off_the_pinned_commit(ltx_env):
    make, caller, _ = ltx_env
    q = make(commit="0" * 40)
    out = q.wait(caller, q.backends["rollout_ltx25"].submit(
        q, caller, {"items": [{"prompt": "p", "seed": 1}]})["job_id"], 60)
    assert out["state"] == "failed" and "generators.ltx25.commit" in out["error"] and "0" * 40 in out["error"]


def test_ltx_build_gpu_backends_includes_it_only_when_a_variant_is_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((("dev",), True), ((), False)):
        names = [b.name for b in build_gpu_backends(small_ltx_cfg(enabled=enabled), tmp_path / "run",
                                                     [0, 1, 2, 3], TokenRegistry(rec), rec)]
        assert ("rollout_ltx25" in names) is present


def _ltx_backend(tmp_path, **over):
    rec = Recorder(tmp_path / "run")
    return Ltx25Backend(small_ltx_cfg(**over), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)


def _ltx_job(**args):
    from types import SimpleNamespace
    return SimpleNamespace(args={"variant": "distilled", "frames": 49, "height": 576, "width": 1024, **args})


def test_ltx_finish_refuses_a_non_24fps_render(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=30, width=1024, height=576)
    with pytest.raises(ValueError, match="fps"):
        _ltx_backend(tmp_path).finish(_ltx_job(), {"index": 0, "prompt": "p"}, out)


def test_ltx_finish_refuses_a_render_of_the_wrong_frame_count(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=1024, height=576)      # 48 frames, job asks 49
    with pytest.raises(ValueError, match="48 frames, not 49"):
        _ltx_backend(tmp_path).finish(_ltx_job(), {"index": 0, "prompt": "p"}, out)


def test_ltx_finish_refuses_a_render_off_the_requested_size(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=960, height=544)
    with pytest.raises(ValueError, match="1024x576"):
        _ltx_backend(tmp_path).finish(_ltx_job(), {"index": 0, "prompt": "p"}, out)


def test_ltx_finished_candidate_passes_the_real_ingestor_as_static(tmp_path):
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    run_dir = tmp_path / "run"
    out = run_dir / "staging" / "rollouts"
    out.mkdir(parents=True)
    make_mp4(out / "0.mp4", seconds=121 / 24, fps=24, width=1024, height=576)
    frames = probe_video(out / "0.mp4").frames
    res = _ltx_backend(tmp_path).finish(_ltx_job(frames=frames), {"index": 0, "prompt": "A quiet forest path at dawn."}, out)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), caption=Path(res["caption"]), pose=None,
                                camera_motion="static", provenance={"kind": "rollout", "generator": "ltx-2.5-distilled",
                                "job_id": "j1", "inputs_hash": "x", "seed": 1})], node_id="n1")
    assert r.accepted, r.reasons
    assert "video_caption_static" in r.formats


# ---- real LTX-2.5 gpu smoke ----

@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["distilled", "dev"])
def test_real_ltx25_rollout(tmp_path, variant):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_rollouts.py -m gpu -k ltx25 -s --basetemp=.cache/pytest/gpu

    Per enabled variant: 2 T2V items, one with a first keyframe and one with first and last
    keyframes (default frames and size; four, so four workers run at once when the host-RAM rule
    allows), workers per that rule over AR_TEST_GPUS. The keyframes are generate_images (Z-Image)
    frames -- the tools chain. Each candidate must ingest as video_caption_static (camera_motion: static)."""
    import os
    import shutil
    import time
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from ar_kernel.tools.images import ImageBackend

    if variant not in REAL.get("generators.ltx25.variants"):
        pytest.skip(f"ltx25 {variant} is disabled in configs/kernel.yaml")
    gpus = [int(g) for g in os.environ.get("AR_TEST_GPUS", "0,1,2,3").split(",")]
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    for b in (ImageBackend, Ltx25Backend):
        q.register(b(REAL, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    host = lambda p: staging / Path(p).relative_to("/workspace/staging")
    peak, stop = _peak_sampler(gpus)
    try:
        img = _wait(q, caller, q.backends["generate_images"].submit(
            q, caller, {"items": [{"prompt": "a red barn in an open field, photorealistic", "seed": 9},
                                  {"prompt": "a cup of coffee on a wooden table, photorealistic", "seed": 10}]})["job_id"])
        assert img["state"] == "done", img.get("error")
        frames = [i["image"] for i in job_result(img)["items"]]
        items = [{"prompt": "Ocean waves gently rolling onto a quiet beach at sunset, camera steady.", "seed": 2},
                 {"prompt": "The red barn under slowly moving clouds, grass swaying, camera steady.",
                  "keyframes": [{"image": frames[0], "frame": 0}], "seed": 3},
                 {"prompt": "A slow walk through a sunlit forest path, camera steady.", "seed": 4},
                 {"prompt": "The camera pulls back from a red barn in a field until a cup of coffee on a wooden "
                            "table fills the frame.",
                  "keyframes": [{"image": frames[0], "frame": 0}, {"image": frames[1], "frame": -1}], "seed": 5}]
        t0 = time.monotonic()
        out = _wait(q, caller, q.backends["rollout_ltx25"].submit(q, caller, {"items": items, "variant": variant})["job_id"])
        wall = time.monotonic() - t0
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
        assert all("candidate" in by[i] for i in range(4)), by
        ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
        rows, failures = [], []
        for i in range(4):
            c = by[i]["candidate"]
            stage = staging / "ingest" / f"static{i}"
            stage.mkdir(parents=True)
            shutil.copy(host(c["video"]), stage / "v.mp4")
            shutil.copy(host(c["caption"]), stage / "c.json")
            [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=None,
                                          camera_motion="static", provenance=c["provenance"],
                                          license=c["license"])], node_id="gpu")
            rows.append({"item": i, "video": c["video"], "frames": c["frames"], "worker": by[i]["worker"],
                         "accepted": res.accepted, "formats": res.formats, "reasons": res.reasons})
            if not (res.accepted and "video_caption_static" in res.formats):
                failures.append(f"{i}: {res.reasons}")
        print(json.dumps({"variant": variant, "gpus": gpus, "wall_s": round(wall, 1), "peak_mib": peak,
                          "gpu_memory_mib": job_result(out)["gpu_memory_mib"], "rows": rows}, indent=1))
    finally:
        stop.set()
        q.shutdown()
    assert job_result(out)["gpu_memory_released"] is True
    assert not failures, failures
