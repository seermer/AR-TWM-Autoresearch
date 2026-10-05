"""GpuJob plumbing with a fake worker (no GPU)."""
from conftest import job_result
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.gpu_jobs import GpuJob, canonical_hash, register_gpu_tools, split_gpus
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError, ToolKit, new_mcp
from tests.conftest import make_mp4

FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


class FakeJob(GpuJob):
    name = tool = "rollout_fake"
    kind, generator, license = "rollout", "fake-gen", "test-license"
    file_keys = ("src",)

    def check_args(self, args):
        if not args.get("items"):
            raise ToolError("items is empty")

    def produce(self, job, items, work, out, cancel, report):
        groups = split_gpus(self.gpus, 1, None)
        return self.run_workers("autoresearcher", lambda r, w: ["python", str(FAKE), "--items",
                                str(work / "items.json"), "--out", str(out), "--rank", str(r), "--world", str(w)],
                                groups, job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        return {"video": out / f"{item['index']}.mp4",
                "caption": self.write_caption(out, item, {"caption": item.get("prompt", "x")})}


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2.5, fps=24, width=736, height=414)
    q.register(FakeJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                       gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, rec, ws, staging, tmp_path / "run"
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_fake"].submit(q, caller, {"items": items, **params})


def test_split_gpus():
    assert split_gpus([0, 1, 4, 5], 1, None) == [[0], [1], [4], [5]]
    assert split_gpus([0, 1, 4, 5], 2, None) == [[0, 1], [4, 5]]
    assert split_gpus([0, 1, 4, 5], 1, 2) == [[0], [1]]
    assert split_gpus([0, 1, 2], 2, None) == [[0, 1]]


def test_run_workers_refuses_empty_gpu_groups(env):
    q, caller, rec, ws, staging, run = env
    backend = q.backends["rollout_fake"]
    with pytest.raises(ValueError):
        backend.run_workers("autoresearcher", lambda r, w: [], [], job=None, work=run, out=run,
                            total=0, cancel=threading.Event(), report=lambda p: None)


def test_the_timeout_comes_from_the_backends_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)

    class ConfiguredJob(FakeJob):
        config_key = "annotate"        # configs/kernel.yaml: timeout_s: 21600

    backend = ConfiguredJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec)
    assert backend.timeout_s == 21600
    assert FakeJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec).timeout_s is None


def test_items_fan_out_one_worker_per_gpu_and_publish_candidates(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4", "prompt": f"p{i}", "seed": i} for i in range(6)]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    got = job_result(out)["items"]
    assert [g["index"] for g in got] == list(range(6))
    c = got[0]["candidate"]
    job = out["id"]
    assert c["video"] == f"/workspace/staging/rollouts/{job}/0.mp4"
    assert (staging / "rollouts" / job / "0.mp4").is_file()
    assert json.loads((staging / "rollouts" / job / "0.json").read_text()) == {"caption": "p0"}
    assert c["provenance"] == {"kind": "rollout", "generator": "fake-gen", "job_id": job,
                               "inputs_hash": c["provenance"]["inputs_hash"], "seed": 0}
    assert c["license"] == "test-license" and "pose" not in c
    assert got[0]["worker"]["gpus"] == "0" and got[1]["worker"]["gpus"] == "1" and got[4]["worker"]["gpus"] == "0"
    assert not (run / "jobs" / job / "in").exists() and not (run / "jobs" / job / "out").exists()
    assert job_result(out)["gpu_memory_released"] is True


def test_inputs_hash_depends_on_file_content_not_path(env):
    q, caller, rec, ws, staging, run = env
    (ws / "b.mp4").write_bytes((ws / "a.mp4").read_bytes())
    out = q.wait(caller, submit(q, caller, [{"src": "a.mp4", "seed": 1}, {"src": "b.mp4", "seed": 1},
                                           {"src": "a.mp4", "seed": 2}])["job_id"], 120)
    h = [i["candidate"]["provenance"]["inputs_hash"] for i in job_result(out)["items"]]
    assert h[0] == h[1] != h[2]


def test_per_item_failure_and_worker_crash_are_item_errors(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4"}, {"src": "a.mp4", "fail": True}, {"src": "a.mp4", "crash": True}, {"src": "a.mp4"}]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done"
    by = {i["index"]: i for i in job_result(out)["items"]}
    assert "candidate" in by[0] and "candidate" in by[3]
    assert by[1]["error"] == "boom"
    assert "exit code 3" in by[2]["error"] and "log tail" in by[2]["error"]


def test_a_truncated_status_file_is_an_item_error_not_a_job_failure(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4"}, {"src": "a.mp4"}, {"src": "a.mp4", "truncated": True}, {"src": "a.mp4"}]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    by = {i["index"]: i for i in job_result(out)["items"]}
    assert "candidate" in by[0] and "candidate" in by[1] and "candidate" in by[3]
    assert "error" in by[2] and "JSONDecodeError" in by[2]["error"]


def test_run_workers_kills_workers_past_its_own_timeout(env):
    q, caller, rec, ws, staging, run = env
    q.backends["rollout_fake"].timeout_s = 1
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 60}])["job_id"]
    out = q.wait(caller, job_id, 60)
    assert out["state"] == "done", out
    assert "timeout" in job_result(out)["items"][0]["error"]


def test_missing_or_escaping_input_is_refused_at_submit(env):
    q, caller, rec, ws, staging, run = env
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "nope.mp4"}])
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "/etc/passwd"}])
    with pytest.raises(ToolError):
        submit(q, caller, [])


def test_a_planted_link_in_staging_fails_the_item_not_the_host(env, tmp_path):
    q, caller, rec, ws, staging, run = env
    outside = tmp_path / "outside"; outside.mkdir()
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 2}])["job_id"]
    (staging / "rollouts").mkdir()
    (staging / "rollouts" / job_id).symlink_to(outside)        # swapped in while the job runs
    out = q.wait(caller, job_id, 120)
    assert "error" in job_result(out)["items"][0]
    assert list(outside.iterdir()) == []


def test_cancel_kills_every_worker(env):
    q, caller, rec, ws, staging, run = env
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 300} for _ in range(4)])["job_id"]
    out_dir = run / "jobs" / job_id / "out"
    deadline = time.monotonic() + 30
    while len(list(out_dir.glob("worker*.pid"))) < 4 and time.monotonic() < deadline:
        time.sleep(0.2)
    pids = [int(p.read_text()) for p in out_dir.glob("worker*.pid")]
    assert len(pids) == 4
    q.cancel(caller, job_id)
    assert q.wait(caller, job_id, 120)["state"] == "cancelled"
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_registration_exposes_only_tools_for_backends_on_the_queue(env):
    q, caller, rec, ws, staging, run = env

    class Wan22Fake(FakeJob):
        name = tool = "rollout_wan22"

    q.register(Wan22Fake(KernelConfig.load(), run, [0, 1, 4, 5], TokenRegistry(rec), rec,
                         gpu_memory=lambda g: {i: 100 for i in g}))
    kit, mcp = ToolKit(TokenRegistry(rec), rec), new_mcp()
    register_gpu_tools(mcp, kit, q)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "rollout_wan22" in names                     # registered backend: the tool appears
    assert "annotate_camera" not in names and "generate_images" not in names   # not registered: absent


def test_commanded_camera_is_published_under_its_own_name_not_as_pose(env):
    """rollout_alayaworld's commanded camera path is metadata: its own file, never the pose role."""
    import numpy as np
    q, caller, rec, ws, staging, run = env
    backend = q.backends["rollout_fake"]

    def finish(job, item, out):
        np.savez(out / f"{item['index']}.cmd.npz", cam_c2w=np.tile(np.eye(4), (3, 1, 1)))
        return {**FakeJob.finish(backend, job, item, out), "commanded_camera": out / f"{item['index']}.cmd.npz"}
    backend.finish = finish
    out = q.wait(caller, submit(q, caller, [{"src": "a.mp4", "seed": 1}])["job_id"], 120)
    assert out["state"] == "done", out
    c, job = job_result(out)["items"][0]["candidate"], out["id"]
    assert c["commanded_camera"] == f"/workspace/staging/rollouts/{job}/0.commanded_camera.npz"
    assert (staging / "rollouts" / job / "0.commanded_camera.npz").is_file()
    assert "pose" not in c and c["video"] == f"/workspace/staging/rollouts/{job}/0.mp4"


# ---- build_gpu_backends: the services a run gets ----

def toggled_cfg(annotate=True, images=True, alaya=("dmd4", "ar30"), wan=True, ltx=("distilled",)):
    import copy
    base = KernelConfig.load()
    raw = copy.deepcopy(base.raw)
    raw["annotate"]["enabled"], raw["images"]["enabled"] = annotate, images
    g = raw["generators"]
    for v in g["alayaworld"]["variants"]:
        g["alayaworld"]["variants"][v]["enabled"] = v in alaya
    g["wan22"]["variants"]["ti2v-5b"]["enabled"] = wan
    for v in g["ltx25"]["variants"]:
        g["ltx25"]["variants"][v]["enabled"] = v in ltx
    return KernelConfig(raw=raw, repo_root=base.repo_root)


@pytest.mark.parametrize("kw,expected", [
    ({}, {"annotate_camera", "generate_images", "rollout_alayaworld", "rollout_wan22", "rollout_ltx25"}),
    ({"annotate": False, "images": False, "alaya": (), "wan": False, "ltx": ()}, set()),
    ({"alaya": ("ar30",), "wan": False, "ltx": ("dev",)},
     {"annotate_camera", "generate_images", "rollout_alayaworld", "rollout_ltx25"}),
    ({"annotate": False, "alaya": (), "ltx": ()}, {"generate_images", "rollout_wan22"}),
])
def test_build_gpu_backends_returns_exactly_the_enabled_backends_plus_the_captioner(tmp_path, kw, expected):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    names = [b.name for b in build_gpu_backends(toggled_cfg(**kw), tmp_path / "run", [0, 1, 2, 3],
                                                TokenRegistry(rec), rec)]
    assert sorted(names) == sorted(expected | {"caption_videos"})     # the captioner is always present


def test_listed_tools_show_only_enabled_tools_and_enabled_variants(tmp_path):
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=60)
    try:
        for b in build_gpu_backends(toggled_cfg(images=False, alaya=("dmd4",), wan=False, ltx=("distilled",)),
                                    tmp_path / "run", [0, 1, 2, 3], reg, rec):
            q.register(b)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_gpu_tools(mcp, kit, q)
        register_caption_tool(mcp, kit, q)
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    finally:
        q.shutdown()
    assert set(tools) == {"annotate_camera", "rollout_alayaworld", "rollout_ltx25", "caption_videos"}
    alaya, ltx = tools["rollout_alayaworld"].description, tools["rollout_ltx25"].description
    assert "'dmd4'" in alaya and "'ar30'" not in alaya
    assert "'distilled'" in ltx and "'dev'" not in ltx


# ---- item schemas: the listed tools say each item field's type ----

def _listed_tools(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=60)
    try:
        backends = build_gpu_backends(toggled_cfg(), tmp_path / "run", [0, 1, 2, 3], reg, rec)
        for b in backends:
            q.register(b)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_gpu_tools(mcp, kit, q)
        return {t.name: t for t in asyncio.run(mcp.list_tools())}, {b.name: b for b in backends}
    finally:
        q.shutdown()


def test_listed_item_schemas_type_every_field(tmp_path):
    """A real agent sent every value inside an untyped item as a string ("seed": "1")
    while every schema-typed value came as an int: the schema must type the item fields."""
    tools, _ = _listed_tools(tmp_path)
    for name in ("generate_images", "rollout_wan22", "rollout_ltx25"):
        item = tools[name].input_schema["properties"]["items"]["anyOf"][0]["items"]
        assert item["properties"]["seed"]["type"] == "integer", name
        assert item["properties"]["prompt"]["type"] == "string", name
        assert set(item["required"]) == {"prompt", "seed"}, name
    for name in ("rollout_wan22", "rollout_ltx25"):
        assert tools[name].input_schema["properties"]["items"]["anyOf"][0]["items"]["properties"]["image"]["type"] == "string"
    alaya = tools["rollout_alayaworld"].input_schema["properties"]["items"]["anyOf"][0]["items"]
    from ar_kernel.tools.rollouts import VIEWPOINTS
    assert alaya["properties"]["viewpoint"]["enum"] == list(VIEWPOINTS)
    assert set(alaya["required"]) == {"image", "viewpoint", "scene_prompt", "turns"}
    turn = alaya["properties"]["turns"]["items"]
    assert set(turn["properties"]) == {"action", "event", "subject_action", "viewpoint_change"}
    for field in (*alaya["properties"].values(), *turn["properties"].values()):
        assert field["description"]                    # every item field says what it is
    assert tools["rollout_alayaworld"].input_schema["properties"]["rounds_per_turn"]["description"]
    assert turn["properties"]["action"]["type"] == "string" and turn["required"] == ["action"]


@pytest.mark.parametrize("name", ["generate_images", "rollout_wan22", "rollout_ltx25"])
def test_integer_text_seed_is_taken_as_an_int(tmp_path, name):
    _, backends = _listed_tools(tmp_path)
    args = {"items": [{"prompt": "p", "seed": "7"}, {"prompt": "q", "seed": 8}]}
    for item in args["items"]:
        backends[name].check_item(item)
    assert [i["seed"] for i in args["items"]] == [7, 8]


@pytest.mark.parametrize("name", ["generate_images", "rollout_wan22", "rollout_ltx25"])
@pytest.mark.parametrize("seed,match", [("abc", "seed must be an int: got 'abc'"), (1.5, "seed must be an int"),
                                        (True, "seed must be an int"), (None, "seed is missing")])
def test_bad_or_missing_seed_is_refused_clearly(tmp_path, name, seed, match):
    _, backends = _listed_tools(tmp_path)
    item = {"prompt": "p"} if seed is None else {"prompt": "p", "seed": seed}
    with pytest.raises(ToolError, match=match):
        backends[name].check_item(item)


def test_a_prompt_copied_from_the_evaluation_is_refused_before_the_job_is_queued(tmp_path):
    from ar_kernel.config import KernelConfig
    from ar_kernel.isolation import EXCLUDED_PROMPT
    from ar_kernel.tools.images import ImageBackend
    from ar_kernel.tools.server import ToolError
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    backend = ImageBackend(KernelConfig.load(), tmp_path / "run", [0], reg, rec)

    class Queue:
        submitted = 0

        def submit(self, caller, name, args):
            self.submitted += 1
            return "job1"
    q = Queue()
    copied = "A hall. Torch sconces on both walls cast flickering orange light."
    with pytest.raises(ToolError, match="item 1: " + EXCLUDED_PROMPT) as refused:
        backend.submit(q, caller, {"items": [{"prompt": "a quiet beach", "seed": 1}, {"prompt": copied, "seed": 2},
                                             {"prompt": "", "seed": 3}, {"prompt": "a pier"}]})
    # every bad item in the one refusal, whatever is wrong with each
    assert "3 of 4 items are bad" in str(refused.value)
    assert "item 2: prompt must be a non-empty string" in str(refused.value) and "item 3: seed is missing" in str(refused.value)
    assert q.submitted == 0
    assert backend.submit(q, caller, {"items": [{"prompt": "a quiet beach", "seed": 1}]}) == {"job_id": "job1"}
    assert [e["type"] for e in rec.read_events("n1")].count("isolation.refused") == 1
