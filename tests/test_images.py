"""generate_images: the real backend with a fake subprocess worker, plus one real
Z-Image-Turbo gpu test."""
import copy
import threading
from pathlib import Path

import pytest
from PIL import Image

from ar_kernel.config import KernelConfig, resolve_gpus
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import images
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.gpu_jobs import build_gpu_backends
from ar_kernel.tools.images import ImageBackend
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError

REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def small_cfg(env="autoresearcher"):
    raw = copy.deepcopy(REAL.raw)
    raw["images"].update(env=env)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The real ImageBackend.produce/run_workers, with the Z-Image bridge swapped for the fake
    worker (image mode: it writes a PNG of the requested size and echoes weights/steps/offload)."""
    monkeypatch.setattr(images, "BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    q.register(ImageBackend(small_cfg(env="autoresearcher"), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                            gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def run(q, caller, args):
    out = q.wait(caller, q.backends["generate_images"].submit(q, caller, args)["job_id"], 120)
    assert out["state"] == "done", out
    return out["id"], {i["index"]: i for i in out["result"]["items"]}


def _open(staging, item):
    return Image.open(staging / Path(item["image"]).relative_to("/workspace/staging"))


def test_image_is_published_with_the_requested_size(env):
    q, caller, staging = env
    job, by = run(q, caller, {"items": [{"prompt": "a red fox in snow", "seed": 1}]})
    item = by[0]
    assert item["image"] == f"/workspace/staging/images/{job}/0.png"
    assert item["prompt"] == "a red fox in snow" and item["seed"] == 1
    assert item["generator"] == "z-image-turbo" and item["license"] == "Apache-2.0"
    with _open(staging, item) as im:
        assert im.size == (1280, 720)              # config default width/height


def test_custom_size_is_honored(env):
    q, caller, staging = env
    _, by = run(q, caller, {"items": [{"prompt": "p", "seed": 2}], "width": 960, "height": 544})
    with _open(staging, by[0]) as im:
        assert im.size == (960, 544)


def test_size_not_a_multiple_of_16_is_refused_at_submit(env):
    q, caller, staging = env
    with pytest.raises(ToolError, match="multiples of 16"):
        q.backends["generate_images"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}],
                                                          "width": 1281, "height": 720})


def test_size_out_of_range_is_refused_at_submit(env):
    q, caller, staging = env
    with pytest.raises(ToolError, match="multiples of 16"):
        q.backends["generate_images"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1}],
                                                          "width": 2000, "height": 720})


def test_missing_prompt_is_refused_at_submit(env):
    q, caller, staging = env
    with pytest.raises(ToolError, match="prompt"):
        q.backends["generate_images"].submit(q, caller, {"items": [{"seed": 1}]})


def test_missing_seed_is_refused_at_submit(env):
    q, caller, staging = env
    with pytest.raises(ToolError, match="seed"):
        q.backends["generate_images"].submit(q, caller, {"items": [{"prompt": "p"}]})


def test_wrong_size_output_is_an_item_error(env):
    q, caller, staging = env
    _, by = run(q, caller, {"items": [{"prompt": "p", "seed": 1, "bad_size": True}]})
    assert "error" in by[0]


def test_workers_get_the_images_config_and_one_gpu_each(env):
    q, caller, staging = env
    _, by = run(q, caller, {"items": [{"prompt": "p", "seed": i} for i in range(5)]})
    for i, item in by.items():
        assert item["worker"]["gpus"] == str([0, 1, 4, 5][i % 4]) and item["worker"]["rank"] == i % 4
        assert item["worker"]["weights"] == str(REAL.repo_root / REAL.get("images.weights"))
        assert item["worker"]["steps"] == REAL.get("images.steps")
        assert item["worker"]["offload"] == REAL.get("images.offload")


def test_limits_come_from_the_images_config_block(tmp_path):
    rec = Recorder(tmp_path / "run")
    b = ImageBackend(REAL, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    assert (b.max_items, b.timeout_s) == (REAL.get("images.max_items"), REAL.get("images.timeout_s"))


def test_build_gpu_backends_includes_images_only_when_enabled(tmp_path):
    rec = Recorder(tmp_path / "run")
    names = lambda cfg: [b.name for b in build_gpu_backends(cfg, tmp_path / "run", [0, 1, 2, 3],
                                                             TokenRegistry(rec), rec)]
    for enabled in (True, False):
        raw = copy.deepcopy(REAL.raw)
        raw["images"]["enabled"] = enabled
        assert ("generate_images" in names(KernelConfig(raw=raw, repo_root=REAL.repo_root))) is enabled


# ---- real Z-Image-Turbo gpu test ----

@pytest.mark.gpu
def test_real_zimage_generates_plausible_images(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_images.py -m gpu -s -- 8 prompts on 4 GPUs."""
    gpus = resolve_gpus(REAL, {"CUDA_VISIBLE_DEVICES": __import__("os").environ.get("AR_TEST_GPUS", "0,1,2,3")})
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=1800)
    q.register(ImageBackend(REAL, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    prompts = [f"a photorealistic first-person view, scene {i}: a kitchen counter with fruit" for i in range(8)]
    try:
        job = q.backends["generate_images"].submit(q, caller, {"items": [
            {"prompt": p, "seed": i} for i, p in enumerate(prompts)]})
        out = q.wait(caller, job["job_id"], 1800)
        while out["state"] not in ("done", "failed", "cancelled"):
            out = q.wait(caller, job["job_id"], 1800)
    finally:
        q.shutdown()
    assert out["state"] == "done", out.get("error")
    for item in out["result"]["items"]:
        assert "error" not in item, item
        with Image.open(staging / Path(item["image"]).relative_to("/workspace/staging")) as im:
            assert im.size == (1280, 720)
    assert out["result"]["gpu_memory_released"] is True
    print({"gpus": gpus, "gpu_memory_mib": out["result"]["gpu_memory_mib"],
          "seconds": [i["worker"].get("seconds") for i in out["result"]["items"]]})
