"""rollout_h3 worker (runs in the gen-lightx2v conda-prefix env, under torchrun with one rank per
GPU): MiniMax H3 through LightX2V's shared block offload recipe.

torchrun --standalone --nproc_per_node=N h3_generate.py --items items.json --prompts prompts.json \
    --out DIR --weights <weights/minimax-h3> --config <LightX2V json> --frames N --height H --width W

The ranks render every item together (sequence parallel over one host copy of the weights), in
index order; rank 0 writes <out>/<index>.mp4 then <out>/<index>.json ({"ok": true, "seconds": t}).
An error ends the job: the ranks cannot stay in step past it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

TASKS = {(): "t2av", (0,): "i2av", (-1,): "l2av", (-1, 0): "fl2av"}


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop and resize to the canvas (the runner would stretch a first frame instead)."""
    return ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS)


def request_of(item: dict, prompt: str, frames: int, height: int, width: int) -> dict:
    images = {k["frame"]: fit(Image.open(k["image"]), (width, height)) for k in item.get("keyframes") or []}
    return {"task": TASKS[tuple(sorted(images))], "prompt": prompt, "seed": int(item["seed"]),
            "size": [height, width], "num_frames": frames,
            "image_path": images.get(0), "last_frame_path": images.get(-1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--prompts", "--out", "--weights", "--config"):
        ap.add_argument(name, required=True)
    for name in ("--frames", "--height", "--width"):
        ap.add_argument(name, type=int, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))

    import torch.distributed as dist
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config, init_parallel
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    # the fl2av variant is the base transformer: it serves every task in TASKS
    startup = build_startup_config({"model_cls": "minimax_h3", "model_variant": "fl2av", "model_path": a.weights,
                                    "config_json": a.config})
    if startup["parallel"]:
        PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None).init_parallel_env()
        init_parallel(startup)
    runner = build_runner(startup)
    rank0 = not dist.is_initialized() or dist.get_rank() == 0

    for item in items:
        index, t0 = item["index"], time.monotonic()
        request = request_of(item, prompts[str(index)], a.frames, a.height, a.width)
        runner.run_request(runner.prepare_request({**request, "save_result_path": str(out / f"{index}.mp4")}))
        if rank0:
            (out / f"{index}.json").write_text(json.dumps({"ok": True, "seconds": round(time.monotonic() - t0, 2)}),
                                               encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
