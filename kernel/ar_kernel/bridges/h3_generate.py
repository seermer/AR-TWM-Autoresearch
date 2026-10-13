"""rollout_h3 worker (runs in the gen-lightx2v conda-prefix env, under torchrun with one rank per
GPU): MiniMax H3 through LightX2V's shared block offload recipe.

torchrun --standalone --nproc_per_node=N h3_generate.py --items items.json --requests requests.json \
    --out DIR --weights <weights/minimax-h3> --config <LightX2V json> --height H --width W

requests.json: {index: {"prompt": the model prompt, "frames": the clip's frame count}}.

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

from PIL import Image

def request_of(item: dict, request: dict, height: int, width: int) -> dict:
    # the kernel has checked that a first frame is exactly width x height, so the runner uses it as it is
    image = Image.open(item["image"]) if item.get("image") else None
    return {"task": "i2av" if image else "t2av", "prompt": request["prompt"], "seed": int(item["seed"]),
            "size": [height, width], "num_frames": int(request["frames"]),
            "image_path": image, "last_frame_path": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--requests", "--out", "--weights", "--config"):
        ap.add_argument(name, required=True)
    for name in ("--height", "--width"):
        ap.add_argument(name, type=int, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    requests = json.loads(Path(a.requests).read_text(encoding="utf-8"))
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))

    import torch.distributed as dist
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config, init_parallel
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    # the fl2av variant is the base transformer: it serves t2av and i2av too
    startup = build_startup_config({"model_cls": "minimax_h3", "model_variant": "fl2av", "model_path": a.weights,
                                    "config_json": a.config})
    if startup["parallel"]:
        PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None).init_parallel_env()
        init_parallel(startup)
    runner = build_runner(startup)
    rank0 = not dist.is_initialized() or dist.get_rank() == 0

    for item in items:
        index, t0 = item["index"], time.monotonic()
        request = request_of(item, requests[str(index)], a.height, a.width)
        runner.run_request(runner.prepare_request({**request, "save_result_path": str(out / f"{index}.mp4")}))
        if rank0:
            (out / f"{index}.json").write_text(json.dumps({"ok": True, "seconds": round(time.monotonic() - t0, 2)}),
                                               encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
