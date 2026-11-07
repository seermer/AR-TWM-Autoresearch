"""generate_images worker (runs in the gen-zimage conda-prefix env): text prompt -> first-frame
image via Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo, Apache-2.0, diffusers ZImagePipeline).

python zimage_generate.py --job <work>/job.json (bridges/protocol.py): one worker per GPU group; per item
<out>/<index>.png. Job file: paths.weights, args.width, args.height, config.steps, config.offload ("none" or
"model"). The model loads once per worker. Turbo runs `num_inference_steps` DiT steps with `guidance_scale=0.0`
(model card, pinned revision).

`offload: model` (the kernel default) calls `enable_model_cpu_offload()`: every allowed size fits
(peak 12.8 GB). `offload: none` keeps the pipeline on the GPU and OOMs at 1920x1088.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import protocol


def main() -> int:
    job = protocol.load_job()
    if job["worker"] is None:
        return protocol.spawn_workers(job)
    out, a, c = Path(job["out"]), job["args"], job["config"]

    def load():
        import torch
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(job["paths"]["weights"], torch_dtype=torch.bfloat16)
        if c["offload"] == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return torch, pipe

    def one(model, item):
        (torch, pipe), t0 = model, time.monotonic()
        with torch.inference_mode():
            image = pipe(prompt=item["prompt"], width=a["width"], height=a["height"],
                         num_inference_steps=c["steps"], guidance_scale=0.0,
                         generator=torch.Generator("cuda").manual_seed(item["seed"])).images[0]
        path = out / f"{item['index']}.png"
        image.save(path)
        return {"image": path}, {"seconds": round(time.monotonic() - t0, 2)}

    protocol.run_items(job, load, one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
