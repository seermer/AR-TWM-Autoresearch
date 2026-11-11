"""Generate images with one Z-Image-Turbo pipeline per GPU worker, using model offload or resident weights."""
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
