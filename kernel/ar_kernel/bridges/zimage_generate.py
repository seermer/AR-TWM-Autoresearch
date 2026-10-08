"""generate_images worker (runs in the gen-zimage conda-prefix env): text prompt -> first-frame
image via Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo, Apache-2.0, diffusers ZImagePipeline).

python zimage_generate.py --items items.json --out DIR --rank R --world W \
    --weights <z-image-turbo dir> --width W --height H --steps N --offload none|model

Bridge protocol: handles items with index % world == rank, in index order; per item
writes <out>/<index>.png then <out>/<index>.json ({"ok": true, "seconds": t} or
{"ok": false, "error": "..."}). The model loads once, before the first item. Turbo runs
`num_inference_steps` DiT steps with `guidance_scale=0.0` (model card, pinned revision).

`--offload model` (the kernel default) calls `enable_model_cpu_offload()`: every allowed size fits
(peak 12.8 GB). `--offload none` keeps the pipeline on the GPU and OOMs at 1920x1088.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import ZImagePipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--out", "--weights"):
        ap.add_argument(name, required=True)
    ap.add_argument("--offload", required=True, choices=("none", "model"))
    for name in ("--rank", "--world", "--width", "--height", "--steps"):
        ap.add_argument(name, type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    mine = [i for i in json.loads(Path(args.items).read_text(encoding="utf-8"))
            if i["index"] % args.world == args.rank]
    if not mine:
        return 0
    pipe = ZImagePipeline.from_pretrained(args.weights, torch_dtype=torch.bfloat16)
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    for item in mine:
        index, t0 = item["index"], time.monotonic()
        try:
            with torch.inference_mode():
                image = pipe(prompt=item["prompt"], width=args.width, height=args.height,
                             num_inference_steps=args.steps, guidance_scale=0.0,
                             generator=torch.Generator("cuda").manual_seed(item["seed"])).images[0]
            image.save(out / f"{index}.png")
            status = {"ok": True, "seconds": round(time.monotonic() - t0, 2)}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        (out / f"{index}.json").write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
