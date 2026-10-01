"""rollout_wan22 worker (runs in the gen-wan22 conda-prefix env): official Wan2.2 code, loaded
once, renders each item's shard with the ti2v-5B pipeline (text-to-video, or image-to-video when
the item carries a first frame).

python wan22_generate.py --items items.json --out DIR --rank R --world W \
    --repo <Wan2.2 checkout> --ckpt-dir <wan2.2-ti2v-5b dir> --frames N \
    [--offload-model/--no-offload-model] [--t5-cpu/--no-t5-cpu]

Bridge protocol: handles items with index % world == rank, in index order; per item
writes <out>/<index>.mp4 then <out>/<index>.json ({"ok": true, "seconds": t} or
{"ok": false, "error": "..."}). The model loads once, before the first item.

Imports and the constructor/generate()/save_video() arguments are copied from generate.py's ti2v
branch at the commit pinned in configs/kernel.yaml (generators.wan22.commit); Wan22Backend refuses
to run against a clone that has moved off it.

A first frame is center-cropped and resized to exactly 1280x704 first (fit_first_frame): WanTI2V.i2v
keeps the input image's own aspect within the 1280*704 area (best_output_size), so a 1280x720
generate_images frame would otherwise render at 1248x704 and a portrait one at 800x1088.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

SIZE = (1280, 704)


def fit_first_frame(img: Image.Image) -> Image.Image:
    """Center-crop to 1280:704 and resize to exactly 1280x704, so Wan's own resize is the identity."""
    return ImageOps.fit(img.convert("RGB"), SIZE, Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--out", "--repo", "--ckpt-dir"):
        ap.add_argument(name, required=True)
    for name in ("--rank", "--world", "--frames"):
        ap.add_argument(name, type=int, required=True)
    ap.add_argument("--offload-model", dest="offload_model", action="store_true", default=True)
    ap.add_argument("--no-offload-model", dest="offload_model", action="store_false")
    ap.add_argument("--t5-cpu", dest="t5_cpu", action="store_true", default=True)
    ap.add_argument("--no-t5-cpu", dest="t5_cpu", action="store_false")
    args = ap.parse_args()

    out = Path(args.out)
    mine = [item for item in json.loads(Path(args.items).read_text(encoding="utf-8"))
            if item["index"] % args.world == args.rank]
    if not mine:
        return 0

    sys.path.insert(0, args.repo)
    import torch
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS["ti2v-5B"]
    pipe = wan.WanTI2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
                       t5_cpu=args.t5_cpu, convert_model_dtype=True)

    for item in mine:
        index, t0 = item["index"], time.monotonic()
        status_path = out / f"{index}.json"
        video = None
        try:
            torch.cuda.reset_peak_memory_stats()
            img = fit_first_frame(Image.open(item["image"])) if item.get("image") else None
            video = pipe.generate(
                item["prompt"], img=img, size=SIZE_CONFIGS["1280*704"],
                max_area=MAX_AREA_CONFIGS["1280*704"], frame_num=args.frames, shift=cfg.sample_shift,
                sample_solver="unipc", sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale,
                seed=item["seed"], offload_model=args.offload_model)
            save_video(tensor=video[None], save_file=str(out / f"{index}.mp4"), fps=cfg.sample_fps,
                       nrow=1, normalize=True, value_range=(-1, 1))
            status = {"ok": True, "seconds": round(time.monotonic() - t0, 2),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # the 24 GB card has no headroom at 1280x704x121 (peak ~22.9 GiB): the previous clip
        # (1.3 GB fp32 on the GPU) must not survive into the next item (the smoke run ran out of memory there)
        del video
        gc.collect()
        torch.cuda.empty_cache()
        status_path.write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
