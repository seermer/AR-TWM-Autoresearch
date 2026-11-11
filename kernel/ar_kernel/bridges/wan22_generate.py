"""Generate Wan2.2 TI2V clips with one pipeline per worker, fitting first frames to 1280x704."""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

import protocol

SIZE = (1280, 704)


def fit_first_frame(img: Image.Image) -> Image.Image:
    """Center-crop to 1280:704 and resize to exactly 1280x704, so Wan's own resize is the identity."""
    return ImageOps.fit(img.convert("RGB"), SIZE, Image.LANCZOS)


def main() -> int:
    job = protocol.load_job()
    if job["worker"] is None:
        return protocol.spawn_workers(job)
    out, frames = Path(job["out"]), job["args"]["frames"]
    extra = job["config"].get("extra_args") or {}

    def load():
        sys.path.insert(0, job["paths"]["repo"])
        import torch
        import wan
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video
        cfg = WAN_CONFIGS["ti2v-5B"]
        pipe = wan.WanTI2V(config=cfg, checkpoint_dir=job["paths"]["ckpt_dir"], device_id=0, rank=0,
                           t5_cpu=extra.get("t5_cpu", True), convert_model_dtype=True)
        return SimpleNamespace(torch=torch, cfg=cfg, pipe=pipe, save_video=save_video,
                               size=SIZE_CONFIGS["1280*704"], area=MAX_AREA_CONFIGS["1280*704"])

    def one(m, item):
        t0, torch, cfg, video = time.monotonic(), m.torch, m.cfg, None
        try:
            torch.cuda.reset_peak_memory_stats()
            img = fit_first_frame(Image.open(item["image"])) if item.get("image") else None
            with torch.inference_mode():
                video = m.pipe.generate(
                    item["prompt"], img=img, size=m.size, max_area=m.area, frame_num=frames, shift=cfg.sample_shift,
                    sample_solver="unipc", sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale,
                    seed=item["seed"], offload_model=extra.get("offload_model", True))
            path = out / f"{item['index']}.mp4"
            m.save_video(tensor=video[None], save_file=str(path), fps=cfg.sample_fps, nrow=1, normalize=True,
                         value_range=(-1, 1))
            return {"video": path}, {"seconds": round(time.monotonic() - t0, 2),
                                     "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                     "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        finally:
            # Release GPU output tensors before generating the next clip.
            del video
            gc.collect()
            torch.cuda.empty_cache()

    protocol.run_items(job, load, one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
