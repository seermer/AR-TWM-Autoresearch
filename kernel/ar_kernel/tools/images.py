"""generate_images: Z-Image-Turbo images for rollouts to start or end on. Images carry no ingest candidate (they are inputs, not training clips); a rollout made
from one folds the image's hash into its own inputs_hash instead."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .gpu_jobs import GpuJob, check_item_seed, check_size, is_int, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "zimage_generate.py"

_MIN_SIDE, _MAX_SIDE = 256, 1920
WIDTH, HEIGHT = 1376, 768        # the defaults


def _valid_side(n) -> bool:
    return is_int(n) and _MIN_SIDE <= n <= _MAX_SIDE and n % 16 == 0


class ImageBackend(GpuJob):
    name = tool = "generate_images"
    kind = "image"
    generator = "z-image-turbo"
    license = "Apache-2.0"
    config_key = "images"           # timeout_s
    description = ("Generate images from text prompts (Z-Image-Turbo), for use as the first or last frame of "
                   "a rollout. A GPU job: returns {job_id} at "
                   "once; collect with job_wait. `width`/`height`: within 2% of 16:9, and multiples of 16 "
                   f"in 256..1920 (default {WIDTH}x{HEIGHT}). A rollout uses a frame image as it is, so make it "
                   "at exactly the size that rollout renders. Items: {'prompt': str, 'seed': int}. Each result "
                   "item gives `image`: a png in /workspace/staging/images/<job_id>/.")

    def check_args(self, args):
        width, height = args.setdefault("width", WIDTH), args.setdefault("height", HEIGHT)
        check_size(self.cfg, width, height)
        if not (_valid_side(width) and _valid_side(height)):
            raise ToolError(f"width/height must be multiples of 16 in [{_MIN_SIDE}, {_MAX_SIDE}]: "
                            f"got {width}x{height}")

    def check_item(self, item):
        if not item.get("prompt") or not isinstance(item["prompt"], str):
            raise ToolError("prompt must be a non-empty string")
        check_item_seed(item)

    def produce(self, job, items, work, out, cancel, report):
        i = self.block
        width, height = job.args["width"], job.args["height"]
        weights = self.cfg.repo_root / i["weights"]
        return self.run_workers(i["env"], lambda r, w: [
            "python", str(BRIDGE), "--items", str(work / "items.json"), "--out", str(out), "--rank", str(r),
            "--world", str(w), "--weights", str(weights), "--width", str(width), "--height", str(height),
            "--steps", str(i["steps"]), "--offload", i["offload"]], split_gpus(self.gpus, 1, None),
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        png = out / f"{item['index']}.png"
        width, height = job.args["width"], job.args["height"]
        with Image.open(png) as im:
            if im.size != (width, height):
                raise ValueError(f"image is {im.size[0]}x{im.size[1]}, not {width}x{height}")
        return {"image": png}
