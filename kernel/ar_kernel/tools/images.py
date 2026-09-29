"""generate_images (spec 10, Plan 3 Task 5): Z-Image-Turbo first frames for rollouts to start
from. Images carry no ingest candidate (they are inputs, not training clips); a rollout made
from one folds the image's hash into its own inputs_hash instead (Task 3)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .gpu_jobs import GpuJob, check_item_seed, is_int, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "zimage_generate.py"

_MIN_SIDE, _MAX_SIDE = 256, 1920


def _valid_side(n) -> bool:
    return is_int(n) and _MIN_SIDE <= n <= _MAX_SIDE and n % 16 == 0


class ImageBackend(GpuJob):
    name = tool = "generate_images"
    kind = "image"
    generator = "z-image-turbo"
    license = "Apache-2.0"
    config_key = "images"           # max_items / timeout_s
    description = ("Generate first-frame images from text prompts (Z-Image-Turbo). AlayaWorld, "
                   "Wan and LTX rollouts start from one of these. A GPU job: returns {job_id} at "
                   "once; collect with job_wait. `width`/`height`: multiples of 16 in 256..1920 "
                   "(default 1280x720, 16:9). Items: {'prompt': str, 'seed': int}. Each result "
                   "item gives `image`: a png in /workspace/staging/images/<job_id>/. Batch many "
                   "prompts per call.")

    def check_args(self, args):
        width, height = args.get("width", 1280), args.get("height", 720)
        if not (_valid_side(width) and _valid_side(height)):
            raise ToolError(f"width/height must be multiples of 16 in [{_MIN_SIDE}, {_MAX_SIDE}]: "
                            f"got {width}x{height}")
        for n, item in enumerate(args["items"]):
            if not item.get("prompt") or not isinstance(item["prompt"], str):
                raise ToolError(f"item {n}: prompt must be a non-empty string")
            check_item_seed(n, item)

    def produce(self, job, items, work, out, cancel, report):
        i = self.block
        width, height = job.args.get("width", 1280), job.args.get("height", 720)
        weights = self.cfg.repo_root / i["weights"]
        return self.run_workers(i["env"], lambda r, w: [
            "python", str(BRIDGE), "--items", str(work / "items.json"), "--out", str(out), "--rank", str(r),
            "--world", str(w), "--weights", str(weights), "--width", str(width), "--height", str(height),
            "--steps", str(i["steps"]), "--offload", i["offload"]], split_gpus(self.gpus, 1, None),
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        png = out / f"{item['index']}.png"
        width, height = job.args.get("width", 1280), job.args.get("height", 720)
        with Image.open(png) as im:
            if im.size != (width, height):
                raise ValueError(f"image is {im.size[0]}x{im.size[1]}, not {width}x{height}")
        return {"image": png}
