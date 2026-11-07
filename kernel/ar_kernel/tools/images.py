"""generate_images: Z-Image-Turbo images for rollouts to start or end on. Images carry no ingest candidate (they are inputs, not training clips); a rollout made
from one folds the image's hash into its own inputs_hash instead."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .gpu_jobs import GpuJob, check_item_seed, check_size, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "zimage_generate.py"

_MIN_SIDE, _MAX_SIDE = 256, 1920


def _valid_side(n) -> bool:
    return _MIN_SIDE <= n <= _MAX_SIDE and n % 16 == 0


class ImageBackend(GpuJob):
    name = tool = "generate_images"
    kind = "image"
    generator = "z-image-turbo"
    config_key = "images"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.height, self.width = self.block["default_size"]
        self.description = (
            "Generate images from text prompts (Z-Image-Turbo). A GPU job: returns {job_id} at "
            "once; collect with job_wait. `width`/`height`: within 2% of 16:9, and multiples of 16 "
            f"in 256..1920 (default {self.width}x{self.height}). "
            "Items: {'prompt': str, 'seed': int}. Each result "
            "item gives `image`: a png in /workspace/staging/images/<job_id>/.")

    def check_args(self, args):
        width, height = args.setdefault("width", self.width), args.setdefault("height", self.height)
        check_size(self.cfg, width, height)
        if not (_valid_side(width) and _valid_side(height)):
            raise ToolError(f"width/height must be multiples of 16 in [{_MIN_SIDE}, {_MAX_SIDE}]: "
                            f"got {width}x{height}")

    def check_item(self, item):
        if not item.get("prompt") or not isinstance(item["prompt"], str):
            raise ToolError("prompt must be a non-empty string")
        check_item_seed(item)

    def prepare(self, job, items, work):
        i = self.block
        return {"bridge": BRIDGE, "env": i["env"], "job": {"groups": split_gpus(self.gpus, 1, i["workers"]),
                                                           "paths": {"weights": str(self.cfg.repo_root / i["weights"])}}}

    def finish(self, job, item, outputs, info, out):
        png = outputs["image"]
        width, height = job.args["width"], job.args["height"]
        with Image.open(png) as im:
            if im.size != (width, height):
                raise ValueError(f"image is {im.size[0]}x{im.size[1]}, not {width}x{height}")
        return {"image": png}
