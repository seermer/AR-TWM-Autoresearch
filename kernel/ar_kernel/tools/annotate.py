"""annotate_camera: per-frame camera poses for agent clips, via ViGeo."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.ingest import find_cut
from ..data.probe import probe_video
from .gpu_jobs import GpuJob, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "vigeo_poses.py"


class AnnotateBackend(GpuJob):
    name = tool = "annotate_camera"
    kind = "annotation"
    config_key = "annotate"
    file_keys = ("video",)
    description = ("Estimate per-frame camera poses for video clips (ViGeo). A GPU job: returns {job_id} at "
                   "once; collect with job_wait. `paths`: mp4 files under /workspace, at most {max_frames} frames each. Each result item gives "
                   "`pose`: an npz in /workspace/staging/annotations/<job_id>/ holding cam_c2w [N,4,4] "
                   "(N = the clip's frame count, OpenCV camera-to-world, first frame identity) and pixel "
                   "`intrinsics`; pass it as `pose` to data_ingest. Measure the clip you will ingest, at its final "
                   "size. `cut` is null, or says where the clip cuts to another shot; data_ingest rejects a "
                   "clip with a cut.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.description = self.description.replace("{max_frames}", str(self.block["max_frames"]))

    def check_item(self, item):
        if not str(item.get("video", "")).lower().endswith(".mp4"):
            raise ToolError(f"{item.get('video')!r} is not an .mp4")

    def prepare(self, job, items, work):
        a, wm = self.block, self.cfg.worldmodel
        return {"bridge": BRIDGE, "env": a["env"], "job": {
            "groups": split_gpus(self.gpus, 1, a["workers"]),
            "paths": {"repo": str(wm / a["repo"]), "checkpoint": str(wm / a["checkpoint"])}}}

    def finish(self, job, item, outputs, info, out):
        pose = outputs["pose"]
        frames = probe_video(Path(item["video"])).frames
        with np.load(pose) as z:
            cam_c2w = z["cam_c2w"]
        n = len(cam_c2w)
        if n != frames:
            raise ValueError(f"pose has {n} frames, the video {frames}")
        cut = find_cut(cam_c2w, Path(item["video"]))
        return {"pose": pose, "frames": frames, "intrinsics": info["intrinsics"], "cut": cut}
