"""annotate_camera (spec 10, 16.3 item 6): per-frame camera poses for agent clips, via ViGeo."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.probe import probe_video
from .gpu_jobs import GpuJob, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "vigeo_poses.py"


class AnnotateBackend(GpuJob):
    name = tool = "annotate_camera"
    kind = "annotation"
    config_key = "annotate"             # max_items / timeout_s
    file_keys = ("video",)
    description = ("Estimate per-frame camera poses for video clips (ViGeo). A GPU job: returns {job_id} at "
                   "once; collect with job_wait. `paths`: mp4 files under /workspace. Each result item gives "
                   "`pose`: an npz in /workspace/staging/annotations/<job_id>/ holding cam_c2w [N,4,4] "
                   "(N = the clip's frame count, OpenCV camera-to-world, first frame identity) and pixel "
                   "`intrinsics`; pass it as `pose` to data_ingest with camera_motion 'moving'. Batch many clips "
                   "per call.")

    def check_args(self, args):
        for item in args["items"]:
            if not str(item.get("video", "")).lower().endswith(".mp4"):
                raise ToolError(f"{item.get('video')!r} is not an .mp4")

    def produce(self, job, items, work, out, cancel, report):
        a = self.cfg.get("annotate")
        wm = self.cfg.worldmodel
        return self.run_workers(a["env"], lambda r, w: [
            "python", str(BRIDGE), "--items", str(work / "items.json"), "--out", str(out), "--rank", str(r),
            "--world", str(w), "--repo", str(wm / a["repo"]), "--checkpoint", str(wm / a["checkpoint"]),
            "--max-frames", str(a["max_frames"])], split_gpus(self.gpus, 1, None),
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        pose = out / f"{item['index']}.npz"
        frames = probe_video(Path(item["video"])).frames
        with np.load(pose) as z:
            n = len(z["cam_c2w"])
        if n != frames:
            raise ValueError(f"pose has {n} frames, the video {frames}")
        status = json.loads((out / f"{item['index']}.json").read_text())
        return {"pose": pose, "frames": frames, "intrinsics": status["intrinsics"]}
