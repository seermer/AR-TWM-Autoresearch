"""A stand-in GPU-job worker for the GpuJob tests (no GPU, no model): implements the bridge
protocol only (see Plan 3's global constraints).

python fake_gen_worker.py --items items.json --out DIR --rank R --world W

For each item with index % world == rank, in index order, it copies item["src"] to
<out>/<index>.mp4 and writes <out>/<index>.json with {"ok": true, "rank": R, "gpus":
CUDA_VISIBLE_DEVICES}. Special item fields: "fail": true writes {"ok": false, "error": "boom"}
instead; "crash": true exits the process with code 3 before writing anything; "truncated": true
writes an incomplete status file (as an OOM-killed worker might, mid-write); "sleep": s sleeps
first (the cancel and timeout tests use this). It writes its pid to <out>/worker<rank>.pid before
processing any item.

With --max-frames it stands in for vigeo_poses.py instead (annotate mode, see annotate()); it
also accepts and echoes that bridge's --repo and --checkpoint.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--items", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world", type=int, required=True)
parser.add_argument("--max-frames", type=int)        # set: annotate mode (stands in for vigeo_poses.py)
parser.add_argument("--repo")
parser.add_argument("--checkpoint")
args, _ = parser.parse_known_args()


def annotate(item, status_path):
    """vigeo_poses.py's contract without a model: identity poses of the clip's frame count
    (plus item["extra_frames"]), and the bridge's own refusal past --max-frames."""
    import numpy as np
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", item["video"]],
                           capture_output=True, text=True, check=True)
    n = int(probe.stdout.strip())
    if n > args.max_frames:
        status_path.write_text(json.dumps({"ok": False, "error": f"{n} frames > max_frames ({args.max_frames})"}))
        return
    c2w = np.tile(np.eye(4, dtype=np.float32), (n + item.get("extra_frames", 0), 1, 1))
    k = np.array([[500, 0, 368], [0, 500, 207], [0, 0, 1]], dtype=np.float32)
    np.savez(out / f"{item['index']}.npz", cam_c2w=c2w, intrinsics=k)
    status_path.write_text(json.dumps({"ok": True, "frames": n, "intrinsics": [500.0, 500.0, 368.0, 207.0],
                                       "rank": args.rank, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                       "repo": args.repo, "checkpoint": args.checkpoint}))


out = Path(args.out)
(out / f"worker{args.rank}.pid").write_text(str(os.getpid()))
items = json.loads(Path(args.items).read_text(encoding="utf-8"))
mine = [item for item in items if item["index"] % args.world == args.rank]

for item in mine:
    index = item["index"]
    if item.get("sleep"):
        time.sleep(item["sleep"])
    if item.get("crash"):
        sys.exit(3)
    status_path = out / f"{index}.json"
    if item.get("truncated"):
        status_path.write_text('{"ok": true, "ran')      # deliberately incomplete JSON
        continue
    if item.get("fail"):
        status_path.write_text(json.dumps({"ok": False, "error": "boom"}))
        continue
    if args.max_frames is not None:
        annotate(item, status_path)
        continue
    shutil.copy(item["src"], out / f"{index}.mp4")
    status_path.write_text(json.dumps({"ok": True, "rank": args.rank,
                                       "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}))
