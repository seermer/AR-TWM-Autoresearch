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

With --weights set it stands in for zimage_generate.py instead (image mode, see image(): --weights
is the one flag only that bridge's argv carries, so its presence alone selects the mode -- an
image item has no "video"/"src" for the other modes' argparse-driven dispatch to key off): it
writes a PNG of --width x --height (item["bad_size"]: true writes one pixel larger, so the
backend's size check can be tested) and echoes --weights/--steps/--offload.

With --precache or --cases it stands in for WorldModel's prompt precache and run_wbench.py
instead (wbench mode, see wbench(): neither takes the bridge arguments).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path



def wbench():
    """The precache (--precache: records its argv and env next to the config, exits 0) and
    run_wbench.py (--cases) without a model. For each case it writes what the eval writes, in the
    layout measured in Plan 3 Task 6 Step 2: case_<id>_combined.mp4 of rounds x 32 - 7 frames
    (960x544, 24 fps, round r = frames [32r - 7, 32r + 25)), the sidecar JSON (actions, nominal
    turn_segments as WorldModel writes them, prompt_schedule) and the camera npz (one pose per frame, frame 0 identity). A case whose
    environment_prompt contains NO_VIDEO renders nothing."""
    import numpy as np
    import yaml
    ap = argparse.ArgumentParser()
    for name in ("--config", "--cases", "--gpus", "--master-port", "--device-map"):
        ap.add_argument(name)
    ap.add_argument("--precache", action="store_true")
    a, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    record = {"argv": sys.argv[1:], "cwd": os.getcwd(), "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
              "gemma": os.environ.get("ALAYA_GEMMA_MAX_MEMORY")}
    (Path(a.config).parent / ("fake_precache.json" if a.precache else "fake_wbench.json")).write_text(json.dumps(record))
    if a.precache:
        return
    mode = cfg["validation"]["modes"]["wbench"]
    root, videos, cpt = Path(mode["dataset"]["root"]), Path(mode["wbench_output_dir"]), mode["wbench_chunks_per_turn"]
    videos.mkdir(parents=True, exist_ok=True)
    for cid in a.cases.split(","):
        case = json.loads((root / "cases" / f"case_{cid}.json").read_text())
        if "NO_VIDEO" in case["environment_prompt"]:
            continue
        its = case["interactions"]
        turns = max(i["turn"] for i in its)
        actions = [next(i["action"] for i in its if i["turn"] == t and i["type"] == "navigation")
                   for t in range(1, turns + 1)]
        prompts, acc = [], [case["environment_prompt"]]
        for t in range(1, turns + 1):
            acc += [i["action"] for i in its if i["turn"] == t and i["type"] != "navigation"]
            prompts.append(" ".join(acc))
        frames = turns * cpt * 32 - 7
        stem = videos / f"case_{cid}_combined"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size=960x544:rate=24", "-frames:v", str(frames), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(stem) + ".mp4"], check=True)
        c2w = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
        c2w[:, 2, 3] = np.arange(frames) * 0.01
        np.savez(str(stem) + "_camera.npz", cam_c2w=c2w)
        segments = [{"turn_index": t, "action": act, "chunk_start": t * cpt, "chunk_end_exclusive": (t + 1) * cpt,
                     "chunk_count": cpt, "frame_start": t * cpt * 32,
                     "frame_end_exclusive": min(frames, (t + 1) * cpt * 32),
                     "frame_count": min(frames, (t + 1) * cpt * 32) - t * cpt * 32} for t, act in enumerate(actions)]
        sidecar = {"case_id": cid, "actions": actions, "turn_segments": segments, "chunks_per_turn": cpt,
                   "output_rounds": turns * cpt, "prompt_schedule": [{"round": r, "turn": r // cpt, "prompt": prompts[r // cpt]}
                                       for r in range(turns * cpt)],
                   "num_frames": frames, "fps": 24, "camera_file": stem.name + "_camera.npz"}
        (Path(str(stem) + ".json")).write_text(json.dumps(sidecar))


if "--precache" in sys.argv or "--cases" in sys.argv:
    wbench()
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--items", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world", type=int, required=True)
parser.add_argument("--max-frames", type=int)        # set: annotate mode (stands in for vigeo_poses.py)
parser.add_argument("--repo")
parser.add_argument("--checkpoint")
parser.add_argument("--weights")                     # set: image mode (stands in for zimage_generate.py)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--steps", type=int)
parser.add_argument("--offload")
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


def image(item, status_path):
    """zimage_generate.py's contract without a model: a solid PNG of the requested size."""
    from PIL import Image
    w, h = args.width, args.height
    if item.get("bad_size"):
        w, h = w + 1, h
    Image.new("RGB", (w, h), color=(100, 150, 200)).save(out / f"{item['index']}.png")
    status_path.write_text(json.dumps({"ok": True, "seconds": 0.01, "rank": args.rank,
                                       "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                       "weights": args.weights, "steps": args.steps, "offload": args.offload}))


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
    if args.weights is not None:
        image(item, status_path)
        continue
    shutil.copy(item["src"], out / f"{index}.mp4")
    status_path.write_text(json.dumps({"ok": True, "rank": args.rank,
                                       "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}))
