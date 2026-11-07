"""A stand-in for WorldModel's prompt precache (--precache) and scripts/tools/run_wbench.py (--cases): no model.

The rollout_alayaworld tests run the real bridges/alayaworld_render.py with these two commands.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def wbench():
    """The precache (--precache: records its argv and env next to the config, exits 0) and
    run_wbench.py (--cases) without a model. For each case it writes what the eval writes, in the
    layout: case_<id>_combined.mp4 of rounds x 32 + 1 frames
    (960x544, 24 fps, the input frame then round r = frames [32r + 1, 32r + 33)), the sidecar JSON (actions, nominal
    turn_segments as WorldModel writes them, prompt_schedule) and the camera npz (one pose per frame, frame 0
    identity), after printing WorldModel's case_start marker. A case prompt containing PRECACHE_HANG makes the
    precache hang (the timeout test); PRECACHE_FAIL makes it exit 4. A case whose environment_prompt contains
    NO_VIDEO renders nothing; WBENCH_FAIL makes the render exit 1 after every case."""
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
        cases = Path(cfg["validation"]["modes"]["wbench"]["dataset"]["root"]) / "cases"
        if any("PRECACHE_HANG" in c.read_text() for c in cases.glob("case_*.json")):
            time.sleep(600)
        if any("PRECACHE_FAIL" in c.read_text() for c in cases.glob("case_*.json")):
            print("Gemma did not load")
            sys.exit(4)
        return
    mode = cfg["validation"]["modes"]["wbench"]
    root, videos, cpt = Path(mode["dataset"]["root"]), Path(mode["wbench_output_dir"]), mode["wbench_chunks_per_turn"]
    videos.mkdir(parents=True, exist_ok=True)
    failed = False
    for cid in a.cases.split(","):
        case = json.loads((root / "cases" / f"case_{cid}.json").read_text())
        print(f"[rank0]: [Validation] case_start case={cid} rank=0", flush=True)
        failed |= "WBENCH_FAIL" in case["environment_prompt"]
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
        frames = turns * cpt * 32 + 1
        stem = videos / f"case_{cid}_combined"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=960x544:rate=24", "-frames:v", str(frames), "-c:v", "libx264",
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
    if failed:
        print("[run_wbench] generation FAILED (rc=1)", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    wbench()
