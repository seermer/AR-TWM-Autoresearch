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
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--items", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world", type=int, required=True)
args, _ = parser.parse_known_args()

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
    shutil.copy(item["src"], out / f"{index}.mp4")
    status_path.write_text(json.dumps({"ok": True, "rank": args.rank,
                                       "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}))
