"""A stand-in bridge for the GPU job tests (no GPU, no model), on the real bridges/protocol.py.

With `groups` in the job file it spawns one worker per group, as the generator bridges do. Per item it makes
the outputs of the job's `tool` (`OUTPUTS`). Item fields: "fail" raises (an item error), "crash" prints "dying"
then exits 3 mid-item, "truncate" leaves an incomplete message and exits 0 mid-item, "sleep": s sleeps first. Job
args: "fake": "exit_early" exits 2 before any item, "orphan" starts a `sleep` it never stops (its pid in
<work>/orphan.pid). Every process writes its pid to <work>/pids/.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernel" / "ar_kernel" / "bridges"))
import protocol  # noqa: E402


def copy_src(job, item, out):
    dst = out / f"{item['index']}.mp4"
    shutil.copy(item["src"], dst)
    return {"video": dst}


def placeholder(job, item, out):
    dst = out / f"{item['index']}.mp4"
    dst.write_bytes(b"not a video")
    return {"video": dst}


OUTPUTS = {"rollout_fake": copy_src, "rollout_keyed": placeholder}


def main() -> int:
    job = protocol.load_job()
    work, out, mode = Path(job["work"]), Path(job["out"]), job["args"].get("fake")
    (work / "pids").mkdir(exist_ok=True)
    (work / "pids" / str(os.getpid())).touch()
    if mode == "orphan" and job["worker"] is None:
        child = subprocess.Popen(["sleep", "600"])
        (work / "orphan.pid").write_text(str(child.pid))
    if job["worker"] is None and job.get("groups"):
        return protocol.spawn_workers(job)
    if mode == "exit_early":
        print("cannot start", file=sys.stderr)
        return 2

    def one(model, item):
        print(f"item {item['index']}")
        if item.get("sleep"):
            time.sleep(item["sleep"])
        if item.get("crash"):
            print("dying", flush=True)
            os._exit(3)
        if item.get("fail"):
            raise ValueError("boom")
        if item.get("truncate"):
            name = "messages.jsonl" if job["worker"] is None else f"messages-{job['worker']}.jsonl"
            with open(work / name, "a") as f:
                f.write('{"type": "item", "index": ')
            os._exit(0)
        return OUTPUTS[job["tool"]](job, item, out), {"rank": job["worker"], "gpus": os.environ.get(
            "CUDA_VISIBLE_DEVICES", ""), **({"keyframes": item["keyframes"]} if "keyframes" in item else {})}

    protocol.run_items(job, lambda: None, one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
