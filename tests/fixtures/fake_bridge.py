"""A stand-in bridge for the GPU job tests (no GPU, no model), on the real bridges/protocol.py.

With `groups` in the job file it spawns one worker per group, as the generator bridges do. Per item it makes
the outputs of the job's `tool` (`OUTPUTS`: (outputs, info echoing what the job file gave)). Item fields: "fail" raises (an item error), "crash" prints "dying"
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
    return {"video": dst}, {}


def placeholder(job, item, out):
    dst = out / f"{item['index']}.mp4"
    dst.write_bytes(b"not a video")
    return {"video": dst}, {"keyframes": item["keyframes"]}


def image(job, item, out):
    """zimage_generate.py without a model: a solid PNG of the job's size (item "bad_size": one pixel wider)."""
    from PIL import Image
    a = job["args"]
    dst = out / f"{item['index']}.png"
    Image.new("RGB", (a["width"] + bool(item.get("bad_size")), a["height"]), (100, 150, 200)).save(dst)
    return {"image": dst}, {"weights": job["paths"]["weights"], "steps": job["config"]["steps"],
                            "offload": job["config"]["offload"]}


def poses(job, item, out):
    """vigeo_poses.py without a model: identity poses of the clip's frame count (plus item "extra_frames"),
    refusing a clip over config.max_frames as the bridge does."""
    import numpy as np
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", item["video"]],
                           capture_output=True, text=True, check=True).stdout.strip())
    if n > job["config"]["max_frames"]:
        raise ValueError(f"{n} frames > max_frames ({job['config']['max_frames']})")
    dst = out / f"{item['index']}.npz"
    np.savez(dst, cam_c2w=np.tile(np.eye(4, dtype=np.float32), (n + item.get("extra_frames", 0), 1, 1)),
             intrinsics=np.array([[500, 0, 368], [0, 500, 207], [0, 0, 1]], dtype=np.float32))
    return {"pose": dst}, {"frames": n, "intrinsics": [500.0, 500.0, 368.0, 207.0], **job["paths"]}


def mp4(dst, width, height, frames, fps=24, audio=False):
    sound = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-shortest", "-c:a", "aac"] if audio else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}",
                    *sound[:4], "-frames:v", str(frames), *sound[4:], "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)
    return dst


def wan(job, item, out):
    """wan22_generate.py without a model: a 1280x704, 24 fps mp4 of args.frames frames."""
    extra = job["config"].get("extra_args") or {}
    return {"video": mp4(out / f"{item['index']}.mp4", 1280, 704, job["args"]["frames"])}, {
        **job["paths"], "frames": job["args"]["frames"], "offload_model": extra.get("offload_model", True),
        "t5_cpu": extra.get("t5_cpu", True), "image": item.get("image")}


def ltx(job, item, out):
    """ltx25_generate.py without a model: an mp4 of the job's size and frames WITH an audio track, as LTX-2.5's
    own output has one (item "fps" overrides the rate)."""
    a, c = job["args"], job["config"]
    dst = mp4(out / f"{item['index']}.mp4", a["width"], a["height"], a["frames"], item.get("fps", 24), audio=True)
    return {"video": dst}, {**job["paths"], "frames": a["frames"], "height": a["height"], "width": a["width"],
                            "quantization": c.get("quantization", "fp8-cast"), "offload": c.get("offload", "cpu"),
                            "keyframes": item.get("keyframes")}


OUTPUTS = {"rollout_fake": copy_src, "rollout_keyed": placeholder, "generate_images": image,
           "annotate_camera": poses, "rollout_wan22": wan, "rollout_ltx25": ltx}


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
        outputs, info = OUTPUTS[job["tool"]](job, item, out)
        return outputs, {"rank": job["worker"], "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""), **info}

    protocol.run_items(job, lambda: None, one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
