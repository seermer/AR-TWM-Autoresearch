"""Run MiniMax H3 with one torchrun rank per GPU and a shared block-offloaded model.
Rank zero publishes outputs; any rank failure ends the job."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

import protocol


def request_of(item: dict, request: dict, height: int, width: int) -> dict:
    # the kernel has checked that a first frame is exactly width x height, so the runner uses it as it is
    image = Image.open(item["image"]) if item.get("image") else None
    return {"task": "i2av" if image else "t2av", "prompt": request["prompt"], "seed": int(item["seed"]),
            "size": [height, width], "num_frames": int(request["frames"]),
            "image_path": image, "last_frame_path": None}


def main() -> int:
    job = protocol.load_job()
    if "RANK" not in os.environ:             # the bridge itself: torchrun sets RANK in each rank
        return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                               f"--nproc_per_node={len(job['gpus'])}", os.path.abspath(__file__),
                               "--job", job["file"]]).returncode
    rank, ranks, p = int(os.environ["RANK"]), len(job["gpus"]), job["paths"]
    out, (height, width) = Path(job["out"]), p["size"]
    requests = json.loads(Path(p["requests"]).read_text(encoding="utf-8"))
    messages = protocol.messages(job) if rank == 0 else None

    import torch.distributed as dist
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config, init_parallel
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    t0 = time.monotonic()
    # the fl2av variant is the base transformer: it serves t2av and i2av too
    startup = build_startup_config({"model_cls": "minimax_h3", "model_variant": "fl2av", "model_path": p["weights"],
                                    "config_json": p["config"]})
    if startup["parallel"]:
        PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None).init_parallel_env()
        init_parallel(startup)
    runner = build_runner(startup)
    if messages:
        messages.send("ready", info={"load_s": round(time.monotonic() - t0, 2)})

    for item in job["items"]:
        index, started = item["index"], time.time()
        logs = [str(protocol.log_path(job, f"{index}.rank{r}")) for r in range(ranks)]
        if messages:
            messages.send("item_started", index=index, log=logs)
        path = out / f"{index}.mp4"
        try:
            with protocol.item_log(job, f"{index}.rank{rank}"):
                request = request_of(item, requests[str(index)], height, width)
                runner.run_request(runner.prepare_request({**request, "save_result_path": str(path)}))
        except Exception as exc:
            if messages:
                messages.send("item", index=index, ok=False, error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc(), started=started, ended=time.time(), log=logs)
            raise
        if messages:
            messages.send("item", index=index, ok=True, outputs={"video": str(path)},
                          info={"seconds": round(time.time() - started, 2)}, started=started, ended=time.time(),
                          log=logs)
    if messages:
        messages.send("job_done")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
