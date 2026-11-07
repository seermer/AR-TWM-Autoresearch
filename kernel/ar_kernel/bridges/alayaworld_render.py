"""rollout_alayaworld bridge (runs in the alayaworld env): WorldModel's prompt precache, then its run_wbench.py,
both as they are.

python alayaworld_render.py --job <work>/job.json (bridges/protocol.py). Job file: precache and render ({"argv",
"cwd", "env"}: the two commands), videos (WorldModel's output folder). A `[Validation] case_start case=<id>` line
of the render starts that case; a case is done when its `case_<id>_combined.json` exists (WorldModel writes it
after the video and the camera path). The files stay where WorldModel wrote them: run_wbench checks at its end
that every video is there. Every case gets an item message, so the bridge exits 0 once the render has ended.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import protocol

MARKER = re.compile(rb"\[Validation\] case_start case=(\S+) rank=\d+")


def tail(path: Path, limit: int = 2000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run(command: dict, log: Path, on_line=None) -> int:
    with open(log, "ab") as sink:
        proc = subprocess.Popen(command["argv"], cwd=command["cwd"], env={**os.environ, **command["env"]},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sink.write(line)
            sink.flush()
            if on_line:
                on_line(line)
        return proc.wait()


def main() -> int:
    job = protocol.load_job()
    messages, videos = protocol.messages(job), Path(job["videos"])
    cases = {str(item["index"]): item["index"] for item in job["items"]}
    precache_log, render_log = protocol.log_path(job, "precache"), protocol.log_path(job, "run_wbench")
    t0 = time.monotonic()
    code = run(job["precache"], precache_log)
    if code != 0:
        print(f"prompt precache failed (exit {code}):\n{tail(precache_log)}", file=sys.stderr)
        return 1
    messages.send("ready", info={"precache_s": round(time.monotonic() - t0, 2)})

    started: dict[str, float] = {}
    reported: set[str] = set()

    def on_line(line: bytes) -> None:
        found = MARKER.search(line)
        case = found and found.group(1).decode()
        if case in cases and case not in started:
            started[case] = time.time()
            messages.send("item_started", index=cases[case], log=str(render_log))

    def report_done() -> None:
        for case, index in cases.items():
            stem = videos / f"case_{case}_combined"
            if case not in reported and stem.with_suffix(".json").exists():
                reported.add(case)
                messages.send("item", index=index, ok=True, log=str(render_log), started=started.get(case),
                              ended=time.time(), info={},
                              outputs={"video": str(stem.with_suffix(".mp4")),
                                       "camera": str(stem.with_name(stem.name + "_camera.npz"))})

    result: list[int] = []
    render = threading.Thread(target=lambda: result.append(run(job["render"], render_log, on_line)), daemon=True)
    render.start()
    while render.is_alive():
        report_done()
        time.sleep(1.0)
    render.join()
    report_done()
    for case, index in cases.items():
        if case not in reported:
            messages.send("item", index=index, ok=False, log=str(render_log), started=started.get(case),
                          ended=time.time(), error=f"no video rendered (run_wbench exit {result[0]}); log tail:\n"
                                                   f"{tail(render_log)}")
    messages.send("job_done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
