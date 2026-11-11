"""Run WorldModel prompt precache and rendering commands; publish completed cases and worker exit codes.
Keep renderer outputs in place until exit and include log tails for cases without videos."""
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
DRAIN_S = 5.0


def tail(path: Path, limit: int = 2000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run(command: dict, log: Path, on_line=None) -> int:
    """The command's exit code, once it exits: a process it leaves behind may keep its output pipe open, so the
    output is read on a thread that gets a few seconds to drain."""
    sink = open(log, "ab")
    proc = subprocess.Popen(command["argv"], cwd=command["cwd"], env={**os.environ, **command["env"]},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def read() -> None:
        with sink:
            for line in proc.stdout:
                sink.write(line)
                sink.flush()
                if on_line:
                    on_line(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    code = proc.wait()
    reader.join(DRAIN_S)
    return code


def main() -> int:
    job = protocol.load_job()
    messages, videos = protocol.messages(job), Path(job["videos"])
    cases = {str(item["index"]): item["index"] for item in job["items"]}
    precache_log, render_log = protocol.log_path(job, "precache"), protocol.log_path(job, "run_wbench")
    t0 = time.monotonic()
    code = run(job["precache"], precache_log)
    if code != 0:
        print(f"prompt precache failed:\n{tail(precache_log)}", file=sys.stderr)
        messages.send("worker_exit", worker=0, exit_code=code)
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
    messages.send("worker_exit", worker=0, exit_code=result[0])
    if len(reported) < len(cases):
        print(tail(render_log), file=sys.stderr)
    messages.send("job_done")
    return int(result[0] != 0)


if __name__ == "__main__":
    sys.exit(main())
