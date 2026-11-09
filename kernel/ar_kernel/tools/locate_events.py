"""locate_events: when each of a clip's events happens, from a local vision model, as a GPU job.

The model answers one question per event and frame, at every frame of the clip on the training grid (24
a second). Each job runs a bridge (bridges/locate_vllm.py) that starts `vllm serve` (the `captioner` model, serving
images), asks about every frame of every clip, and always stops it again.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import Context

from .. import fs, telemetry
from ..data.annotation import FPS
from ..data.probe import readable_video
from ..subproc import free_port
from .captioner import clip_host_path, container_path, input_error, stage_clip
from .context import PathError
from .gpu_jobs import items_schema
from .jobs import job_dir
from .server import ToolError, listed, refuse
from .vllm_server import gpu_memory_mib, serve_command

TOOL = "locate_events"
BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "locate_vllm.py"
LEAD_FRAMES = 6         # a span starts this many frames before the first frame its event holds in: early beats late
SYSTEM = (
    "You are the frame reader of a video research pipeline. Each request gives one frame of a video and names one "
    "event. Say what the frame shows about the event.\n\n"
    "Answer with one JSON object and nothing else, with these fields:\n"
    "- happening_now: true when the event is happening in this frame, otherwise false.\n"
    "- already_happened: true when the event happened before this frame and its result is visible in it, "
    "otherwise false.")


def best_span(there: list[bool]) -> tuple[int, int, int] | None:
    """(first frame, last frame, disagreeing frames) of the run of frames that fits `there` best: a frame
    without the event inside the run and a frame with it outside each disagree. The earliest start wins a
    tie, then the longest run. None when no frame has the event."""
    if not any(there):
        return None
    before = [0]                                    # frames with the event before each frame
    for x in there:
        before.append(before[-1] + x)
    best = None
    for i in range(len(there)):
        for j in range(i, len(there)):
            inside = before[j + 1] - before[i]
            off = (j + 1 - i - inside) + (before[-1] - inside)
            if best is None or (off, i, -j) < (best[2], best[0], -best[1]):
                best = (i, j, off)
    return best


def spans(events: list[str], answers: list[list[dict]]) -> list[dict]:
    """Per event: when it starts and ends in the clip (or null), and for how many seconds of the clip the
    answers disagree. `answers[n][k]` is the answer about event n in frame k."""
    out = []
    for text, per in zip(events, answers):
        found = best_span([a["happening_now"] or a["already_happened"] for a in per])
        if found is None:
            out.append({"event": text, "start_s": None, "end_s": None, "disagreeing_s": 0})
            continue
        first, last, off = found
        out.append({"event": text, "start_s": round(max(0, first - LEAD_FRAMES) / FPS, 4),
                    "end_s": round((last + 1) / FPS, 4), "disagreeing_s": round(off / FPS, 4)})
    return out


class LocateEventsBackend:
    """JobQueue backend for locate_events. Args: {"items": [{"video": container path, "events": [str]}]}.
    Result: {"clips": {video: {"events": [...]} | {"error"}}, "load_s", "gpu_memory_mib", "gpu_memory_released"}."""
    name = tool = TOOL

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.gpu_memory = registry, gpu_memory
        self.block = cfg.get(TOOL)
        self.max_items = self.block.get("max_items")

    def server_command(self, port: int) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "locator", mm_limits={"image": 1, "video": 0})

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = job_dir(self.run_dir, job)
        media = work / "media"
        fs.mkdir(media)
        results: dict[str, dict] = {}
        items = []
        for i, item in enumerate(job.args["items"]):
            video = item["video"]
            dst = media / f"{i}{Path(video).suffix}"
            try:
                stage_clip(caller, video, dst)
                info = readable_video(dst)
            except (PathError, OSError) as exc:
                telemetry.error(exc, "the item's input error")
                results[video] = {"error": input_error(exc)[:500]}
                continue
            except ValueError as exc:
                telemetry.error(exc, "the item's error")
                results[video] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                continue
            items.append({"index": i, "video": str(dst), "width": info.width, "height": info.height,
                          "events": item["events"]})
        ran = {"ready": None, "gpu_memory_mib": None, "gpu_memory_released": None}
        try:
            if items:
                from .gpu_jobs import run_bridge
                port = free_port()
                spec = {"bridge": BRIDGE, "env": c["env"], "job": {
                    "serve": {"argv": self.server_command(port), "env": {"HF_HUB_OFFLINE": "1"}, "port": port},
                    "system": SYSTEM, "fps": FPS, "concurrent_requests": c["concurrent_requests"],
                    **{k: self.block[k] for k in ("frame_width", "request_timeout_s", "reasoning_effort")}}}
                events = {item["index"]: item["events"] for item in items}

                def row(m: dict) -> dict:     # the agent gets the spans; each frame's answer stays in the messages
                    return {"events": spans(events[m["index"]], m["info"]["answers"])} if m["ok"] \
                        else {"error": m["error"][:500]}     # whole in telemetry

                ran = run_bridge(self, job, work, spec, items, cancel, report, row, done=len(results))
                for i, r in ran["rows"].items():
                    results[job.args["items"][i]["video"]] = {"error": r["error"]} if "index" in r else r
        finally:
            fs.rmtree(media, ignore_errors="the job's own links")
        ready = ran["ready"]
        return {"clips": {item["video"]: results[item["video"]] for item in job.args["items"]},
                "load_s": ready["info"]["load_s"] if ready else None,
                "gpu_memory_mib": ran["gpu_memory_mib"], "gpu_memory_released": ran["gpu_memory_released"]}


def submit(q, caller, items) -> dict:
    """Validate a locate_events call and queue its job; returns {"job_id"}."""
    items = listed(caller, items, "items")
    if not items:
        raise ToolError("items is empty")
    cap = getattr(q.backends.get(TOOL), "max_items", None)
    if cap and len(items) > cap:
        raise ToolError(f"at most {cap} items per job: got {len(items)}; send the rest in another job")
    bad, seen = {}, set()
    for n, item in enumerate(items):
        if not (isinstance(item, dict) and set(item) == {"video", "events"} and isinstance(item["video"], str)):
            bad[n] = "an item is {'video': path, 'events': [sentences]}"
            continue
        events = item["events"]
        if not (isinstance(events, list) and events and all(isinstance(e, str) and e.strip() for e in events)):
            bad[n] = "events must be a non-empty list of sentences"
        elif container_path(item["video"]) in seen:
            bad[n] = "this video is listed twice: give all its events in one item"
        else:
            try:
                readable_video(clip_host_path(caller, item["video"]))
            except PathError as exc:
                telemetry.error(exc, "a bad item")
                bad[n] = str(exc)
            except ValueError as exc:
                telemetry.error(exc, "a bad item")
                bad[n] = f"not a readable video: {exc}"[:300]
        seen.add(container_path(item["video"]))
    refuse(caller, TOOL, items, bad)
    return {"job_id": q.submit(caller, TOOL, {"items": [
        {"video": container_path(i["video"]), "events": [" ".join(e.split()) for e in i["events"]]} for i in items]})}


Items = Annotated[list[dict[str, Any]] | str, items_schema("the clips to look at, one item each", {
    "video": {"type": "string", "description": "a video file under /workspace"},
    "events": {"type": "array", "items": {"type": "string"},
               "description": "the events to find, one sentence each, as you would write them in the annotation"}},
    ["video", "events"])]


def register_locate_tool(mcp, kit, q) -> None:
    cap = getattr(q.backends.get(TOOL), "max_items", None)

    @mcp.tool(name=TOOL, description="Find when each event happens in a clip, with the kernel's local vision "
              "model. A GPU job: returns {job_id} at once; collect "
              "it with job_wait. Every job loads the model again, so send every clip in one call. The finished "
              "job's result file (job_wait gives its path) holds `clips`: each video mapped to {events} or "
              "{error}. Each event has `start_s` and `end_s`, the seconds of the clip between which the event is "
              "happening or its result is visible, both null when the event was not found, and "
              "`disagreeing_s`: how many seconds of the clip the model's answers contradict that span. "
              "More in knowledge file locate_events."
              + (f" At most {cap} clips per job." if cap else ""))
    async def locate_events(items: Items, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, TOOL, {"items": items}, lambda c: submit(q, c, items))
