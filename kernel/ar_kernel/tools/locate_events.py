"""locate_events: when each of a clip's events happens, from a local vision model, as a GPU job.

The model answers one question per event and frame, at every frame of the clip on the training grid (24
a second). Each job starts `vllm serve` (the `captioner` model, serving images), asks about every frame
of every clip, and always stops it again.
"""
from __future__ import annotations

import base64
import io
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any

from mcp.server.mcpserver import Context
from PIL import Image
from pydantic import BaseModel

from ..data.annotation import FPS
from ..data.probe import probe_video, readable_video
from ..subproc import free_port
from .captioner import clip_host_path, container_path, input_error, stage_clip
from .context import PathError
from .gpu_jobs import items_schema
from .jobs import job_dir
from .server import ToolError, listed, refuse
from .vllm_server import VllmServer, gpu_memory_mib, local_client, serve_command, wait_gpu_release

TOOL = "locate_events"
LEAD_FRAMES = 6         # a span starts this many frames before the first frame its event holds in: early beats late
SYSTEM = (
    "You are the frame reader of a video research pipeline. Each request gives one frame of a video and names one "
    "event. Say what the frame shows about the event.\n\n"
    "Answer with one JSON object and nothing else, with these fields:\n"
    "- happening_now: true when the event is happening in this frame, otherwise false.\n"
    "- already_happened: true when the event happened before this frame and its result is visible in it, "
    "otherwise false.")


class Answer(BaseModel):
    happening_now: bool
    already_happened: bool


def decode(video: Path, width: int) -> list[Image.Image]:
    """The clip's frames at 24 a second (the training grid), `width` pixels wide."""
    info = probe_video(video)
    height = round(info.height * width / info.width / 2) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FPS},scale={width}:{height}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    size = width * height * 3
    return [Image.frombytes("RGB", (width, height), raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]


def data_url(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


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

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib, poll_s: float = 2.0) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory, self.poll_s = registry, recorder, gpu_memory, poll_s
        self.block = cfg.get(TOOL)
        self.max_items = self.block.get("max_items")

    def server_command(self, port: int) -> list[str]:
        return serve_command(self.cfg.get("captioner"), self.gpus, port, "locator", mm_limits={"image": 1, "video": 0})

    def ask(self, client, url: str, event: str) -> tuple[dict, str | None]:
        """(the parsed answer about one event in one frame, the model's reasoning). `url` is the frame."""
        done = client.chat.completions.parse(
            model="locator", reasoning_effort=self.block["reasoning_effort"], response_format=Answer,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                   {"type": "text", "text": f"Event: {event}"}]}])
        message = done.choices[0].message
        if message.parsed is None:
            raise ValueError(f"the model gave no answer in the asked shape: {message.content!r}"[:500])
        return message.parsed.model_dump(), getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = job_dir(self.run_dir, job)
        media = work / "media"
        media.mkdir(parents=True)
        results: dict[str, dict] = {}
        calls = []                                  # (video, event number, frame number, the frame as a data URL)
        frames_of: dict[str, int] = {}
        port, before = free_port(), self.gpu_memory(self.gpus)
        server = VllmServer(c["env"], self.server_command(port), self.gpus, port, cwd=work,
                            log_path=work / "vllm.log", recorder=self.recorder, node=job.node,
                            phase=TOOL, poll_s=self.poll_s, label="locate server")
        events = {item["video"]: item["events"] for item in job.args["items"]}
        load_s, answers, failed, lock, done = None, {}, {}, threading.Lock(), [0]
        asked: dict[tuple, list] = {}               # (video, event number) -> each frame's (answer, reasoning) once asked
        abandon = threading.Event()                 # ends a start still waiting when there is nothing to ask
        loader = ThreadPoolExecutor(max_workers=1)  # the model loads while the frames are decoded
        starting = loader.submit(server.start, SimpleNamespace(is_set=lambda: cancel.is_set() or abandon.is_set()))
        try:
            for i, item in enumerate(job.args["items"]):
                video = item["video"]
                dst = media / f"{i}{Path(video).suffix}"
                try:
                    stage_clip(caller, video, dst)
                except (PathError, OSError) as exc:
                    results[video] = {"error": input_error(exc)[:500]}
                    continue
                try:
                    urls = [data_url(frame) for frame in decode(dst, int(self.block["frame_width"]))]
                    frames_of[video] = len(urls)
                    calls += [(video, n, k, url) for n in range(len(item["events"])) for k, url in enumerate(urls)]
                except (PathError, OSError, subprocess.SubprocessError, ValueError) as exc:
                    results[video] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            if calls:
                load_s = starting.result()
                with local_client(server.base_url, float(self.block["request_timeout_s"]), server.api_key) as client:
                    def one(video, n, k, url) -> None:
                        if cancel.is_set() or load_s is None:
                            return
                        try:
                            answer, reasoning = self.ask(client, url, events[video][n])
                        except Exception as exc:        # noqa: BLE001 -- one frame's failure is its clip's error
                            answer, reasoning = None, f"{type(exc).__name__}: {exc}"[:2000]
                        with lock:
                            if answer is None:
                                failed[video] = reasoning
                            else:
                                answers.setdefault(video, {})[n, k] = answer
                            frames = asked.setdefault((video, n), [None] * frames_of[video])
                            frames[k] = (answer, reasoning)
                            if None not in frames:
                                self.recorder.event("locate.event", node=job.node, component="tools", job_id=job.id,
                                                    ok=all(a for a, _ in frames), payload={
                                                        "video": video, "event": n, "answers": [a for a, _ in frames],
                                                        "reasonings": [r for _, r in frames]})
                            done[0] += 1
                            report({"asked": done[0], "total": len(calls)})

                    with ThreadPoolExecutor(max_workers=max(1, min(int(c["concurrent_requests"]), len(calls)))) as pool:
                        for f in [pool.submit(one, *call) for call in calls]:
                            f.result()
        finally:
            abandon.set()
            loader.shutdown()                       # start has returned, so stop finds its process
            exit_code = server.stop()
            shutil.rmtree(media, ignore_errors=True)
            self.recorder.event("locate.server_stopped", node=job.node, component="tools", job_id=job.id,
                                exit_code=exit_code)
        for item in job.args["items"]:
            video = item["video"]
            if video in failed:
                results[video] = {"error": failed[video]}
            elif video in frames_of and len(answers.get(video, {})) == len(item["events"]) * frames_of[video]:
                results[video] = {"events": spans(item["events"], [
                    [answers[video][n, k] for k in range(frames_of[video])] for n in range(len(item["events"]))])}
            else:
                results.setdefault(video, {"error": "not asked (canceled)"})
        timeout = float(c["memory_release_timeout_s"])
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, before,
                                           min(timeout, 5.0) if cancel.is_set() else timeout)
        return {"clips": {item["video"]: results[item["video"]] for item in job.args["items"]}, "load_s": load_s,
                "gpu_memory_mib": {"before": before, "after": after}, "gpu_memory_released": released}


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
                bad[n] = str(exc)
            except ValueError as exc:
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
