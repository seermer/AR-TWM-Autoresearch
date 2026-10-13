"""locate_events: when each of a clip's events happens, from a local vision model, as a GPU job.

The clip is cut into one-second pieces. Each piece becomes one sheet of its frames with the time
under each, and the model answers one question set per sheet. Each job starts `vllm serve` (the
`captioner` model, serving images), asks about every piece of every clip, and always stops it again.
"""
from __future__ import annotations

import base64
import io
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import httpx
from mcp.server.mcpserver import Context
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import create_model

from ..data.probe import probe_video
from ..subproc import free_port
from .captioner import clip_host_path, container_path, stage_clip
from .context import PathError
from .gpu_jobs import items_schema
from .server import ToolError, listed, refuse
from .vllm_server import VllmServer, gpu_memory_mib, serve_command, wait_gpu_release

TOOL = "locate_events"
SHEET_FPS, GRID = 16, 4             # a piece's sheet: 16 frames a second, 4 x 4
GAP, CAPTION_H = 12, 28             # white space between frames, and the caption strip under each
SHEET_TIMES = {f"{i / SHEET_FPS:.2f}": i / SHEET_FPS for i in range(SHEET_FPS)}     # a caption -> its time
QUESTION = (
    "The sheet shows {n} consecutive frames of a video, in reading order (left to right, top to bottom). "
    "Under each frame is its time in seconds. For each event below, answer three things.\n"
    "- happening_now: is the event happening in these frames?\n"
    "- already_happened: has the event already happened before these frames, with its result visible in them?\n"
    "- start: the time under the first frame in which the event is happening, or null when the event does "
    "not start in these frames.\n\nEvents:\n{events}")


def pieces(frames: int, fps: float) -> list[tuple[float, float]]:
    """(start, end) of each one-second piece. A last piece shorter than a second is moved back to
    cover the clip's final full second."""
    duration = frames / fps
    whole = int(duration)
    out = [(float(k), float(k + 1)) for k in range(whole)]
    if duration - whole > 1e-6:
        out.append((max(0.0, duration - 1.0), duration))
    return out


def sheet_times(start: float, end: float) -> list[float]:
    """Seconds from the piece's start of each frame on its sheet."""
    return [i / SHEET_FPS for i in range(SHEET_FPS) if start + i / SHEET_FPS < end - 1e-6]


def decode(video: Path, width: int) -> list[Image.Image]:
    """Every frame of the clip, `width` pixels wide."""
    info = probe_video(video)
    height = round(info.height * width / info.width / 2) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={width}:{height}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True,
                         timeout=600).stdout
    size = width * height * 3
    return [Image.frombytes("RGB", (width, height), raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]


def sheet(frames: list[Image.Image], fps: float, start: float, times: list[float]) -> Image.Image:
    """One piece as a grid of its frames (the nearest source frame for each time), each with its time
    under it, on white."""
    w, h = frames[0].size
    font = ImageFont.load_default(size=CAPTION_H - 6)
    out = Image.new("RGB", (GRID * w + (GRID + 1) * GAP, GRID * (h + CAPTION_H) + (GRID + 1) * GAP), "white")
    draw = ImageDraw.Draw(out)
    for i, t in enumerate(times):
        x, y = GAP + (i % GRID) * (w + GAP), GAP + (i // GRID) * (h + CAPTION_H + GAP)
        out.paste(frames[min(len(frames) - 1, round((start + t) * fps))], (x, y))
        draw.text((x + w // 2, y + h + CAPTION_H // 2), f"{t:.2f}", fill="black", font=font, anchor="mm")
    return out


def answer_model(n_events: int, times: list[float]):
    """What one piece's answer must look like: per event two yes/no answers and a start that is one of
    the sheet's captions, or none."""
    start = Optional[Literal[tuple(f"{t:.2f}" for t in times)]]
    event = create_model("Event", happening_now=(bool, ...), already_happened=(bool, ...), start=(start, ...))
    return create_model("Answer", **{f"event_{i + 1}": (event, ...) for i in range(n_events)})


def best_span(there: list[bool]) -> tuple[int, int, int] | None:
    """(first piece, last piece, disagreeing pieces) of the run of pieces that fits `there` best: a
    piece without the event inside the run and a piece with it outside each disagree once. The earliest
    start wins a tie, then the longest run. None when no piece has the event."""
    if not any(there):
        return None
    n, best = len(there), None
    for i in range(n):
        for j in range(i, n):
            off = sum(not x for x in there[i:j + 1]) + sum(there[:i]) + sum(there[j + 1:])
            if best is None or (off, i, -j) < (best[2], best[0], -best[1]):
                best = (i, j, off)
    return best


def spans(events: list[str], cut: list[tuple[float, float]], answers: list[dict]) -> list[dict]:
    """Per event: when it starts and ends in the clip (or null), and how many pieces disagree."""
    out = []
    for n, text in enumerate(events, start=1):
        per = [a[f"event_{n}"] for a in answers]
        found = best_span([p["happening_now"] or p["already_happened"] for p in per])
        if found is None:
            out.append({"event": text, "start_s": None, "end_s": None, "disagreeing_pieces": 0})
            continue
        first, last, off = found
        picked = SHEET_TIMES.get(per[first]["start"], 0.0)
        out.append({"event": text, "start_s": round(cut[first][0] + picked, 4),
                    "end_s": round(cut[last][1], 4), "disagreeing_pieces": off})
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

    def ask(self, client, image: Image.Image, events: list[str], times: list[float]) -> tuple[dict, str | None]:
        """(the parsed answer of one piece, the model's reasoning)."""
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=92)
        url = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
        question = QUESTION.format(n=len(times), events="\n".join(f"{i}. {e}" for i, e in enumerate(events, 1)))
        done = client.chat.completions.parse(
            model="locator", temperature=0.0, reasoning_effort=self.block["reasoning_effort"],
            response_format=answer_model(len(events), times),
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                   {"type": "text", "text": question}]}])
        message = done.choices[0].message
        if message.parsed is None:
            raise ValueError(f"the model gave no answer in the asked shape: {message.content!r}"[:500])
        return message.parsed.model_dump(), getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

    def run(self, job, cancel: threading.Event, report) -> dict:
        c = self.cfg.get("captioner")
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = self.run_dir / "jobs" / job.id
        media = work / "media"
        media.mkdir(parents=True)
        results: dict[str, dict] = {}
        calls = []                                  # (video, events, piece number, sheet, its times)
        cuts: dict[str, list] = {}
        for i, item in enumerate(job.args["items"]):
            video = item["video"]
            dst = media / f"{i}{Path(video).suffix}"
            try:
                stage_clip(caller, video, dst)
                info = probe_video(dst)
                frames = decode(dst, int(self.block["frame_width"]))
                cuts[video] = pieces(len(frames), info.fps)
                for k, (start, end) in enumerate(cuts[video]):
                    times = sheet_times(start, end)
                    calls.append((video, item["events"], k, sheet(frames, info.fps, start, times), times))
            except (PathError, OSError, subprocess.SubprocessError, ValueError) as exc:
                results[video] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        if not calls:
            return {"clips": results, "load_s": None, "gpu_memory_mib": None, "gpu_memory_released": None}
        port, before = free_port(), self.gpu_memory(self.gpus)
        server = VllmServer(c["env"], self.server_command(port), self.gpus, port, cwd=work,
                            log_path=work / "vllm.log", recorder=self.recorder, node=job.node,
                            phase=TOOL, poll_s=self.poll_s, label="locate server")
        load_s, answers, failed, lock, done = None, {}, {}, threading.Lock(), [0]
        try:
            load_s = server.start(float(c["startup_timeout_s"]), cancel)
            # no proxy lookup: the server is on this machine
            with httpx.Client(timeout=float(self.block["request_timeout_s"]), trust_env=False) as http:
                client = OpenAI(base_url=f"{server.base_url}/v1", api_key="none", http_client=http, max_retries=0)

                def one(video, events, k, image, times) -> None:
                    if cancel.is_set() or load_s is None:
                        return
                    try:
                        answer, reasoning = self.ask(client, image, events, times)
                    except Exception as exc:        # noqa: BLE001 -- one piece's failure is its clip's error
                        answer, reasoning = None, f"{type(exc).__name__}: {exc}"[:2000]
                    with lock:
                        if answer is None:
                            failed[video] = reasoning
                        else:
                            answers.setdefault(video, {})[k] = answer
                        done[0] += 1
                        self.recorder.event("locate.piece", node=job.node, component="tools", job_id=job.id,
                                            ok=answer is not None,
                                            payload={"video": video, "piece": k, "answer": answer, "reasoning": reasoning})
                        report({"asked": done[0], "total": len(calls)})

                with ThreadPoolExecutor(max_workers=max(1, min(int(c["max_num_seqs"]), len(calls)))) as pool:
                    for f in [pool.submit(one, *call) for call in calls]:
                        f.result()
        finally:
            exit_code = server.stop()
            shutil.rmtree(media, ignore_errors=True)
            self.recorder.event("locate.server_stopped", node=job.node, component="tools", job_id=job.id,
                                exit_code=exit_code)
        for item in job.args["items"]:
            video = item["video"]
            if video in failed:
                results[video] = {"error": failed[video]}
            elif len(answers.get(video, {})) == len(cuts.get(video, [])) and video in cuts:
                results[video] = {"events": spans(item["events"], cuts[video],
                                                  [answers[video][k] for k in range(len(cuts[video]))])}
            else:
                results.setdefault(video, {"error": "not asked (cancelled)"})
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
                clip_host_path(caller, item["video"])
            except PathError as exc:
                bad[n] = str(exc)
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
              "model, which looks at the clip second by second. A GPU job: returns {job_id} at once; collect "
              "it with job_wait. Every job loads the model again, so send every clip in one call. The finished "
              "job's result file (job_wait gives its path) holds `clips`: each video mapped to {events} or "
              "{error}. Each event has `start_s` and `end_s`, the seconds of the clip between which the event is "
              "happening or its result is visible, both null when the event was not found, and "
              "`disagreeing_pieces`: how many seconds of the clip the model's answers contradict that span. "
              "The spans are for you to set the turns' starts in the annotation: skill locate_events."
              + (f" At most {cap} clips per job." if cap else ""))
    async def locate_events(items: Items, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, TOOL, {"items": items}, lambda c: submit(q, c, items))
