"""Locate events across decoded frames with one local server per job; retain answers and reasoning."""
from __future__ import annotations

import base64
import io
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from pydantic import BaseModel

import protocol
from caption_vllm import client, serve, stop


class Answer(BaseModel):
    happening_now: bool
    already_happened: bool


def decode(video: str, width: int, height: int, frame_width: int, fps: int) -> list[Image.Image]:
    """The clip's frames at `fps` (the training grid), `frame_width` pixels wide; (width, height) is the clip's."""
    h = round(height * frame_width / width / 2) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},scale={frame_width}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    size = frame_width * h * 3
    return [Image.frombytes("RGB", (frame_width, h), raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]


def data_url(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


def ask(client, url: str, system: str, event: str, reasoning_effort: str) -> tuple[dict, str | None]:
    """(the parsed answer about one event in one frame, the model's reasoning). `url` is the frame."""
    done = client.chat.completions.parse(
        model="locator", reasoning_effort=reasoning_effort, response_format=Answer,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                               {"type": "text", "text": f"Event: {event}"}]}])
    message = done.choices[0].message
    if message.parsed is None:
        raise ValueError(f"the model gave no answer in the asked shape: {message.content!r}")
    return message.parsed.model_dump(), getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)


def main() -> int:
    job = protocol.load_job()
    messages, log = protocol.messages(job), str(protocol.log_path(job, "vllm"))
    items = {item["index"]: item for item in job["items"]}
    starts = {}

    def frames(item) -> list[str]:
        return [data_url(f) for f in decode(item["video"], item["width"], item["height"], job["frame_width"], job["fps"])]

    def failed(index, exc) -> None:
        messages.send("item", index=index, ok=False, error=f"{type(exc).__name__}: {exc}",
                      traceback="".join(traceback.format_exception(exc)), started=starts.get(index), ended=time.time(),
                      log=log)

    with ThreadPoolExecutor(max_workers=1) as decoder:          # the frames are decoded while the model loads
        decoded = decoder.submit(lambda: {i: _attempt(frames, item) for i, item in items.items()})
        proc, url, key, load_s = serve(job)
        urls = decoded.result()
    try:
        messages.send("ready", info={"load_s": round(load_s, 2)})
        calls, answers, reasonings, errors, left = [], {}, {}, {}, {}
        for index, got in urls.items():
            starts[index] = time.time()
            messages.send("item_started", index=index, log=log)
            if isinstance(got, Exception):
                failed(index, got)
                continue
            events = items[index]["events"]
            answers[index] = [[None] * len(got) for _ in events]
            reasonings[index] = [[None] * len(got) for _ in events]
            left[index] = len(events) * len(got)
            calls += [(index, n, k, u) for n in range(len(events)) for k, u in enumerate(got)]
        lock = threading.Lock()

        def done(index) -> None:
            info = {"answers": answers[index], "reasonings": reasonings[index]}
            if index in errors:
                messages.send("item", index=index, ok=False, error=errors[index], info=info, started=starts[index],
                              ended=time.time(), log=log)
            else:
                messages.send("item", index=index, ok=True, outputs={}, info=info, started=starts[index],
                              ended=time.time(), log=log)

        def one(c, index, n, k, u) -> None:
            try:
                answer, reasoning = ask(c, u, job["system"], items[index]["events"][n], job["reasoning_effort"])
            except Exception as exc:        # noqa: BLE001 -- one frame's failure is its clip's error
                answer, reasoning = None, f"{type(exc).__name__}: {exc}"
            with lock:
                answers[index][n][k], reasonings[index][n][k] = answer, reasoning
                if answer is None:
                    errors.setdefault(index, reasoning)
                left[index] -= 1
                if left[index] == 0:
                    done(index)

        for index in [i for i, n in left.items() if n == 0]:          # a clip without frames
            done(index)
        with client(url, key, float(job["request_timeout_s"])) as c, \
                ThreadPoolExecutor(max_workers=max(1, min(int(job["concurrent_requests"]), len(calls)))) as pool:
            for f in [pool.submit(one, c, *call) for call in calls]:
                f.result()
    finally:
        stop(proc)
    messages.send("job_done")
    return 0


def _attempt(fn, *args):
    """fn(*args), or the exception it raised."""
    try:
        return fn(*args)
    except Exception as exc:          # noqa: BLE001 -- reported as that clip's error
        return exc


if __name__ == "__main__":
    sys.exit(main())
