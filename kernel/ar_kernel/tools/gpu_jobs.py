"""Shared plumbing for the GPU data-source jobs: rollout_*, annotate_camera and
generate_images.

A backend stages the agent's input files into a kernel-private job dir (never trusting the
path between check and use), runs the tool's bridge (bridges/) in its own conda env, and publishes each finished
item into the caller's staging dir as it is reported, by hard links placed through O_NOFOLLOW directory fds, so a
link the agent plants cannot redirect a kernel write. A per-item failure is an item error, not a failed job.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any

from mcp.server.mcpserver import Context
from PIL import Image
from pydantic import Field, WithJsonSchema

from ..data.annotation import ACTIONS, PERSPECTIVES
from ..data.probe import TARGET_ASPECT
from ..faults import disk_fault
from ..isolation import EXCLUDED_PROMPT, copies_held_out, strings
from ..subproc import file_tail
from .captioner import clip_host_path, container_path, input_error, stage_clip
from .context import STAGING, PathError
from .hf_tools import publish_file
from .jobs import job_dir, run_cancellable
from .server import FROM_FILE, ToolError, listed, refuse
from .vllm_server import gpu_memory_mib, wait_gpu_release


def items_schema(description: str, properties: dict, required: list[str]) -> WithJsonSchema:
    """The listed JSON schema of an `items` list. Only the schema: the values still reach the tool
    as plain dicts and each backend's check_args validates them, so a bad value is a recorded
    tool.error rather than a failure inside the MCP layer."""
    return WithJsonSchema({"description": f"{description}, {FROM_FILE}", "anyOf": [
        {"type": "array", "items": {"type": "object", "properties": properties, "required": required,
                                    "additionalProperties": False}},
        {"type": "string"}]})


def _str(description: str) -> dict:
    return {"type": "string", "description": description}


_SEED = {"type": "integer", "description": "random seed; the same item and seed give the same output"}
_FRAME = _str("first frame: an image file under /workspace (any size; it is cropped and resized)")
ImageItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the images to make, one item each",
    {"prompt": _str("what the image shows"), "seed": _SEED}, ["prompt", "seed"])]
ClipItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"prompt": _str("what the clip shows"), "image": _FRAME, "seed": _SEED}, ["prompt", "seed"])]
KEYFRAMES_SCHEMA = {"type": "array", "description": "an image the clip must show at a given frame (at most one)",
                    "maxItems": 1, "items": {
    "type": "object", "additionalProperties": False, "required": ["image", "frame"], "properties": {
        "image": _str("an image file under /workspace, exactly the clip's width x height (it is used as it is)"),
        "frame": {"type": "integer", "description": "0-based frame index; 0 is the first frame, -1 the last"}}}}
LtxItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"prompt": _str("what the clip shows"), "keyframes": KEYFRAMES_SCHEMA, "seed": _SEED}, ["prompt", "seed"])]
_TURN = {"type": "object", "additionalProperties": False, "required": ["action", "perspective"], "properties": {
    "action": {"type": "string", "enum": list(ACTIONS),
               "description": "the navigation action of this turn; `stop` holds still"},
    "event": _str("one sentence: something that happens during this turn. Leave it out when nothing does"),
    "perspective": {"type": "string", "enum": list(PERSPECTIVES),
                    "description": "whose eyes the turn is seen through. A turn whose perspective differs from "
                                   "the turn before it switches the view: the kernel adds the sentence that says so, "
                                   "and its action must be `stop`"}}}
_SCENE = _str("the scene: place, objects, light")
_CHARACTER = _str("the subject, if there is one")
_TURNS = {"type": "array", "items": _TURN, "description": "the clip, turn by turn"}
H3Items = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"scene_prompt": _SCENE, "character_prompt": _CHARACTER, "seed": _SEED,
     "image": _str("first frame: an image file under /workspace, exactly 960x544 (it is used as it is)"),
     "seconds": {"type": "number", "description": "the clip's length in seconds; the tool description gives the range"},
     "turns": _TURNS},
    ["scene_prompt", "seed", "turns"])]
WorldItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"image": _str("first frame: an image file under /workspace (.jpg, .png, ...), exactly 960x544 (it is "
                   "used as it is)"),
     "scene_prompt": _SCENE, "character_prompt": _CHARACTER, "seed": _SEED,
     "subject_mask": _str("an image under /workspace of the first frame's size, white where the subject is in "
                          "the first frame"),
     "turns": _TURNS},
    ["image", "scene_prompt", "seed", "turns"])]


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")                 # alaya/data/wbench.py _IMAGE_EXTS


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_item_seed(item: dict) -> None:
    """An item's `seed` must be an int; integer text ("7") is taken as that int, in place."""
    if "seed" not in item:
        raise ToolError("seed is missing; give an int")
    seed = item["seed"]
    if isinstance(seed, str) and seed.strip().lstrip("+-").isdigit():
        item["seed"] = seed = int(seed)
    if not is_int(seed):
        raise ToolError(f"seed must be an int: got {seed!r}")


def check_keyframes(item: dict) -> None:
    """An item's optional `keyframes`: at most one {'image': path, 'frame': int}, frame >= -1 (-1 = the last)."""
    frames = item.get("keyframes")
    if frames is None:
        return
    if not isinstance(frames, list):
        raise ToolError("keyframes must be a list of {'image': path, 'frame': int}")
    if len(frames) > 1:
        raise ToolError(f"at most one keyframe per item: got {len(frames)}")
    for k in frames:
        if not (isinstance(k, dict) and set(k) == {"image", "frame"} and isinstance(k["image"], str)
                and is_int(k["frame"])):
            raise ToolError(f"each keyframe must be {{'image': path, 'frame': int}}: got {k!r}")
        if not k["image"].lower().endswith(IMAGE_EXTS):
            raise ToolError(f"a keyframe's image must be an image file ({', '.join(IMAGE_EXTS)}): got {k['image']!r}")
        if k["frame"] < -1:
            raise ToolError(f"frame must be -1 (the last frame) or a frame index from 0: got {k['frame']}")


def _near_16_9(cfg, width: int, height: int) -> bool:
    return abs(width / height - TARGET_ASPECT) <= TARGET_ASPECT * float(cfg.get("ingest.aspect_tolerance"))


def check_size(cfg, width, height) -> None:
    """A job's `width` x `height`: within the ingest tolerance of 16:9, the first check of every tool."""
    if not (is_int(width) and is_int(height) and width > 0 and height > 0):
        raise ToolError(f"width and height must be positive ints: got {width!r} and {height!r}")
    if not _near_16_9(cfg, width, height):
        raise ToolError(f"{width}x{height} is not within {cfg.get('ingest.aspect_tolerance'):.0%} of 16:9")


def check_listed_size(cfg, args: dict, resolutions: list) -> None:
    """Fill `height`/`width` in from the first of the tool's `resolutions` ([height, width] pairs), then
    check the size: 16:9 first, then that it is on the list."""
    h, w = resolutions[0]
    size = [args.setdefault("height", h), args.setdefault("width", w)]
    check_size(cfg, size[1], size[0])
    if size not in [list(r) for r in resolutions]:
        raise ToolError(f"width x height must be one of the resolutions {sizes_text(resolutions)}: "
                        f"got {size[1]}x{size[0]}")


def sizes_text(resolutions: list) -> str:
    return ", ".join(f"{w}x{h}" for h, w in resolutions)


def check_frame_image(cfg, path: Path, size: tuple[int, int], what: str) -> None:
    """An image a clip starts or ends on is used as it is: it must be within the ingest tolerance of
    16:9, exactly the job's (width, height), and stored unrotated (some models apply an EXIF
    orientation and some ignore it)."""
    try:
        with Image.open(path) as im:
            got, orientation = im.size, im.getexif().get(0x0112, 1)
    except Exception as exc:            # noqa: BLE001 -- PIL raises many types on a broken file
        raise ToolError(f"{what} is not a readable image: {type(exc).__name__}: {exc}") from None
    if orientation != 1:
        raise ToolError(f"{what} carries an EXIF orientation: save it with the orientation applied and no tag")
    if not _near_16_9(cfg, *got):
        raise ToolError(f"{what} is {got[0]}x{got[1]}, which is not within "
                        f"{cfg.get('ingest.aspect_tolerance'):.0%} of 16:9")
    if got != tuple(size):
        raise ToolError(f"{what} is {got[0]}x{got[1]}, but the job renders {size[0]}x{size[1]} and uses the "
                        f"image as it is: make it exactly that size")


def split_gpus(gpus: list[int], per_worker: int, workers: int) -> list[list[int]]:
    if workers * per_worker > len(gpus):
        raise ValueError(f"{workers} workers x {per_worker} GPUs do not fit GPUs {gpus}")
    return [gpus[i * per_worker:(i + 1) * per_worker] for i in range(workers)]


TAIL = 2000                 # characters of a log an item's error carries


def _messages(work: Path, offsets: dict[Path, int]) -> list[dict]:
    """The complete lines appended to every messages*.jsonl in `work` since the last call."""
    new = []
    for path in sorted(work.glob("messages*.jsonl")):
        with open(path, "rb") as f:
            f.seek(offsets.get(path, 0))
            data = f.read()
        end = data.rfind(b"\n") + 1
        offsets[path] = offsets.get(path, 0) + end
        new += [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return new


CANCELED = "canceled before this item finished"


def _unfinished(work: Path, items: list[dict], rows: dict, started: dict, codes: dict, exit_code, world: int) -> dict:
    """Error rows for the items without an `item` message after the bridge ended. A log's tail (one log, or a
    list: one per rank, each under its file name) goes into the first such row that names it; later rows point to
    that item."""
    shown: dict[tuple, int] = {}

    def tail(log, index: int) -> str:
        logs = (log,) if isinstance(log, str) else tuple(log)
        if logs in shown:
            return f"; see item {shown[logs]}'s error"
        shown[logs] = index
        if len(logs) == 1:
            return f"; log tail:\n{file_tail(Path(logs[0]), TAIL)}"
        return "; log tails:\n" + "\n".join(f"--- {Path(p).name} ---\n{file_tail(Path(p), TAIL)}" for p in logs)

    def worker(index: int) -> int:
        return index % world if world else 0

    out = {}
    for item in items:
        index = item["index"]
        if index in rows:
            continue
        w = worker(index)
        code = codes.get(w, exit_code)
        ended = f"(exit code {code})" if isinstance(code, int) else f"({code})"
        working = [i for i in started if i not in rows and worker(i) == w]     # started; the worker ended first
        if index in started:
            error = (f"worker {w} started this item but ended without its result {ended}"
                     + tail(started[index]["log"], index))
        elif working:
            which = f"item {working[0]}" if len(working) == 1 else "items " + ", ".join(map(str, working))
            see = f"see item {working[0]}'s error" if len(working) == 1 else "see their errors"
            error = f"not run: worker {w} ended while working on {which} {ended}; {see}"
        else:
            log = work / "logs" / f"worker{w}.log"
            log = log if world and log.exists() else work / "bridge.log"
            before = "reaching this item" if any(worker(i) == w for i in started) else "starting any item"
            error = f"not run: worker {w} ended before {before} {ended}" + tail(str(log), index)
        out[index] = {"index": index, "error": error}
    return out


def run_bridge(backend, job, work: Path, spec: dict, items: list[dict], cancel, report, on_item,
               done: int = 0, poll=None) -> dict:
    """Writes `work/job.json` and runs `spec["bridge"]` in `spec["env"]` on the backend's GPUs; each `item` message
    becomes `on_item(message)`, the item's row. Items with no `item` message get an error row. Returns {"rows":
    {index: row}, "ready": the first `ready` message or None, "gpu_memory_mib": {"before", "after"},
    "gpu_memory_released"}. `done`: rows the caller already has, for progress; `poll`: called every 2 s while
    the bridge runs. If handling a message raises, the bridge is killed and waited for before the error goes on."""
    total = done + len(items)
    job_file = {"tool": backend.tool, "args": {k: v for k, v in job.args.items() if k != "items"}, "items": items,
                "gpus": backend.gpus, "work": str(work), "out": str(work / "out"),
                "config": getattr(backend, "block", None), **spec.get("job", {})}
    (work / "job.json").write_text(json.dumps(job_file, ensure_ascii=False, indent=1), encoding="utf-8")
    before = backend.gpu_memory(backend.gpus)
    kill, code = threading.Event(), []

    def launch() -> None:
        try:
            code.append(run_cancellable(
                spec["env"], ["python", str(spec["bridge"]), "--job", str(work / "job.json")],
                cwd=spec.get("cwd", work), log_path=work / "bridge.log",
                cancel=SimpleNamespace(is_set=lambda: cancel.is_set() or kill.is_set()),
                extra_env={**spec.get("env_vars", {}), "CUDA_VISIBLE_DEVICES": ",".join(map(str, backend.gpus)),
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                recorder=backend.recorder, node=job.node, phase=backend.tool))
        except Exception as exc:                 # noqa: BLE001 -- every item gets it below
            code.append(f"launch failed: {type(exc).__name__}: {exc}")

    rows: dict[int, dict] = {}
    started: dict[int, dict] = {}
    codes: dict[int, int] = {}                  # a spawned worker's exit code
    ready, offsets, polled = None, {}, 0.0
    thread = threading.Thread(target=launch, name=f"ar-{backend.tool}-bridge", daemon=True)
    thread.start()
    try:
        while True:
            alive = thread.is_alive()
            for m in _messages(work, offsets):
                if m["type"] == "ready" and ready is None:
                    ready = m
                elif m["type"] == "item_started":
                    started[m["index"]] = m
                elif m["type"] == "worker_exit":
                    codes[m["worker"]] = m["exit_code"]
                elif m["type"] == "item" and m["index"] not in rows:
                    rows[m["index"]] = on_item(m)
            report({"done": done + len(rows), "total": total})
            if poll and time.monotonic() - polled >= 2.0:
                poll()
                polled = time.monotonic()
            if not alive:
                break
            time.sleep(0.2)
    except BaseException:
        kill.set()
        thread.join()
        raise
    thread.join()
    if cancel.is_set():
        rows.update({i["index"]: {"index": i["index"], "error": CANCELED} for i in items if i["index"] not in rows})
    rows.update(_unfinished(work, items, rows, started, codes, code[0] if code else None,
                            len(spec.get("job", {}).get("groups") or [])))
    timeout = 5.0 if cancel.is_set() else float(backend.cfg.get("captioner.memory_release_timeout_s", 120))
    after, released = wait_gpu_release(backend.gpu_memory, backend.gpus, before, timeout)
    if released is False:
        backend.recorder.event(f"{backend.tool}.gpu_not_released", node=job.node, component="tools",
                               job_id=job.id, payload={"before": before, "after": after})
    return {"rows": rows, "ready": ready, "gpu_memory_mib": {"before": before, "after": after},
            "gpu_memory_released": released}


class GpuJob:
    """Base JobQueue backend. Subclasses set name/tool, kind ("rollout" | "annotation" | "image"),
    generator (the name in `derived_from`), file_keys (item fields naming workspace files), frame_keys
    (those of them holding an image that must have the job's size, with `frame_size`),
    optionally config_key (the kernel.yaml block, `self.block`) and implement check_args, prepare and finish. A job takes any number of items, or at most the block's
    `max_items`."""
    name = tool = kind = generator = description = config_key = ""
    file_keys: tuple[str, ...] = ()
    frame_keys: tuple[str, ...] = ()
    takes_keyframes = False     # items may carry `keyframes` (check_keyframes)

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory = registry, recorder, gpu_memory
        self.block = (cfg.get(self.config_key) if self.config_key else None) or {}
        self.max_items = self.block.get("max_items")

    # ---- submit (tool call; fast; ToolError goes back to the agent) ----
    def submit(self, q, caller, args: dict) -> dict:
        items = listed(caller, args.get("items"), "items")
        if not items:
            raise ToolError("items is empty")
        if self.max_items and len(items) > self.max_items:
            raise ToolError(f"at most {self.max_items} items per job: got {len(items)}; send the rest in another job")
        refuse(caller, self.name, items, {n: "not an object" for n, item in enumerate(items)
                                          if not isinstance(item, dict)})
        args = {**args, "items": items}
        self.check_args(args)
        bad = {}
        for n, item in enumerate(items):
            try:
                self.check_item(item)
                self.check_item_for(item, args)
            except ToolError as exc:
                bad[n] = str(exc)
            if copies_held_out(self.cfg, *strings(item)):       # before a GPU is scheduled
                self.recorder.event("isolation.refused", node=caller.node, phase=caller.phase,
                                    attempt=caller.attempt, component="tools", tool=self.name,
                                    payload={"item": n})
                bad.setdefault(n, EXCLUDED_PROMPT)
            for key in self.file_keys:
                if isinstance(item.get(key), str):
                    try:
                        host = clip_host_path(caller, item[key])
                        if key in self.frame_keys:
                            check_frame_image(self.cfg, host, self.frame_size(args), key)
                    except PathError as exc:
                        bad.setdefault(n, f"{key}: {exc}")
                    except ToolError as exc:
                        bad.setdefault(n, str(exc))
                    item = {**item, key: container_path(item[key])}
            if self.takes_keyframes and n not in bad:
                frames = []
                for k in item.get("keyframes") or []:
                    try:
                        check_frame_image(self.cfg, clip_host_path(caller, k["image"]), self.frame_size(args),
                                          f"the keyframe at frame {k['frame']}")
                    except PathError as exc:
                        bad.setdefault(n, f"keyframes: {exc}")
                    except ToolError as exc:
                        bad.setdefault(n, str(exc))
                    frames.append({**k, "image": container_path(k["image"])})
                item = {**item, "keyframes": frames}
            items[n] = item
        refuse(caller, self.name, items, bad)
        return {"job_id": q.submit(caller, self.name, args)}

    def check_args(self, args: dict) -> None:
        """The arguments of the whole job; may fill defaults in. Raises ToolError."""

    def check_item(self, item: dict) -> None:
        """One item. Raises ToolError with what is wrong: `submit` refuses the call for all bad items at once."""

    def check_item_for(self, item: dict, args: dict) -> None:
        """Checks of one item that need the job's arguments (after check_args filled defaults in). Raises ToolError."""

    def frame_size(self, args: dict) -> tuple[int, int]:
        """(width, height) the job renders: what a `frame_keys` image and every keyframe must be."""
        return args["width"], args["height"]

    # ---- run (worker thread, under the GPU lock) ----
    def run(self, job, cancel: threading.Event, report) -> dict:
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = job_dir(self.run_dir, job)
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True)
        out.mkdir()
        results: dict[int, dict] = {}
        memory = {"gpu_memory_mib": None, "gpu_memory_released": None}
        try:
            staged = []
            for index, item in enumerate(job.args["items"]):
                try:
                    staged.append(self._stage(caller, index, item, inp, job.args))
                except (PathError, OSError, ToolError) as exc:
                    results[index] = {"index": index, "error": input_error(exc)}
            if staged and not cancel.is_set():
                spec = self.prepare(job, staged, work)
                results.update(spec.get("rows", {}))
                send = [item for item in staged if item["index"] not in results]
                if send:
                    by_index = {item["index"]: item for item in send}
                    ran = run_bridge(self, job, work, spec, send, cancel, report,
                                     lambda m: self._item(caller, job, by_index[m["index"]], out, m), done=len(results))
                    results.update(ran["rows"])
                    memory = {k: ran[k] for k in memory}
            for item in staged:
                results.setdefault(item["index"], {"index": item["index"], "error": CANCELED})
        finally:
            shutil.rmtree(inp, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
        return {"items": [results[i] for i in sorted(results)], **memory}

    def _stage(self, caller, index: int, item: dict, inp: Path, args: dict) -> dict:
        """The item with its files copied into `inp`. The agent can change a file after submit, so a
        frame image's size is checked again on the copy the job uses."""
        staged = {**item, "index": index}
        for key in self.file_keys:
            if isinstance(item.get(key), str):
                dst = inp / f"{index}_{key}{Path(item[key]).suffix}"
                stage_clip(caller, item[key], dst)
                if key in self.frame_keys:
                    check_frame_image(self.cfg, dst, self.frame_size(args), key)
                staged[key] = str(dst)
        if self.takes_keyframes:
            staged["keyframes"] = []
            for n, k in enumerate(item.get("keyframes") or []):
                dst = inp / f"{index}_keyframe{n}{Path(k['image']).suffix}"
                stage_clip(caller, k["image"], dst)
                check_frame_image(self.cfg, dst, self.frame_size(args),
                                  f"the keyframe at frame {k['frame']}")
                staged["keyframes"].append({**k, "image": str(dst)})
        return staged

    def prepare(self, job, items: list[dict], work: Path) -> dict:
        """The bridge for `items` (staged): {"bridge": script, "env": its conda env, "cwd"?: its working folder
        (default `work`), "env_vars"?: extra environment, "job"?: the tool's part of the job file, "rows"?: {index:
        row} of items whose input is bad, left out of the job}. Raise to fail the whole job."""
        raise NotImplementedError

    def finish(self, job, item: dict, outputs: dict[str, Path], info: dict, out: Path) -> dict:
        """Post-process one finished item from the bridge's `outputs` and `info` (files it makes itself go into
        `out`); return {"video": Path, "annotation"?: Path, "pose"?: Path, plus any extra JSON-able fields}. Raise
        to report an item error."""
        raise NotImplementedError

    def _item(self, caller, job, item: dict, out: Path, message: dict) -> dict:
        """The row of an `item` message: its error, or its outputs finished and published by hard link."""
        index = item["index"]
        if not message["ok"]:
            return {"index": index, "error": message["error"]}
        info = message.get("info") or {}
        try:
            files = self.finish(job, item, {k: Path(v) for k, v in message["outputs"].items()}, info, out)
            published = {}
            for role in ("video", "annotation", "pose", "image", "commanded_camera"):
                if files.get(role):
                    src = Path(files[role])
                    if not src.resolve().is_relative_to(out.resolve()):    # only the job's own files get a link
                        raise ValueError(f"{role} {src} is not under the job's output folder")
                    # metadata files get their role in the name, so they never pass for a pose
                    name = "" if role in ("video", "pose", "image") else f".{role}"
                    rel = f"{self.kind}s/{job.id}/{index}{name}{src.suffix}"
                    publish_file(src, caller.staging_host, rel, link=True)
                    published[role] = str(STAGING / rel)
        except Exception as exc:            # noqa: BLE001 -- a finish() bug must be this item's error,
            # not a job failure that orphans the others.
            if disk_fault(exc):                 # the machine's: the phase stops
                raise
            return {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        extra = {k: v for k, v in files.items() if k not in ("video", "annotation", "pose", "commanded_camera")}
        if self.kind == "annotation":
            return {"index": index, "video": job.args["items"][index]["video"], **published, **extra,
                    "worker": info}
        if self.kind == "image":
            return {"index": index, "image": published["image"], "prompt": item.get("prompt"),
                    "seed": item.get("seed"), "generator": self.generator_name(job), "worker": info}
        candidate = {**published, **extra, "derived_from": {"kind": "model", "name": self.generator_name(job)}}
        return {"index": index, "candidate": candidate, "worker": info}

    def generator_name(self, job) -> str:
        return self.generator

    def write_annotation(self, out: Path, item: dict, annotation: dict) -> Path:
        path = out / f"{item['index']}.annotation.json"
        path.write_text(json.dumps(annotation, ensure_ascii=False, indent=1), encoding="utf-8")
        return path


JOB_NOTE = (" One call is one job: send every item in it, however many (jobs run one at a time, so many small "
            "jobs only wait on each other). The finished job's full result is written to a file under "
            "/workspace/staging/results/; job_wait returns that path with a summary. Read the file with a "
            "script: never copy results by hand.")


def job_description(backend) -> str:
    cap = getattr(backend, "max_items", None)
    most = f" At most {cap} items per job." if cap else ""
    return backend.description + JOB_NOTE + most


def register_gpu_tools(mcp, kit, q) -> None:
    """Registers each GPU data tool whose backend is on the queue. Disabled tools are simply absent."""
    b = q.backends

    def submit(ctx: Context, name: str, **args):
        """Queue a job for backend `name`; unset (None) parameters are left out."""
        args = {k: v for k, v in args.items() if v is not None}
        return kit.call(ctx, name, args, lambda c: b[name].submit(q, c, dict(args)))

    if "annotate_camera" in b:
        @mcp.tool(name="annotate_camera", description=job_description(b["annotate_camera"]))
        async def annotate_camera(
                paths: Annotated[list[str] | str, Field(description=f"mp4 files under /workspace, {FROM_FILE}")],
                ctx: Context) -> dict[str, Any]:
            return await kit.call(ctx, "annotate_camera", {"paths": paths}, lambda c: b["annotate_camera"].submit(
                q, c, {"items": [{"video": p} for p in listed(c, paths, "paths")]}))

    if "rollout_alayaworld" in b:
        @mcp.tool(name="rollout_alayaworld", description=job_description(b["rollout_alayaworld"]))
        async def rollout_alayaworld(
                items: WorldItems, ctx: Context,
                node: Annotated[str | None, Field(description="render with this scored node's fine-tune instead of the released model")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_alayaworld", items=items, node=node)

    if "generate_images" in b:
        @mcp.tool(name="generate_images", description=job_description(b["generate_images"]))
        async def generate_images(
                items: ImageItems, ctx: Context,
                width: Annotated[int | None, Field(description="multiple of 16 in 256..1920; the default is in the tool description")] = None,
                height: Annotated[int | None, Field(description="multiple of 16 in 256..1920; the default is in the tool description")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "generate_images", items=items, width=width, height=height)

    if "rollout_wan22" in b:
        @mcp.tool(name="rollout_wan22", description=job_description(b["rollout_wan22"]))
        async def rollout_wan22(
                items: ClipItems, ctx: Context,
                frames: Annotated[int | None, Field(description="4k+1 frames at 24 fps; one value for the job")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_wan22", items=items, frames=frames)

    if "rollout_ltx25" in b:
        @mcp.tool(name="rollout_ltx25", description=job_description(b["rollout_ltx25"]))
        async def rollout_ltx25(
                items: LtxItems, ctx: Context,
                frames: Annotated[int | None, Field(description="8k+1 frames at 24 fps; one value for the job")] = None,
                height: Annotated[int | None, Field(description="with width, one of the listed resolutions")] = None,
                width: Annotated[int | None, Field(description="with height, one of the listed resolutions")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_ltx25", items=items, frames=frames,
                                height=height, width=width)

    if "rollout_h3" in b:
        @mcp.tool(name="rollout_h3", description=job_description(b["rollout_h3"]))
        async def rollout_h3(items: H3Items, ctx: Context) -> dict[str, Any]:
            return await submit(ctx, "rollout_h3", items=items)


def build_gpu_backends(cfg, run_dir: Path, gpus: list[int], registry, recorder) -> list:
    """Every GPU data backend a run gets: the captioner and locate_events always, the others only when
    their `annotate`/`images`/`generators` config enables them. The caller registers each on the run's JobQueue, then calls `register_gpu_tools`
    `register_caption_tool` and `register_locate_tool`."""
    from .annotate import AnnotateBackend          # imports this module
    from .captioner import CaptionBackend
    from .h3 import H3Backend
    from .images import ImageBackend
    from .locate_events import LocateEventsBackend
    from .rollouts import AlayaWorldBackend, Ltx25Backend, Wan22Backend

    def generator_on(name: str) -> bool:
        block = cfg.get(f"generators.{name}") or {}
        return bool(block.get("enabled"))

    enabled = [(True, CaptionBackend), (True, LocateEventsBackend), (cfg.get("annotate.enabled"), AnnotateBackend),
               (cfg.get("images.enabled"), ImageBackend), (generator_on("alayaworld"), AlayaWorldBackend),
               (generator_on("wan22"), Wan22Backend), (generator_on("ltx25"), Ltx25Backend),
               (generator_on("h3"), H3Backend)]
    return [backend(cfg, run_dir, gpus, registry, recorder) for on, backend in enabled if on]
