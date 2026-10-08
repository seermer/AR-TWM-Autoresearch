"""Shared plumbing for the GPU data-source jobs: rollout_*, annotate_camera and
generate_images.

A backend stages the agent's input files into a kernel-private job dir (never trusting the
path between check and use), runs one worker per GPU group in the generator's own conda env, and publishes each finished item into the
caller's staging dir with O_NOFOLLOW moves, so a link the agent plants cannot redirect a
kernel write. A per-item failure is an item error, not a failed job.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Callable

from mcp.server.mcpserver import Context
from pydantic import Field, WithJsonSchema

from ..archive.blobs import sha256_file
from ..isolation import EXCLUDED_PROMPT, copies_held_out, strings
from ..subproc import file_tail
from .captioner import clip_host_path, container_path, stage_clip
from .context import STAGING, PathError
from .hf_tools import move_into
from .jobs import run_cancellable
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
KEYFRAMES_SCHEMA = {"type": "array", "description": "images the clip must show at given frames", "items": {
    "type": "object", "additionalProperties": False, "required": ["image", "frame"], "properties": {
        "image": _str("an image file under /workspace (any size; it is cropped and resized)"),
        "frame": {"type": "integer", "description": "0-based frame index; 0 is the first frame, -1 the last"}}}}
LtxItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"prompt": _str("what the clip shows"), "keyframes": KEYFRAMES_SCHEMA, "seed": _SEED}, ["prompt", "seed"])]
_TURN = {"type": "object", "additionalProperties": False, "required": ["action"], "properties": {
    "action": _str("camera move for this turn: W, A, S, D (translate), left, right, up, down (rotate), stop, "
                   "or two joined with '+', e.g. 'W+left'"),
    "event": _str("something that happens in the scene during this turn"),
    "subject_action": _str("something the subject does during this turn"),
    "viewpoint_change": _str("a change of viewpoint during this turn: 'fp_to_tp', 'tp_to_fp', 'fp_to_scope', "
                             "or 'tp_to_tp: <the new view>'")}}
WorldItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"image": _str("first frame: an image file under /workspace (.jpg, .png, ...; any size)"),
     "viewpoint": {"type": "string", "enum": ["first_person", "third_person"],     # rollouts.VIEWPOINTS
                   "description": "whose eyes the first frame is seen through"},
     "scene_prompt": _str("the scene: place, objects, light"),
     "character_prompt": _str("the subject, if there is one"),
     "viewpoint_prompt": _str("how the camera sees the scene at the start"),
     "subject_mask": _str("an image under /workspace, white where the subject is in the first frame"),
     "turns": {"type": "array", "items": _TURN,
               "description": "the clip, turn by turn; each turn lasts `rounds_per_turn` rounds"}},
    ["image", "viewpoint", "scene_prompt", "turns"])]


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
    """An item's optional `keyframes`: [{'image': path, 'frame': int}], frame >= -1 (-1 = the last), none repeated."""
    frames = item.get("keyframes")
    if frames is None:
        return
    if not isinstance(frames, list):
        raise ToolError("keyframes must be a list of {'image': path, 'frame': int}")
    for k in frames:
        if not (isinstance(k, dict) and set(k) == {"image", "frame"} and isinstance(k["image"], str)
                and is_int(k["frame"])):
            raise ToolError(f"each keyframe must be {{'image': path, 'frame': int}}: got {k!r}")
        if k["frame"] < -1:
            raise ToolError(f"frame must be -1 (the last frame) or a frame index from 0: got {k['frame']}")
    seen = [k["frame"] for k in frames]
    if len(set(seen)) != len(seen):
        raise ToolError(f"keyframes repeat a frame: {seen}")


def enabled_variants(block: dict, names: tuple[str, ...]) -> list[str]:
    """The variants config `block` lists (`variants: [a, b]`) that the backend knows, in config order."""
    return [v for v in block.get("variants") or [] if v in names]


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_gpus(gpus: list[int], per_worker: int, workers: int | None) -> list[list[int]]:
    groups = [gpus[i:i + per_worker] for i in range(0, len(gpus) - per_worker + 1, per_worker)]
    return groups[:workers] if workers else groups


class GpuJob:
    """Base JobQueue backend. Subclasses set name/tool, kind ("rollout" | "annotation" | "image"),
    generator (provenance name), license, file_keys (item fields naming workspace files),
    optionally config_key (the kernel.yaml block, `self.block`, whose timeout_s and license override
    the class defaults) and implement check_args, produce and finish. A job takes any number of items, or at most the block's
    `max_items`."""
    name = tool = kind = generator = license = description = config_key = ""
    file_keys: tuple[str, ...] = ()
    takes_keyframes = False     # items may carry `keyframes` (check_keyframes)
    timeout_s: float | None = None   # per-job wall-clock cap enforced by run_workers; None = no cap

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory = registry, recorder, gpu_memory
        self.block = (cfg.get(self.config_key) if self.config_key else None) or {}
        self.timeout_s = self.block.get("timeout_s", self.timeout_s)
        self.license = self.block.get("license", self.license)
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
                        clip_host_path(caller, item[key])
                    except PathError as exc:
                        bad.setdefault(n, f"{key}: {exc}")
                    item = {**item, key: container_path(item[key])}
            if self.takes_keyframes and n not in bad:
                frames = []
                for k in item.get("keyframes") or []:
                    try:
                        clip_host_path(caller, k["image"])
                    except PathError as exc:
                        bad.setdefault(n, f"keyframes: {exc}")
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

    # ---- run (worker thread, under the GPU lock) ----
    def run(self, job, cancel: threading.Event, report) -> dict:
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = self.run_dir / "jobs" / job.id
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True)
        out.mkdir()
        before = self.gpu_memory(self.gpus)
        results: dict[int, dict] = {}
        missing: dict[int, str] = {}
        try:
            staged = []
            for index, item in enumerate(job.args["items"]):
                try:
                    staged.append(self._stage(caller, index, item, inp))
                except (PathError, OSError) as exc:
                    results[index] = {"index": index, "error": f"input: {exc}"}
            (work / "items.json").write_text(json.dumps(staged), encoding="utf-8")
            if staged and not cancel.is_set():
                produced = self.produce(job, staged, work, out, cancel, report)
                if produced:
                    _, missing = produced
            for item in staged:
                results[item["index"]] = self._collect(caller, job, item, out, missing)
        finally:
            shutil.rmtree(inp, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
        timeout = 5.0 if cancel.is_set() else float(self.cfg.get("captioner.memory_release_timeout_s", 120))
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, before, timeout)
        if released is False:
            self.recorder.event(f"{self.tool}.gpu_not_released", node=job.node, component="tools",
                                job_id=job.id, payload={"before": before, "after": after})
        return {"items": [results[i] for i in sorted(results)],
                "gpu_memory_mib": {"before": before, "after": after}, "gpu_memory_released": released}

    def _stage(self, caller, index: int, item: dict, inp: Path) -> dict:
        staged = {**item, "index": index, "hashes": {}}
        for key in self.file_keys:
            if isinstance(item.get(key), str):
                dst = inp / f"{index}_{key}{Path(item[key]).suffix}"
                stage_clip(caller, item[key], dst)
                staged[key] = str(dst)
                staged["hashes"][key] = sha256_file(dst)
        if self.takes_keyframes:
            staged["keyframes"] = []
            for n, k in enumerate(item.get("keyframes") or []):
                dst = inp / f"{index}_keyframe{n}{Path(k['image']).suffix}"
                stage_clip(caller, k["image"], dst)
                staged["keyframes"].append({**k, "image": str(dst)})
                staged["hashes"][f"keyframe{n}"] = sha256_file(dst)
        return staged

    def produce(self, job, items: list[dict], work: Path, out: Path, cancel, report):
        """Run the workers that fill `out` for `items` (typically via `run_workers`, whose
        result is returned directly). Raise to fail the whole job."""
        raise NotImplementedError

    def finish(self, job, item: dict, out: Path) -> dict:
        """Post-process one finished item; return {"video": Path, "caption": Path, "pose"?: Path,
        plus any extra JSON-able fields}. Raise to report an item error."""
        raise NotImplementedError

    def _collect(self, caller, job, item: dict, out: Path, missing: dict[int, str]) -> dict:
        index = item["index"]
        status_path = out / f"{index}.json"
        if not status_path.exists():
            return {"index": index, "error": missing.get(index, "no output (cancelled or not reached)")}
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError(f"status is a {type(status).__name__}, not an object")
            if not status.get("ok"):
                return {"index": index, "error": status.get("error", "failed")}
            files = self.finish(job, item, out)
            published = {}
            for role in ("video", "caption", "pose", "image", "commanded_camera"):
                if files.get(role):
                    src = Path(files[role])
                    # metadata files get their role in the name, so they never pass for a pose
                    name = "" if role in ("video", "caption", "pose", "image") else f".{role}"
                    rel = f"{self.kind}s/{job.id}/{index}{name}{src.suffix}"
                    move_into(src, caller.staging_host, rel)
                    published[role] = str(STAGING / rel)
        except Exception as exc:            # noqa: BLE001 -- a bad/truncated status or a finish()
            # bug must be this item's error, not a job failure that orphans the others.
            return {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        extra = {k: v for k, v in files.items() if k not in ("video", "caption", "pose", "commanded_camera")}
        worker = {k: v for k, v in status.items() if k != "ok"}
        if self.kind == "annotation":
            return {"index": index, "video": job.args["items"][index]["video"], **published, **extra,
                    "worker": worker}
        if self.kind == "image":
            return {"index": index, "image": published["image"], "prompt": item.get("prompt"),
                    "seed": item.get("seed"), "generator": self.generator_name(job), "license": self.license,
                    "worker": worker}
        params = {k: v for k, v in job.args.items() if k != "items"}
        spec = {k: v for k, v in item.items() if k not in (*self.file_keys, "index", "hashes", "keyframes")}
        if self.takes_keyframes:
            spec["keyframe_frames"] = [k["frame"] for k in item["keyframes"]]
        inputs_hash = canonical_hash({"generator": self.generator_name(job), "params": params,
                                      "item": spec, "files": item["hashes"]})
        candidate = {**published, "provenance": {"kind": "rollout", "generator": self.generator_name(job),
                     "job_id": job.id, "inputs_hash": inputs_hash, "seed": item.get("seed")},
                     "license": self.license}
        return {"index": index, "candidate": {**candidate, **extra}, "worker": worker}

    def generator_name(self, job) -> str:
        return self.generator

    def write_caption(self, out: Path, item: dict, caption: dict) -> Path:
        path = out / f"{item['index']}.caption.json"
        path.write_text(json.dumps(caption, ensure_ascii=False), encoding="utf-8")
        return path

    # ---- workers ----
    def run_workers(self, env: str, argv_for: Callable[[int, int], list[str]], groups: list[list[int]], *,
                    job, work: Path, out: Path, total: int, cancel, report, cwd: Path | None = None,
                    extra_env: dict | None = None, deadline: float | None = None,
                    done: Callable[[], int] | None = None) -> tuple[list[int | str], dict[int, str]]:
        """One worker per GPU group, in parallel; each handles items with index % world == rank.
        A cancel kills every worker's process group, and so does this backend's own `timeout_s`
        (the deadline is checked in the polling loop, and once passed acts on the workers exactly
        like an external cancel; `deadline`, a time.monotonic() value, replaces the one computed
        from `timeout_s` when a backend spent part of that budget before its workers). An item whose worker died without writing its status becomes an
        item error carrying the worker's exit code (or "timeout after Ns") and log tail.

        Returns (codes, missing): `missing` maps such an item's index to that error message, for
        `run` to pass into `_collect`. The staged items' indices are read back from
        `work/items.json` (items that failed staging never reach here); `total` is only the
        progress-report denominator, and `done` counts the finished items of a worker that does not
        write `out/<index>.json` itself.
        """
        done = done or (lambda: len([p for p in out.glob("*.json") if p.stem.isdigit()]))
        if not groups:
            raise ValueError("run_workers: no GPU groups (check the GPU list, gpus_per_worker and workers)")
        indices = [item["index"] for item in json.loads((work / "items.json").read_text(encoding="utf-8"))]
        codes: list[int | str | None] = [None] * len(groups)
        if deadline is None and self.timeout_s:
            deadline = time.monotonic() + self.timeout_s
        timed_out = threading.Event()
        worker_cancel = cancel if deadline is None else \
            SimpleNamespace(is_set=lambda: cancel.is_set() or timed_out.is_set())

        def one(rank: int, group: list[int]) -> None:
            try:
                codes[rank] = run_cancellable(
                    env, argv_for(rank, len(groups)), cwd=cwd or work, cancel=worker_cancel,
                    extra_env={**(extra_env or {}), "CUDA_VISIBLE_DEVICES": ",".join(map(str, group)),
                               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                    log_path=work / f"worker{rank}.log", recorder=self.recorder, node=job.node, phase=self.tool)
            except Exception as exc:                 # noqa: BLE001 -- reported per item below
                codes[rank] = f"launch failed: {type(exc).__name__}: {exc}"

        threads = [threading.Thread(target=one, args=(r, g), name=f"ar-{self.tool}-w{r}", daemon=True)
                   for r, g in enumerate(groups)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            if deadline is not None and time.monotonic() > deadline:
                timed_out.set()
            report({"done": done(), "total": total})
            time.sleep(0.2)             # short poll so a finished job returns promptly
        for t in threads:
            t.join()
        report({"done": done(), "total": total})
        world = len(groups)
        missing: dict[int, str] = {}
        for index in indices:
            code = codes[index % world]
            if code not in (0, None) and not (out / f"{index}.json").exists():
                reason = f"timeout after {self.timeout_s}s" if timed_out.is_set() else f"exit code {code}"
                missing[index] = (f"worker {index % world} failed ({reason}); log tail:\n"
                                  f"{file_tail(work / f'worker{index % world}.log', 2000)}")
        return codes, missing


JOB_NOTE = (" One call is one job: send every item in it, however many (jobs run one at a time, so many small "
            "jobs only wait on each other). The finished job's full result is written to a file under "
            "/workspace/staging/results/; job_wait returns that path with a summary. Read the file with a "
            "script: never copy results by hand.")


def job_description(backend) -> str:
    limit = getattr(backend, "timeout_s", None)
    stop = f" A job is stopped after {limit / 3600:g} h; items not finished by then are item errors." if limit else ""
    cap = getattr(backend, "max_items", None)
    most = f" At most {cap} items per job." if cap else ""
    return backend.description + JOB_NOTE + most + stop


def register_gpu_tools(mcp, kit, q) -> None:
    """Registers each GPU data tool whose backend is on the queue. Disabled tools/variants are
    simply absent (disabled variants are omitted from tool schemas)."""
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
                rounds_per_turn: Annotated[int | None, Field(description="1..3, default 3; a round is 32 frames at 24 fps")] = None,
                seed: Annotated[int | None, Field(description="default 42; one seed for the whole job")] = None,
                node: Annotated[str | None, Field(description="render with this scored node's fine-tune instead of the released model")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_alayaworld", items=items,
                                rounds_per_turn=rounds_per_turn, seed=seed, node=node)

    if "generate_images" in b:
        @mcp.tool(name="generate_images", description=job_description(b["generate_images"]))
        async def generate_images(
                items: ImageItems, ctx: Context,
                width: Annotated[int | None, Field(description="multiple of 16 in 256..1920, default 1280")] = None,
                height: Annotated[int | None, Field(description="multiple of 16 in 256..1920, default 720")] = None,
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
                variant: Annotated[str | None, Field(description="default the first one listed above")] = None,
                frames: Annotated[int | None, Field(description="8k+1 frames at 24 fps; one value for the job")] = None,
                height: Annotated[int | None, Field(description="with width, one of the listed resolutions")] = None,
                width: Annotated[int | None, Field(description="with height, one of the listed resolutions")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_ltx25", items=items, variant=variant, frames=frames,
                                height=height, width=width)


def build_gpu_backends(cfg, run_dir: Path, gpus: list[int], registry, recorder) -> list:
    """Every GPU data backend a run gets: the captioner always, the others only when
    their `annotate`/`images`/`generators` config enables them (a generator needs `enabled` or a listed
    variant). The caller registers each on the run's JobQueue, then calls `register_gpu_tools`
    and `register_caption_tool`."""
    from .annotate import AnnotateBackend          # imports this module
    from .captioner import CaptionBackend
    from .images import ImageBackend
    from .rollouts import AlayaWorldBackend, Ltx25Backend, Wan22Backend

    def generator_on(name: str) -> bool:
        block = cfg.get(f"generators.{name}") or {}
        return bool(block.get("enabled") or block.get("variants"))

    enabled = [(True, CaptionBackend), (cfg.get("annotate.enabled"), AnnotateBackend),
               (cfg.get("images.enabled"), ImageBackend), (generator_on("alayaworld"), AlayaWorldBackend),
               (generator_on("wan22"), Wan22Backend), (generator_on("ltx25"), Ltx25Backend)]
    return [backend(cfg, run_dir, gpus, registry, recorder) for on, backend in enabled if on]
