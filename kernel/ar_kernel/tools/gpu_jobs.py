"""Shared plumbing for the GPU data-source jobs (spec 10): rollout_*, annotate_camera and
generate_images.

A backend stages the agent's input files into a kernel-private job dir (never trusting the
path between check and use), runs one worker per GPU group in the generator's own conda env
(bridge protocol: see the Plan 3 file structure), and publishes each finished item into the
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
from pydantic import WithJsonSchema

from ..archive.blobs import sha256_file
from ..subproc import file_tail
from .captioner import clip_host_path, container_path, stage_clip
from .context import STAGING, PathError
from .hf_tools import move_into
from .jobs import run_cancellable
from .server import ToolError
from .vllm_server import gpu_memory_mib, wait_gpu_release


def _items(properties: dict, required: list[str]) -> WithJsonSchema:
    """The listed JSON schema of an `items` list. Only the schema: the values still reach the tool
    as plain dicts and each backend's check_args validates them, so a bad value is a recorded
    tool.error rather than a failure inside the MCP layer."""
    return WithJsonSchema({"type": "array", "items": {"type": "object", "properties": properties,
                                                      "required": required, "additionalProperties": False}})


_STR, _INT = {"type": "string"}, {"type": "integer"}
ImageItems = Annotated[list[dict[str, Any]], _items({"prompt": _STR, "seed": _INT}, ["prompt", "seed"])]
ClipItems = Annotated[list[dict[str, Any]], _items({"prompt": _STR, "image": _STR, "seed": _INT},
                                                   ["prompt", "seed"])]
_TURN = {"type": "object", "properties": {"action": _STR, "subject_action": _STR, "event_edit": _STR,
                                          "perspective_switch": _STR},
         "required": ["action"], "additionalProperties": False}
CaseItems = Annotated[list[dict[str, Any]], _items(
    {"image": _STR, "perspective": {"type": "string", "enum": ["first_person", "third_person"]},  # rollouts.PERSPECTIVES
     "environment_prompt": _STR, "character_prompt": _STR, "perspective_prompt": _STR, "subject_mask": _STR,
     "turns": {"type": "array", "items": _TURN}},
    ["image", "perspective", "environment_prompt", "turns"])]


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_item_seed(n: int, item: dict) -> None:
    """An item's `seed` must be an int; integer text ("7") is taken as that int, in place."""
    if "seed" not in item:
        raise ToolError(f"item {n}: seed is missing; give an int")
    seed = item["seed"]
    if isinstance(seed, str) and seed.strip().lstrip("+-").isdigit():
        item["seed"] = seed = int(seed)
    if not is_int(seed):
        raise ToolError(f"item {n}: seed must be an int: got {seed!r}")


def enabled_variants(block: dict, names: tuple[str, ...]) -> list[str]:
    """The `names` whose `variants.<name>.enabled` is set in config `block`, in `names` order."""
    return [v for v in names if ((block.get("variants") or {}).get(v) or {}).get("enabled")]


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_gpus(gpus: list[int], per_worker: int, workers: int | None) -> list[list[int]]:
    groups = [gpus[i:i + per_worker] for i in range(0, len(gpus) - per_worker + 1, per_worker)]
    return groups[:workers] if workers else groups


class GpuJob:
    """Base JobQueue backend. Subclasses set name/tool, kind ("rollout" | "annotation" | "image"),
    generator (provenance name), license, file_keys (item fields naming workspace files),
    optionally config_key (the kernel.yaml block, `self.block`, whose max_items, timeout_s and
    license override the class defaults) and implement check_args, produce and finish."""
    name = tool = kind = generator = license = description = config_key = ""
    file_keys: tuple[str, ...] = ()
    max_items = 16
    timeout_s: float | None = None   # per-job wall-clock cap enforced by run_workers; None = no cap

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory = registry, recorder, gpu_memory
        self.block = (cfg.get(self.config_key) if self.config_key else None) or {}
        self.max_items = int(self.block.get("max_items", self.max_items))
        self.timeout_s = self.block.get("timeout_s", self.timeout_s)
        self.license = self.block.get("license", self.license)

    # ---- submit (tool call; fast; ToolError goes back to the agent) ----
    def submit(self, q, caller, args: dict) -> dict:
        items = args.get("items") or []
        if not items:
            raise ToolError("items is empty")
        if len(items) > self.max_items:
            raise ToolError(f"at most {self.max_items} items per job")
        self.check_args(args)
        for n, item in enumerate(items):
            for key in self.file_keys:
                if isinstance(item.get(key), str):
                    try:
                        clip_host_path(caller, item[key])
                    except PathError as exc:
                        raise ToolError(f"item {n}: {key}: {exc}") from exc
                    item = {**item, key: container_path(item[key])}
            items[n] = item
        return {"job_id": q.submit(caller, self.name, {**args, "items": items})}

    def check_args(self, args: dict) -> None:
        raise NotImplementedError

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
        spec = {k: v for k, v in item.items() if k not in (*self.file_keys, "index", "hashes")}
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
                    extra_env: dict | None = None,
                    deadline: float | None = None) -> tuple[list[int | str], dict[int, str]]:
        """One worker per GPU group, in parallel; each handles items with index % world == rank.
        A cancel kills every worker's process group, and so does this backend's own `timeout_s`
        (the deadline is checked in the polling loop, and once passed acts on the workers exactly
        like an external cancel; `deadline`, a time.monotonic() value, replaces the one computed
        from `timeout_s` when a backend spent part of that budget before its workers). An item whose worker died without writing its status becomes an
        item error carrying the worker's exit code (or "timeout after Ns") and log tail.

        Returns (codes, missing): `missing` maps such an item's index to that error message, for
        `run` to pass into `_collect`. The staged items' indices are read back from
        `work/items.json` (items that failed staging never reach here); `total` is only the
        progress-report denominator.
        """
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
            report({"done": len([p for p in out.glob("*.json") if p.stem.isdigit()]), "total": total})
            time.sleep(0.2)             # short poll so a finished job returns promptly
        for t in threads:
            t.join()
        report({"done": len([p for p in out.glob("*.json") if p.stem.isdigit()]), "total": total})
        world = len(groups)
        missing: dict[int, str] = {}
        for index in indices:
            code = codes[index % world]
            if code not in (0, None) and not (out / f"{index}.json").exists():
                reason = f"timeout after {self.timeout_s}s" if timed_out.is_set() else f"exit code {code}"
                missing[index] = (f"worker {index % world} failed ({reason}); log tail:\n"
                                  f"{file_tail(work / f'worker{index % world}.log', 2000)}")
        return codes, missing


def register_gpu_tools(mcp, kit, q) -> None:
    """Registers each GPU data tool whose backend is on the queue. Disabled tools/variants are
    simply absent (spec 10: disabled variants are omitted from tool schemas)."""
    b = q.backends

    def submit(ctx: Context, name: str, **args):
        """Queue a job for backend `name`; unset (None) parameters are left out."""
        args = {k: v for k, v in args.items() if v is not None}
        return kit.call(ctx, name, args, lambda c: b[name].submit(q, c, dict(args)))

    if "annotate_camera" in b:
        @mcp.tool(name="annotate_camera", description=b["annotate_camera"].description)
        async def annotate_camera(paths: list[str], ctx: Context) -> dict[str, Any]:
            return await kit.call(ctx, "annotate_camera", {"paths": paths},
                                  lambda c: b["annotate_camera"].submit(q, c, {"items": [{"video": p} for p in paths]}))

    if "rollout_alayaworld" in b:
        @mcp.tool(name="rollout_alayaworld", description=b["rollout_alayaworld"].description)
        async def rollout_alayaworld(items: CaseItems, ctx: Context, variant: str | None = None,
                                     rounds_per_turn: int | None = None, seed: int | None = None) -> dict[str, Any]:
            return await submit(ctx, "rollout_alayaworld", items=items, variant=variant,
                                rounds_per_turn=rounds_per_turn, seed=seed)

    if "generate_images" in b:
        @mcp.tool(name="generate_images", description=b["generate_images"].description)
        async def generate_images(items: ImageItems, ctx: Context, width: int | None = None,
                                  height: int | None = None) -> dict[str, Any]:
            return await submit(ctx, "generate_images", items=items, width=width, height=height)

    if "rollout_wan22" in b:
        @mcp.tool(name="rollout_wan22", description=b["rollout_wan22"].description)
        async def rollout_wan22(items: ClipItems, ctx: Context, frames: int | None = None) -> dict[str, Any]:
            return await submit(ctx, "rollout_wan22", items=items, frames=frames)

    if "rollout_ltx25" in b:
        @mcp.tool(name="rollout_ltx25", description=b["rollout_ltx25"].description)
        async def rollout_ltx25(items: ClipItems, ctx: Context, variant: str | None = None,
                                frames: int | None = None, height: int | None = None,
                                width: int | None = None) -> dict[str, Any]:
            return await submit(ctx, "rollout_ltx25", items=items, variant=variant, frames=frames,
                                height=height, width=width)


def build_gpu_backends(cfg, run_dir: Path, gpus: list[int], registry, recorder) -> list:
    """Every GPU data backend a run gets (spec 10): the captioner always, the others only when
    their `annotate`/`images`/`generators` config enables them (a generator needs an enabled
    variant). The caller registers each on the run's JobQueue, then calls `register_gpu_tools`
    and `register_caption_tool`."""
    from .annotate import AnnotateBackend          # imports this module
    from .captioner import CaptionBackend
    from .images import ImageBackend
    from .rollouts import AlayaWorldBackend, Ltx25Backend, Wan22Backend

    def generator_on(name: str) -> bool:
        return any((v or {}).get("enabled") for v in (cfg.get(f"generators.{name}.variants") or {}).values())

    enabled = [(True, CaptionBackend), (cfg.get("annotate.enabled"), AnnotateBackend),
               (cfg.get("images.enabled"), ImageBackend), (generator_on("alayaworld"), AlayaWorldBackend),
               (generator_on("wan22"), Wan22Backend), (generator_on("ltx25"), Ltx25Backend)]
    return [backend(cfg, run_dir, gpus, registry, recorder) for on, backend in enabled if on]
