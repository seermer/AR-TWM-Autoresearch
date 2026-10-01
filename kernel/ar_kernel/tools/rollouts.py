"""rollout_alayaworld: WBench-style cases rendered by AlayaWorld through
the WBench eval's own render path (WorldModel scripts/tools/run_wbench.py, configs/wbench_full.yaml).

The agent writes cases (first frame, perspective, prompts, per-turn actions). The kernel stages
them as WBench case files, pre-encodes their prompts (the eval runs with the text encoder off),
renders them, and publishes each clip with its round boundaries on the per_chunk grid (25 + 32k
frames) and one caption segment per round from the prompts the render used.

No pose is published (user decision 2026-09-26): the renders follow the commanded translation
but only weakly the commanded turns and orbits, so the commanded camera path is
not a pose label. It is published as metadata (`commanded_camera`); the agent runs
annotate_camera (ViGeo) on the clip for the pose it ingests with.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from ..archive.db import open_db
from ..archive.nodes import NodeStore
from ..data.probe import aspect_ok, probe_video
from ..eval.lora import concat_eval_lora
from ..subproc import file_tail, free_port, meminfo_gib
from .gpu_jobs import GpuJob, check_item_seed, enabled_variants, is_int, split_gpus
from .jobs import run_cancellable
from .server import ToolError

WAN22_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "wan22_generate.py"
LTX25_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "ltx25_generate.py"

# The launches, as argv after "python" (run in the generator env, cwd WorldModel). The tests
# swap both for the fake worker.
PRECACHE = ["-m", "scripts.tools.precache_wbench_text_embeds"]
RUN_WBENCH = ["scripts/tools/run_wbench.py"]

# Action tokens WorldModel understands, from alaya/trainer/rollout_trainer.py
# (_wbench_action_to_nav: its `single` and `aliases` keys, lower-case wasd and the arrows).
# An action is one token or several joined by "+" (or ","); WorldModel silently ignores an
# unknown token, so it is refused here instead.
ACTION_TOKENS = frozenset({
    "W", "S", "A", "D", "w", "a", "s", "d", "left", "right", "up", "down", "stop",
    "forward", "backward", "cam_left", "cam_right", "cam_up", "cam_down", "look_left", "look_right",
    "look_up", "look_down", "pitch_up", "pitch_down", "yaw_left", "yaw_right", "->", "→", "<-", "←"})
TURN_TYPES = ("subject_action", "event_edit", "perspective_switch")     # WBench interaction types
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")                 # alaya/data/wbench.py _IMAGE_EXTS
PERSPECTIVES = ("first_person", "third_person")
VARIANTS = ("dmd4", "ar30")
VARIANT_TEXT = {"dmd4": "the eval's 4-step student", "ar30": "the 30-step AR teacher"}
# ar30 = the AR teacher without the DMD LoRA, sampled as configs/infer_i2v_camera_ar.yaml does.
AR30 = {("paths", "dmd_resume"): None, ("validation", "sampling_steps"): 30,
        ("validation", "scheduler"): "shift", ("validation", "cfg_scale"): 3.0}
ROUND_FRAMES = 32           # one rollout round: 4 latents x temporal stride 8
GRID = 25                   # per_chunk round boundaries sit at 25 + 32k frames
FPS = 24
FFMPEG_TIMEOUT_S = 600      # a wedged/oversize ffmpeg must fail its item, not hang the job


def valid_action(action) -> bool:
    parts = str(action).replace(",", "+").split("+") if isinstance(action, str) else [""]
    return all(p.strip() in ACTION_TOKENS for p in parts)


def case_json(index: int, item: dict, image_rel: str, mask_rel: str | None) -> dict:
    """One item as a WBench case (the schema of WBench/data/cases/*.json): a navigation entry per
    turn, plus that turn's subject_action / event_edit / perspective_switch."""
    interactions = []
    for turn, t in enumerate(item["turns"], 1):
        interactions.append({"type": "navigation", "action": t["action"], "turn": turn})
        interactions += [{"type": k, "action": t[k], "turn": turn} for k in TURN_TYPES if t.get(k)]
    return {"id": str(index), "environment_prompt": item["environment_prompt"],
            "character_prompt": item.get("character_prompt", ""),
            "perspective_prompt": item.get("perspective_prompt", ""),
            "settings": {"perspective": item["perspective"], "subject": {"type": "unknown", "desc": ""},
                         "tracking_object": None, "initial_image": image_rel, "subject_mask": mask_rel},
            "interactions": interactions, "metric_list": []}


def render_config(cfg, *, variant: str, rounds_per_turn: int, seed: int, indices: list[int], work: Path,
                  text_cache: Path, node_lora: Path | None = None, node_rank: int = 0,
                  history_encoder: Path | None = None) -> dict:
    """configs/wbench_full.yaml as eval/render.py:build_render_config adapts it, pointed at this
    job's cases. The prompt cache is the run's (the eval's own holds only WBench's prompts).
    `node_lora`: a node's fine-tune (rank `node_rank`) with its `history_encoder`: for dmd4 the
    directory of the student LoRA concatenated with the node's, for ar30 the node's checkpoint."""
    wm = cfg.worldmodel
    c = yaml.safe_load((wm / "configs" / "wbench_full.yaml").read_text(encoding="utf-8"))
    for key, value in c["paths"].items():
        if isinstance(value, str) and value:
            c["paths"][key] = str(wm / value)
    c["run"].update(output_dir=str(work / "rollout"), log_dir=str(work / "logs"), seed=int(seed))
    c["runtime"]["text_embed_cache_dir"] = str(text_cache)
    c["validation"].update(per_sample_seed=True, save_joystick=False)
    mode = c["validation"]["modes"]["wbench"]
    mode["dataset"].update(root=str(work / "data"), case_ids=[str(i) for i in indices])
    mode["wbench_output_dir"] = str(work / "videos")
    mode["wbench_chunks_per_turn"] = int(rounds_per_turn)
    if variant == "ar30":
        for (section, key), value in AR30.items():
            c[section][key] = value
    if node_lora is not None:
        c["paths"].update(dmd_resume=str(node_lora), history_encoder=str(history_encoder))
        c["lora"]["rank"] = node_rank + (0 if variant == "ar30" else c["lora"]["rank"])
        c["lora"]["alpha"] = c["lora"]["rank"]
    return c


def checked_repo(cfg, key: str, label: str) -> Path:
    """The pinned third-party checkout of generator block `key`: refuses to run against a clone
    that has moved off the commit configs/kernel.yaml pins (`<key>.commit`), so a silent `git pull`
    there cannot change what a backend renders without a new review."""
    block = cfg.get(key)
    repo, pinned = cfg.repo_root / block["repo"], block.get("commit")
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    if git.returncode != 0:
        raise RuntimeError(f"no {label} checkout at {repo} (git: {git.stderr.strip()}); clone it at "
                           f"commit {pinned} into that folder")
    head = git.stdout.strip()
    if head != pinned:
        raise RuntimeError(f"{repo} is at commit {head}, but {key}.commit pins {pinned}; re-clone the "
                           f"pinned commit or update the pin")
    return repo


def check_clip_items(items: list[dict]) -> None:
    """Items of a text/image-to-video job: {'prompt': str, 'image'?: path, 'seed': int}."""
    for n, item in enumerate(items):
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ToolError(f"item {n}: prompt must be a non-empty string")
        if item.get("image") is not None and not isinstance(item["image"], str):
            raise ToolError(f"item {n}: image must be a file path")
        check_item_seed(n, item)


def check_published(info) -> None:
    """A published clip is 24 fps and within 2% of 16:9."""
    if round(info.fps) != FPS:
        raise ValueError(f"published clip is {info.fps} fps, not 24")
    if not aspect_ok(info, 0.02):
        raise ValueError(f"published clip's aspect {info.display_aspect:.4f} is not within "
                         f"2% of 16:9 ({info.width}x{info.height})")


def round_segments(schedule: list[dict], first: int, trim: int, frames: int) -> list[dict]:
    """One caption segment per round (rounds start at mp4 frame first + 32r), in seconds of the
    trimmed clip; adjacent equal prompts merged; spans [0, frames/24]."""
    segs: list[dict] = []
    for entry in sorted(schedule, key=lambda e: e["round"]):
        start = max(0, first + ROUND_FRAMES * int(entry["round"]) - trim)
        if start >= frames:
            break
        if segs and segs[-1]["prompt"] == entry["prompt"]:
            continue
        if segs:
            segs[-1]["time_range_s"][1] = start / FPS
        segs.append({"time_range_s": [0.0 if not segs else start / FPS, None], "prompt": entry["prompt"]})
    segs[-1]["time_range_s"][1] = frames / FPS
    return segs


class AlayaWorldBackend(GpuJob):
    name = tool = "rollout_alayaworld"
    kind = "rollout"
    config_key = "generators.alayaworld"      # max_items / timeout_s
    file_keys = ("image", "subject_mask")
    max_turns = 9
    description = (
        "Render WBench-style cases with AlayaWorld exactly as the WBench eval does. The clips come from "
        "the released model, the same model every node fine-tunes, or with `node` from that scored node's "
        "fine-tune. A GPU job: returns "
        "{job_id} at once; collect with job_wait. Params: `variant` ({variants}; default the first), "
        "`node` (the id of a scored node other than the root; default the released model), "
        "`rounds_per_turn` 1..3 (default 3; a round "
        "is 32 frames at 24 fps), `seed` (int, default 42). Item: {'image': first frame under /workspace "
        "(.jpg/.png/..., any size), 'perspective': 'first_person'|'third_person', 'environment_prompt', "
        "'character_prompt'?, 'perspective_prompt'?, 'subject_mask'? (image, white = subject), 'turns': "
        "[{'action': WBench navigation action (W, A, S, D, left, right, up, down, stop, or combined like "
        "'W+left'), 'subject_action'?: text, 'event_edit'?: text, 'perspective_switch'?: a WBench code such as "
        "'fp_to_tp', 'tp_to_fp', 'fp_to_scope', or 'tp_to_tp: <new view>'}, ...]}. Each turn holds its action "
        "and prompt for all its rounds. Actions steer translation reliably, rotation (turns, orbits) only "
        "weakly. Each result item gives a `candidate` (mp4, caption with one segment per round, "
        "provenance) with NO pose and no camera_motion: run annotate_camera on candidate.video, then "
        "data_ingest it with that pose and camera_motion 'moving' (eligible for "
        "video_timed_prompts_camera:per_chunk). Metadata, not labels: `commanded_camera` (npz of the camera "
        "path the actions commanded, one pose per frame; not what the video shows), `actions` and "
        "`turn_segments` (frame ranges in the published clip).")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = int(self.block.get("max_turns", self.max_turns))
        variants = "; ".join(f"'{v}': {VARIANT_TEXT[v]}" for v in self.enabled_variants())
        self.description = self.description.replace("{variants}", variants)   # disabled variants omitted

    def enabled_variants(self) -> list[str]:
        return enabled_variants(self.block, VARIANTS)

    def check_args(self, args):
        args.setdefault("variant", (self.enabled_variants() or ["dmd4"])[0])
        args.setdefault("rounds_per_turn", 3)
        args.setdefault("seed", 42)
        if args["variant"] not in self.enabled_variants():
            raise ToolError(f"variant must be one of the enabled variants {self.enabled_variants()}: "
                            f"got {args['variant']!r}")
        rpt = args["rounds_per_turn"]
        if not (is_int(rpt) and 1 <= rpt <= 3):
            raise ToolError(f"rounds_per_turn must be an int in 1..3: got {rpt!r}")
        if not is_int(args["seed"]):
            raise ToolError(f"seed must be an int: got {args['seed']!r}")
        if "node" in args:
            self._node(args["node"])
        for n, item in enumerate(args["items"]):
            self._check_item(n, item)

    def _check_item(self, n: int, item: dict) -> None:
        def bad(msg):
            raise ToolError(f"item {n}: {msg}")
        if not str(item.get("image", "")).lower().endswith(IMAGE_EXTS):
            bad(f"image must be an image file ({', '.join(IMAGE_EXTS)}): got {item.get('image')!r}")
        if item.get("perspective") not in PERSPECTIVES:
            bad(f"perspective must be one of {PERSPECTIVES}: got {item.get('perspective')!r}")
        if not isinstance(item.get("environment_prompt"), str) or not item["environment_prompt"].strip():
            bad("environment_prompt must be a non-empty string")
        for key in ("character_prompt", "perspective_prompt"):
            if not isinstance(item.get(key, ""), str):
                bad(f"{key} must be a string")
        mask = item.get("subject_mask")
        if mask is not None and not str(mask).lower().endswith(IMAGE_EXTS):
            bad(f"subject_mask must be an image file ({', '.join(IMAGE_EXTS)}): got {mask!r}")
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            bad("turns must be a non-empty list")
        if len(turns) > self.max_turns:
            bad(f"at most {self.max_turns} turns: got {len(turns)}")
        for t, turn in enumerate(turns, 1):
            if not isinstance(turn, dict):
                bad(f"turn {t} must be an object")
            if not valid_action(turn.get("action")):
                bad(f"turn {t}: unknown action {turn.get('action')!r}; use W, A, S, D, left, right, up, "
                    f"down or stop, alone or joined with '+'")
            for key, value in turn.items():
                if key != "action" and (key not in TURN_TYPES or not isinstance(value, str)):
                    bad(f"turn {t}: {key!r} is not one of {TURN_TYPES} with a text value")

    def _node(self, node_id) -> dict:
        """The archive row of a node whose fine-tune can be rendered."""
        conn = open_db(self.run_dir)
        try:
            node = NodeStore(conn).get(node_id)
        except KeyError:
            raise ToolError(f"node {node_id!r} is not a node of this run") from None
        finally:
            conn.close()
        if node["status"] != "scored" or not node["checkpoint_path"]:
            raise ToolError(f"node {node_id!r} has no fine-tune to render (status {node['status']}); give a "
                            f"scored node other than the root, or omit `node` for the released model")
        return node

    def generator_name(self, job) -> str:
        node = job.args.get("node")
        return f"alayaworld-{job.args['variant']}" + (f"@{node}" if node else "")

    def produce(self, job, items, work, out, cancel, report):
        data = work / "data"
        for sub in ("cases", "images", "masks"):
            (data / sub).mkdir(parents=True, exist_ok=True)
        indices = []
        for item in items:
            i = item["index"]
            item["seed"] = job.args["seed"]         # provenance: cases are seeded from run.seed + case id
            try:
                image = f"images/case_{i}{Path(item['image']).suffix.lower()}"
                with Image.open(item["image"]) as im:
                    im.convert("RGB").save(data / image)
                mask = None
                if item.get("subject_mask"):
                    mask = f"masks/case_{i}_mask.png"
                    with Image.open(item["subject_mask"]) as im:
                        im.convert("L").save(data / mask)
            except Exception as exc:            # noqa: BLE001 -- a bad image/mask (e.g. PIL raising
                # DecompressionBombError, SyntaxError or ValueError on a broken file) must be this
                # item's error, not a job failure that orphans the others.
                (out / f"{i}.json").write_text(json.dumps(
                    {"ok": False, "error": f"input: {type(exc).__name__}: {exc}"}))
                continue
            (data / "cases" / f"case_{i}.json").write_text(
                json.dumps(case_json(i, item, image, mask), ensure_ascii=False, indent=1), encoding="utf-8")
            indices.append(i)
        if not indices:
            return None
        env, wm = self.block["env"], self.cfg.worldmodel
        # One timeout_s budget for precache + render: the precache is killed at the deadline like a
        # cancel, and the render gets what is left.
        deadline = time.monotonic() + self.timeout_s if self.timeout_s else None
        late = SimpleNamespace(is_set=lambda: deadline is not None and time.monotonic() > deadline)
        try:
            fine_tune = {}
            if job.args.get("node"):
                node = self._node(job.args["node"])
                checkpoint = self.run_dir / node["checkpoint_path"]
                # dmd4 renders as the eval does: the student LoRA and the node's as one adapter.
                lora = checkpoint if job.args["variant"] == "ar30" else concat_eval_lora(
                    self.cfg, checkpoint, work, self.recorder, job.node)
                fine_tune = dict(node_lora=lora, node_rank=node["lora_rank"],
                                 history_encoder=checkpoint / "history_encoder.pt")
            config = work / "render_config.yaml"
            config.write_text(yaml.safe_dump(render_config(
                self.cfg, variant=job.args["variant"], rounds_per_turn=job.args["rounds_per_turn"],
                seed=job.args["seed"], indices=indices, work=work,
                text_cache=self.run_dir / "cache" / "text_embed", **fine_tune), sort_keys=True), encoding="utf-8")
            # The eval renders with the text encoder off (a 24 GB card cannot hold Gemma next to the
            # DiT), so every prompt of these cases is encoded first, into the run's prompt cache.
            code = run_cancellable(
                env, ["python", *PRECACHE, "--config", str(config), "--device-map", "auto"], cwd=wm,
                cancel=SimpleNamespace(is_set=lambda: cancel.is_set() or late.is_set()),
                log_path=work / "precache.log", recorder=self.recorder, node=job.node,
                phase=self.tool, extra_env={"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus[:2])),
                                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"})
            if cancel.is_set():
                return None
            if late.is_set():
                raise RuntimeError(f"prompt precache timed out (job timeout_s {self.timeout_s}s)")
            if code != 0:
                raise RuntimeError(f"prompt precache failed (exit {code}):\n"
                                   f"{file_tail(work / 'precache.log', 2000)}")
            # run_wbench.py sets CUDA_VISIBLE_DEVICES from --gpus itself (PCI order is inherited).
            codes, missing = self.run_workers(env, lambda r, w: [
                "python", *RUN_WBENCH, "--config", str(config), "--gpus", ",".join(map(str, self.gpus)),
                "--cases", ",".join(map(str, indices)), "--master-port", str(free_port())],
                [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report, cwd=wm,
                deadline=deadline)
            videos = work / "videos"
            for i in indices:
                src = videos / f"case_{i}_combined"
                if src.with_suffix(".mp4").exists() and src.with_suffix(".json").exists():
                    shutil.move(src.with_suffix(".mp4"), out / f"{i}.mp4")
                    shutil.move(src.with_suffix(".json"), out / f"{i}.sidecar.json")
                    camera = src.with_name(src.name + "_camera.npz")
                    if camera.exists():
                        shutil.move(camera, out / f"{i}.camera.npz")
                    (out / f"{i}.json").write_text(json.dumps({"ok": True}))
                elif codes[0] == 0:
                    (out / f"{i}.json").write_text(json.dumps({"ok": False, "error": "no video rendered"}))
            return codes, missing
        finally:
            shutil.rmtree(data, ignore_errors=True)
            shutil.rmtree(work / "videos", ignore_errors=True)
            shutil.rmtree(work / "eval", ignore_errors=True)        # the concatenated LoRA

    def finish(self, job, item, out):
        i = item["index"]
        sidecar = json.loads((out / f"{i}.sidecar.json").read_text(encoding="utf-8"))
        video = out / f"{i}.mp4"
        frames = probe_video(video).frames
        with np.load(out / f"{i}.camera.npz") as z:
            saved = z["cam_c2w"].astype(np.float64)
        if len(saved) != frames:
            raise ValueError(f"camera path has {len(saved)} frames, the video {frames}")
        # Round r occupies mp4 frames [first + 32r, first + 32r + 32). The first latent of the
        # rollout decodes to one frame, so the mp4 starts 7 frames into round 0 (first = -7,
        # measured); turn_segments' frame ranges are nominal and ignore this.
        first = frames - ROUND_FRAMES * int(sidecar["output_rounds"])
        if not -ROUND_FRAMES < first <= 0:
            raise ValueError(f"{frames} frames for {sidecar['output_rounds']} rounds of {ROUND_FRAMES}")
        trim = (first - GRID) % ROUND_FRAMES
        trimmed = video if trim == 0 else out / f"{i}.trim.mp4"     # no re-encode when already aligned
        if trim:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf",
                            f"select='gte(n\\,{trim})',setpts=N/{FPS}/TB", "-r", str(FPS), "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(trimmed)],
                           check=True, timeout=FFMPEG_TIMEOUT_S)
        n = probe_video(trimmed).frames
        poses = saved[trim:]
        if n != len(poses):
            raise ValueError(f"trimmed video has {n} frames, the poses {len(poses)}")
        commanded = out / f"{i}.commanded.npz"     # metadata, never the pose (see the module docstring)
        np.savez(commanded, cam_c2w=(np.linalg.inv(poses[0]) @ poses).astype(np.float32))
        schedule = sidecar["prompt_schedule"]
        caption = self.write_caption(out, item, {
            "caption": schedule[0]["prompt"], "segments": round_segments(schedule, first, trim, n)})
        turns = [{**t, "frame_start": max(0, first + ROUND_FRAMES * t["chunk_start"] - trim),
                  "frame_end_exclusive": min(n, first + ROUND_FRAMES * t["chunk_end_exclusive"] - trim)}
                 for t in sidecar["turn_segments"]]
        turns = [{**t, "frame_count": t["frame_end_exclusive"] - t["frame_start"]} for t in turns]
        return {"video": trimmed, "caption": caption, "commanded_camera": commanded, "frames": n, "trim": trim, "actions": sidecar["actions"], "turn_segments": turns}


class Wan22Backend(GpuJob):
    """rollout_wan22: training clips from Wan2.2 TI2V-5B, the official
    code in its own env. Text-to-video, or image-to-video when the item
    carries a first frame (the bridge fits it to 1280x704). Every render is 1280x704; the published
    clip is center-cropped to 1248x704 and carries no pose/camera_motion: the agent adds one
    (annotate_camera then 'moving', or 'static'), like a clip it shot itself."""
    name = tool = "rollout_wan22"
    kind = "rollout"
    generator = "wan2.2-ti2v-5b"
    config_key = "generators.wan22"      # max_items / timeout_s
    file_keys = ("image",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        default, maximum = self.block["frames"]
        self.description = (
            "Render training clips with Wan 2.2 TI2V-5B (text-to-video, or image-to-video when an "
            "item carries a first frame). A GPU job: returns {job_id} at once; collect with job_wait. "
            f"`frames`: 4k+1, 1 < frames <= {maximum}, default {default} (one value "
            "for the whole job). Item: {'prompt': str, 'image'?: first frame under /workspace (any "
            "size; center-cropped and resized to 1280x704, e.g. a generate_images frame), 'seed': "
            "int}. Each result item gives a `candidate` for data_ingest "
            "(a 1248x704 (~16:9), 24 fps mp4 center-cropped from Wan's 1280x704, caption, provenance); it "
            "carries no pose/camera_motion -- add one (annotate_camera then 'moving', or 'static') "
            "before ingesting. Wan often ignores camera instructions like 'camera steady'; never label a clip 'static' from its prompt; run annotate_camera, or check the frames, first. Batch many prompts per call.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.get("frames", default)
        if not (is_int(frames) and 1 < frames <= maximum and frames % 4 == 1):
            raise ToolError(f"frames must be 4k+1 with 1 < frames <= {maximum}: got {frames!r}")
        args["frames"] = frames
        check_clip_items(args["items"])

    def produce(self, job, items, work, out, cancel, report):
        repo = checked_repo(self.cfg, self.config_key, "Wan2.2")
        ckpt = self.cfg.repo_root / self.block["weights"]
        extra = self.block.get("extra_args") or {}
        offload = "--offload-model" if extra.get("offload_model", True) else "--no-offload-model"
        t5_cpu = "--t5-cpu" if extra.get("t5_cpu", True) else "--no-t5-cpu"
        groups = split_gpus(self.gpus, int(self.block.get("gpus_per_worker", 1)), self.block.get("workers"))
        return self.run_workers(self.block["env"], lambda r, w: [
            "python", str(WAN22_BRIDGE), "--items", str(work / "items.json"), "--out", str(out),
            "--rank", str(r), "--world", str(w), "--repo", str(repo), "--ckpt-dir", str(ckpt),
            "--frames", str(job.args["frames"]), offload, t5_cpu], groups,
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report,
            cwd=self.cfg.repo_root,
            # the fit spike OOM'd at VAE decode with the default caching allocator even
            # though sampling itself stayed under budget (fragmentation, not a real shortage --
            # PyTorch's own OOM message suggests exactly this flag).
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    def finish(self, job, item, out):
        rendered = out / f"{item['index']}.mp4"
        raw = probe_video(rendered)
        if (raw.width, raw.height) != (1280, 704):     # the bridge fits every first frame to this
            raise ValueError(f"rendered clip is {raw.width}x{raw.height}, not 1280x704")
        cropped = out / f"{item['index']}.cropped.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(rendered), "-vf",
                        "crop=1248:704:16:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-an", str(cropped)], check=True, timeout=FFMPEG_TIMEOUT_S)
        info = probe_video(cropped)
        check_published(info)
        caption = self.write_caption(out, item, {"caption": item["prompt"]})
        return {"video": cropped, "caption": caption, "frames": info.frames}


class Ltx25Backend(GpuJob):
    """rollout_ltx25: training clips from LTX-2.5 (Lightricks'
    ltx-pipelines in its own env): `distilled` (DistilledPipeline) or `dev`
    (TI2VidTwoStagesPipeline + the distilled LoRA). On 24 GB cards each worker runs one GPU with
    fp8-cast weights and CPU offload, so each worker also holds the model in host RAM: the worker
    count shrinks to what MemAvailable holds (`worker_groups`), and a job that cannot fit one
    worker fails before any starts rather than meeting the host OOM killer. The clip carries no
    pose/camera_motion, like Wan's."""
    name = tool = "rollout_ltx25"
    kind = "rollout"
    config_key = "generators.ltx25"      # max_items / timeout_s
    file_keys = ("image",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        (h, w), (default, maximum) = self.block["resolutions"][0], self.block["frames"]
        self.description = (
            "Render training clips with LTX-2.5 (text-to-video, or image-to-video when an item carries a "
            "first frame). A GPU job: returns {job_id} at once; collect with job_wait. Params (one value per "
            f"job): `variant` (one of {self.enabled_variants()}; default the first), `frames` (8k+1, "
            f"1 < frames <= {maximum}, default {default}), `height`/`width` (one of "
            f"{self.block['resolutions']} as [height, width]; default {h}x{w}). Item: {{'prompt': str, "
            "'image'?: first frame under /workspace (any size; center-cropped and resized to the clip size, "
            "e.g. a generate_images frame), 'seed': int}. Each result item gives a `candidate` for "
            "data_ingest (a 24 fps, 16:9, silent mp4, caption, provenance); it carries no "
            "pose/camera_motion -- add one (annotate_camera then 'moving', or 'static') before ingesting. "
            "LTX often ignores camera instructions like 'camera steady'; never label a clip 'static' from its prompt; run annotate_camera, or check the frames, first. Batch many prompts per call.")

    def enabled_variants(self) -> list[str]:
        return enabled_variants(self.block, ("distilled", "dev"))

    def generator_name(self, job) -> str:
        return f"ltx-2.5-{job.args['variant']}"

    def check_args(self, args):
        enabled = self.enabled_variants()
        args.setdefault("variant", enabled[0] if enabled else None)
        if args["variant"] not in enabled:
            raise ToolError(f"variant must be one of the enabled variants {enabled}: got {args['variant']!r}")
        default, maximum = self.block["frames"]
        frames = args.setdefault("frames", default)
        if not (is_int(frames) and 1 < frames <= maximum):
            raise ToolError(f"frames must be 8k+1 with 1 < frames <= {maximum}: got {frames!r}")
        if frames % 8 != 1:
            raise ToolError(f"frames must be 8k+1: got {frames!r}")
        h, w = self.block["resolutions"][0]
        size = [args.setdefault("height", h), args.setdefault("width", w)]
        if not all(is_int(v) for v in size):
            raise ToolError(f"height and width must be ints: got {size}")
        if size not in [list(r) for r in self.block["resolutions"]]:
            raise ToolError(f"[height, width] must be one of the resolutions {self.block['resolutions']}: got {size}")
        check_clip_items(args["items"])

    def worker_groups(self, variant: str, n_items: int) -> list[list[int]]:
        """One GPU per worker; workers = min(GPUs (capped by the config's `workers`), items,
        floor((MemAvailable - host_reserve_gib) / peak_rss_gib)): the job shrinks to what free
        host RAM holds (with the reserve kept for the kernel and OS) and refuses only below one."""
        rss = float(self.block["variants"][variant]["peak_rss_gib"])
        reserve = float(self.block.get("host_reserve_gib", 60))
        avail = meminfo_gib()["MemAvailable"]
        by_ram = int((avail - reserve) // rss)
        groups = split_gpus(self.gpus, 1, self.block.get("workers"))
        n = min(len(groups), n_items, by_ram)
        if n < 1:
            raise RuntimeError(f"not enough free host RAM for one ltx-2.5-{variant} worker: MemAvailable "
                               f"{avail:.0f} GiB - host_reserve_gib {reserve:g} < peak_rss_gib {rss:g}; "
                               f"free host memory and retry")
        return groups[:n]

    def produce(self, job, items, work, out, cancel, report):
        repo = checked_repo(self.cfg, self.config_key, "LTX-2")
        a = job.args
        groups = self.worker_groups(a["variant"], len(items))
        return self.run_workers(self.block["env"], lambda r, w: [
            "python", str(LTX25_BRIDGE), "--items", str(work / "items.json"), "--out", str(out),
            "--rank", str(r), "--world", str(w), "--weights", str(self.cfg.repo_root / self.block["weights"]),
            "--variant", a["variant"], "--frames", str(a["frames"]), "--height", str(a["height"]),
            "--width", str(a["width"]), "--quantization", self.block.get("quantization", "fp8-cast"),
            "--offload", self.block.get("offload", "cpu")], groups,
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report, cwd=repo,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    def finish(self, job, item, out):
        a = job.args
        silent = out / f"{item['index']}.silent.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{item['index']}.mp4"), "-an",
                        "-c:v", "copy", str(silent)], check=True, timeout=FFMPEG_TIMEOUT_S)
        info = probe_video(silent)
        if (info.width, info.height) != (a["width"], a["height"]):
            raise ValueError(f"rendered clip is {info.width}x{info.height}, not {a['width']}x{a['height']}")
        check_published(info)
        if info.frames != a["frames"]:
            raise ValueError(f"rendered clip has {info.frames} frames, not {a['frames']}")
        caption = self.write_caption(out, item, {"caption": item["prompt"]})
        return {"video": silent, "caption": caption, "frames": info.frames}
