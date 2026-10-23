"""rollout_alayaworld: clips rendered by AlayaWorld from the agent's own items, through WorldModel's
turn-based render path (scripts/tools/run_wbench.py, configs/wbench_full.yaml).

The agent writes items (first frame, scene and subject text, seed, per-turn action, event and
perspective). The kernel stages them in the render script's input schema, pre-encodes their prompts,
renders them, and publishes each whole clip with its annotation: the item's texts and the frame each
turn starts on. Nothing the agent sees names that script, its schema or the evaluation.

No pose is published: the renders follow the commanded translation
but only weakly the commanded turns and orbits, so the commanded camera path is
not a pose label. It is published as metadata (`commanded_camera`); the agent runs
annotate_camera (ViGeo) on the clip for the pose it ingests with.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..archive.db import open_db
from ..archive.nodes import NodeStore
from ..data.annotation import ACTIONS, FPS, PERSPECTIVES, SWITCH
from ..data.ingest import SIZE
from ..data.probe import aspect_ok, probe_video
from ..subproc import file_tail, free_port, meminfo_gib
from .gpu_jobs import (IMAGE_EXTS, GpuJob, check_item_seed, check_keyframes, check_listed_size, is_int, sizes_text,
                       split_gpus)
from .jobs import run_cancellable
from .server import ToolError

WAN22_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "wan22_generate.py"
LTX25_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "ltx25_generate.py"

# The launches, as argv after "python" (run in the generator env, cwd WorldModel). The tests
# swap both for the fake worker.
PRECACHE = ["-m", "scripts.tools.precache_wbench_text_embeds"]
RUN_WBENCH = ["scripts/tools/run_wbench.py"]

ITEM_KEYS = ("image", "scene_prompt", "character_prompt", "seed", "subject_mask", "turns")
TURN_KEYS = ("action", "event", "perspective")
SWITCH_CODE = {"third_person": "fp_to_tp", "first_person": "tp_to_fp"}
# The AR teacher without the DMD LoRA, sampled as configs/infer_i2v_camera_ar.yaml does.
AR30 = {("paths", "dmd_resume"): None, ("validation", "sampling_steps"): 30,
        ("validation", "scheduler"): "shift", ("validation", "cfg_scale"): 3.0}
TURN_FRAMES = 96            # one turn of the render path, as in the evaluation


def check_turn_item(item: dict, keys: tuple[str, ...], max_turns: int | None = None) -> None:
    """The fields rollout_alayaworld and rollout_h3 share: scene and subject text, seed and turns."""
    def bad(msg):
        raise ToolError(msg)
    unknown = sorted(set(item) - set(keys))
    if unknown:
        bad(f"{', '.join(unknown)} is not a field of an item: the fields are {', '.join(keys)}")
    if not isinstance(item.get("scene_prompt"), str) or not item["scene_prompt"].strip():
        bad("scene_prompt must be a non-empty string")
    if not isinstance(item.get("character_prompt", ""), str):
        bad("character_prompt must be a string")
    check_item_seed(item)
    turns = item.get("turns")
    if not isinstance(turns, list) or not turns:
        bad("turns must be a non-empty list")
    if max_turns and len(turns) > max_turns:
        bad(f"at most {max_turns} turns: got {len(turns)}")
    previous = None
    for t, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            bad(f"turn {t} must be an object")
        unknown = sorted(set(turn) - set(TURN_KEYS))
        if unknown:
            bad(f"turn {t}: {', '.join(unknown)} is not a field of a turn: the fields are {', '.join(TURN_KEYS)}")
        if turn.get("action") not in ACTIONS:
            bad(f"turn {t}: action must be one of {', '.join(ACTIONS)}: got {turn.get('action')!r}")
        if turn.get("perspective") not in PERSPECTIVES:
            bad(f"turn {t}: perspective must be one of {', '.join(PERSPECTIVES)}: got {turn.get('perspective')!r}")
        if "event" in turn and (not isinstance(turn["event"], str) or not turn["event"].strip()):
            bad(f"turn {t}: event must be one sentence; leave it out when nothing happens")
        if previous and turn["perspective"] != previous and turn["action"] != "stop":
            bad(f"turn {t} switches the perspective, so its action must be stop: got {turn['action']!r}")
        previous = turn["perspective"]


def item_annotation(item: dict, starts: list[int]) -> dict:
    """The annotation of a rendered item: its texts, and each turn's planned first frame as seconds."""
    turns = [{"start_s": start / FPS, **({"event": t["event"]} if t.get("event") else {}),
              "perspective": t["perspective"]} for t, start in zip(item["turns"], starts)]
    return {"scene_prompt": item["scene_prompt"],
            **({"character_prompt": item["character_prompt"]} if item.get("character_prompt") else {}),
            "turns": turns}


def case_json(index: int, item: dict, image_rel: str, mask_rel: str | None) -> dict:
    """One item in the render script's input schema: a navigation entry per turn, the turn's event,
    and on a turn that changes the perspective the switch sentence as that entry's own prompt."""
    interactions, previous = [], None
    for turn, t in enumerate(item["turns"], 1):
        interactions.append({"type": "navigation", "action": t["action"], "turn": turn})
        if t.get("event"):
            interactions.append({"type": "event_edit", "action": t["event"], "turn": turn})
        if previous and t["perspective"] != previous:
            interactions.append({"type": "perspective_switch", "action": SWITCH_CODE[t["perspective"]],
                                 "prompt": SWITCH[t["perspective"]], "turn": turn})
        previous = t["perspective"]
    return {"id": str(index), "seed": item["seed"], "environment_prompt": item["scene_prompt"],
            "character_prompt": item.get("character_prompt", ""),
            "perspective_prompt": "",          # camera text: the render removes it before the model sees the prompt
            "settings": {"perspective": item["turns"][0]["perspective"], "subject": {"type": "unknown", "desc": ""},
                         "tracking_object": None, "initial_image": image_rel, "subject_mask": mask_rel},
            "interactions": interactions, "metric_list": []}


def render_config(cfg, *, indices: list[int], work: Path,
                  text_cache: Path, node_lora: Path | None = None, node_rank: int = 0,
                  history_encoder: Path | None = None) -> dict:
    """configs/wbench_full.yaml as eval/render.py:build_render_config adapts it, pointed at this
    job's cases and sampled with the AR teacher (AR30). The prompt cache is the run's (the eval's
    own holds only WBench's prompts). `node_lora`: a node's checkpoint (LoRA rank `node_rank`) with
    its `history_encoder`."""
    wm = cfg.worldmodel
    c = yaml.safe_load((wm / "configs" / "wbench_full.yaml").read_text(encoding="utf-8"))
    for key, value in c["paths"].items():
        if isinstance(value, str) and value:
            c["paths"][key] = str(wm / value)
    c["run"].update(output_dir=str(work / "rollout"), log_dir=str(work / "logs"))
    c["runtime"]["text_embed_cache_dir"] = str(text_cache)
    c["validation"]["save_joystick"] = False
    mode = c["validation"]["modes"]["wbench"]
    mode["dataset"].update(root=str(work / "data"), case_ids=[str(i) for i in indices])
    mode["wbench_output_dir"] = str(work / "videos")
    mode["wbench_keep_last_frame"] = True      # every turn is TURN_FRAMES long, the last one too
    for (section, key), value in AR30.items():
        c[section][key] = value
    if node_lora is not None:
        c["paths"].update(dmd_resume=str(node_lora), history_encoder=str(history_encoder))
        c["lora"]["rank"] = c["lora"]["alpha"] = node_rank
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


def check_clip_item(item: dict) -> None:
    """An item of a text/image-to-video job: {'prompt': str, 'image'?: path, 'seed': int}."""
    if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
        raise ToolError("prompt must be a non-empty string")
    if item.get("image") is not None and not isinstance(item["image"], str):
        raise ToolError("image must be a file path")
    check_item_seed(item)


def check_published(info) -> None:
    """A published clip is 24 fps and within 2% of 16:9."""
    if round(info.fps) != FPS:
        raise ValueError(f"published clip is {info.fps} fps, not 24")
    if not aspect_ok(info, 0.02):
        raise ValueError(f"published clip's aspect {info.display_aspect:.4f} is not within "
                         f"2% of 16:9 ({info.width}x{info.height})")


class AlayaWorldBackend(GpuJob):
    name = tool = "rollout_alayaworld"
    kind = "rollout"
    config_key = "generators.alayaworld"
    file_keys = frame_keys = ("image", "subject_mask")
    description = (
        "Use this tool when a clip's camera has to follow commanded moves, or the clip has to come from a node's "
        "fine-tune. "
        "Render clips with AlayaWorld itself: the released model that every node fine-tunes, or with `node` a "
        "scored node's fine-tune. A GPU job: returns {job_id} at once; collect with job_wait. Each item is a "
        "first frame, the scene and subject text, a seed and a list of turns. The first frame, and a subject mask, "
        "must be exactly {size}: it is used as it is. A turn lasts 4 seconds. It has a navigation action, "
        "a perspective, and may have one event: one sentence about something that happens during the turn. "
        "The kernel writes each turn's text from the scene, the subject and the turn's event, and adds its own "
        "sentence on a perspective change and on every third-person turn. "
        "The camera keeps the first turn's mode for the whole clip. "
        "Actions steer translation reliably, rotation (turns, orbits) only weakly; an event may show up "
        "early, late, weakly or not at all. Each result item gives a `candidate`: `video` ({size}, 24 fps), "
        "`annotation` (a file: your scene and subject text and the turns, each with the time it starts in the "
        "clip) and `derived_from`. It has no pose: run annotate_camera on candidate.video, then data_ingest the "
        "video, the annotation and that pose. Use locate_events to find each event's span in the clip. Metadata, "
        "not labels: `commanded_camera` (npz of the camera path the actions commanded, one pose per frame; not "
        "what the video shows) and `actions`. "
        "At most {max_turns} turns per item.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = int(self.block["max_turns"])
        self.size = SIZE
        self.description = self.description.replace("{max_turns}", str(self.max_turns)).replace(
            "{size}", f"{self.size[0]}x{self.size[1]}")

    def frame_size(self, args):
        return self.size

    def check_args(self, args):
        if "node" in args:
            self._node(args["node"])

    def check_item(self, item: dict) -> None:
        if not str(item.get("image", "")).lower().endswith(IMAGE_EXTS):
            raise ToolError(f"image must be an image file ({', '.join(IMAGE_EXTS)}): got {item.get('image')!r}")
        mask = item.get("subject_mask")
        if mask is not None and not str(mask).lower().endswith(IMAGE_EXTS):
            raise ToolError(f"subject_mask must be an image file ({', '.join(IMAGE_EXTS)}): got {mask!r}")
        check_turn_item(item, ITEM_KEYS, self.max_turns)

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
        return "alayaworld-ar30" + (f"@{node}" if node else "")

    def produce(self, job, items, work, out, cancel, report):
        data = work / "data"
        for sub in ("cases", "images", "masks"):
            (data / sub).mkdir(parents=True, exist_ok=True)
        indices = []
        for item in items:
            i = item["index"]
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
        try:
            fine_tune = {}
            if job.args.get("node"):
                node = self._node(job.args["node"])
                checkpoint = self.run_dir / node["checkpoint_path"]
                fine_tune = dict(node_lora=checkpoint, node_rank=node["lora_rank"],
                                 history_encoder=checkpoint / "history_encoder.pt")
            config = work / "render_config.yaml"
            config.write_text(yaml.safe_dump(render_config(
                self.cfg, indices=indices, work=work,
                text_cache=self.run_dir / "cache" / "text_embed", **fine_tune), sort_keys=True), encoding="utf-8")
            # The eval renders with the text encoder off (a 24 GB card cannot hold Gemma next to the
            # DiT), so every prompt of these cases is encoded first, into the run's prompt cache.
            code = run_cancellable(
                env, ["python", *PRECACHE, "--config", str(config), "--device-map", "auto"], cwd=wm,
                cancel=cancel,
                log_path=work / "precache.log", recorder=self.recorder, node=job.node,
                phase=self.tool, extra_env={"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus[:2])),
                                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"})
            if cancel.is_set():
                return None
            if code != 0:
                raise RuntimeError(f"prompt precache failed (exit {code}):\n"
                                   f"{file_tail(work / 'precache.log', 2000)}")
            videos = work / "videos"
            # run_wbench.py sets CUDA_VISIBLE_DEVICES from --gpus itself (PCI order is inherited).
            codes, missing = self.run_workers(env, lambda r, w: [
                "python", *RUN_WBENCH, "--config", str(config), "--gpus", ",".join(map(str, self.gpus)),
                "--cases", ",".join(map(str, indices)), "--master-port", str(free_port())],
                [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report, cwd=wm,
                done=lambda: len(items) - len(indices) + len(list(videos.glob("case_*_combined.json"))))
            for i in indices:
                src = videos / f"case_{i}_combined"
                if src.with_suffix(".mp4").exists() and src.with_suffix(".json").exists():
                    shutil.move(src.with_suffix(".mp4"), out / f"{i}.mp4")
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

    def finish(self, job, item, out):
        i = item["index"]
        video = out / f"{i}.mp4"
        info = probe_video(video)
        check_published(info)
        turns = len(item["turns"])
        # mp4 frame 0 is the input frame, so turn k starts on frame 1 + TURN_FRAMES * k (the first on 0)
        if info.frames != 1 + TURN_FRAMES * turns:
            raise ValueError(f"{info.frames} frames for {turns} turns of {TURN_FRAMES}")
        with np.load(out / f"{i}.camera.npz") as z:
            poses = z["cam_c2w"].astype(np.float64)
        if len(poses) != info.frames:
            raise ValueError(f"camera path has {len(poses)} frames, the video {info.frames}")
        commanded = out / f"{i}.commanded.npz"     # metadata, never the pose (see the module docstring)
        np.savez(commanded, cam_c2w=(np.linalg.inv(poses[0]) @ poses).astype(np.float32))
        starts = [0] + [1 + TURN_FRAMES * k for k in range(1, turns)]
        return {"video": video, "annotation": self.write_annotation(out, item, item_annotation(item, starts)),
                "commanded_camera": commanded, "frames": info.frames,
                "actions": [t["action"] for t in item["turns"]]}


class Wan22Backend(GpuJob):
    """rollout_wan22: training clips from Wan2.2 TI2V-5B, the official
    code in its own env. Text-to-video, or image-to-video when the item
    carries a first frame (the bridge fits it to 1280x704). Every render is 1280x704; the published
    clip is center-cropped to 1248x704 and carries no pose and no annotation: the agent adds both,
    like a clip it found itself."""
    name = tool = "rollout_wan22"
    kind = "rollout"
    generator = "wan2.2-ti2v-5b"
    config_key = "generators.wan22"
    file_keys = ("image",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        default, maximum = self.block["frames"]
        self.description = (
            "Render training clips with Wan 2.2 TI2V-5B (text-to-video, or image-to-video when an "
            "item carries a first frame). A GPU job: returns {job_id} at once; collect with job_wait. "
            f"`frames`: 4k+1, 1 < frames <= {maximum}, default {default} (one value "
            "for the whole job). Item: {'prompt': str, 'image'?: first frame under /workspace (any "
            "size; center-cropped and resized to 1280x704), 'seed': "
            "int}. Each result item gives a `candidate`: `video` (a 1248x704 (~16:9), 24 fps mp4 "
            "center-cropped from Wan's 1280x704), the `prompt` used and `derived_from`. To ingest it, bring "
            "the clip to 960x544, run annotate_camera on the result and write its annotation (knowledge file "
            "data_formats). Wan often ignores camera instructions like 'camera steady'. Batch many prompts "
            "per call.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.get("frames", default)
        if not (is_int(frames) and 1 < frames <= maximum and frames % 4 == 1):
            raise ToolError(f"frames must be 4k+1 with 1 < frames <= {maximum}: got {frames!r}")
        args["frames"] = frames

    check_item = staticmethod(check_clip_item)

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
                        "-an", str(cropped)], check=True)
        info = probe_video(cropped)
        check_published(info)
        return {"video": cropped, "prompt": item["prompt"], "frames": info.frames}


class Ltx25Backend(GpuJob):
    """rollout_ltx25: training clips from LTX-2.5 (Lightricks'
    ltx-pipelines in its own env): the dev model (TI2VidTwoStagesPipeline + the distilled LoRA
    for stage 2). On 24 GB cards each worker runs one GPU with
    fp8-cast weights and CPU offload, so each worker also holds the model in host RAM: the worker
    count shrinks to what MemAvailable holds (`worker_groups`), and a job that cannot fit one
    worker fails before any starts rather than meeting the host OOM killer. The clip carries no
    pose and no annotation, like Wan's."""
    name = tool = "rollout_ltx25"
    kind = "rollout"
    config_key = "generators.ltx25"
    generator = "ltx-2.5-dev"
    takes_keyframes = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        (h, w), (default, maximum) = self.block["resolutions"][0], self.block["frames"]
        self.description = (
            "Use this tool when a clip has to show a given image at a frame other than its first, which the other "
            "rollout tools cannot. "
            "Render training clips with LTX-2.5 from a text prompt, optionally pinned to one image at a chosen "
            "frame. A GPU job: returns {job_id} at once; collect with job_wait. Params (one value per "
            "job): `frames` (8k+1, "
            f"1 < frames <= {maximum}, default {default}), `width`/`height` (one of "
            f"{sizes_text(self.block['resolutions'])}; default {w}x{h}). Item: {{'prompt': str, "
            "'keyframes'?: [{'image': file under /workspace, 'frame': int}] (at most one), 'seed': int}. A "
            "keyframe at frame 0 is the first frame and one at -1 the last; any frame in between also works. A "
            "keyframe image must be exactly the job's width x height: it is used as it is. "
            "Each result item gives a `candidate`: "
            "`video` (a 24 fps, 16:9, silent mp4), the `prompt` used and `derived_from`. To ingest it, bring "
            "the clip to 960x544, run annotate_camera on the result and write its annotation (knowledge file "
            "data_formats). LTX often ignores camera instructions like 'camera steady'. Batch many prompts "
            "per call.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.setdefault("frames", default)
        if not (is_int(frames) and 1 < frames <= maximum):
            raise ToolError(f"frames must be 8k+1 with 1 < frames <= {maximum}: got {frames!r}")
        if frames % 8 != 1:
            raise ToolError(f"frames must be 8k+1: got {frames!r}")
        check_listed_size(self.cfg, args, self.block["resolutions"])

    def check_item(self, item):
        if "image" in item:
            raise ToolError("image is not a field of this tool: use keyframes [{'image': path, 'frame': 0}]")
        check_clip_item(item)
        check_keyframes(item)

    def check_item_for(self, item, args):
        frames = args["frames"]
        seen = [frames - 1 if k["frame"] == -1 else k["frame"] for k in item.get("keyframes") or []]
        if any(f >= frames for f in seen):
            raise ToolError(f"a keyframe frame is outside the clip: frames are 0..{frames - 1}, or -1 for the last")

    def worker_groups(self, n_items: int) -> list[list[int]]:
        """One GPU per worker; workers = min(GPUs (capped by the config's `workers`), items,
        floor((MemAvailable - host_reserve_gib) / peak_rss_gib)): the job shrinks to what free
        host RAM holds (with the reserve kept for the kernel and OS) and refuses only below one."""
        rss = float(self.block["peak_rss_gib"])
        reserve = float(self.block.get("host_reserve_gib", 60))
        avail = meminfo_gib()["MemAvailable"]
        by_ram = int((avail - reserve) // rss)
        groups = split_gpus(self.gpus, 1, self.block.get("workers"))
        n = min(len(groups), n_items, by_ram)
        if n < 1:
            raise RuntimeError(f"not enough free host RAM for one ltx-2.5 worker: MemAvailable "
                               f"{avail:.0f} GiB - host_reserve_gib {reserve:g} < peak_rss_gib {rss:g}; "
                               f"free host memory and retry")
        return groups[:n]

    def produce(self, job, items, work, out, cancel, report):
        repo = checked_repo(self.cfg, self.config_key, "LTX-2")
        a = job.args
        groups = self.worker_groups(len(items))
        return self.run_workers(self.block["env"], lambda r, w: [
            "python", str(LTX25_BRIDGE), "--items", str(work / "items.json"), "--out", str(out),
            "--rank", str(r), "--world", str(w), "--weights", str(self.cfg.repo_root / self.block["weights"]),
            "--frames", str(a["frames"]), "--height", str(a["height"]),
            "--width", str(a["width"]), "--quantization", self.block.get("quantization", "fp8-cast"),
            "--offload", self.block.get("offload", "cpu")], groups,
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report, cwd=repo,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    def finish(self, job, item, out):
        a = job.args
        silent = out / f"{item['index']}.silent.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{item['index']}.mp4"), "-an",
                        "-c:v", "copy", str(silent)], check=True)
        info = probe_video(silent)
        if (info.width, info.height) != (a["width"], a["height"]):
            raise ValueError(f"rendered clip is {info.width}x{info.height}, not {a['width']}x{a['height']}")
        check_published(info)
        if info.frames != a["frames"]:
            raise ValueError(f"rendered clip has {info.frames} frames, not {a['frames']}")
        return {"video": silent, "prompt": item["prompt"], "frames": info.frames}
