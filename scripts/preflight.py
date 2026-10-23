"""Run the riskiest GPU paths of a run once, at their largest sizes, on the machine a config describes.

    python scripts/preflight.py --config configs/h100.yaml                     # the default steps
    python scripts/preflight.py --config configs/h100.yaml --steps vllm h3     # any of STEPS

Run from AutoResearcher/ in the autoresearcher env, after scripts/download_weights.py, with the three repos
committed (it starts a run, runs/preflight-<time>, like `ar run`). Every step runs even after one fails;
the exit code is 1 when any failed. Each step reports its peak GPU memory per card and the lowest
MemAvailable seen, also written to runs/preflight-<time>/preflight.json.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import download_weights
from ar_kernel.archive.commits import CommitStore
from ar_kernel.config import KernelConfig, load_dotenv
from ar_kernel.data.ingest import Candidate, Ingestor
from ar_kernel.doctor import gpu_checks, run_checks
from ar_kernel.eval.lora import concat_eval_lora
from ar_kernel.eval.render import build_render_config, render_proxy
from ar_kernel.run import bootstrap_run
from ar_kernel.subproc import cache_env, free_port, meminfo_gib
from ar_kernel.telemetry import files
from ar_kernel.tools.annotate import AnnotateBackend
from ar_kernel.tools.captioner import CaptionBackend
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.h3 import H3Backend
from ar_kernel.tools.images import ImageBackend
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.locate_events import LocateEventsBackend
from ar_kernel.tools.rollouts import AlayaWorldBackend, Ltx25Backend
from ar_kernel.tools.vllm_server import VllmServer, gpu_memory_mib, local_client, serve_command
from ar_kernel.train.gate import Gate, resolve_config
from ar_kernel.train.recipe import lora_of
from ar_kernel.train.runner import TrainRunner

STEPS = ("doctor", "weights", "vllm", "train", "render", "ltx25", "images", "annotate", "alayaworld", "h3")
DEFAULT = ("doctor", "weights", "vllm", "train", "render", "ltx25")
ACTIONS = ("W", "left", "W+D", "stop", "right", "S", "up", "W+A", "down")


class Failed(Exception):
    pass


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


class Preflight:
    def __init__(self, cfg: KernelConfig) -> None:
        self.cfg = cfg
        self.ctx = bootstrap_run(cfg, run_id=f"preflight-{time.strftime('%Y%m%d-%H%M%S')}", env=os.environ)
        self.run_dir, self.gpus, self.rec = self.ctx.run_dir, self.ctx.gpus, self.ctx.recorder
        self.ws, self.staging = self.run_dir / "workspace", self.run_dir / "staging"
        self.ws.mkdir(), self.staging.mkdir()
        self.examples = cfg.worldmodel / "data" / "examples" / "video_timed_prompts_camera"
        self.clips = sorted((self.examples / "videos").glob("*.mp4"))
        ffmpeg("-i", self.clips[0], "-frames:v", "1", self.ws / "first.png")
        self.checkpoint = self.rank = None

    # --- GPU jobs, through the same queue the agents use ---
    def job(self, backend_cls, tool: str, args: dict):
        reg = TokenRegistry(self.rec)
        q = JobQueue(self.rec, threading.Lock(), wait_cap_s=3600)
        q.register(backend_cls(self.cfg, self.run_dir, self.gpus, reg, self.rec))
        caller = reg.issue(node="preflight", phase="improve_recipe", attempt=1, workspace_host=self.ws,
                           staging_host=self.staging)
        try:
            job = q.backends[tool].submit(q, caller, args)
            out = q.wait(caller, job["job_id"], 3600)
            while out["state"] not in ("done", "failed", "canceled"):
                out = q.wait(caller, job["job_id"], 3600)
        finally:
            q.shutdown()
        if out["state"] != "done":
            raise Failed(f"{tool} job {out['state']}: {out['error']}")
        result = json.loads((self.staging / Path(out["result_file"]).relative_to("/workspace/staging")).read_text())
        if result.get("gpu_memory_released") is False:
            raise Failed(f"{tool}: GPU memory was not released after the job")
        return result

    def items_ok(self, result: dict, key: str = "items") -> dict:
        errors = [i for i in result[key] if "error" in i]
        if errors:
            raise Failed(f"{len(errors)} of {len(result[key])} items failed, e.g. {errors[0]}")
        return {"items": len(result[key]), "workers": [i.get("worker") for i in result[key]][:2]}

    # --- steps ---
    def doctor(self):
        bad = [f for f in run_checks(self.cfg) + gpu_checks(self.cfg) if f.level == "fail"]
        if bad:
            raise Failed("; ".join(f"{f.check}: {f.detail}" for f in bad))

    def weights(self):
        missing = [e["name"] for e in download_weights.entries(self.cfg) if not download_weights.done(self.cfg, e)]
        if missing:
            raise Failed(f"not downloaded by scripts/download_weights.py: {missing}")

    def vllm(self):
        """One captioner server for all its uses: launched with the local judge's media limits (the most per
        prompt) and a media folder, it gets caption_videos', locate_events' and the judge's requests, each kind
        `concurrent_requests` at once. The tools launch the same command with smaller limits."""
        c = self.cfg.get("captioner")
        n = int(c["concurrent_requests"])
        media = self.run_dir / "vllm_media"
        media.mkdir()
        clips = []
        for i, clip in zip(range(n), itertools.cycle(self.clips)):
            os.link(clip, media / f"{i}.mp4")
            clips.append(media / f"{i}.mp4")
        frames = self.run_dir / "vllm_frames"
        frames.mkdir()
        ffmpeg("-i", self.clips[0], "-vf", "select=not(mod(n\\,2))", "-vsync", "vfr",
               "-frames:v", self.cfg.get("eval.judge.max_images"), frames / "%03d.jpg")
        urls = ["data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode() for f in sorted(frames.glob("*.jpg"))]
        video = "data:video/mp4;base64," + base64.b64encode(self.clips[0].read_bytes()).decode()
        judge = c["model"]
        port = free_port()
        command = serve_command(c, self.gpus, port, "captioner", media,
                                mm_limits={"image": self.cfg.get("eval.judge.max_images"), "video": 1})
        names = command.index("--served-model-name") + 1
        command[names:names + 1] = ["captioner", "locator", judge]       # the names the three callers use
        server = VllmServer(c["env"], command, self.gpus, port, cwd=self.run_dir, log_path=self.run_dir / "vllm.log",
                            recorder=self.rec, node="preflight", phase="vllm", label="vLLM server")
        caption, locate = CaptionBackend(self.cfg, self.run_dir, self.gpus, None, self.rec), \
            LocateEventsBackend(self.cfg, self.run_dir, self.gpus, None, self.rec)

        def ask_judge(content):
            done = client.chat.completions.create(
                model=judge, max_tokens=10, extra_body=self.cfg.get("eval.judge.extra_body"),
                messages=[{"role": "user", "content": [*content, {"type": "text", "text": "Is it daytime? Yes or no."}]}])
            if not (done.choices[0].message.content or "").strip():
                raise Failed("the judge gave an empty answer")
            return done.choices[0].message.content.strip()

        try:
            load_s = server.start()
            with local_client(server.base_url, float(c["clip_timeout_s"]), server.api_key) as client, \
                    ThreadPoolExecutor(max_workers=n) as pool:
                captions = list(pool.map(lambda f: caption._caption(client, f, "Describe the scene in one sentence.", c),
                                         clips))
                answers = list(pool.map(lambda u: locate.ask(client, u, "A car drives past.")[0], urls[:n]))
                judged = list(pool.map(ask_judge, [[{"type": "video_url", "video_url": {"url": video}}],
                                                   [{"type": "image_url", "image_url": {"url": u}} for u in urls]] * (n // 2)))
        finally:
            server.stop()
        bad = [x for x in captions if len(x.get("caption", "")) < 20]
        if bad:
            raise Failed(f"{len(bad)} of {n} captions failed, e.g. {bad[0]}")
        return {"load_s": round(load_s), "captions": n, "caption": captions[0]["caption"][:200],
                "locate": answers[0], "judge": judged[:2], "judge_images": len(urls)}

    def images(self):
        return self.items_ok(self.job(ImageBackend, "generate_images", {"width": 1920, "height": 1088, "items": [
            {"prompt": f"a photorealistic first-person view of a busy market street, variant {i}", "seed": i}
            for i in range(len(self.gpus))]}))

    def annotate(self):
        frames = int(self.cfg.get("annotate.max_frames"))
        ffmpeg("-stream_loop", "-1", "-i", self.clips[0], "-frames:v", frames, "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", self.ws / "long.mp4")
        return self.items_ok(self.job(AnnotateBackend, "annotate_camera",
                                      {"items": [{"video": "long.mp4"} for _ in self.gpus]}))

    def alayaworld(self):
        turns = int(self.cfg.get("generators.alayaworld.max_turns"))
        items = [{"image": "first.png", "seed": i,
                  "scene_prompt": "A quiet city street lined with trees and parked cars under a grey sky.",
                  "turns": [{"action": ACTIONS[(i + t) % len(ACTIONS)], "perspective": "first_person",
                             **({"event": "A cyclist rides past."} if t % 2 else {})} for t in range(turns)]}
                 for i in range(len(self.gpus))]
        return self.items_ok(self.job(AlayaWorldBackend, "rollout_alayaworld", {"items": items}))

    def ltx25(self):
        (h, w), frames = self.cfg.get("generators.ltx25.resolutions")[0], self.cfg.get("generators.ltx25.frames")[1]
        ffmpeg("-i", self.ws / "first.png", "-vf", f"scale={w}:{h}", self.ws / "ltx_key.png")
        items = [{"prompt": "A slow walk along a quiet city street lined with trees, camera steady.", "seed": i,
                  **({"keyframes": [{"image": "ltx_key.png", "frame": 0}]} if i % 2 else {})}
                 for i in range(len(self.gpus))]
        return self.items_ok(self.job(Ltx25Backend, "rollout_ltx25",
                                      {"frames": frames, "width": w, "height": h, "items": items}))

    def h3(self):
        seconds = self.cfg.get("generators.h3.frames")[1] / 24
        items = [{"scene_prompt": "Live-action, a quiet harbor at dawn, fishing boats", "seed": i, "seconds": seconds,
                  **({"image": "first.png"} if i else {}),
                  "turns": [{"action": "W", "perspective": "first_person"},
                            {"action": "right", "event": "A seagull lands on the nearest boat.",
                             "perspective": "first_person"}]} for i in range(2)]
        return self.items_ok(self.job(H3Backend, "rollout_h3", {"items": items}))

    def train(self):
        """A few optimizer steps of the config's base recipe on the WorldModel example clips (and re-encoded
        copies of them, enough for one optimizer step)."""
        ing = Ingestor(self.cfg, self.run_dir, self.ctx.conn, self.rec)
        accum = yaml.safe_load(self.cfg.base_recipe.read_text())["optimizer"]["grad_accum_steps"]
        clip_ids = []
        for copy in range(-(-len(self.gpus) * accum // len(self.clips))):
            for video in self.clips:
                d = self.staging / "train" / f"{video.stem}-{copy}"
                d.mkdir(parents=True)
                caption = json.loads((self.examples / "captions" / f"{video.stem}.json").read_text())
                turns = [{"start_s": s["time_range_s"][0], "perspective": "first_person",
                          **({"event": e} if (e := s["prompt"].removeprefix(caption["caption"]).strip()) else {})}
                         for s in caption["segments"]]
                (d / "annotation.json").write_text(json.dumps({"scene_prompt": caption["caption"], "turns": turns}))
                ffmpeg("-i", video, "-c:v", "libx264", "-crf", 16 + copy, "-pix_fmt", "yuv420p", d / "video.mp4")
                shutil.copy2(self.examples / "poses" / f"{video.stem}.npz", d / "pose.npz")
                named = {k: f"/workspace/staging/train/{d.name}/{f}"
                         for k, f in (("video", "video.mp4"), ("annotation", "annotation.json"), ("pose", "pose.npz"))}
                with files.scope(self.rec, node="preflight", phase="improve_recipe", attempt=1, tool="data_ingest"):
                    res = ing.ingest([Candidate(video=d / "video.mp4", annotation=d / "annotation.json",
                                                pose=d / "pose.npz", named=named,
                                                derived_from={"kind": "dataset", "name": "WorldModel example clips"})],
                                     node_id="preflight")[0]
                if not res.accepted:
                    raise Failed(f"ingest refused {d.name}: {res.reasons}")
                clip_ids.append(res.clip_id)
        commits = CommitStore(self.ctx.conn, ing.blobs, ing.clips)
        commit = commits.commit(None, {"examples": {"weight": 1.0, "clips": clip_ids}}, "preflight", node_id="preflight")
        node_dir = self.run_dir / "nodes" / "preflight"
        recipe = {"optimizer.max_steps": 3, "optimizer.epochs": 500}
        failures = Gate(self.cfg, commits, self.rec).check(recipe, commit, None, "preflight", self.run_dir / "gate",
                                                          self.run_dir, self.gpus)
        if failures:
            raise Failed(f"gate: {failures}")
        resolved = resolve_config(self.cfg, commits, self.run_dir, recipe, commit, "preflight", node_dir)
        runner = TrainRunner(self.cfg, self.rec)
        runner.precache(resolved, self.gpus, "preflight")
        self.checkpoint = runner.train(resolved, self.gpus, "preflight", node_dir)
        self.rank = lora_of(resolved)[0]
        return {"clips": len(clip_ids), "checkpoint": str(self.checkpoint)}

    def render(self):
        """The eval's render of the trained node (the released model when train failed) on the proxy cases with
        the most turns, one per GPU: the longest rollout and VAE decode a run renders."""
        cases = self.run_dir / "config" / "proxy_cases.txt"
        turns = lambda c: max(int(i.get("turn", 0) or 0) for i in json.loads(
            (self.cfg.eval_data / "cases" / f"case_{c}.json").read_text())["interactions"])
        chosen = sorted(cases.read_text().strip().split(","), key=turns, reverse=True)[:len(self.gpus)]
        node_dir = self.run_dir / "nodes" / "preflight-render"
        lora = None
        if self.checkpoint is not None:
            lora = concat_eval_lora(self.cfg, self.checkpoint, node_dir, self.rec, "preflight-render")
        history = self.checkpoint / "history_encoder.pt" if self.checkpoint else \
            self.cfg.worldmodel / "weights/alaya-world-ar/history_encoder.pt"
        config = build_render_config(self.cfg, lora, self.rank or 0, history, node_dir / "videos", chosen, node_dir)
        videos = render_proxy(self.cfg, config, self.gpus, "preflight-render", self.rec, chosen)
        return {"cases": chosen, "turns": [turns(c) for c in chosen], "trained": self.checkpoint is not None,
                "videos": len(list(videos.glob("case_*_combined.mp4")))}


class Sampler:
    """The peak GPU memory per card and the lowest MemAvailable while the `with` block runs."""
    def __init__(self, gpus: list[int]) -> None:
        self.gpus, self.peak, self.low, self.stop = gpus, {}, float("inf"), threading.Event()

    def loop(self) -> None:
        while not self.stop.is_set():
            for g, used in (gpu_memory_mib(self.gpus) or {}).items():
                self.peak[g] = max(self.peak.get(g, 0), used)
            self.low = min(self.low, meminfo_gib()["MemAvailable"])
            self.stop.wait(2)

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join()
        self.stats = {"peak_gpu_mib": self.peak, "min_mem_available_gib": round(self.low, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=[], metavar="FILE", help="kernel.yaml overlays, as for `ar run`")
    ap.add_argument("--steps", nargs="+", default=list(DEFAULT), choices=STEPS)
    args = ap.parse_args(argv)
    os.environ.update(cache_env())
    cfg = KernelConfig.load(overlays=args.config)
    load_dotenv(cfg.repo_root / ".env", os.environ)
    pf = Preflight(cfg)
    print(f"preflight run {pf.run_dir.name} on GPUs {pf.gpus}", flush=True)
    report = {}
    for step in [s for s in STEPS if s in args.steps]:
        print(f"== {step}", flush=True)
        t0, sampler = time.monotonic(), Sampler(pf.gpus)
        try:
            with sampler:
                entry = {"ok": True, "detail": getattr(pf, step)()}
        except Exception as exc:                         # noqa: BLE001 -- reported; the next step still runs
            entry = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        report[step] = {**entry, "seconds": round(time.monotonic() - t0), **sampler.stats}
        print(json.dumps({k: v for k, v in report[step].items() if k != "traceback"}, indent=1, default=str), flush=True)
        (pf.run_dir / "preflight.json").write_text(json.dumps(report, indent=1, default=str))
    failed = [s for s, r in report.items() if not r["ok"]]
    print(f"preflight {pf.run_dir.name}: " + (f"FAILED {failed}" if failed else f"all {len(report)} steps passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
