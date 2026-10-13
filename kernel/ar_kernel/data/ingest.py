from __future__ import annotations
import hashlib, json, os, shutil, stat, subprocess, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig, run_config_path
from ..isolation import EXCLUDED_CLIP, copies_held_out
from . import annotation as ann
from .checker import CheckerError, check_clip
from .leakage import LeakageChecker
from .probe import probe_video

SIZE = (960, 544)       # the only frame size ingest takes
# A cut: a jump of the measured camera and a jump of the picture no more than CUT_NEAR frames apart.
CUT_STEP_RATIO, CUT_TURN_DEG, CUT_PICTURE_RATIO, CUT_NEAR = 12.0, 10.0, 5.0, 3


def picture_steps(video: Path) -> np.ndarray:
    """Mean absolute difference between consecutive frames, grey at 160x90."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=160:90:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=600).stdout
    grey = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 90, 160).astype(np.float32)
    return np.abs(np.diff(grey, axis=0)).mean(axis=(1, 2))


def find_cut(cam_c2w: np.ndarray, video: Path) -> str | None:
    """Where the clip cuts to another shot, or None when it is one shot."""
    poses = np.asarray(cam_c2w, dtype=np.float64)
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    relative = np.einsum("nij,nik->njk", poses[:-1, :3, :3], poses[1:, :3, :3])
    turns = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    picture = picture_steps(video)

    def over(signal: np.ndarray, ratio: float) -> np.ndarray:
        median = float(np.median(signal)) if len(signal) else 0.0
        return signal >= ratio * median if median > 0 else np.zeros(len(signal), dtype=bool)

    pose_jumps = np.flatnonzero(over(steps, CUT_STEP_RATIO) | (turns >= CUT_TURN_DEG))
    for at in np.flatnonzero(over(picture, CUT_PICTURE_RATIO)):
        if len(pose_jumps) and np.abs(pose_jumps - at).min() <= CUT_NEAR:
            return f"a cut between frames {at} and {at + 1}: the picture and the measured camera both jump there"
    return None


@dataclass
class Candidate:
    video: Path
    annotation: Path
    pose: Path
    derived_from: dict

@dataclass
class IngestResult:
    accepted: bool
    clip_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

class Ingestor:
    def __init__(self, cfg: KernelConfig, run_dir: Path, conn, recorder,
                 leakage: LeakageChecker | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.blobs = BlobStore(run_dir, conn)
        self.clips = ClipStore(conn)
        self.recorder = recorder
        self.leakage = leakage or LeakageChecker(cfg)
        self.base_recipe = run_config_path(cfg, run_dir, "base_recipe.yaml")
        self.workers = int(cfg.get("ingest.workers"))

    def ingest(self, candidates: list[Candidate], node_id: str) -> list[IngestResult]:
        """A clip's checks are subprocesses, so `ingest.workers` clips are checked at a time; the archive
        is then written one clip at a time, in order. The kernel works on its own copy of each staged
        file and never writes or deletes in staging, which the agent owns and can relink meanwhile."""
        with ThreadPoolExecutor(self.workers) as pool:
            checks = [pool.submit(self._check, c, node_id) for c in candidates]
        results = []
        try:
            for candidate, check in zip(candidates, checks):
                result, held, facts = check.result()
                results.append(result or self._store(candidate, held, facts, node_id))
        finally:
            for check in checks:
                if not check.exception():
                    _discard(check.result()[1])
        return results

    def _outside_staging(self, path: Path) -> bool:
        """Only staged files may be ingested: store/ and other nodes' views are under the run dir too."""
        return not Path(path).resolve().is_relative_to((self.run_dir / "staging").resolve())

    def _take(self, src: Path, dst: Path) -> Path:
        """Copy the staged file to `dst` without trusting its path between check and use: the file is
        opened without following a link, and the open file must be a regular file that still lies in
        staging (a directory on the path may have been swapped for a link to a host folder). A copy,
        never a rename: the agent may hold a hard link to the staged file and rewrite it later."""
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)      # NONBLOCK: a swapped-in FIFO
        with open(fd, "rb") as fin:
            real = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if not stat.S_ISREG(os.fstat(fd).st_mode) or self._outside_staging(real):
                raise OSError("the staged file changed while it was being read")
            with open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return dst

    def _event(self, candidate: Candidate, kind: str, node_id: str, payload: dict) -> None:
        """Every ingest event names its candidate by the paths the agent staged."""
        staged = {k: str(v) if v is not None else None
                  for k, v in (("video", candidate.video), ("annotation", candidate.annotation), ("pose", candidate.pose))}
        self.recorder.event(kind, node=node_id, phase="ingest", payload={"candidate": staged, **payload})

    def _reject(self, candidate: Candidate, node_id: str, reasons: list[str], **payload) -> IngestResult:
        self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons, **payload})
        return IngestResult(accepted=False, reasons=reasons)

    def _check(self, candidate: Candidate, node_id: str) -> tuple[IngestResult | None, dict[str, Path], dict | None]:
        """(the rejection or None, the quarantined files, what `_store` records of a clip that passed)."""
        reasons: list[str] = []
        origin = candidate.derived_from
        if not isinstance(origin, dict) or origin.get("kind") not in ann.DERIVED_KINDS \
                or not isinstance(origin.get("name"), str) or not origin["name"].strip():
            reasons.append(f"derived_from must be {{kind, name}} with kind one of {', '.join(ann.DERIVED_KINDS)}")
        for label, path in (("video", candidate.video), ("annotation", candidate.annotation),
                            ("pose", candidate.pose)):
            if self._outside_staging(path):
                reasons.append(
                    f"{label} path {path} is outside the staging directory "
                    f"{self.run_dir / 'staging'}; ingest only accepts files staged there")
        if reasons:
            return self._reject(candidate, node_id, reasons), {}, None

        # Quarantine before ANY check, so every check and the store act on the same bytes (the
        # agent keeps running and could swap a staged file). quarantine/ is kernel-private.
        staged = {"video": candidate.video, "annotation": candidate.annotation, "pose": candidate.pose}
        qdir = self.run_dir / "quarantine" / uuid.uuid4().hex
        qdir.mkdir(parents=True)
        held = {k: qdir / f"{k}{Path(v).suffix}" for k, v in staged.items()}
        try:
            for key, qpath in held.items():
                self._take(staged[key], qpath)
            checked = self._check_files(candidate, held, node_id)
        except (subprocess.SubprocessError, OSError, ValueError, IndexError, KeyError, CheckerError) as exc:
            # A file ffprobe, decoding or the checker bridge cannot handle: record a rejection
            # and carry on with the batch instead of failing the whole call.
            checked = self._reject(candidate, node_id, [
                f"could not read the candidate: {type(exc).__name__}: {exc}"[:500]])
        except BaseException:
            _discard(held)
            raise
        return (checked, held, None) if isinstance(checked, IngestResult) else (None, held, checked)

    def _check_files(self, candidate: Candidate, held: dict[str, Path], node_id: str) -> IngestResult | dict:
        video, pose = held["video"], held["pose"]
        info = probe_video(video)
        if info.rotation:
            return self._reject(candidate, node_id, [
                f"video carries a {info.rotation} degree display rotation; WorldModel decodes "
                f"frames in stored orientation, so they would train rotated. re-encode with "
                f"the rotation applied (ffmpeg applies it when transcoding: "
                f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4)"])
        if (info.width, info.height) != SIZE:
            return self._reject(candidate, node_id, [
                f"the video is {info.width}x{info.height}; ingest takes {SIZE[0]}x{SIZE[1]} only. Crop or resize "
                "it first, then measure the camera on the result"])
        try:
            annotation = json.loads(held["annotation"].read_text(encoding="utf-8"))
        except ValueError as exc:
            return self._reject(candidate, node_id, [f"the annotation is not valid JSON: {exc}"])
        problems = ann.check_annotation(annotation)
        if not problems and annotation["turns"][-1]["start_s"] >= info.duration:
            problems = [f"turn {len(annotation['turns'])} starts at {annotation['turns'][-1]['start_s']} s, "
                        f"at or after the end of the video ({info.duration:.2f} s)"]
        if problems:
            return self._reject(candidate, node_id, [f"annotation: {p}" for p in problems])
        caption_json = ann.caption_file(annotation, info.duration)
        held["caption"] = held["video"].with_name("caption.json")
        held["caption"].write_text(json.dumps(caption_json, indent=1), encoding="utf-8")

        probe_root = self.run_dir / "tmp" / f"probe_{uuid.uuid4().hex}"     # parallel ingests never share it
        shutil.rmtree(probe_root, ignore_errors=True)
        for sub, key, name in (("videos", "video", "c.mp4"), ("captions", "caption", "c.json"), ("poses", "pose", "c.npz")):
            (probe_root / sub).mkdir(parents=True)
            shutil.copy2(held[key], probe_root / sub / name)
        try:
            report = check_clip(self.cfg, probe_root, self.base_recipe, recorder=self.recorder, node=node_id)
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)

        def named(text: str) -> str:
            """The checker's message about the candidate's files, without the kernel's own paths."""
            for kind, name in (("videos/c.mp4", "the video"), ("captions/c.json", "the annotation"),
                               ("poses/c.npz", "the pose file")):
                text = text.replace(str(probe_root / kind), name)
            return text.replace(str(probe_root), "the clip")

        if not report["ok"]:
            frames = int(info.frames / (info.fps / ann.FPS)) if info.fps else 0     # the loader's 24 fps count
            reasons = set()
            for error in report["errors"]:
                if "shorter than one training window" in error:
                    reasons.add(f"the video lasts {frames / ann.FPS:.2f} s at {ann.FPS} fps ({info.frames} frames at "
                                f"{info.fps:.2f} fps); a clip must last at least {ann.FIRST_TURN_S} s")
                elif "to be trained on" in error and ann.short_turns(annotation, frames):
                    reasons.update(ann.short_turns(annotation, frames))
                else:
                    reasons.add(named(error))
            return self._reject(candidate, node_id, sorted(reasons), report=report)

        texts = [caption_json["caption"], *[s["prompt"] for s in caption_json["segments"]]]
        verdict = self.leakage.check(video)
        excluded = ("image" if verdict.rejected else "text" if copies_held_out(self.cfg, *texts) else None)
        self._event(candidate, "ingest.leakage", node_id,
                    {"matches": verdict.matches, "near": verdict.near_matches, "excluded": excluded})
        if excluded:                    # the agent gets one sentence; telemetry keeps the reason
            return self._reject(candidate, node_id, [EXCLUDED_CLIP], excluded=excluded)

        with np.load(pose) as arrays:
            has_intrinsics = "intrinsics" in arrays.files
            cut = find_cut(arrays["cam_c2w"], video)
        if cut:
            return self._reject(candidate, node_id, [f"the clip has {cut}; a training clip is one continuous shot"])
        return {"metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                             "height": info.height, "duration": info.duration,
                             "turns": len(annotation["turns"]), "has_intrinsics": has_intrinsics},
                "warnings": sorted({named(w) for w in report["warnings"]})}

    def _store(self, candidate: Candidate, held: dict[str, Path], facts: dict, node_id: str) -> IngestResult:
        digests = {key: self.blobs.put(held[key], key) for key in ("video", "caption", "annotation", "pose")}
        clip_id = hashlib.sha256(json.dumps(
            {key: digests[key] for key in ("video", "caption", "pose")}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, **{f"{key}_digest": digest for key, digest in digests.items()}, **facts,
            "derived_from": {"kind": candidate.derived_from["kind"], "name": candidate.derived_from["name"]},
            "ingested_by": node_id,
        })
        self._event(candidate, "ingest.accepted", node_id, {"clip_id": clip_id, "warnings": facts["warnings"]})
        return IngestResult(accepted=True, clip_id=clip_id, warnings=facts["warnings"])


def _discard(held: dict[str, Path]) -> None:
    """Remove a candidate's quarantine folder: the archive took what it wanted, the originals are in staging."""
    for qdir in {qpath.parent for qpath in held.values()}:
        shutil.rmtree(qdir, ignore_errors=True)
