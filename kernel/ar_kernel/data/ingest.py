from __future__ import annotations
import hashlib, json, shutil, subprocess, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig, run_config_path
from ..isolation import EXCLUDED_CLIP, blocked, copies_held_out
from .checker import CheckerError, check_clip_formats
from .leakage import LeakageChecker
from .probe import aspect_ok, probe_video

# A camera step this many times the clip's median step, or a turn of this many degrees between two
# frames, is reported. Measured on two runs' clips: single-shot clips stay under 5x and 1 degree;
# clips made by joining two renders jump 16x to 155x or 40 to 110 degrees at the join.
JUMP_STEP_RATIO, JUMP_ROTATION_DEG = 15.0, 20.0


def pose_jump(cam_c2w: np.ndarray) -> str | None:
    """A warning naming the largest frame-to-frame camera jump, or None when the path has none."""
    poses = np.asarray(cam_c2w, dtype=np.float64)
    if len(poses) < 3:
        return None
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    relative = np.einsum("nij,nik->njk", poses[:-1, :3, :3], poses[1:, :3, :3])
    turns = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    median = float(np.median(steps))
    ratio = steps / median if median > 0 else np.zeros_like(steps)
    at = int(np.argmax(np.maximum(ratio / JUMP_STEP_RATIO, turns / JUMP_ROTATION_DEG)))
    if ratio[at] < JUMP_STEP_RATIO and turns[at] < JUMP_ROTATION_DEG:
        return None
    return (f"camera pose jumps between frames {at} and {at + 1}: the step is {ratio[at]:.1f}x the clip's "
            f"median step and the camera turns {turns[at]:.1f} degrees")


@dataclass
class Candidate:
    video: Path
    caption: Path
    pose: Path | None
    camera_motion: str
    provenance: dict
    license: str | None = None
    derived_from: list[str] = field(default_factory=list)

@dataclass
class IngestResult:
    accepted: bool
    clip_id: str | None = None
    formats: list[str] = field(default_factory=list)
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
        self.tolerance = float(cfg.get("ingest.aspect_tolerance"))
        self.workers = int(cfg.get("ingest.workers"))

    def ingest(self, candidates: list[Candidate], node_id: str) -> list[IngestResult]:
        """A clip's checks are subprocesses, so `ingest.workers` clips are checked at a time; the archive
        is then written one clip at a time, in order."""
        with ThreadPoolExecutor(self.workers) as pool:
            checks = [pool.submit(self._check, c, node_id) for c in candidates]
        results = []
        try:
            for candidate, check in zip(candidates, checks):
                result, held, facts = check.result()
                results.append(result or self._store(candidate, held, facts, node_id))
        finally:
            for candidate, check in zip(candidates, checks):
                if not check.exception():
                    self._give_back(candidate, check.result()[1])
        return results

    def _outside_staging(self, path: Path) -> bool:
        """Only staged files may be ingested: store/ and other nodes' views are under the run
        dir too, and ingesting consumes the source file."""
        return not Path(path).resolve().is_relative_to((self.run_dir / "staging").resolve())

    def _event(self, candidate: Candidate, kind: str, node_id: str, payload: dict) -> None:
        """Every ingest event names its candidate by the paths the agent staged."""
        staged = {k: str(v) if v is not None else None
                  for k, v in (("video", candidate.video), ("caption", candidate.caption), ("pose", candidate.pose))}
        self.recorder.event(kind, node=node_id, phase="ingest", payload={"candidate": staged, **payload})

    def _reject(self, candidate: Candidate, node_id: str, reasons: list[str], **payload) -> IngestResult:
        self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons, **payload})
        return IngestResult(accepted=False, reasons=reasons)

    def _give_back(self, candidate: Candidate, held: dict[str, Path]) -> None:
        """Anything the archive did not take goes back where the agent staged it."""
        for key, qpath in held.items():
            if qpath.exists():
                _move(qpath, _free_path(Path(getattr(candidate, key))))
        for qdir in {qpath.parent for qpath in held.values()}:
            shutil.rmtree(qdir, ignore_errors=True)

    def _check(self, candidate: Candidate, node_id: str) -> tuple[IngestResult | None, dict[str, Path], dict | None]:
        """(the rejection or None, the quarantined files, what `_store` records of a clip that passed)."""
        reasons: list[str] = []
        if not candidate.provenance:
            reasons.append("provenance is required")
        if candidate.camera_motion not in {"moving", "static"}:
            reasons.append(f"camera_motion must be moving or static, got {candidate.camera_motion!r}")
        if candidate.camera_motion == "static" and candidate.pose is not None:
            reasons.append("static clips must not carry poses (video_caption_static uses identity poses)")
        if candidate.camera_motion == "moving" and candidate.pose is None:
            reasons.append("moving clips need poses/<id>.npz")
        # BlobStore.put() consumes its source: anything outside staging would be deleted.
        for label, path in (("video", candidate.video), ("caption", candidate.caption),
                            ("pose", candidate.pose)):
            if path is not None and self._outside_staging(path):
                reasons.append(
                    f"{label} path {path} is outside the staging directory "
                    f"{self.run_dir / 'staging'}; ingest only accepts files staged there")
        if reasons:
            return self._reject(candidate, node_id, reasons), {}, None

        # Quarantine before ANY check, so every check and the store act on the same bytes (the
        # agent keeps running and could swap a staged file). quarantine/ is kernel-private.
        staged = {"video": candidate.video, "caption": candidate.caption, "pose": candidate.pose}
        qdir = self.run_dir / "quarantine" / uuid.uuid4().hex
        qdir.mkdir(parents=True)
        held = {k: _move(v, qdir / f"{k}{Path(v).suffix}") for k, v in staged.items() if v is not None}
        try:
            checked = self._check_files(candidate, held, node_id)
        except (subprocess.CalledProcessError, ValueError, IndexError, KeyError, CheckerError) as exc:
            # A file ffprobe, decoding or the checker bridge cannot handle: record a rejection
            # and carry on with the batch instead of failing the whole call.
            checked = self._reject(candidate, node_id, [
                f"could not read the candidate video: {type(exc).__name__}: {exc}"[:500]])
        except BaseException:
            self._give_back(candidate, held)
            raise
        return (checked, held, None) if isinstance(checked, IngestResult) else (None, held, checked)

    def _check_files(self, candidate: Candidate, held: dict[str, Path], node_id: str) -> IngestResult | dict:
        video, caption, pose = held["video"], held["caption"], held.get("pose")
        info = probe_video(video)
        if info.rotation:
            return self._reject(candidate, node_id, [
                f"video carries a {info.rotation} degree display rotation; WorldModel decodes "
                f"frames in stored orientation, so they would train rotated. re-encode with "
                f"the rotation applied (ffmpeg applies it when transcoding: "
                f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4)"])
        if not aspect_ok(info, self.tolerance):
            return self._reject(candidate, node_id, [
                f"display aspect ratio {info.display_aspect:.4f} (coded {info.width}x{info.height}, "
                f"sar {info.sar:.4f}) is not within {self.tolerance:.0%} of 16:9"])

        probe_root = self.run_dir / "tmp" / f"probe_{uuid.uuid4().hex}"     # parallel ingests never share it
        shutil.rmtree(probe_root, ignore_errors=True)
        (probe_root / "videos").mkdir(parents=True)
        (probe_root / "captions").mkdir(parents=True)
        shutil.copy2(video, probe_root / "videos" / "c.mp4")
        shutil.copy2(caption, probe_root / "captions" / "c.json")
        if pose is not None:
            (probe_root / "poses").mkdir(parents=True)
            shutil.copy2(pose, probe_root / "poses" / "c.npz")
        try:
            reports = check_clip_formats(self.cfg, probe_root, candidate.camera_motion,
                                         self.base_recipe, recorder=self.recorder, node=node_id)
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)

        def named(text: str) -> str:
            """The checker's message about the candidate's files, without the kernel's own paths."""
            for kind, name in (("videos/c.mp4", "the video"), ("captions/c.json", "the caption file"),
                               ("poses/c.npz", "the pose file")):
                text = text.replace(str(probe_root / kind), name)
            if "shorter than one training window" in text:
                text += (f" ({info.frames} frames at {info.fps:.2f} fps are fewer than a window's frames once "
                         "resampled to 24 fps: write the clip at exactly 24 fps)")
            return text.replace(str(probe_root), "the clip")

        formats = [key for key, report in reports.items() if report["ok"]]
        warnings = sorted({named(w) for report in reports.values() for w in report["warnings"]})
        if not formats:
            return self._reject(candidate, node_id,
                                sorted({named(e) for report in reports.values() for e in report["errors"]}),
                                reports=reports)

        caption_json = json.loads(caption.read_text(encoding="utf-8"))
        segments = caption_json.get("segments") if isinstance(caption_json.get("segments"), list) else []
        texts = [caption_json.get("caption"), *[s.get("prompt") for s in segments if isinstance(s, dict)]]
        verdict = self.leakage.check(video)
        excluded = ("image" if verdict.rejected else
                    "source" if blocked(self.cfg, json.dumps(candidate.provenance)) else
                    "text" if copies_held_out(self.cfg, *texts) else None)
        self._event(candidate, "ingest.leakage", node_id,
                    {"matches": verdict.matches, "near": verdict.near_matches, "excluded": excluded})
        if excluded:                    # the agent gets one sentence; telemetry keeps the reason
            return self._reject(candidate, node_id, [EXCLUDED_CLIP], excluded=excluded)

        has_segments = bool(caption_json.get("segments"))
        has_intrinsics = False
        if pose is not None:
            with np.load(pose) as arrays:
                has_intrinsics = "intrinsics" in arrays.files
                jump = pose_jump(arrays["cam_c2w"])
            if jump:
                warnings.append(jump)
        return {"metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                             "height": info.height, "duration": info.duration,
                             "has_segments": has_segments, "has_intrinsics": has_intrinsics},
                "formats": formats, "warnings": warnings}

    def _store(self, candidate: Candidate, held: dict[str, Path], facts: dict, node_id: str) -> IngestResult:
        video_digest = self.blobs.put(held["video"], "video")
        caption_digest = self.blobs.put(held["caption"], "caption")
        pose_digest = self.blobs.put(held["pose"], "pose") if "pose" in held else None
        clip_id = hashlib.sha256(json.dumps(
            {"video": video_digest, "caption": caption_digest, "pose": pose_digest,
             "camera_motion": candidate.camera_motion}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, "video_digest": video_digest, "caption_digest": caption_digest,
            "pose_digest": pose_digest, "camera_motion": candidate.camera_motion, **facts,
            "provenance": candidate.provenance,
            "license": candidate.license, "derived_from": candidate.derived_from,
            "ingested_by": node_id,
        })
        self._event(candidate, "ingest.accepted", node_id,
                    {"clip_id": clip_id, "formats": facts["formats"], "warnings": facts["warnings"]})
        return IngestResult(accepted=True, clip_id=clip_id, formats=facts["formats"], warnings=facts["warnings"])


def _move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(src), str(dst)))


def _free_path(path: Path) -> Path:
    """`path`, or a sibling name if the agent has since put something there."""
    if not path.exists():
        return path
    n = 1
    while (candidate := path.with_name(f"{path.stem}.returned{n}{path.suffix}")).exists():
        n += 1
    return candidate
