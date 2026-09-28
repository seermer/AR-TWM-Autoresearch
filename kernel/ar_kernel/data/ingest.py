from __future__ import annotations
import hashlib, json, shutil, subprocess, uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig, run_config_path
from .checker import check_clip_formats
from .leakage import LeakageChecker
from .probe import aspect_ok, probe_video

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

def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

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

    def ingest(self, candidates: list[Candidate], node_id: str) -> list[IngestResult]:
        return [self._one(c, node_id) for c in candidates]

    def _outside_staging(self, path: Path) -> bool:
        """Only files under <run_dir>/staging/ may be ingested.

        The whole run dir was too wide: it includes store/ (a stored blob handed
        back in would be unlinked as a duplicate source) and other nodes' views
        (ingesting one removes that view's hardlink).
        """
        return not Path(path).resolve().is_relative_to((self.run_dir / "staging").resolve())

    def _event(self, candidate: Candidate, kind: str, node_id: str, payload: dict) -> None:
        """Every ingest event names its candidate by the paths the agent staged."""
        staged = {k: str(v) if v is not None else None
                  for k, v in (("video", candidate.video), ("caption", candidate.caption), ("pose", candidate.pose))}
        self.recorder.event(kind, node=node_id, phase="ingest", payload={"candidate": staged, **payload})

    def _one(self, candidate: Candidate, node_id: str) -> IngestResult:
        reasons: list[str] = []
        if not candidate.provenance:
            reasons.append("provenance is required")
        if candidate.camera_motion not in {"moving", "static"}:
            reasons.append(f"camera_motion must be moving or static, got {candidate.camera_motion!r}")
        if candidate.camera_motion == "static" and candidate.pose is not None:
            reasons.append("static clips must not carry poses (video_caption_static uses identity poses)")
        if candidate.camera_motion == "moving" and candidate.pose is None:
            reasons.append("moving clips need poses/<id>.npz")
        # BlobStore.put() moves/consumes its source file (tests/test_blobs.py asserts this
        # intentionally). Ingest must never be handed a path outside the run's own directory,
        # or it will silently delete files it does not own (see the Task 14 incident where the
        # manual test's real WorldModel example clips were consumed this way).
        for label, path in (("video", candidate.video), ("caption", candidate.caption),
                            ("pose", candidate.pose)):
            if path is not None and self._outside_staging(path):
                reasons.append(
                    f"{label} path {path} is outside the staging directory "
                    f"{self.run_dir / 'staging'}; ingest only accepts files staged there")
        if reasons:
            self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        # Quarantine before ANY check: every check and the store then act on the
        # same bytes. Checking the staged path and re-reading it at put() time let
        # an agent process swap the file after the leakage check (review I2).
        # quarantine/ is kernel-private; agents only ever see staging/.
        staged = {"video": candidate.video, "caption": candidate.caption, "pose": candidate.pose}
        qdir = self.run_dir / "quarantine" / uuid.uuid4().hex
        qdir.mkdir(parents=True)
        held = {k: _move(v, qdir / f"{k}{Path(v).suffix}") for k, v in staged.items() if v is not None}
        try:
            try:
                result = self._check_and_store(candidate, held, node_id)
            except (subprocess.CalledProcessError, ValueError, IndexError, KeyError) as exc:
                # A file ffprobe/decoding cannot read is the candidate's fault: record
                # a rejection and carry on with the batch instead of crashing the node.
                reasons = [f"could not read the candidate video: {type(exc).__name__}: {exc}"[:500]]
                self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons})
                result = IngestResult(accepted=False, reasons=reasons)
        finally:
            # Anything not consumed by the store goes back where the agent staged it.
            for key, qpath in held.items():
                if qpath.exists():
                    _move(qpath, _free_path(Path(staged[key])))
            shutil.rmtree(qdir, ignore_errors=True)
        return result

    def _check_and_store(self, candidate: Candidate, held: dict[str, Path],
                         node_id: str) -> IngestResult:
        video, caption, pose = held["video"], held["caption"], held.get("pose")
        info = probe_video(video)
        if info.rotation:
            reasons = [f"video carries a {info.rotation} degree display rotation; WorldModel decodes "
                       f"frames in stored orientation, so they would train rotated. re-encode with "
                       f"the rotation applied (ffmpeg applies it when transcoding: "
                       f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4)"]
            self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)
        if not aspect_ok(info, self.tolerance):
            reasons = [f"display aspect ratio {info.display_aspect:.4f} (coded {info.width}x{info.height}, "
                       f"sar {info.sar:.4f}) is not within {self.tolerance:.0%} of 16:9"]
            self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        probe_root = self.run_dir / "tmp" / f"probe_{_digest(video)[:12]}"
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

        formats = [key for key, report in reports.items() if report["ok"]]
        warnings = sorted({w for report in reports.values() for w in report["warnings"]})
        if not formats:
            reasons = sorted({e for report in reports.values() for e in report["errors"]})
            self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons, "reports": reports})
            return IngestResult(accepted=False, reasons=reasons)

        verdict = self.leakage.check(video)
        self._event(candidate, "ingest.leakage", node_id, {"matches": verdict.matches, "near": verdict.near_matches})
        if verdict.rejected:
            reasons = [f"matches WBench case {m['case_id']} "
                       f"(phash {m['phash_distance']}, ncc {m['ncc']:.3f})" for m in verdict.matches]
            self._event(candidate, "ingest.rejected", node_id, {"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        has_segments = bool(json.loads(caption.read_text(encoding="utf-8")).get("segments"))
        has_intrinsics = False
        if pose is not None:
            with np.load(pose) as arrays:
                has_intrinsics = "intrinsics" in arrays.files
        video_digest = self.blobs.put(video, "video")
        caption_digest = self.blobs.put(caption, "caption")
        pose_digest = self.blobs.put(pose, "pose") if pose else None
        clip_id = hashlib.sha256(json.dumps(
            {"video": video_digest, "caption": caption_digest, "pose": pose_digest,
             "camera_motion": candidate.camera_motion}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, "video_digest": video_digest, "caption_digest": caption_digest,
            "pose_digest": pose_digest, "camera_motion": candidate.camera_motion,
            "metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                         "height": info.height, "duration": info.duration,
                         "has_segments": has_segments, "has_intrinsics": has_intrinsics},
            "formats": formats, "warnings": warnings, "provenance": candidate.provenance,
            "license": candidate.license, "derived_from": candidate.derived_from,
            "ingested_by": node_id,
        })
        self._event(candidate, "ingest.accepted", node_id, {"clip_id": clip_id, "formats": formats, "warnings": warnings})
        return IngestResult(accepted=True, clip_id=clip_id, formats=formats, warnings=warnings)


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
