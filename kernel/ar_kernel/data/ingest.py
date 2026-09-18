from __future__ import annotations
import hashlib, json, shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig
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
    def __init__(self, cfg: KernelConfig, run_dir: Path, conn, recorder) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.blobs = BlobStore(run_dir, conn)
        self.clips = ClipStore(conn)
        self.recorder = recorder
        self.leakage = LeakageChecker(cfg)
        self.base_recipe = cfg.repo_root / "configs" / "base_recipe.yaml"
        self.tolerance = float(cfg.get("ingest.aspect_tolerance"))

    def ingest(self, candidates: list[Candidate], node_id: str) -> list[IngestResult]:
        return [self._one(c, node_id) for c in candidates]

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
        if reasons:
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        info = probe_video(candidate.video)
        if not aspect_ok(info, self.tolerance):
            reasons.append(f"aspect ratio {info.width}/{info.height} is not within "
                           f"{self.tolerance:.0%} of 16:9")
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        probe_root = self.run_dir / "tmp" / f"probe_{_digest(candidate.video)[:12]}"
        shutil.rmtree(probe_root, ignore_errors=True)
        (probe_root / "videos").mkdir(parents=True)
        (probe_root / "captions").mkdir(parents=True)
        shutil.copy2(candidate.video, probe_root / "videos" / "c.mp4")
        shutil.copy2(candidate.caption, probe_root / "captions" / "c.json")
        if candidate.pose is not None:
            (probe_root / "poses").mkdir(parents=True)
            shutil.copy2(candidate.pose, probe_root / "poses" / "c.npz")
        try:
            reports = check_clip_formats(self.cfg, probe_root, candidate.camera_motion,
                                         self.base_recipe, recorder=self.recorder, node=node_id)
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)

        formats = [key for key, report in reports.items() if report["ok"]]
        warnings = sorted({w for report in reports.values() for w in report["warnings"]})
        if not formats:
            reasons = sorted({e for report in reports.values() for e in report["errors"]})
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons, "reports": reports})
            return IngestResult(accepted=False, reasons=reasons)

        verdict = self.leakage.check(candidate.video)
        self.recorder.event("ingest.leakage", node=node_id, phase="ingest",
                            payload={"matches": verdict.matches, "near": verdict.near_matches})
        if verdict.rejected:
            reasons = [f"matches WBench case {m['case_id']} "
                       f"(phash {m['phash_distance']}, ncc {m['ncc']:.3f})" for m in verdict.matches]
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)

        video_digest = self.blobs.put(candidate.video, "video")
        caption_digest = self.blobs.put(candidate.caption, "caption")
        pose_digest = self.blobs.put(candidate.pose, "pose") if candidate.pose else None
        clip_id = hashlib.sha256(json.dumps(
            {"video": video_digest, "caption": caption_digest, "pose": pose_digest,
             "camera_motion": candidate.camera_motion}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, "video_digest": video_digest, "caption_digest": caption_digest,
            "pose_digest": pose_digest, "camera_motion": candidate.camera_motion,
            "metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                         "height": info.height, "duration": info.duration},
            "formats": formats, "warnings": warnings, "provenance": candidate.provenance,
            "license": candidate.license, "derived_from": candidate.derived_from,
            "ingested_by": node_id,
        })
        self.recorder.event("ingest.accepted", node=node_id, phase="ingest",
                            payload={"clip_id": clip_id, "formats": formats, "warnings": warnings})
        return IngestResult(accepted=True, clip_id=clip_id, formats=formats, warnings=warnings)
