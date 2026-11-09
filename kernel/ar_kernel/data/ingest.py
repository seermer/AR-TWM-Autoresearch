from __future__ import annotations
import dataclasses, hashlib, json, os, stat, subprocess, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import fs, subproc, telemetry
from ..archive.activity import add_activity
from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..config import KernelConfig, run_config_path
from ..faults import disk_fault
from ..isolation import EXCLUDED_CLIP, copies_held_out, host_roots, scrub
from ..process_digest import _shape
from . import annotation as ann
from .checker import CheckerError, check_clip
from .leakage import LeakageChecker
from .probe import probe_video

SIZE = (960, 544)       # the only frame size ingest takes
# A cut: a jump of the measured camera and a jump of the picture no more than CUT_NEAR frames apart.
CUT_STEP_RATIO, CUT_TURN_DEG, CUT_PICTURE_RATIO, CUT_NEAR = 12.0, 10.0, 5.0, 3


def picture_steps(video: Path) -> np.ndarray:
    """Mean absolute difference between consecutive frames, grey at 160x90."""
    raw = subproc.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=160:90:flags=area,format=gray",
                       "-fps_mode", "passthrough", "-f", "rawvideo", "-"],      # the clip's own frames, none repeated
                      check=True, text=False).stdout
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
    named: dict             # role -> the container path the agent gave

@dataclass
class IngestResult:
    accepted: bool
    clip_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    checker_failed: bool = False    # rejected because the clip checker itself failed

class Ingestor:
    def __init__(self, cfg: KernelConfig, run_dir: Path, conn, leakage: LeakageChecker | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.blobs = BlobStore(run_dir, conn)
        self.clips = ClipStore(conn)
        self.leakage = leakage or LeakageChecker(cfg)
        self.base_recipe = run_config_path(cfg, run_dir, "base_recipe.yaml")
        self.workers = int(cfg.get("ingest.workers"))

    def ingest(self, candidates: list[Candidate], node_id: str, attempt: int) -> list[IngestResult]:
        """A clip's checks are subprocesses, so `ingest.workers` clips are checked at a time; the archive
        is then written one clip at a time, in order. The kernel works on its own copy of each staged
        file and never writes or deletes in staging, which the agent owns and can relink meanwhile."""
        with ThreadPoolExecutor(self.workers) as pool:
            checks = [telemetry.submit(pool, self._check, c, node_id, attempt) for c in candidates]
        results = []
        try:
            for candidate, check in zip(candidates, checks):
                result, held, facts = check.result()
                if result is None:
                    result = self._store(candidate, held, facts, node_id, attempt)
                else:
                    _keep(candidate, held)
                results.append(result)
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
            fs.write_bytes(dst, fin.read())
        return dst

    def _row(self, node_id: str, attempt: int, clip_id: str | None, reasons: list[str] = ()) -> None:
        """The candidate's activity row: accepted with its clip id, or rejected with its reasons."""
        names, roots = self.cfg.get("isolation.blocked_names") or [], host_roots(self.cfg)
        add_activity(self.run_dir, node=node_id, phase="ingest", attempt=attempt, kind="ingest", name=clip_id,
                     ok=clip_id is not None,
                     detail="\n".join(scrub(_shape(str(r)), names, roots) for r in reasons) or None)

    def _reject(self, node_id: str, attempt: int, reasons: list[str]) -> IngestResult:
        self._row(node_id, attempt, None, reasons)
        return IngestResult(accepted=False, reasons=reasons)

    def _passes(self, check: str, evidence: dict, reasons: list[str]) -> bool:
        """One check of a candidate, as a decision with its evidence."""
        telemetry.decision("ingest.check", {"check": check, **evidence}, ["pass", "reject"],
                           "reject" if reasons else "pass", "\n".join(reasons) or "passed")
        return not reasons

    def _check(self, candidate: Candidate, node_id: str,
               attempt: int) -> tuple[IngestResult | None, dict[str, Path], dict | None]:
        """(the rejection or None, the quarantined files, what `_store` records of a clip that passed)."""
        with telemetry.span("ingest.candidate", {"named": candidate.named, "derived_from": candidate.derived_from}) as sp:
            out = self._checked(candidate, node_id, attempt)
            sp.output = {"accepted": out[0] is None, "reasons": out[0].reasons if out[0] else []}
            return out

    def _checked(self, candidate: Candidate, node_id: str,
                 attempt: int) -> tuple[IngestResult | None, dict[str, Path], dict | None]:
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
        if not self._passes("origin and paths", {"derived_from": origin}, reasons):
            return self._reject(node_id, attempt, reasons), {}, None

        # Quarantine before ANY check, so every check and the store act on the same bytes (the
        # agent keeps running and could swap a staged file). quarantine/ is kernel-private.
        staged = {"video": candidate.video, "annotation": candidate.annotation, "pose": candidate.pose}
        qdir = self.run_dir / "quarantine" / uuid.uuid4().hex
        fs.mkdir(qdir)
        held = {k: qdir / f"{k}{Path(v).suffix}" for k, v in staged.items()}
        try:
            for key, qpath in held.items():
                self._take(staged[key], qpath)
            checked = self._check_files(held, node_id, attempt)
        except (subprocess.SubprocessError, OSError, ValueError, IndexError, KeyError, CheckerError) as exc:
            # A file ffprobe, decoding or the checker bridge cannot handle: record a rejection
            # and carry on with the batch instead of failing the whole call. A full disk is the machine's.
            if disk_fault(exc):
                _discard(held)
                raise
            telemetry.error(exc, "a rejected candidate")
            self._passes("readable", {}, [f"{type(exc).__name__}: {exc}"])
            checked = self._reject(node_id, attempt, [
                f"could not read the candidate: {type(exc).__name__}: {exc}"[:500]])
            checked.checker_failed = isinstance(exc, CheckerError)
        except BaseException:
            _discard(held)
            raise
        return (checked, held, None) if isinstance(checked, IngestResult) else (None, held, checked)

    def _check_files(self, held: dict[str, Path], node_id: str,
                     attempt: int) -> IngestResult | dict:
        video, pose = held["video"], held["pose"]
        info = probe_video(video)
        rotated = [f"video carries a {info.rotation} degree display rotation; WorldModel decodes "
                   f"frames in stored orientation, so they would train rotated. re-encode with "
                   f"the rotation applied (ffmpeg applies it when transcoding: "
                   f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4)"] if info.rotation else []
        if not self._passes("rotation", {"probe": dataclasses.asdict(info)}, rotated):
            return self._reject(node_id, attempt, rotated)
        size = [f"the video is {info.width}x{info.height}; ingest takes {SIZE[0]}x{SIZE[1]} only. Crop or resize "
                "it first, then measure the camera on the result"] if (info.width, info.height) != SIZE else []
        if not self._passes("size", {"width": info.width, "height": info.height}, size):
            return self._reject(node_id, attempt, size)
        try:
            annotation = json.loads(held["annotation"].read_text(encoding="utf-8"))
        except ValueError as exc:
            telemetry.error(exc, "a rejected candidate")
            self._passes("annotation", {}, [f"the annotation is not valid JSON: {exc}"])
            return self._reject(node_id, attempt, [f"the annotation is not valid JSON: {exc}"])
        problems = ann.check_annotation(annotation)
        duration = info.frames / info.fps if info.fps else info.duration        # to the frame, not ffprobe's rounding
        if not problems and annotation["turns"][-1]["start_s"] >= duration:
            problems = [f"turn {len(annotation['turns'])} starts at {annotation['turns'][-1]['start_s']} s, "
                        f"at or after the end of the video ({duration:.2f} s)"]
        if not self._passes("annotation", {"duration_s": duration}, [f"annotation: {p}" for p in problems]):
            return self._reject(node_id, attempt, [f"annotation: {p}" for p in problems])
        caption_json = ann.caption_file(annotation, duration)
        held["caption"] = held["video"].with_name("caption.json")
        fs.write_text(held["caption"], json.dumps(caption_json, indent=1))

        probe_root = self.run_dir / "tmp" / f"probe_{uuid.uuid4().hex}"     # parallel ingests never share it
        for sub, key, name in (("videos", "video", "c.mp4"), ("captions", "caption", "c.json"), ("poses", "pose", "c.npz")):
            fs.mkdir(probe_root / sub)
            fs.copy(held[key], probe_root / sub / name)
        try:
            report = check_clip(self.cfg, probe_root, self.base_recipe)
        finally:
            fs.rmtree(probe_root, ignore_errors="the checker's scratch copy")

        def named(text: str) -> str:
            """The checker's message about the candidate's files, without the kernel's own paths."""
            for kind, name in (("videos/c.mp4", "the video"), ("captions/c.json", "the annotation"),
                               ("poses/c.npz", "the pose file")):
                text = text.replace(str(probe_root / kind), name)
            return text.replace(str(probe_root), "the clip")

        if not self._passes("clip checker", {"report": report}, [] if report["ok"] else report["errors"]):
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
            return self._reject(node_id, attempt, sorted(reasons))

        texts = [caption_json["caption"], *[s["prompt"] for s in caption_json["segments"]]]
        verdict = self.leakage.check(video)
        excluded = ("image" if verdict.rejected else "text" if copies_held_out(self.cfg, *texts) else None)
        if not self._passes("held-out cases", {"leakage": dataclasses.asdict(verdict), "excluded": excluded},
                            [f"it copies a held-out case's {excluded}"] if excluded else []):     # the agent gets one sentence
            return self._reject(node_id, attempt, [EXCLUDED_CLIP])

        with np.load(pose) as arrays:
            has_intrinsics = "intrinsics" in arrays.files
            cut = find_cut(arrays["cam_c2w"], video)
        if not self._passes("one shot", {"has_intrinsics": has_intrinsics}, [cut] if cut else []):
            return self._reject(node_id, attempt,
                                [f"the clip has {cut}; a training clip is one continuous shot"])
        return {"metadata": {"frames": info.frames, "fps": info.fps, "width": info.width,
                             "height": info.height, "duration": info.duration,
                             "turns": len(annotation["turns"]), "has_intrinsics": has_intrinsics},
                "warnings": sorted({named(w) for w in report["warnings"]})}

    def _store(self, candidate: Candidate, held: dict[str, Path], facts: dict, node_id: str,
               attempt: int) -> IngestResult:
        digests = {key: self.blobs.put(held[key], key) for key in ("video", "caption", "annotation", "pose")}
        _keep(candidate, {key: self.blobs.path(digests[key], key) for key in digests})     # a link to each blob
        clip_id = hashlib.sha256(json.dumps(
            {key: digests[key] for key in ("video", "caption", "pose")}, sort_keys=True).encode()).hexdigest()
        self.clips.add({
            "clip_id": clip_id, **{f"{key}_digest": digest for key, digest in digests.items()}, **facts,
            "derived_from": {"kind": candidate.derived_from["kind"], "name": candidate.derived_from["name"]},
            "ingested_by": node_id, "attempt": attempt,
        })
        self._row(node_id, attempt, clip_id)
        return IngestResult(accepted=True, clip_id=clip_id, warnings=facts["warnings"])


def _keep(candidate: Candidate, files: dict[str, Path]) -> None:
    """The candidate's files in the file store, as the agent named them."""
    for key in ("video", "annotation", "pose"):
        if key in files:
            with files[key].open("rb") as handle:
                telemetry.keep(candidate.named.get(key, str(getattr(candidate, key))), handle, "read")


def _discard(held: dict[str, Path]) -> None:
    """Remove a candidate's quarantine folder: the archive took what it wanted, the originals are in staging."""
    for qdir in {qpath.parent for qpath in held.values()}:
        fs.rmtree(qdir, ignore_errors="a candidate's quarantine copy")
