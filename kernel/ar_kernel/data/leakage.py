from __future__ import annotations
import subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from ..config import KernelConfig
from .probe import probe_video

SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
NCC_SIZE = 32

@dataclass
class LeakageVerdict:
    rejected: bool
    matches: list[dict] = field(default_factory=list)
    near_matches: list[dict] = field(default_factory=list)

def _grayscale_vector(image: Image.Image) -> np.ndarray:
    small = np.asarray(image.convert("L").resize((NCC_SIZE, NCC_SIZE)), dtype=np.float64).ravel()
    centered = small - small.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered

def _entropy(image: Image.Image) -> float:
    hist = np.asarray(image.convert("L").histogram(), dtype=np.float64)
    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())

def _extract(video: Path, timestamp: float) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                        "-i", str(video), "-frames:v", "1", str(out)], check=True, timeout=120)
        return Image.open(out).copy()

class LeakageChecker:
    def __init__(self, cfg: KernelConfig) -> None:
        self.max_distance = int(cfg.get("leakage.phash_max_distance"))
        self.min_ncc = float(cfg.get("leakage.min_ncc"))
        self.min_entropy = float(cfg.get("leakage.min_entropy"))
        self.references: list[tuple[str, imagehash.ImageHash, np.ndarray]] = []
        for path in sorted((cfg.eval_data / "images").glob("case_*.jpg")):
            image = Image.open(path)
            case_id = path.stem.replace("case_", "")
            self.references.append((case_id, imagehash.phash(image), _grayscale_vector(image)))

    def check(self, video: Path) -> LeakageVerdict:
        info = probe_video(video)
        verdict = LeakageVerdict(rejected=False)
        for fraction in SAMPLE_FRACTIONS:
            frame = _extract(video, fraction * info.duration)
            frame_hash = imagehash.phash(frame)
            frame_vector = _grayscale_vector(frame)
            flat = _entropy(frame) < self.min_entropy
            for case_id, ref_hash, ref_vector in self.references:
                distance = frame_hash - ref_hash
                if distance > self.max_distance:
                    continue
                ncc = float(np.dot(frame_vector, ref_vector))
                record = {"case_id": case_id, "fraction": fraction,
                          "phash_distance": int(distance), "ncc": ncc, "flat": flat}
                if ncc >= self.min_ncc and not flat:
                    verdict.matches.append(record)
                    verdict.rejected = True
                else:
                    verdict.near_matches.append(record)
        return verdict
