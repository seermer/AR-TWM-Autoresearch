from __future__ import annotations
import json
from pathlib import Path

from ..config import KernelConfig
from ..subproc import run_in_env

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "check_clip.py"
MARK = "===AR_JSON==="

class CheckerError(RuntimeError):
    """The checker bridge did not return a report."""

def check_clip_formats(cfg: KernelConfig, clip_dir: Path, camera_motion: str,
                       base_recipe: Path, recorder=None, node: str = "run") -> dict[str, dict]:
    # `python <script>.py` puts the script's own directory on sys.path[0], not the
    # process cwd, so `alaya` would not be importable without this: point PYTHONPATH
    # at the WorldModel checkout the bridge needs to import from.
    proc = run_in_env(
        "alayaworld",
        ["python", str(BRIDGE), "--config", str(base_recipe), "--root", str(clip_dir),
         "--camera-motion", camera_motion],
        cwd=cfg.worldmodel, extra_env={"PYTHONPATH": str(cfg.worldmodel)},
        timeout=600, recorder=recorder, node=node, phase="ingest",
    )
    if MARK not in proc.stdout:
        raise CheckerError(f"checker bridge failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.split(MARK)[1])
