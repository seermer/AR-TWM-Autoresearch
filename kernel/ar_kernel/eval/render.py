from __future__ import annotations
import copy
from pathlib import Path
import yaml

from ..config import KernelConfig
from ..liveness import Liveness, tree_mark
from ..subproc import output_tail, run_in_env

def build_render_config(cfg: KernelConfig, merged: Path | None, history_encoder: Path,
                        videos_dir: Path, case_ids: list[str], node_dir: Path) -> Path:
    source = yaml.safe_load((cfg.worldmodel / "configs" / "wbench_full.yaml").read_text())
    config = copy.deepcopy(source)
    config["paths"]["resume_checkpoint"] = str(merged or (cfg.worldmodel / "weights/alaya-world-ar"))
    config["paths"]["history_encoder"] = str(history_encoder)
    config["paths"]["dmd_resume"] = str(cfg.worldmodel / "weights/alaya-world-dmd")
    config["run"]["output_dir"] = str(Path(node_dir) / "eval" / "rollout")
    config["run"]["log_dir"] = str(Path(node_dir) / "eval" / "logs")
    config["validation"]["per_sample_seed"] = True
    mode = config["validation"]["modes"]["wbench"]
    mode["dataset"]["root"] = str(cfg.wbench / "data")
    mode["dataset"]["case_ids"] = list(case_ids)
    mode["wbench_output_dir"] = str(videos_dir)
    target = Path(node_dir) / "eval" / "render_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return target

def render_proxy(cfg: KernelConfig, render_config: Path, gpus: list[int], node_id: str,
                 recorder, case_ids: list[str]) -> Path:
    videos_dir = Path(yaml.safe_load(render_config.read_text())
                      ["validation"]["modes"]["wbench"]["wbench_output_dir"])
    videos_dir.mkdir(parents=True, exist_ok=True)
    with recorder.span("render", node=node_id, phase="render", payload={"cases": case_ids}):
        liveness = Liveness.from_config(cfg, float(cfg.get("timeouts.eval_s")),
                                        signals=[lambda: tree_mark(videos_dir, render_config.parent / "logs")])
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/run_wbench.py", "--config", str(render_config),
             "--gpus", ",".join(str(g) for g in gpus), "--cases", ",".join(case_ids)],
            cwd=cfg.worldmodel, timeout=None, liveness=liveness, recorder=recorder, node=node_id,
            phase="render")
    if proc.returncode != 0:
        raise RuntimeError(f"render failed (rc={proc.returncode}): {output_tail(proc)}")
    rendered = sorted(videos_dir.glob("case_*_combined.mp4"))
    if len(rendered) != len(case_ids):
        raise RuntimeError(f"rendered {len(rendered)} of {len(case_ids)} proxy cases")
    return videos_dir
