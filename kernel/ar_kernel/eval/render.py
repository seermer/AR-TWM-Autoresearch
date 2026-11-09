from __future__ import annotations
from pathlib import Path
import yaml

from .. import fs, telemetry
from ..config import KernelConfig
from ..subproc import output_tail, run_in_env

def build_render_config(cfg: KernelConfig, eval_lora: Path | None, node_rank: int, history_encoder: Path,
                        videos_dir: Path, case_ids: list[str], node_dir: Path) -> Path:
    """`eval_lora`: the directory holding the student LoRA concatenated with the node's (rank
    `node_rank`); None renders the released model."""
    config = cfg.render_config()
    config["paths"]["resume_checkpoint"] = str(cfg.worldmodel / "weights/alaya-world-ar")
    config["paths"]["history_encoder"] = str(history_encoder)
    config["paths"]["dmd_resume"] = str(eval_lora or (cfg.worldmodel / "weights/alaya-world-dmd"))
    if eval_lora is not None:
        config["lora"]["rank"] += node_rank
        config["lora"]["alpha"] = config["lora"]["rank"]
    config["run"]["output_dir"] = str(Path(node_dir) / "eval" / "rollout")
    config["run"]["log_dir"] = str(Path(node_dir) / "eval" / "logs")
    mode = config["validation"]["modes"]["wbench"]
    mode["dataset"]["root"] = str(cfg.eval_data)
    mode["dataset"]["case_ids"] = list(case_ids)
    mode["wbench_output_dir"] = str(videos_dir)
    target = Path(node_dir) / "eval" / "render_config.yaml"
    fs.mkdir(target.parent)
    fs.write_text(target, yaml.safe_dump(config, sort_keys=True))
    return target

def render_proxy(cfg: KernelConfig, render_config: Path, gpus: list[int], case_ids: list[str]) -> Path:
    videos_dir = Path(yaml.safe_load(render_config.read_text())
                      ["validation"]["modes"]["wbench"]["wbench_output_dir"])
    fs.mkdir(videos_dir)
    with telemetry.span("eval.render", {"config": str(render_config), "gpus": list(gpus), "cases": list(case_ids)}) as sp:
        _render(cfg, render_config, gpus, case_ids, videos_dir)
        sp.output = {"files": fs.files(videos_dir)}
    return videos_dir


def _render(cfg: KernelConfig, render_config: Path, gpus: list[int], case_ids: list[str], videos_dir: Path) -> None:
    # The render runs with the text encoder off, so the cases' prompts not yet in the cache are encoded first.
    proc = run_in_env(
        "alayaworld",
        ["python", "-m", "scripts.tools.precache_wbench_text_embeds", "--config", str(render_config),
         "--device-map", "auto"],
        cwd=cfg.worldmodel,
        extra_env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus[:2]),
                   "ALAYA_GEMMA_MAX_MEMORY": "0=13GiB,1=13GiB"})
    if proc.returncode != 0:
        raise RuntimeError(f"prompt precache failed (rc={proc.returncode}): {output_tail(proc)}")
    proc = run_in_env(
        "alayaworld",
        ["python", "scripts/tools/run_wbench.py", "--config", str(render_config),
         "--gpus", ",".join(str(g) for g in gpus), "--cases", ",".join(case_ids)],
        cwd=cfg.worldmodel)
    if proc.returncode != 0:
        raise RuntimeError(f"render failed (rc={proc.returncode}): {output_tail(proc)}")
    rendered = sorted(videos_dir.glob("case_*_combined.mp4"))
    if len(rendered) != len(case_ids):
        raise RuntimeError(f"rendered {len(rendered)} of {len(case_ids)} proxy cases")
