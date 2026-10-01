from __future__ import annotations
from pathlib import Path

from ..config import KernelConfig
from ..subproc import output_tail, run_in_env

def concat_eval_lora(cfg: KernelConfig, checkpoint: Path, node_dir: Path, recorder, node_id: str,
                     cancel=None) -> Path:
    """The adapter the eval renders with: the few-step student LoRA and the node's LoRA as one
    adapter whose delta is exactly their sum. The base weights are left as released."""
    out = Path(node_dir) / "eval" / "lora"
    with recorder.span("concat_lora", node=node_id, phase="eval", payload={"checkpoint": str(checkpoint)}):
        proc = run_in_env(
            "alayaworld",
            ["python", "scripts/tools/concat_loras.py", "--loras",
             str(cfg.worldmodel / "weights/alaya-world-dmd/lora.safetensors"),
             str(Path(checkpoint) / "lora.safetensors"), "--output", str(out)],
            cwd=cfg.worldmodel, timeout=1800, recorder=recorder, node=node_id, phase="eval", cancel=cancel)
    if proc.returncode != 0:
        raise RuntimeError(f"LoRA concatenation failed (rc={proc.returncode}): {output_tail(proc)}")
    return out
