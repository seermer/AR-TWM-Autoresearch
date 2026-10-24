"""Download the weights and data a run needs, for the machine its config describes.

    python scripts/download_weights.py                            # 4 x 24 GB (configs/kernel.yaml)
    python scripts/download_weights.py --config configs/h100.yaml # 8 x H100: Qwen3.8-Flash-Next instead of 27B-FP8
    python scripts/download_weights.py --check [--config ...]     # list what is missing; exit 1 if anything is

Run from AutoResearcher/ in the autoresearcher env, logged in with `hf auth login`. Generators that are
switched off in the config are skipped. A finished download leaves a marker in .cache/downloads/ and is
skipped on the next call; an interrupted one resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ar_kernel.config import KernelConfig
from ar_kernel.subproc import cache_env

LTX25_FILES = ["diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
               "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
               "vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors",
               "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
               "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"]
H3_FILES = ["model_index.json", "modular_model_index.json", "transformer/*", "text_encoder/*", "vae/*", "audio_vae/*",
            "tokenizer/*", "processor/*", "scheduler/*", "audio_scheduler/*"]


def entries(cfg: KernelConfig) -> list[dict]:
    """What this config needs: {name, repo, dest (None: the HF cache), and snapshot_download options}."""
    wm, wb, ar = cfg.worldmodel, cfg.wbench, cfg.repo_root
    gen = cfg.get("generators") or {}
    out = [
        dict(name="alaya-world-ar", repo="AlayaLab/AlayaWorld-v1.1-stage2b", dest=wm / "weights/alaya-world-ar"),
        dict(name="alaya-world-dmd", repo="AlayaLab/AlayaWorld-v1.1-stage3", dest=wm / "weights/alaya-world-dmd"),
        dict(name="ltx-2.3", repo="Lightricks/LTX-2.3", dest=wm / "weights/ltx-2.3",
             allow_patterns=["ltx-2.3-22b-dev.safetensors"]),
        dict(name="gemma-3-12b", repo="google/gemma-3-12b-it-qat-q4_0-unquantized",
             dest=wm / "weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized"),
        dict(name="vigeo-1.1", repo="pkqbajng/ViGeo1.1", dest=wm / "third_party/ViGeo/checkpoints/ViGeo1.1"),
        dict(name="wbench-data", repo="meituan-longcat/WBench", repo_type="dataset", dest=wb / "data",
             ignore_patterns=["splits/*"]),
        dict(name="wbench-weights", repo="meituan-longcat/WBench-weights", dest=wb / "weights"),
        dict(name="captioner", repo=cfg.get("captioner.model"), dest=None),
    ]
    if cfg.get("eval.data_repo"):
        out.append(dict(name="eval-data", repo=cfg.get("eval.data_repo"), repo_type="dataset", dest=cfg.eval_data))
    if (cfg.get("images") or {}).get("enabled"):
        out.append(dict(name="z-image-turbo", repo="Tongyi-MAI/Z-Image-Turbo", dest=ar / cfg.get("images.weights"),
                        revision=cfg.get("images.revision"), ignore_patterns=["assets/*"]))
    if (gen.get("wan22") or {}).get("enabled"):
        out.append(dict(name="wan2.2", repo="Wan-AI/Wan2.2-TI2V-5B", dest=ar / gen["wan22"]["weights"]))
    if (gen.get("ltx25") or {}).get("enabled"):
        out.append(dict(name="ltx-2.5", repo="Lightricks/LTX-2.5", dest=ar / gen["ltx25"]["weights"],
                        allow_patterns=LTX25_FILES))
    if (gen.get("h3") or {}).get("enabled"):
        out.append(dict(name="minimax-h3", repo="MiniMaxAI/MiniMax-H3", dest=ar / gen["h3"]["weights"],
                        revision=gen["h3"]["revision"], allow_patterns=H3_FILES))
    return out


def marker(cfg: KernelConfig, e: dict) -> Path:
    tag = f"{e['repo']}@{e.get('revision') or 'main'}".replace("/", "--")
    return cfg.repo_root / ".cache" / "downloads" / f"{e['name']}--{tag}.done"


def done(cfg: KernelConfig, e: dict) -> bool:
    return marker(cfg, e).exists() and (e["dest"] is None or e["dest"].exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=[], metavar="FILE", help="kernel.yaml overlays, as for `ar run`")
    ap.add_argument("--check", action="store_true", help="only list what is missing")
    args = ap.parse_args(argv)
    os.environ.update(cache_env())                 # the HF cache inside the project
    cfg = KernelConfig.load(overlays=args.config)
    todo = [e for e in entries(cfg) if not done(cfg, e)]
    for e in entries(cfg):
        where = os.path.relpath(e["dest"], cfg.repo_root.parent) if e["dest"] else "HF cache"
        print(f"{'missing' if e in todo else 'ok':8} {e['name']:16} {e['repo']} -> {where}")
    if args.check:
        return 1 if todo else 0
    from huggingface_hub import snapshot_download
    for e in todo:
        print(f"downloading {e['name']} ({e['repo']})", flush=True)
        options = {k: e[k] for k in ("repo_type", "revision", "allow_patterns", "ignore_patterns") if k in e}
        snapshot_download(e["repo"], local_dir=e["dest"], **options)
        marker(cfg, e).parent.mkdir(parents=True, exist_ok=True)
        marker(cfg, e).touch()
    if (cfg.get("generators.h3") or {}).get("enabled") and not (cfg.repo_root / "weights/minimax-h3-adaln").exists():
        print("rollout_h3 also needs its AdaLN cache: see README step 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
