"""rollout_ltx25 worker (runs in the gen-ltx25 conda-prefix env): Lightricks' ltx-pipelines, one
pipeline built once, renders each item's shard (text-to-video, or image-to-video when the item
carries a first frame, conditioned at frame 0 with strength 1.0).

python ltx25_generate.py --items items.json --out DIR --rank R --world W \
    --weights <weights/ltx-2.5> --variant distilled|dev --frames N --height H --width W \
    [--quantization fp8-cast] [--offload cpu]

Bridge protocol: handles items with index % world == rank, in index order; per item
writes <out>/<index>.mp4 (24 fps, video only) then <out>/<index>.json ({"ok": true, ...} or
{"ok": false, "error": "..."}). The pipeline is built once, before the first item.

The pipeline is configured by the module's own CLI parser (`upstream_args`), fed the flags its
`main()` takes, and constructed and called exactly as that `main()` does at the commit pinned in
configs/kernel.yaml (generators.ltx25.commit): distilled = ltx_pipelines.distilled
(DistilledPipeline), dev = ltx_pipelines.ti2vid_two_stages (TI2VidTwoStagesPipeline, dev
transformer + the distilled LoRA at 1.0 for stage 2). So the quantization policy, offload mode,
split ModelPaths and every sampling default (steps, guidance) are upstream's own.

The audio LTX-2.5 generates alongside is not muxed (encode_video(audio=None)).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TRANSFORMER = {"distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
               "dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"}
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSAMPLER = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DISTILLED_LORA = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
FPS = 24


def upstream_argv(a) -> list[str]:
    """The flags the variant's own CLI would get (--prompt/--output-path are per item, below)."""
    w = Path(a.weights)
    argv = ["--transformer-path", str(w / TRANSFORMER[a.variant]), "--text-encoder-path", str(w / TEXT_ENCODER),
            "--video-vae-path", str(w / VIDEO_VAE), "--audio-vae-path", str(w / AUDIO_VAE),
            "--spatial-upsampler-path", str(w / UPSAMPLER), "--num-frames", str(a.frames),
            "--height", str(a.height), "--width", str(a.width), "--frame-rate", str(FPS),
            "--prompt", "-", "--output-path", "-"]
    if a.quantization != "none":
        argv += ["--quantization", a.quantization]
    if a.offload != "none":
        argv += ["--offload", a.offload]
    if a.variant == "dev":
        argv += ["--distilled-lora", str(w / DISTILLED_LORA), "1.0"]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--out", "--weights"):
        ap.add_argument(name, required=True)
    for name in ("--rank", "--world", "--frames", "--height", "--width"):
        ap.add_argument(name, type=int, required=True)
    ap.add_argument("--variant", choices=sorted(TRANSFORMER), required=True)
    ap.add_argument("--quantization", default="fp8-cast")
    ap.add_argument("--offload", default="cpu")
    a = ap.parse_args()

    out = Path(a.out)
    mine = [item for item in json.loads(Path(a.items).read_text(encoding="utf-8"))
            if item["index"] % a.world == a.rank]
    if not mine:
        return 0

    import torch
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput, add_generated_keyframes_arg
    from ltx_pipelines.utils.constants import detect_params
    from ltx_pipelines.utils.media_io import encode_video

    argv = upstream_argv(a)
    params = detect_params(str(Path(a.weights) / TRANSFORMER[a.variant]))
    if a.variant == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
        args = add_generated_keyframes_arg(default_2_stage_distilled_arg_parser(
            params=params, supports_auto_duration=True)).parse_args(argv)
        pipeline = DistilledPipeline(
            model_paths=args.model_paths, spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (), quantization=args.quantization,
            compilation_config=args.compile, offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization)
        call = {}
    else:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.args import default_2_stage_arg_parser
        args = add_generated_keyframes_arg(default_2_stage_arg_parser(
            params=params, supports_auto_duration=True)).parse_args(argv)
        pipeline = TI2VidTwoStagesPipeline(
            model_paths=args.model_paths, distilled_lora=args.distilled_lora,
            spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (), quantization=args.quantization,
            compilation_config=args.compile, offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization)
        call = {"negative_prompt": args.negative_prompt, "num_inference_steps": args.num_inference_steps,
                "video_guider_params": MultiModalGuiderParams(
                    cfg_scale=args.video_cfg_guidance_scale, stg_scale=args.video_stg_guidance_scale,
                    rescale_scale=args.video_rescale_scale, modality_scale=args.a2v_guidance_scale,
                    skip_step=args.video_skip_step, stg_blocks=args.video_stg_blocks),
                "audio_guider_params": MultiModalGuiderParams(
                    cfg_scale=args.audio_cfg_guidance_scale, stg_scale=args.audio_stg_guidance_scale,
                    rescale_scale=args.audio_rescale_scale, modality_scale=args.v2a_guidance_scale,
                    skip_step=args.audio_skip_step, stg_blocks=args.audio_stg_blocks),
                "max_batch_size": args.max_batch_size}

    for item in mine:
        index, t0 = item["index"], time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            images = [ImageConditioningInput(item["image"], 0, 1.0)] if item.get("image") else []
            with torch.inference_mode():
                result = pipeline(prompt=item["prompt"], seed=int(item["seed"]), height=a.height, width=a.width,
                                  num_frames=a.frames, frame_rate=float(FPS), images=images,
                                  vae_dtype=torch.bfloat16, tiling_config=AUTO_TILING, **call)
                encode_video(video=result.video, fps=FPS, audio=None, output_path=str(out / f"{index}.mp4"),
                             video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config))
            status = {"ok": True, "seconds": round(time.monotonic() - t0, 2),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = None
        torch.cuda.empty_cache()
        (out / f"{index}.json").write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
