"""Generate silent 24-fps LTX-2.5 clips with the dev model and distilled second-stage LoRA.
Each worker uses upstream CLI defaults for sampling, quantization, and offload; keyframes pin images to frames."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import protocol

TRANSFORMER = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSAMPLER = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DISTILLED_LORA = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
FPS = 24


def upstream_argv(a) -> list[str]:
    """The flags the pipeline's own CLI would get (--prompt/--output-path are per item, below)."""
    w = Path(a.weights)
    argv = ["--transformer-path", str(w / TRANSFORMER), "--text-encoder-path", str(w / TEXT_ENCODER),
            "--video-vae-path", str(w / VIDEO_VAE), "--audio-vae-path", str(w / AUDIO_VAE),
            "--spatial-upsampler-path", str(w / UPSAMPLER), "--num-frames", str(a.frames),
            "--height", str(a.height), "--width", str(a.width), "--frame-rate", str(FPS),
            "--prompt", "-", "--output-path", "-"]
    if a.quantization != "none":
        argv += ["--quantization", a.quantization]
    if a.offload != "none":
        argv += ["--offload", a.offload]
    return argv + ["--distilled-lora", str(w / DISTILLED_LORA), "1.0"]


def main() -> int:
    job = protocol.load_job()
    if job["worker"] is None:
        return protocol.spawn_workers(job)
    out, c = Path(job["out"]), job["config"]
    a = SimpleNamespace(weights=job["paths"]["weights"], frames=job["args"]["frames"], height=job["args"]["height"],
                        width=job["args"]["width"], quantization=c.get("quantization", "fp8-cast"),
                        offload=c.get("offload", "cpu"))

    def load():
        import torch
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.args import ImageConditioningInput, add_generated_keyframes_arg, default_2_stage_arg_parser
        from ltx_pipelines.utils.constants import detect_params
        from ltx_pipelines.utils.media_io import encode_video

        params = detect_params(str(Path(a.weights) / TRANSFORMER))
        args = add_generated_keyframes_arg(default_2_stage_arg_parser(
            params=params, supports_auto_duration=True)).parse_args(upstream_argv(a))
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
        return SimpleNamespace(torch=torch, pipeline=pipeline, call=call, tiling=AUTO_TILING,
                               chunks=get_video_chunks_number, encode_video=encode_video,
                               conditioning=ImageConditioningInput)

    def one(m, item):
        t0, torch, result = time.monotonic(), m.torch, None
        try:
            torch.cuda.reset_peak_memory_stats()
            images = [m.conditioning(k["image"], a.frames - 1 if k["frame"] == -1 else k["frame"], 1.0)
                      for k in item.get("keyframes") or []]
            path = out / f"{item['index']}.mp4"
            with torch.inference_mode():
                result = m.pipeline(prompt=item["prompt"], seed=int(item["seed"]), height=a.height, width=a.width,
                                    num_frames=a.frames, frame_rate=float(FPS), images=images,
                                    vae_dtype=torch.bfloat16, tiling_config=m.tiling, **m.call)
                m.encode_video(video=result.video, fps=FPS, audio=None, output_path=str(path),
                               video_chunks_number=m.chunks(result.num_frames, result.tiling_config))
            return {"video": path}, {"seconds": round(time.monotonic() - t0, 2),
                                     "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                     "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        finally:
            result = None
            torch.cuda.empty_cache()

    protocol.run_items(job, load, one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
