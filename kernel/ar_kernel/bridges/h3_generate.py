"""rollout_h3 worker (runs in the gen-h3 conda-prefix env, diffusers 0.40.0): MiniMax H3 through
diffusers' MiniMaxH3ModularPipeline, on every GPU it is given.

python h3_generate.py --items items.json --prompts prompts.json --out DIR --rank R --world W \
    --weights <weights/minimax-h3> --frames N --height H --width W --gpu0-reserve-gib G

Bridge protocol: handles items with index % world == rank, in index order; per item writes
<out>/<index>.mp4 then <out>/<index>.json ({"ok": true, "seconds": t} or {"ok": false, "error": "..."}).

Two phases, because the 62 GiB text encoder and the transformer do not fit together:
1. the text encoder at bf16 across the cards encodes every item (prompt and keyframes), then is freed;
2. the transformer, int8 weight-only (torchao), is placed with an explicit device map: the small
   layers and both VAEs on GPU 0 (the forward pass indexes its outputs with tensors that live
   there, so device_map="balanced" fails), the 50 blocks spread by card memory.
The pipeline index names the hub repo, so every component is pointed at the local folder
(`_component_specs` is private to diffusers; the env pins the version).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

FPS = 24
N_BLOCKS = 50
SMALL = ("proj_in", "audio_proj_in", "context_embedder", "token_refiner", "time_embedder", "time_proj", "rope",
         "norm_out", "proj_out", "audio_proj_out")
NOT_QUANTIZED = [m for m in SMALL if m != "rope"]
TEXT_ENCODER_GIB = 70       # 62 GiB of bf16 weights and room to run them


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop and resize to the canvas (the pipeline would stretch a first frame instead)."""
    return ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS)


UNFIT = 78                  # exit code: the machine is too small (gpu_jobs.UNFIT_EXIT stops the run on it)


class TooSmall(RuntimeError):
    pass


def require_memory(mem_gib: list[float]) -> None:
    if sum(mem_gib) < TEXT_ENCODER_GIB:
        raise TooSmall(f"MiniMax H3 needs about {TEXT_ENCODER_GIB} GiB of GPU memory across the job's cards "
                           f"for its text encoder: got {sum(mem_gib):.0f} GiB on {len(mem_gib)} card(s)")


def does_not_fit(what: str, mem_gib: list[float]) -> TooSmall:
    sizes = ", ".join(f"{m:.0f}" for m in mem_gib)
    return TooSmall(f"MiniMax H3 ran out of GPU memory loading {what} on {len(mem_gib)} card(s) of {sizes} GiB; "
                        "it was measured on 4 cards of 24 GiB")


def block_device_map(mem_gib: list[float], reserve0_gib: float) -> dict[str, int]:
    """Small layers on GPU 0; the blocks in order across the cards, in proportion to each card's
    memory less `reserve0_gib` on GPU 0 (the VAEs, the small layers, decoding)."""
    room = [max(0.0, m - (reserve0_gib if i == 0 else 0.0)) for i, m in enumerate(mem_gib)]
    dmap = {name: 0 for name in SMALL}
    start = 0
    for device in range(len(room)):
        upto = round(N_BLOCKS * sum(room[:device + 1]) / sum(room))
        dmap.update({f"transformer_blocks.{b}": device for b in range(start, upto)})
        start = upto
    return dmap


def local(pipe, weights: str):
    for spec in pipe._component_specs.values():
        if getattr(spec, "pretrained_model_name_or_path", None):
            spec.pretrained_model_name_or_path = weights
    return pipe


def workflow_of(item: dict) -> str:
    return "fl2va" if item.get("keyframes") else "t2va"


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--prompts", "--out", "--weights"):
        ap.add_argument(name, required=True)
    for name in ("--rank", "--world", "--frames", "--height", "--width"):
        ap.add_argument(name, type=int, required=True)
    ap.add_argument("--gpu0-reserve-gib", type=float, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    mine = [i for i in json.loads(Path(a.items).read_text(encoding="utf-8")) if i["index"] % a.world == a.rank]
    if not mine:
        return 0

    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.utils.export_utils import encode_video
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration

    mem = [torch.cuda.mem_get_info(i)[1] / 2**30 for i in range(torch.cuda.device_count())]
    require_memory(mem)
    full = ModularPipeline.from_pretrained(a.weights)
    flows = {}                              # workflow -> (its encode blocks, the rest)
    for name in sorted({workflow_of(i) for i in mine}):
        rest = full.blocks.get_workflow(name)
        encode = SequentialPipelineBlocks.from_blocks_dict(
            {k: rest.sub_blocks.pop(k) for k in ("before_encode", "text_encoder") if k in rest.sub_blocks})
        flows[name] = (encode, rest)

    def pipelines(blocks: dict, **given) -> dict:
        """One pipeline per workflow over the same loaded components."""
        pipes = {name: local(b.init_pipeline(a.weights), a.weights) for name, b in blocks.items()}
        first = next(iter(pipes.values()))
        first.update_components(**given)
        first.load_components(dtype=torch.bfloat16)
        for other in list(pipes.values())[1:]:
            other.update_components(**{n: getattr(first, n) for n in other._component_specs
                                       if getattr(first, n, None) is not None})
            other.load_components(dtype=torch.bfloat16)
        return pipes

    def failed(index: int, exc: Exception) -> None:
        (out / f"{index}.json").write_text(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                                           encoding="utf-8")

    # Phase 1: conditioning for every item.
    try:
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            a.weights, subfolder="text_encoder", dtype=torch.bfloat16, device_map="balanced",
            max_memory={i: f"{m - 2:.0f}GiB" for i, m in enumerate(mem)})
    except torch.OutOfMemoryError as exc:
        raise does_not_fit("the text encoder", mem) from exc
    encoders = pipelines({n: e for n, (e, _) in flows.items()}, text_encoder=encoder)
    states = {}
    for item in mine:
        index = item["index"]
        try:
            call = {"prompt": prompts[str(index)], "height": a.height, "width": a.width}
            if item.get("keyframes"):
                images = {k["frame"]: fit(Image.open(k["image"]), (a.width, a.height)) for k in item["keyframes"]}
                if 0 in images:
                    call["image"] = images[0]
                if -1 in images:
                    call["last_image"] = images[-1]
            with torch.inference_mode():
                states[index] = encoders[workflow_of(item)](**call)
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            failed(index, exc)
    del encoders, encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: the transformer, then every clip.
    try:
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            a.weights, subfolder="transformer", dtype=torch.bfloat16,
            device_map=block_device_map(mem, a.gpu0_reserve_gib),
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=NOT_QUANTIZED))
    except torch.OutOfMemoryError as exc:
        raise does_not_fit("the transformer", mem) from exc
    transformer.requires_grad_(False)
    renderers = pipelines({n: r for n, (_, r) in flows.items()}, transformer=transformer)
    first = next(iter(renderers.values()))
    first.vae.to("cuda:0")
    first.audio_vae.to("cuda:0")

    for item in mine:
        index, t0 = item["index"], time.monotonic()
        if index not in states:
            continue
        try:
            for i in range(len(mem)):
                torch.cuda.reset_peak_memory_stats(i)
            name = workflow_of(item)
            size = {"height": a.height, "width": a.width} if name == "t2va" else {}    # fl2va: set while encoding
            with torch.inference_mode():
                result = renderers[name](state=states.pop(index), num_frames=a.frames, **size,
                                         generator=torch.Generator().manual_seed(int(item["seed"])),
                                         output=["videos", "audio", "sampling_rate"])
            encode_video(result["videos"][0], fps=FPS, output_path=str(out / f"{index}.mp4"))
            status = {"ok": True, "seconds": round(time.monotonic() - t0, 2),
                      "peak_allocated_gib": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                                             for i in range(len(mem))]}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = None
        torch.cuda.empty_cache()
        (out / f"{index}.json").write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TooSmall as exc:
        print(exc, file=sys.stderr)
        sys.exit(UNFIT)
