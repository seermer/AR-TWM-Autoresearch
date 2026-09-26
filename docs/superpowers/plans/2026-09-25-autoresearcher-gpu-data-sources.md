# AutoResearcher GPU Data Sources Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agents can request generated training clips (`rollout_alayaworld`, `rollout_wan22`, `rollout_ltx25`) and per-frame camera poses (`annotate_camera`) as asynchronous GPU jobs. Every output lands in the agent's staging directory as a ready-to-ingest candidate, and every variant is smoke-verified on this machine before it is enabled.

**Architecture:** Each tool is a `JobQueue` backend (Plan 2, Task 9) built on one shared base, `GpuJob` in `kernel/ar_kernel/tools/gpu_jobs.py`, which does four things:
- It stages the agent's input files race-free into a kernel-private job directory.
- It runs one worker process per GPU group, in the generator's own conda env.
- It post-processes outputs to the standard layout (spec §6).
- It publishes them into `/workspace/staging/{rollouts,annotations}/<job_id>/` without following links the agent planted.

The workers are small bridge scripts in `kernel/ar_kernel/bridges/`. Each loads its model once and loops over its shard of items.

AlayaWorld is the one exception to the bridge scripts. It runs WorldModel's own `scripts/finetune/train.sh` in `VALIDATE_ONLY` mode through the `custom_i2v` validation path. A 5-line WorldModel patch lets each image carry its own per-round prompt schedule.

All caches the kernel's subprocesses create go to the project-local `AutoResearcher/.cache/` (Task 1).

**Tech Stack:**
- **Kernel:** Python 3.12 (`autoresearcher` env).
- **WorldModel / ViGeo:** `alayaworld` env (Python 3.10, torch 2.7.1).
- **Wan 2.2:** new env `gen-wan22`, running the official Wan2.2 code, cloned into `AutoResearcher/third_party/Wan2.2`.
- **LTX-2.5:** new env `gen-ltx25`, running Lightricks `ltx-pipelines`/`ltx-core` from `AutoResearcher/third_party/LTX-2` (torch 2.13 cu132).
- **Media processing:** host `ffmpeg` 6.1.1.

**Spec:** `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`, sections implemented:
- §10 rows `rollout.*` and `annotate.camera`.
- §5.4 rollout provenance.
- §16.3 items 5 (generator fit) and 6 (camera annotation backend).
- §17 `generators`.

Also read:
- `docs/superpowers/plans/verification-log.md`. Findings 1–9 bind every task here that launches or kills a process.
- The Plan 2 plan, `docs/superpowers/plans/2026-09-21-autoresearcher-agent-runtime.md`: Task 9 (JobQueue, `run_cancellable`) and the Plan 4 interface contract at its end.

## Global Constraints

- **Agent boundaries.** Agents never touch WorldModel, WBench, the kernel, weights or `.env`. Generators run in the kernel, and agents reach them only through MCP tools.
- **No paid model for media.** No video or image from these tools is ever sent to the paid agent model (user rule, 2026-09-25).
- **Asynchronous GPU jobs.** Every tool here returns `{job_id}` at once. The job runs under the run's `gpu_lock`, one job at a time, on the node's explicit GPU list (spec §10).
- **GPU policy.**
  - Device lists are always explicit; the default is `0,1,2,3`; any indices are allowed; at least 4 per run.
  - Workers get `CUDA_VISIBLE_DEVICES` set from that list plus `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  - GPU tests read the list from `AR_TEST_GPUS` (default `0,1,2,3`). Check `nvidia-smi` first: another user has held GPUs 2–3 before. If some are busy, use any 4 free ones and record which.
- **Launching.**
  - Every launch goes through `run_cancellable` (process group, `subproc.*` telemetry).
  - `torchrun` jobs get a free `MASTER_PORT`.
  - After a job, GPU memory must return to its pre-job level (the captioner's release check).
- **Python envs.** Never use system or `base` Python. Every command uses `conda run --no-capture-output -n <env>`.
  - Kernel and tests: `autoresearcher`.
  - WorldModel and ViGeo: `alayaworld`.
  - Wan: `gen-wan22` (new).
  - LTX: `gen-ltx25` (new).
  - Do not use or modify the `gen-alaya` env: it is not ours.
- **Large files stay inside the project** (user rule, 2026-09-25): models, datasets, compile caches and package caches.
  - Kernel subprocesses get `HF_HOME`, `XDG_CACHE_HOME` and the other cache variables pointing at `AutoResearcher/.cache/` (Task 1).
  - Env installs set `PIP_CACHE_DIR`/`UV_CACHE_DIR` there too.
  - Conda envs themselves stay in `~/miniforge3/envs` (named envs; `/home` has ~91 GB free; each new env is ~8–12 GB).
- **Disk.** Disk is the binding constraint. Ask before freeing space, and never delete anything outside the project folder. Copying into the project is fine; the originals stay where they are.
- **Portability.** No absolute paths in tracked code or config; everything resolves from `KernelConfig` / `REPO_ROOT` (`docs/PORTABILITY.md`).
- **Tests.** The default unit suite uses no GPU and no network. Real-model tests are marked `gpu` and run explicitly and alone.
- **WorldModel changes.** Only the Task 2 patch is allowed (user-approved 2026-09-25). No other WorldModel or WBench edits except genuine bug fixes, each reported.
- **Commits.** Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit with explicit pathspecs; never `git add -A`.
- **Branching.** Work on branch `feat/gpu-data-sources` in AutoResearcher and on `main` in WorldModel (one small commit). When done, fast-forward AutoResearcher `main` and push all three repos (memory `main-always-current`).

## Verified facts this plan relies on (checked 2026-09-25)

1. **GPUs:** 6 × RTX 4090, 24 GB each (Ada, sm_89; FP8 capable, not Blackwell). The driver is 615.71, with CUDA UMD 13.4, so cu132 wheels run. Host RAM is 251 GB.
2. **Disk:** `/mnt/biometrics` (project) has 2.1 TB free; `/` (`/home`, conda envs, `~/.cache`) has 91 GB free.
3. **Existing out-of-project caches from this project:**
   - `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8` (29 GB), used by the captioner.
   - About one entry in `~/.cache/vllm/torch_compile_cache`. That 23 GB cache is mostly the user's older work.
   - Everything else in `~/.cache/huggingface/hub` for this project is an empty 12 KB stub.
   - WBench already keeps its weights in `WBench/weights/`, including its own torch-hub dir (`src/metrics/weight_utils.py`).
4. **ViGeo** is at `WorldModel/third_party/ViGeo`, with its checkpoint at `checkpoints/ViGeo1.1/vigeo.pt`; the `alayaworld` env already runs it for WorldModel.
   - API: `ViGeo.from_pretrained(<dir>)`, `model.infer(images[T,3,H,W] in [0,1], mode="offline"|"chunk")`.
   - Outputs: `pose_pred [T,3,4]` camera-to-world, OpenCV axes (x right, y down, z forward). Focal comes from `recover_focal_from_xy` (normalized; see `vigeo/utils.py`); `utils/data.py:load_intrinsic` shows the normalization.
5. **AlayaWorld single-image generation:**
   - It is `VALIDATE_ONLY=1 ALAYA_USE_FA3=0 LOG_FILTER=all CONFIG_PATH=<cfg> bash scripts/finetune/train.sh`, with cwd `WorldModel`. It uses the `custom_i2v` validation mode (`configs/infer_i2v_camera.yaml` = 4-step DMD student; `configs/infer_i2v_camera_ar.yaml` = 30-step AR teacher, `cfg_scale` 3.0).
   - Inputs: an image dir, `captions.json` (image name → caption), and `pose.jsonl` (entry i = `{"pose_path": npz with cam_c2w [N,4,4], "intrinsic"?: [fx,fy,cx,cy] normalized}`).
   - Images are sorted by name, and image i pairs with pose entry i.
   - `validation.max_samples` limits how many images are rendered (1 in the shipped configs).
   - Output: `<run.output_dir>/validation/.../<stem>_pred_clean.mp4`, where the stem contains `sample-<idx>`. It is written before any joystick overlay.
   - `train.sh` sets the rank count from `CUDA_VISIBLE_DEVICES` and defaults `MASTER_PORT` to 29500.
6. **Per-round prompts in WorldModel.** `RolloutTrainer._validation_prompt_schedule` already switches prompts per round when a sample's metadata has `wbench_prompt_schedule` (index `r // wbench_chunks_per_turn`). Otherwise it uses the mode-wide `prompt_schedule`. `custom_i2v` never sets that metadata key, which is what Task 2 adds.
7. **LTX-2.5 weights** are in `AutoResearcher/weights/ltx-2.5` as a split, Comfy-aligned pack:
   - `diffusion_models/ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors`: 42 GB each.
   - `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`: 26 GB.
   - `vae/ltx-2.5-{video,audio}-vae-bf16.safetensors`.
   - `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`.
   - `latent_upscale_models/…spatial-upscaler-x2…`.
   - `model_patches/ltx-2.5-duration-head-bf16.safetensors`.
   
   The int8 `comfy` files are ComfyUI-only. NVFP4 needs Blackwell.
   
   The runner is `ltx_pipelines.distilled` (distilled) or `ltx_pipelines.ti2vid_two_stages` (dev + distilled LoRA), with split flags `--transformer-path --text-encoder-path --video-vae-path --audio-vae-path --spatial-upsampler-path`, `--image PATH FRAME_IDX STRENGTH`, `--num-frames` (`% 8 == 1`) and width/height divisible by 32.
   
   **The multi-GPU runners replicate the full working model on every GPU ("a latency tool, not a memory tool")**, so on 24 GB cards the only route is `--quantization fp8-cast` plus `--offload cpu` on one GPU per process. `ltx-core` pins `torch==2.13.0` + `natten` (cu132) for the DiffVAE. The LTX-2 repo is at `https://github.com/Lightricks/LTX-2` (6 MB).
   
   *(This corrects the Plan 2 note that "LTX-2.5 runs sharded across the 4 cards".)*
8. **Wan 2.2 TI2V-5B weights** are in `AutoResearcher/weights/wan2.2-ti2v-5b`, in the official format (`Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, diffusers-style DiT shards, `google/umt5-xxl` tokenizer).
   - Runner: `generate.py --task ti2v-5B --size 1280*704 --ckpt_dir … [--image …] --offload_model True --convert_model_dtype --t5_cpu`, with 121 frames at 24 fps, 50 steps, guide 5.0 and shift 5.0 by default.
   - Requirements: `torch>=2.4`, `transformers>=4.49,<=4.51.3`, `diffusers>=0.31`, `flash_attn`, `decord`-free.
   - Repo: `https://github.com/Wan-Video/Wan2.2` (18 MB).
   - **1280×704 is 1.818:1, 2.3 % off 16:9, over the ingest tolerance (±2 %)**, so the kernel center-crops to 1248×704 (0.3 % off).
9. **Standard-format timing (spec §6.2).** The training window is 25 history + 32 target frames at 24 fps. A `per_chunk` clip must have every segment boundary at `25/24 + k·32/24` s, within half a frame.
10. **Kernel building blocks from Plan 2:**
    - `captioner.stage_clip(caller, path, dst)`: race-free hard link or copy of an agent file.
    - `captioner.clip_host_path`, `captioner.container_path` and `captioner.gpu_memory_mib`.
    - `hf_tools._move_into(src, base, rel)`: an O_NOFOLLOW move into agent-owned dirs.
    - `jobs.run_cancellable`, `JobQueue.register`, and `JobQueue.submit(caller, backend_name, args)`, which raises `ToolError("generator … is not enabled")` for unregistered names.
11. **Nothing reads the current `generators:` block of `kernel.yaml` yet** (grep: no references in `kernel/`), so this plan reshapes it freely.

## Review Focus

1. **An agent swaps a staging subdirectory for a symlink while a job runs.** Publishing must raise per item instead of writing outside staging. Task 3 has the test.
2. **A job is cancelled mid-run with 4 workers alive.** Every worker's process group must die and the GPUs must be released. Task 3 has the test.
3. **One worker crashes (OOM) while the others succeed.** The job still returns `done`, with that worker's items reported as per-item errors that include the log tail, not a failed job and not silently missing items. Task 3 has the test.
4. **A pose file whose frame count differs from the published mp4's**, for example after the 16:9 crop or the AlayaWorld leading-frame trim. Ingest would reject it. Every backend re-probes the final mp4 and asserts `len(cam_c2w) == frames` before publishing. Tasks 4 and 5 have the test.
5. **Very long or huge input videos to `annotate_camera`.** They must be refused per clip (over `annotate.max_frames`), not OOM the whole job. Task 4 has the test.

---

## File structure

| File | Responsibility |
|---|---|
| `kernel/ar_kernel/subproc.py` (modify) | `cache_dir()`, `cache_env()`: project-local cache variables for every `run_in_env` child |
| `kernel/ar_kernel/cli.py` (modify) | Apply `cache_env()` to the kernel process itself at startup |
| `.gitignore` (modify) | `.cache/`, `third_party/` |
| `kernel/ar_kernel/tools/captioner.py` (modify) | Extract `wait_gpu_release()` for reuse (behavior unchanged) |
| `kernel/ar_kernel/tools/gpu_jobs.py` (create) | `GpuJob` base: submit validation, staging, `run_workers`, publishing, candidates, provenance; tool registration for every GPU data tool; `build_gpu_backends(cfg, …)` |
| `kernel/ar_kernel/tools/annotate.py` (create) | `AnnotateBackend` (`annotate_camera`) |
| `kernel/ar_kernel/tools/rollouts.py` (create) | `AlayaWorldBackend`, `Wan22Backend`, `Ltx25Backend` |
| `kernel/ar_kernel/bridges/vigeo_poses.py` (create) | Worker in `alayaworld`: video → `cam_c2w [N,4,4]` + pixel intrinsics |
| `kernel/ar_kernel/bridges/wan22_generate.py` (create) | Worker in `gen-wan22`: loads WanTI2V once, renders its shard |
| `kernel/ar_kernel/bridges/ltx25_generate.py` (create) | Worker in `gen-ltx25`: loads an LTX-2.5 pipeline once, renders its shard |
| `configs/kernel.yaml` (modify) | `generators:` reshaped, `annotate:` added |
| `tests/test_cache_env.py`, `tests/test_gpu_jobs.py`, `tests/test_annotate.py`, `tests/test_rollouts.py` (create) | Unit tests (fake workers) plus one `gpu` test per real backend |
| `tests/fixtures/fake_gen_worker.py` (create) | Fake worker implementing the bridge protocol |
| `WorldModel/alaya/data/custom_i2v.py` (modify), `WorldModel/tests/test_custom_i2v_schedule.py` (create) | A caption may be a list of per-round prompts |
| `seed_agent/knowledge/data_building.md` (modify) | Short section on rollouts and annotation (simplicity rule: a few lines) |
| Spec, Plan 2 contract, `verification-log.md`, `docs/PORTABILITY.md` (modify) | Amendments and measured results |

**Bridge worker protocol** (all three bridge scripts and the fake worker):
- **Arguments:** `python <bridge>.py --items <items.json> --out <dir> --rank <r> --world <w> [bridge options]`.
- **`items.json`:** a list of objects, each with an integer `"index"` plus the item's fields; file fields are absolute host paths inside the job dir.
- **Sharding:** the worker handles exactly the items with `index % world == rank`, in index order.
- **Per item it writes:**
  - the output file(s) as `<out>/<index>.<ext>`;
  - then `<out>/<index>.json` containing `{"ok": true, ...metadata}` or `{"ok": false, "error": "<message>"}`.
- **Error handling:** a per-item exception is caught and reported in that JSON, and the worker moves on to the next item.
- **Loading:** the model loads once, before the first item.
- **Exit code:** non-zero only when the model cannot load.

---

### Task 1: Project-local caches

**Files:**
- Modify: `kernel/ar_kernel/subproc.py`, `kernel/ar_kernel/cli.py`, `.gitignore`, `docs/PORTABILITY.md`, spec §2 (a "Caches" paragraph after §2.2)
- Test: `tests/test_cache_env.py`

**Interfaces:**
- Produces: `subproc.cache_dir() -> Path`, `subproc.cache_env() -> dict[str, str]`. `run_in_env` children always see `cache_env()`, overridable per call by `extra_env`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_env.py
"""Large caches stay inside the project (user rule 2026-09-25): every kernel subprocess gets
cache variables pointing at AutoResearcher/.cache unless AR_CACHE_DIR relocates it."""
import json
from pathlib import Path

from ar_kernel.config import REPO_ROOT
from ar_kernel.subproc import cache_dir, cache_env, run_in_env

VARS = ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "PIP_CACHE_DIR", "UV_CACHE_DIR")


def test_default_cache_dir_is_inside_the_repo(monkeypatch):
    monkeypatch.delenv("AR_CACHE_DIR", raising=False)
    assert cache_dir() == REPO_ROOT / ".cache"
    env = cache_env()
    assert set(env) == set(VARS)
    assert all(Path(v).is_relative_to(REPO_ROOT / ".cache") for v in env.values())
    assert env["HF_HOME"] == str(REPO_ROOT / ".cache" / "huggingface")


def test_ar_cache_dir_relocates(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    assert cache_dir() == tmp_path
    assert cache_env()["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm")


def test_children_get_cache_env_and_extra_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/else")          # a user-level setting must not leak through
    code = "import json, os; print(json.dumps({k: os.environ.get(k) for k in ('HF_HOME', 'TORCH_HOME')}))"
    proc = run_in_env("autoresearcher", ["python", "-c", code], cwd=tmp_path,
                      extra_env={"TORCH_HOME": str(tmp_path / "mine")})
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen == {"HF_HOME": str(tmp_path / "huggingface"), "TORCH_HOME": str(tmp_path / "mine")}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_cache_env.py -q`
Expected: FAIL (`ImportError: cannot import name 'cache_dir'`).

- [ ] **Step 3: Implement**

In `kernel/ar_kernel/subproc.py`, add below the constants:

```python
from .config import REPO_ROOT

CACHE_VARS = {"HF_HOME": "huggingface", "XDG_CACHE_HOME": "", "TORCH_HOME": "torch",
              "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
              "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "nv", "PIP_CACHE_DIR": "pip",
              "UV_CACHE_DIR": "uv"}


def cache_dir() -> Path:
    """Where model downloads, compile caches and package caches go: inside the project
    (AutoResearcher/.cache), or AR_CACHE_DIR. The user's rule: large files stay in the project."""
    return Path(os.environ.get("AR_CACHE_DIR") or REPO_ROOT / ".cache")


def cache_env() -> dict[str, str]:
    root = cache_dir()
    return {var: str(root / sub) if sub else str(root) for var, sub in CACHE_VARS.items()}
```

In `run_in_env`, change the env line to
`process_env = {**os.environ, **cache_env(), **(extra_env or {})}`.

In `cli.py` `main()`, as the first statement, add `os.environ.update(cache_env())`. This makes in-process `huggingface_hub` (HfTools) use the project cache too. Import `cache_env` from `.subproc`.

Before editing, check that `config.py` does not import `subproc`, since a circular import would break this. If it does, move `REPO_ROOT` into a tiny module instead.

`.gitignore`: add the lines `.cache/` and `third_party/`.

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_cache_env.py tests/test_subproc.py tests/test_captioner.py -q`
Expected: all PASS.

- [ ] **Step 5: Copy the captioner model into the project cache and prove it resolves offline**

```bash
mkdir -p AutoResearcher/.cache/huggingface/hub
cp -a ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8 AutoResearcher/.cache/huggingface/hub/
HF_HOME=$PWD/AutoResearcher/.cache/huggingface HF_HUB_OFFLINE=1 conda run --no-capture-output -n zhantaoy-vllm \
  python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.8-27B-FP8', local_files_only=True))"
du -sh AutoResearcher/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8
```

Expected: a path under `AutoResearcher/.cache/huggingface/hub/…/snapshots/<sha>`, size ~29 GB.

Do **not** delete the original in `~/.cache`; it is outside the project. Report it to the user as removable if they choose.

The vLLM compile cache will be rebuilt under `.cache/vllm` on the next captioner start, and that first start will be slower.

- [ ] **Step 6: Docs**
  - `docs/PORTABILITY.md`: add a "Caches" section covering `AutoResearcher/.cache/` (gitignored), `AR_CACHE_DIR`, which variables are set, and the fact that a new machine needs the captioner model copied or downloaded there.
  - Spec: add a "Caches" paragraph after §2.2 with the same content in one paragraph.

- [ ] **Step 7: Commit**

```bash
git add kernel/ar_kernel/subproc.py kernel/ar_kernel/cli.py .gitignore tests/test_cache_env.py docs/PORTABILITY.md docs/superpowers/specs/2026-09-17-autoresearcher-design.md
git commit -m "feat(kernel): project-local caches for every kernel subprocess (AutoResearcher/.cache)"
```

---

### Task 2: WorldModel: per-image prompt schedules in `custom_i2v`

**Files:**
- Modify: `WorldModel/alaya/data/custom_i2v.py`
- Test: `WorldModel/tests/test_custom_i2v_schedule.py`

**Interfaces:**
- Produces: in `captions.json`, a value may be a JSON list of strings, one per rollout round. The sample's caption is then `list[0]` and `metadata["wbench_prompt_schedule"]` is the list. With the mode setting `wbench_chunks_per_turn: 1`, the trainer uses prompt `r` for round `r` (the last prompt repeats). String values behave exactly as before.

- [ ] **Step 1: Write the failing test**

```python
# WorldModel/tests/test_custom_i2v_schedule.py
import json

import numpy as np
from PIL import Image

from alaya.data.custom_i2v import CustomI2VDataset


def _inputs(tmp_path, captions):
    (tmp_path / "images").mkdir()
    for name in ("0000.png", "0001.png"):
        Image.new("RGB", (64, 36), (10, 20, 30)).save(tmp_path / "images" / name)
    np.savez(tmp_path / "cam.npz", cam_c2w=np.tile(np.eye(4, dtype=np.float32), (8, 1, 1)))
    (tmp_path / "pose.jsonl").write_text(
        "".join(json.dumps({"pose_path": str(tmp_path / "cam.npz")}) + "\n" for _ in range(2)))
    (tmp_path / "captions.json").write_text(json.dumps(captions))
    return CustomI2VDataset(image_dir=str(tmp_path / "images"), pose_jsonl=str(tmp_path / "pose.jsonl"),
                            annotation_base_dir=None, width=64, height=32, frames=1, traj_frames=8,
                            captions_json=str(tmp_path / "captions.json"))


def test_list_caption_becomes_a_per_round_schedule(tmp_path):
    ds = _inputs(tmp_path, {"0000": ["walk forward", "turn left"], "0001": "a plain caption"})
    first, second = ds[0], ds[1]
    assert first["caption"] == "walk forward"
    assert first["metadata"]["wbench_prompt_schedule"] == ["walk forward", "turn left"]
    assert second["caption"] == "a plain caption"
    assert "wbench_prompt_schedule" not in second["metadata"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd WorldModel && conda run --no-capture-output -n alayaworld python -m pytest tests/test_custom_i2v_schedule.py -q`
Expected: FAIL. The list is stringified by `{str(k): str(v)}`, so the caption is `"['walk forward', 'turn left']"`.

- [ ] **Step 3: Implement**

In `__init__`, change the captions load so that list values are kept:

```python
                self.captions = {str(k): ([str(p) for p in v] if isinstance(v, list) else str(v))
                                 for k, v in json.load(f).items()}
```

In `__getitem__`, replace the final caption lines with:

```python
        # Per-image caption (try stem then file name), falling back to the generic prompt. A list is a
        # per-round prompt schedule: round r uses entry r (the last one repeats), through the same
        # metadata key the WBench path uses (set the mode's wbench_chunks_per_turn: 1).
        caption = self.captions.get(img_path.stem) or self.captions.get(img_path.name) or self.caption
        if isinstance(caption, list):
            metadata["wbench_prompt_schedule"] = caption
            caption = caption[0]
        return {"video_pixels": video_pixels, "caption": caption, "metadata": metadata}
```

Then check that the validation dataloader's collate passes the list through for batch size 1. `grep -n "collate" alaya/data/dataloader.py alaya/trainer/rollout_trainer.py`. The WBench dataset already places this list in metadata, so whatever collate WBench validation uses works. If `custom_i2v` goes through a different collate that would turn the list into tuples, handle it the way WBench does and say so in the report. Task 5's GPU smoke confirms the rounds really switch prompts.

- [ ] **Step 4: Run the WorldModel tests that touch data**

Run: `cd WorldModel && conda run --no-capture-output -n alayaworld python -m pytest tests/test_custom_i2v_schedule.py tests/test_standard_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit (WorldModel `main`)**

```bash
cd WorldModel
git add alaya/data/custom_i2v.py tests/test_custom_i2v_schedule.py
git commit -m "feat(custom_i2v): a caption may be a per-round prompt list (wbench_prompt_schedule)"
```

Push happens at the end of the plan with the other repos.

---

### Task 3: `GpuJob` base, worker fan-out, publishing, tool registration

**Files:**
- Create: `kernel/ar_kernel/tools/gpu_jobs.py`, `tests/fixtures/fake_gen_worker.py`, `tests/test_gpu_jobs.py`
- Modify: `kernel/ar_kernel/tools/captioner.py` (extract `wait_gpu_release`), `configs/kernel.yaml` (`generators:`, `annotate:`)

**Interfaces:**
- Consumes: `captioner.stage_clip/clip_host_path/container_path/gpu_memory_mib`, `hf_tools._move_into`, `jobs.run_cancellable`, `ToolError`.
- Produces:
  - `wait_gpu_release(gpu_memory, gpus, before, timeout_s) -> tuple[dict|None, bool|None]` (in `captioner.py`).
  - `split_gpus(gpus: list[int], per_worker: int, workers: int | None) -> list[list[int]]`.
  - `run_workers(env, argv_for, groups, *, cwd, cancel, work, out, total, report, recorder, node, phase, extra_env=None) -> list[int | str]`.
  - `canonical_hash(obj) -> str`, `file_sha256(path) -> str`.
  - `class GpuJob`: a `JobQueue` backend. The subclass sets `name`/`tool`/`kind` (`"rollout"`/`"annotation"`) and implements `check_args(args)`, `produce(job, items, work, out, cancel, report)` and `finish(job, item, out) -> dict`.
  - The job result: `{"items": [ {"index", ...published} | {"index", "error"} ], "gpu_memory_mib": {...}, "gpu_memory_released": bool|None}`.
  - `register_gpu_tools(mcp, kit, q)`: registers `annotate_camera` / `rollout_alayaworld` / `rollout_wan22` / `rollout_ltx25` only for backends registered on `q`.
  - `build_gpu_backends(cfg, run_dir, gpus, registry, recorder) -> list`: the enabled backends, filled in by Tasks 4–7.

- [ ] **Step 1: Config shape**

Replace the `generators:` block in `configs/kernel.yaml` with the block below. Everything stays disabled; Tasks 4–7 enable what their smokes verify.

```yaml
annotate:                        # annotate_camera GPU job (spec 10, 16.3 item 6)
  enabled: false                 # Task 4 enables it once ViGeo passes on the example clips
  env: alayaworld
  repo: third_party/ViGeo        # relative to paths.worldmodel
  checkpoint: third_party/ViGeo/checkpoints/ViGeo1.1
  max_frames: 1200               # longer clips are refused per clip (memory)
  max_items: 64
  timeout_s: 21600
generators:                      # rollout_* GPU jobs; a variant is enabled only after its smoke (spec 16.3 item 5)
  alayaworld:
    env: alayaworld
    variants: {dmd4: {enabled: false, config: configs/infer_i2v_camera.yaml},
               ar30: {enabled: false, config: configs/infer_i2v_camera_ar.yaml}}
    max_rounds: 15
    max_items: 16
    timeout_s: 43200
    license: LTX-2 Community License (AlayaWorld is an LTX-2 derivative)
  wan22:
    env: gen-wan22
    repo: third_party/Wan2.2     # relative to the AutoResearcher repo
    weights: weights/wan2.2-ti2v-5b
    variants: {ti2v-5b: {enabled: false}}
    gpus_per_worker: 1
    workers: null                # null = as many as the GPU list allows
    frames: [121, 121]           # [default, max]; 4k+1
    extra_args: {}               # set by Task 6 from the fit measurements
    max_items: 16
    timeout_s: 43200
    license: Apache-2.0
  ltx25:
    env: gen-ltx25
    repo: third_party/LTX-2
    weights: weights/ltx-2.5
    variants: {distilled: {enabled: false}, dev: {enabled: false}}
    gpus_per_worker: 1
    workers: null                # Task 7 sets it from host-RAM measurements
    frames: [121, 241]           # [default, max]; 8k+1
    resolutions: [[576, 1024]]   # [height, width], each divisible by 32, 16:9
    max_items: 16
    timeout_s: 43200
    license: LTX-2 Community License
```

- [ ] **Step 2: Write the fake worker**

`tests/fixtures/fake_gen_worker.py` follows the bridge protocol. For each of its items it copies `item["src"]` to `<out>/<index>.mp4` and writes `{"ok": true, "rank": r, "gpus": CUDA_VISIBLE_DEVICES}`.

Special behaviors, selected by the item:
- `"fail": true` writes `{"ok": false, "error": "boom"}`.
- `"crash": true` exits the process with code 3 before writing anything.
- `"sleep": s` sleeps first, which the cancel test uses.

It also writes its pid to `<out>/worker<rank>.pid`.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_gpu_jobs.py
"""GpuJob plumbing with a fake worker (no GPU)."""
import json, os, threading, time
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.gpu_jobs import GpuJob, canonical_hash, split_gpus
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.tools.server import ToolError
from tests.conftest import make_mp4

FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


class FakeJob(GpuJob):
    name = tool = "rollout_fake"
    kind, generator, license = "rollout", "fake-gen", "test-license"
    file_keys = ("src",)

    def check_args(self, args):
        if not args.get("items"):
            raise ToolError("items is empty")

    def produce(self, job, items, work, out, cancel, report):
        groups = split_gpus(self.gpus, 1, None)
        return self.run_workers("autoresearcher", lambda r, w: ["python", str(FAKE), "--items",
                                str(work / "items.json"), "--out", str(out), "--rank", str(r), "--world", str(w)],
                                groups, job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        return {"video": out / f"{item['index']}.mp4",
                "caption": self.write_caption(out, item, {"caption": item.get("prompt", "x")})}


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    make_mp4(ws / "a.mp4", seconds=2.5, fps=24, width=736, height=414)
    q.register(FakeJob(KernelConfig.load(), tmp_path / "run", [0, 1, 4, 5], reg, rec,
                       gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, rec, ws, staging, tmp_path / "run"
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_fake"].submit(q, caller, {"items": items, **params})


def test_split_gpus():
    assert split_gpus([0, 1, 4, 5], 1, None) == [[0], [1], [4], [5]]
    assert split_gpus([0, 1, 4, 5], 2, None) == [[0, 1], [4, 5]]
    assert split_gpus([0, 1, 4, 5], 1, 2) == [[0], [1]]
    assert split_gpus([0, 1, 2], 2, None) == [[0, 1]]


def test_items_fan_out_one_worker_per_gpu_and_publish_candidates(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4", "prompt": f"p{i}", "seed": i} for i in range(6)]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    got = out["result"]["items"]
    assert [g["index"] for g in got] == list(range(6))
    c = got[0]["candidate"]
    job = out["id"]
    assert c["video"] == f"/workspace/staging/rollouts/{job}/0.mp4"
    assert (staging / "rollouts" / job / "0.mp4").is_file()
    assert json.loads((staging / "rollouts" / job / "0.json").read_text()) == {"caption": "p0"}
    assert c["provenance"] == {"kind": "rollout", "generator": "fake-gen", "job_id": job,
                               "inputs_hash": c["provenance"]["inputs_hash"], "seed": 0}
    assert c["license"] == "test-license" and "pose" not in c
    assert got[0]["worker"]["gpus"] == "0" and got[1]["worker"]["gpus"] == "1" and got[4]["worker"]["gpus"] == "0"
    assert not (run / "jobs" / job / "in").exists() and not (run / "jobs" / job / "out").exists()
    assert out["result"]["gpu_memory_released"] is True


def test_inputs_hash_depends_on_file_content_not_path(env):
    q, caller, rec, ws, staging, run = env
    (ws / "b.mp4").write_bytes((ws / "a.mp4").read_bytes())
    out = q.wait(caller, submit(q, caller, [{"src": "a.mp4", "seed": 1}, {"src": "b.mp4", "seed": 1},
                                           {"src": "a.mp4", "seed": 2}])["job_id"], 120)
    h = [i["candidate"]["provenance"]["inputs_hash"] for i in out["result"]["items"]]
    assert h[0] == h[1] != h[2]


def test_per_item_failure_and_worker_crash_are_item_errors(env):
    q, caller, rec, ws, staging, run = env
    items = [{"src": "a.mp4"}, {"src": "a.mp4", "fail": True}, {"src": "a.mp4", "crash": True}, {"src": "a.mp4"}]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done"
    by = {i["index"]: i for i in out["result"]["items"]}
    assert "candidate" in by[0] and "candidate" in by[3]
    assert by[1]["error"] == "boom"
    assert "exit code 3" in by[2]["error"] and "log tail" in by[2]["error"]


def test_missing_or_escaping_input_is_refused_at_submit(env):
    q, caller, rec, ws, staging, run = env
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "nope.mp4"}])
    with pytest.raises(ToolError):
        submit(q, caller, [{"src": "/etc/passwd"}])
    with pytest.raises(ToolError):
        submit(q, caller, [])


def test_a_planted_link_in_staging_fails_the_item_not_the_host(env, tmp_path):
    q, caller, rec, ws, staging, run = env
    outside = tmp_path / "outside"; outside.mkdir()
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 2}])["job_id"]
    (staging / "rollouts").mkdir()
    (staging / "rollouts" / job_id).symlink_to(outside)        # swapped in while the job runs
    out = q.wait(caller, job_id, 120)
    assert "error" in out["result"]["items"][0]
    assert list(outside.iterdir()) == []


def test_cancel_kills_every_worker(env):
    q, caller, rec, ws, staging, run = env
    job_id = submit(q, caller, [{"src": "a.mp4", "sleep": 300} for _ in range(4)])["job_id"]
    out_dir = run / "jobs" / job_id / "out"
    deadline = time.monotonic() + 30
    while len(list(out_dir.glob("worker*.pid"))) < 4 and time.monotonic() < deadline:
        time.sleep(0.2)
    pids = [int(p.read_text()) for p in out_dir.glob("worker*.pid")]
    assert len(pids) == 4
    q.cancel(caller, job_id)
    assert q.wait(caller, job_id, 120)["state"] == "cancelled"
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
```

- [ ] **Step 4: Run them to verify they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_gpu_jobs.py -q`
Expected: FAIL (`ModuleNotFoundError: ar_kernel.tools.gpu_jobs`).

- [ ] **Step 5: Implement `gpu_jobs.py`**

```python
"""Shared plumbing for the GPU data-source jobs (spec 10): rollout_* and annotate_camera.

A backend stages the agent's input files into a kernel-private job dir (never trusting the
path between check and use), runs one worker per GPU group in the generator's own conda env
(bridge protocol: see the Plan 3 file structure), and publishes each finished item into the
caller's staging dir with O_NOFOLLOW moves, so a link the agent plants cannot redirect a
kernel write. A per-item failure is an item error, not a failed job.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .captioner import clip_host_path, container_path, gpu_memory_mib, stage_clip, wait_gpu_release
from .context import STAGING, PathError
from .hf_tools import _move_into
from .jobs import run_cancellable
from .server import ToolError


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_gpus(gpus: list[int], per_worker: int, workers: int | None) -> list[list[int]]:
    groups = [gpus[i:i + per_worker] for i in range(0, len(gpus) - per_worker + 1, per_worker)]
    return groups[:workers] if workers else groups


def _tail(path: Path, limit: int = 2000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-limit:] if path.exists() else ""


class GpuJob:
    """Base JobQueue backend. Subclasses set name/tool, kind ("rollout" | "annotation"),
    generator (provenance name), license, file_keys (item fields naming workspace files) and
    implement check_args, produce and finish."""
    name = tool = kind = generator = license = ""
    file_keys: tuple[str, ...] = ()
    max_items = 16

    def __init__(self, cfg, run_dir: Path, gpus: list[int], registry, recorder,
                 gpu_memory=gpu_memory_mib) -> None:
        self.cfg, self.run_dir, self.gpus = cfg, Path(run_dir), list(gpus)
        self.registry, self.recorder, self.gpu_memory = registry, recorder, gpu_memory

    # ---- submit (tool call; fast; ToolError goes back to the agent) ----
    def submit(self, q, caller, args: dict) -> dict:
        items = args.get("items") or []
        if not items:
            raise ToolError("items is empty")
        if len(items) > self.max_items:
            raise ToolError(f"at most {self.max_items} items per job")
        self.check_args(args)
        for n, item in enumerate(items):
            for key in self.file_keys:
                if isinstance(item.get(key), str):
                    try:
                        clip_host_path(caller, item[key])
                    except PathError as exc:
                        raise ToolError(f"item {n}: {key}: {exc}") from exc
                    item = {**item, key: container_path(item[key])}
            items[n] = item
        return {"job_id": q.submit(caller, self.name, {**args, "items": items})}

    def check_args(self, args: dict) -> None:
        raise NotImplementedError

    # ---- run (worker thread, under the GPU lock) ----
    def run(self, job, cancel: threading.Event, report) -> dict:
        caller = self.registry.lookup(job.token)
        if caller is None:
            raise RuntimeError("the phase that submitted this job has ended")
        work = self.run_dir / "jobs" / job.id
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True)
        out.mkdir()
        before = self.gpu_memory(self.gpus)
        results: dict[int, dict] = {}
        try:
            staged = []
            for index, item in enumerate(job.args["items"]):
                try:
                    staged.append(self._stage(caller, index, item, inp))
                except (PathError, OSError) as exc:
                    results[index] = {"index": index, "error": f"input: {exc}"}
            (work / "items.json").write_text(json.dumps(staged), encoding="utf-8")
            if staged and not cancel.is_set():
                self.produce(job, staged, work, out, cancel, report)
            for item in staged:
                results[item["index"]] = self._collect(caller, job, item, out)
        finally:
            shutil.rmtree(inp, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
        timeout = 5.0 if cancel.is_set() else float(self.cfg.get("captioner.memory_release_timeout_s", 120))
        after, released = wait_gpu_release(self.gpu_memory, self.gpus, before, timeout)
        if released is False:
            self.recorder.event(f"{self.tool}.gpu_not_released", node=job.node, component="tools",
                                job_id=job.id, payload={"before": before, "after": after})
        return {"items": [results[i] for i in sorted(results)],
                "gpu_memory_mib": {"before": before, "after": after}, "gpu_memory_released": released}

    def _stage(self, caller, index: int, item: dict, inp: Path) -> dict:
        staged = {**item, "index": index, "hashes": {}}
        for key in self.file_keys:
            if isinstance(item.get(key), str):
                dst = inp / f"{index}_{key}{Path(item[key]).suffix}"
                stage_clip(caller, item[key], dst)
                staged[key] = str(dst)
                staged["hashes"][key] = file_sha256(dst)
        return staged

    def produce(self, job, items: list[dict], work: Path, out: Path, cancel, report) -> None:
        raise NotImplementedError

    def finish(self, job, item: dict, out: Path) -> dict:
        """Post-process one finished item; return {"video": Path, "caption": Path, "pose"?: Path,
        plus any extra JSON-able fields}. Raise to report an item error."""
        raise NotImplementedError

    def _collect(self, caller, job, item: dict, out: Path) -> dict:
        index = item["index"]
        status_path = out / f"{index}.json"
        if not status_path.exists():
            return {"index": index, "error": self.missing.get(index, "no output (cancelled or not reached)")}
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("ok"):
            return {"index": index, "error": status.get("error", "failed")}
        try:
            files = self.finish(job, item, out)
            published = {}
            for role in ("video", "caption", "pose"):
                if files.get(role):
                    src = Path(files[role])
                    rel = f"{self.kind}s/{job.id}/{index}{src.suffix}"
                    _move_into(src, caller.staging_host, rel)
                    published[role] = str(STAGING / rel)
        except (OSError, RuntimeError, ValueError, PathError) as exc:
            return {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        extra = {k: v for k, v in files.items() if k not in ("video", "caption", "pose")}
        worker = {k: v for k, v in status.items() if k != "ok"}
        if self.kind == "annotation":
            return {"index": index, **published, **extra, "worker": worker}
        params = {k: v for k, v in job.args.items() if k != "items"}
        spec = {k: v for k, v in item.items() if k not in (*self.file_keys, "index", "hashes")}
        inputs_hash = canonical_hash({"generator": self.generator_name(job), "params": params,
                                      "item": spec, "files": item["hashes"]})
        candidate = {**published, "provenance": {"kind": "rollout", "generator": self.generator_name(job),
                     "job_id": job.id, "inputs_hash": inputs_hash, "seed": item.get("seed")},
                     "license": self.license}
        return {"index": index, "candidate": {**candidate, **extra}, "worker": worker}

    def generator_name(self, job) -> str:
        return self.generator

    def write_caption(self, out: Path, item: dict, caption: dict) -> Path:
        path = out / f"{item['index']}.caption.json"
        path.write_text(json.dumps(caption, ensure_ascii=False), encoding="utf-8")
        return path

    # ---- workers ----
    missing: dict[int, str] = {}

    def run_workers(self, env: str, argv_for: Callable[[int, int], list[str]], groups: list[list[int]], *,
                    job, work: Path, out: Path, total: int, cancel, report, cwd: Path | None = None,
                    extra_env: dict | None = None) -> list[int | str]:
        """One worker per GPU group, in parallel; each handles items with index % world == rank.
        A cancel kills every worker's process group. Items of a worker that died without writing
        their status become item errors carrying the worker's exit code and log tail."""
        codes: list[int | str | None] = [None] * len(groups)
        self.missing = {}

        def one(rank: int, group: list[int]) -> None:
            try:
                codes[rank] = run_cancellable(
                    env, argv_for(rank, len(groups)), cwd=cwd or work, cancel=cancel,
                    extra_env={**(extra_env or {}), "CUDA_VISIBLE_DEVICES": ",".join(map(str, group)),
                               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                    log_path=work / f"worker{rank}.log", recorder=self.recorder, node=job.node, phase=self.tool)
            except Exception as exc:                 # noqa: BLE001 -- reported per item below
                codes[rank] = f"launch failed: {type(exc).__name__}: {exc}"

        threads = [threading.Thread(target=one, args=(r, g), name=f"ar-{self.tool}-w{r}", daemon=True)
                   for r, g in enumerate(groups)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            report({"done": len([p for p in out.glob("*.json") if p.stem.isdigit()]), "total": total})
            time.sleep(2.0)
        for t in threads:
            t.join()
        report({"done": len([p for p in out.glob("*.json") if p.stem.isdigit()]), "total": total})
        world = len(groups)
        for index in range(total):
            code = codes[index % world] if world else "no GPU group"
            if code not in (0, None) and not (out / f"{index}.json").exists():
                self.missing[index] = (f"worker {index % world} failed (exit code {code}); log tail:\n"
                                       f"{_tail(work / f'worker{index % world}.log')}")
        return codes
```

Two corrections to the sketch, both bound by the tests:
- **Map missing items by their real indices.** Workers shard by each item's original `index % world`. Items that failed staging are absent from `items.json`, so `run_workers` must take the staged items' indices (derive them inside `run_workers` from `work / "items.json"`; keep `total` for the progress count), and loop over those, not over `range(total)`.
- **Make `missing` per-job state.** Have `run_workers` return `(codes, missing)`, and have `run` pass `missing` into `_collect`. No class attribute.

The crash case's error must contain `exit code 3` and `log tail`.

In `captioner.py`, move the body of `CaptionBackend._released` into a module function `wait_gpu_release(gpu_memory, gpus, before, timeout_s)` and have `_released` call it. The captioner's tests must still pass.

Registration (in the same file):

```python
def register_gpu_tools(mcp, kit, q) -> None:
    """Registers each GPU data tool whose backend is on the queue. Disabled tools/variants are
    simply absent (spec 10: disabled variants are omitted from tool schemas)."""
    b = q.backends
    if "annotate_camera" in b:
        @mcp.tool(name="annotate_camera", description=b["annotate_camera"].description)
        async def annotate_camera(paths: list[str], ctx: Context) -> dict[str, Any]:
            return await kit.call(ctx, "annotate_camera", {"paths": paths},
                                  lambda c: b["annotate_camera"].submit(q, c, {"items": [{"video": p} for p in paths]}))
    # rollout_alayaworld / rollout_wan22 / rollout_ltx25: same pattern; each tool's parameters are
    # its job-level params plus `items: list[dict]`, and its description lists the enabled variants.
```

Write out all four registrations explicitly, with typed parameters, so the MCP schema is precise:
- `rollout_alayaworld(items, variant, rounds, seed)`
- `rollout_wan22(items, frames)`
- `rollout_ltx25(items, variant, frames, height, width)`

Each backend exposes a `description` string that states:
- the item fields;
- that it returns `{job_id}` at once, with the result collected via `job_wait`;
- where outputs land and that each result item carries a ready `candidate` for `data_ingest`;
- that rollouts without poses need `annotate_camera` (or `camera_motion: static`) before ingest.

`build_gpu_backends(cfg, run_dir, gpus, registry, recorder)` returns `[]` for now. Tasks 4–7 each add their backend when its config says enabled.

Add a registration test: a queue holding only `FakeJob` exposes no `rollout_wan22`. Use `new_mcp()` plus `await mcp.list_tools()`, the way `tests/test_tool_server.py` lists tools.

- [ ] **Step 6: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_gpu_jobs.py tests/test_captioner.py tests/test_jobs.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add kernel/ar_kernel/tools/gpu_jobs.py kernel/ar_kernel/tools/captioner.py configs/kernel.yaml tests/test_gpu_jobs.py tests/fixtures/fake_gen_worker.py
git commit -m "feat(tools): GpuJob base for GPU data sources: staging, per-GPU workers, safe publishing"
```

---

### Task 4: `annotate_camera` (ViGeo)

**Files:**
- Create: `kernel/ar_kernel/bridges/vigeo_poses.py`, `kernel/ar_kernel/tools/annotate.py`, `tests/test_annotate.py`
- Modify: `kernel/ar_kernel/tools/gpu_jobs.py` (`build_gpu_backends`), `configs/kernel.yaml` (`annotate.enabled`)

**Interfaces:**
- Consumes: `GpuJob`, `split_gpus`, `run_workers` (Task 3); `probe_video` (`ar_kernel.data.probe`).
- Produces:
  - `AnnotateBackend(GpuJob)`, name `annotate_camera`, `kind = "annotation"`, `file_keys = ("video",)`.
  - Result items: `{"index", "video": <the agent's original path>, "pose": "/workspace/staging/annotations/<job>/<i>.npz", "frames": N, "intrinsics": [fx, fy, cx, cy]}` or `{"index", "error"}`.
  - The npz holds `cam_c2w [N,4,4]` float32 (first frame = identity; OpenCV axes; translation in ViGeo's units, consistent within the clip) and `intrinsics [3,3]` in pixels of the mp4.

- [ ] **Step 1: Read ViGeo first (no code yet)**

Read these files:
- `WorldModel/third_party/ViGeo/README.md` (Quick Start, inference modes);
- `vigeo/vigeo.py` (`infer`, the input size rules, what `focal` is normalized to);
- `vigeo/utils.py` (`recover_focal_from_xy`);
- `utils/data.py` (`load_image_sequence`, `load_intrinsic`);
- WorldModel's own ViGeo call site: `grep -rn "ViGeo.from_pretrained\|vigeo" WorldModel/alaya --include=*.py | head`.

Write down in the task report:
- the input resolution rule, e.g. a patch-size multiple;
- how the normalized focal maps to pixels for a W×H frame;
- whether `pose_pred` is camera-to-world;
- how chunk mode with `kv_caches` is called for long clips.

Base the bridge on those facts, not on guesses.

- [ ] **Step 2: Write the bridge `kernel/ar_kernel/bridges/vigeo_poses.py`**

- **Arguments:** the bridge protocol plus `--repo <ViGeo dir> --checkpoint <dir> --max-frames N`.
- **Imports:** it imports ViGeo from `--repo` (`sys.path.insert(0, repo)`), so nothing is installed.
- **Per item:**
  1. Decode every frame of `item["video"]` with PyAV or OpenCV (whatever `alayaworld` has; check `conda run -n alayaworld python -c "import av, cv2"`).
  2. Refuse more than `max_frames` frames with `{"ok": false, "error": "N frames > max_frames"}`.
  3. Resize each frame to the model input size.
  4. Run chunk mode (16-frame chunks with the KV cache), or offline mode when the clip is at most the chunk limit you found in Step 1.
  5. Convert the poses to 4×4, re-express them relative to frame 0 (`c2w_0^{-1} @ c2w_t`, so frame 0 is the identity), and cast to float32.
  6. Build pixel intrinsics from the median focal over frames: `fx = f_norm_x * W`, `fy = f_norm_y * H`, principal point at `(W/2, H/2)`. Use the exact normalization found in Step 1.
  7. Write `<out>/<index>.npz` with `cam_c2w` and `intrinsics`, then the status `{"ok": true, "frames": N, "intrinsics": [fx, fy, cx, cy], "seconds": t}`.
- **Memory:** use `torch.inference_mode()` and bf16 autocast if ViGeo's README does, and free the GPU between clips.

- [ ] **Step 3: Write `tools/annotate.py`**

```python
"""annotate_camera (spec 10, 16.3 item 6): per-frame camera poses for agent clips, via ViGeo."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.probe import probe_video
from .gpu_jobs import GpuJob, split_gpus
from .server import ToolError

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "vigeo_poses.py"


class AnnotateBackend(GpuJob):
    name = tool = "annotate_camera"
    kind = "annotation"
    file_keys = ("video",)
    description = ("Estimate per-frame camera poses for video clips (ViGeo). A GPU job: returns {job_id} at "
                   "once; collect with job_wait. `paths`: mp4 files under /workspace. Each result item gives "
                   "`pose`: an npz in /workspace/staging/annotations/<job_id>/ holding cam_c2w [N,4,4] "
                   "(N = the clip's frame count, OpenCV camera-to-world, first frame identity) and pixel "
                   "`intrinsics`; pass it as `pose` to data_ingest with camera_motion 'moving'. Batch many clips "
                   "per call.")

    def __init__(self, cfg, *a, **kw):
        super().__init__(cfg, *a, **kw)
        self.max_items = int(cfg.get("annotate.max_items"))

    def check_args(self, args):
        for item in args["items"]:
            if not str(item.get("video", "")).lower().endswith(".mp4"):
                raise ToolError(f"{item.get('video')!r} is not an .mp4")

    def produce(self, job, items, work, out, cancel, report):
        a = self.cfg.get("annotate")
        wm = self.cfg.worldmodel
        self.run_workers(a["env"], lambda r, w: [
            "python", str(BRIDGE), "--items", str(work / "items.json"), "--out", str(out), "--rank", str(r),
            "--world", str(w), "--repo", str(wm / a["repo"]), "--checkpoint", str(wm / a["checkpoint"]),
            "--max-frames", str(a["max_frames"])], split_gpus(self.gpus, 1, None),
            job=job, work=work, out=out, total=len(items), cancel=cancel, report=report)

    def finish(self, job, item, out):
        pose = out / f"{item['index']}.npz"
        frames = probe_video(Path(item["video"])).frames
        with np.load(pose) as z:
            n = len(z["cam_c2w"])
        if n != frames:
            raise ValueError(f"pose has {n} frames, the video {frames}")
        status = json.loads((out / f"{item['index']}.json").read_text())
        return {"pose": pose, "frames": frames, "intrinsics": status["intrinsics"]}
```

`finish` returns only the pose as a file role, so the video is never re-published. In `GpuJob._collect` (Task 3), the annotation branch becomes `return {"index": index, "video": job.args["items"][index]["video"], **published, **extra, "worker": worker}`, so each result item names the agent's original clip. Make that one-line change here, and keep Task 3's tests green.

In `build_gpu_backends`, add `AnnotateBackend(...)` when `cfg.get("annotate.enabled")`.

- [ ] **Step 4: Unit tests (fake worker)**

In `tests/test_annotate.py`, subclass `AnnotateBackend` so that `produce` runs a fake that writes an identity `cam_c2w` of the probed frame count plus intrinsics. Assert:
- the pose is published under `staging/annotations/<job>/0.npz` and the result item's `video` is the agent's path;
- a pose with the wrong frame count becomes an item error (Review Focus 4);
- a clip over `max_frames` is an item error from the worker (the fake honors `--max-frames`), and the other clips still succeed (Review Focus 5);
- a non-mp4 path is refused at submit.

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_annotate.py -q` → PASS.

- [ ] **Step 5: GPU verification against the example poses (spec §16.3 item 6)**

Add a `gpu` test to `tests/test_annotate.py`. It runs the real backend on `AR_TEST_GPUS` over every clip in `WorldModel/data/examples/video_caption_camera/videos/`, then for each clip:
- **(a)** The published npz passes the kernel ingest checker as `video_caption_camera`. Use `Ingestor` exactly as `tests/test_ingest.py` does; the example caption is the caption.
- **(b)** Compared with the provided `poses/<id>.npz` after expressing both relative to frame 0, the median relative-rotation error between consecutive frames is under 1°. Relative rotation is `acos((trace(R_i^T R_{i+1}) - 1) / 2)`, compared between ViGeo and the ground truth.
- **(c)** After Sim(3) alignment (Umeyama) of the camera centers, the translation ATE is under 10 % of the ground-truth path length.
- **(d)** fx and fy are within 15 % of the provided intrinsics when the example has them.

Print every number. Also record the wall time per clip and the peak GPU memory from the job result.

Run: `AR_TEST_GPUS=0,1,2,3 conda run --no-capture-output -n autoresearcher python -m pytest tests/test_annotate.py -m gpu -s -q`

If (b)–(d) fail, do **not** loosen the thresholds silently, and do not start on Depth-Anything-3. Stop and report the numbers. The controller decides with the user (spec §16.3 item 6 fallback).

- [ ] **Step 6: Enable, record, commit**

- Set `annotate.enabled: true`.
- In `verification-log.md`, add a "Plan 3 — annotate_camera (ViGeo)" section: the GPUs used, per-clip errors against the ground truth, seconds per clip, and peak memory.
- Commit:

```bash
git add kernel/ar_kernel/bridges/vigeo_poses.py kernel/ar_kernel/tools/annotate.py kernel/ar_kernel/tools/gpu_jobs.py configs/kernel.yaml tests/test_annotate.py docs/superpowers/plans/verification-log.md
git commit -m "feat(tools): annotate_camera via ViGeo, verified against the example poses"
```

---

### Task 5: `rollout_alayaworld` (dmd4 / ar30)

**Files:**
- Create: `kernel/ar_kernel/tools/rollouts.py` (`AlayaWorldBackend`), `tests/test_rollouts.py`
- Modify: `gpu_jobs.py` (`build_gpu_backends`), `configs/kernel.yaml`

**Interfaces:**
- Consumes: Task 2 (list captions), Task 3 (`GpuJob`), Task 4 (`AnnotateBackend`, only in the GPU alignment check).
- Produces: `AlayaWorldBackend(GpuJob)`, tool `rollout_alayaworld`.
  - **Job params:** `variant` (`dmd4` | `ar30`, enabled ones only), `rounds` (1..`max_rounds`), `seed` (int).
  - **Item fields:**
    - `first_frame`: an image path.
    - `camera`: either an npz path (`cam_c2w [N,4,4]`, optional `intrinsics [3,3]` in pixels of `first_frame`) or `{"forward": m_per_frame, "yaw": deg_per_frame, "pitch": deg_per_frame}`.
    - `prompts`: a list of strings, one per round (the last repeats).
    - `caption`: optional; defaults to `prompts[0]`.
  - **Candidate:** 24 fps mp4, `pose` npz (`cam_c2w` for every published frame, plus `intrinsics` if given), caption JSON `{"caption", "segments"}`, `camera_motion: "moving"`, provenance generator `alayaworld-<variant>`.
  - `seed` in provenance is the job seed. Record WorldModel's per-sample seed rule in the report.

- [ ] **Step 1: Kernel input builder + config writer (unit-tested, no GPU)**

`AlayaWorldBackend.produce` writes into `work/`:
- `inputs/images/<index:04d>.<ext>`, from the staged `first_frame`;
- `inputs/poses/<index>.npz`: the caller's trajectory, or one synthesized for `rounds*32*3` frames (at least 512). Copy the 15-line `synth_trajectory` from `WorldModel/scripts/infer/prepare_i2v_inputs.py` into `rollouts.py`, with a comment naming the source.
- `inputs/pose.jsonl`, in image order; `intrinsic` is normalized `[fx/W, fy/H, cx/W, cy/H]` when the caller gave pixel intrinsics;
- `inputs/captions.json`: `{"<index:04d>": prompts}`;
- `config.yaml`: the variant's WorldModel config with:
  - `paths.*` made absolute against `cfg.worldmodel`;
  - `run.output_dir = work/out_wm` and `run.log_dir = work/logs`;
  - `run.seed = seed`;
  - `validation.max_samples = len(items)`;
  - `validation.save_joystick = false`;
  - `validation.modes.custom_i2v.rollout_rounds = rounds`;
  - `…custom_i2v.wbench_chunks_per_turn = 1`;
  - `…custom_i2v.dataset.{image_dir,captions_json,pose_jsonl}` = the files above.

It then launches, via `run_workers` with a single group of **all** node GPUs (one torchrun job), `run_cancellable("alayaworld", ["bash", "scripts/finetune/train.sh"], cwd=cfg.worldmodel, extra_env={"CONFIG_PATH": <cfg>, "VALIDATE_ONLY": "1", "ALAYA_USE_FA3": "0", "LOG_FILTER": "all", "MASTER_PORT": <free port>})`.

After it exits, a kernel step maps each `*sample-<k>*_pred_clean.mp4` to item k and writes `out/<index>.mp4` plus `out/<index>.json` (`{"ok": true}`, or an error when the file is missing). This follows the bridge protocol, so Task 3's `_collect` applies unchanged.

Unit tests in `tests/test_rollouts.py`:
- The config writer produces the listed fields (load the YAML and assert).
- `captions.json` has lists.
- `pose.jsonl` is in image order with normalized intrinsics.
- The synthesized trajectory for `{"forward": 0.01, "yaw": 0, "pitch": 0}` moves along +z.
- A wrong `variant` or `rounds` over the max is refused at submit.

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_rollouts.py -q` → PASS.

- [ ] **Step 2: GPU spike to measure the output layout (dmd4, 1 item, 2 rounds)**

Render one item with a real image and a synthesized forward trajectory. Take the image from `WorldModel/data/examples/video_caption_camera`: extract frame 0 with ffmpeg to the scratchpad.

Record:
- the output mp4's frame count F and fps;
- which trajectory index output frame 0 corresponds to;
- where round r's frames start.

Method for the pose mapping: run `annotate_camera` (Task 4) on the output, then find the offset `o` that minimizes the relative-rotation and Sim(3) error between the ViGeo poses and `traj[o : o + F]`. The expectation is `o = 0`, with output frame 0 being the input image; prove it or correct it.

Also confirm the two rounds used the two prompts: the `*_info.txt` or log `Prompt Schedule: chunk0/chunk1` lines should show different prompts.

Write the findings into `verification-log.md` before coding Step 3.

- [ ] **Step 3: `finish`: trim to round boundaries, poses, segments**

Let the measured layout be "round r occupies output frames `[s + 32r, s + 32(r+1))`" (s measured in Step 2). Then:
- **Trim.** `trim = (s - 25) % 32` leading frames are removed, so every round boundary lands at `25 + 32k` (spec §6.2 `per_chunk`). Do the trim with ffmpeg: `-vf select='gte(n\,trim)',setpts=N/24/TB -r 24 -c:v libx264 -crf 18 -pix_fmt yuv420p`.
- **Poses.** `cam_c2w = traj[o + trim : o + trim + F']`, where F' is the frame count of the trimmed mp4 **probed after writing**. Assert `len == F'` (Review Focus 4).
- **Segments.** `[{"time_range_s": [start, end), "prompt": prompts[min(r, len-1)]}]`, one per round, in seconds of the trimmed clip. The first segment starts at 0 and the last ends at the duration. Merge adjacent rounds with identical prompts.
- **Caption.** `caption = item.get("caption") or prompts[0]`.

Unit-test `finish` with a synthetic mp4 (`make_mp4`) of the measured layout and a fake status. Assert:
- the trimmed frame count;
- the pose length equals the probed frames;
- every internal boundary satisfies `abs(t*24 - (25 + 32k)) <= 0.5` for some k;
- the candidate passes the kernel checker as `video_timed_prompts_camera:per_chunk`. Run the real `Ingestor` as `tests/test_ingest.py` does; it uses the `alayaworld` env but no GPU.

- [ ] **Step 4: GPU smoke, both variants**

Add a `gpu` test: for each of dmd4 and ar30, render 2 items with different per-round prompts and 3 rounds. Assert:
- the job is `done`;
- both candidates pass `Ingestor` as `video_timed_prompts_camera:per_chunk`;
- ViGeo poses on the output agree with the published `cam_c2w` under the Task 4 thresholds;
- GPU memory is released.

Record the wall time (load plus per item) and the peak memory per variant.

If a variant does not fit or fails, leave it disabled and report why.

- [ ] **Step 5: Enable, record, commit**

- Set `generators.alayaworld.variants.<v>.enabled: true` for each variant that passed.
- Add `AlayaWorldBackend` to `build_gpu_backends` (it is registered if any variant is enabled).
- Write the verification-log section (layout, offsets, timings).
- Commit:

```bash
git add kernel/ar_kernel/tools/rollouts.py kernel/ar_kernel/tools/gpu_jobs.py configs/kernel.yaml tests/test_rollouts.py docs/superpowers/plans/verification-log.md
git commit -m "feat(tools): rollout_alayaworld through custom_i2v with per-round prompts, poses and per_chunk segments"
```

---

### Task 6: `rollout_wan22` (TI2V-5B)

**Files:**
- Create: `kernel/ar_kernel/bridges/wan22_generate.py`; `Wan22Backend` in `tools/rollouts.py`; tests in `tests/test_rollouts.py`
- Modify: `gpu_jobs.py`, `configs/kernel.yaml`, `docs/PORTABILITY.md` (env setup commands)

**Interfaces:**
- Produces: tool `rollout_wan22`.
  - **Job params:** `frames` (4k+1, at most the configured max; defaults to the configured default).
  - **Items:** `prompt`, optional `image` (first frame, any size; Wan resizes it), `seed`.
  - **Candidate:** a 1248×704, 24 fps mp4 (center-cropped from 1280×704), caption JSON `{"caption": prompt}`, no pose, no `camera_motion`. The agent adds one: `annotate_camera` then `moving`, or `static`. Provenance generator `wan2.2-ti2v-5b`, license `Apache-2.0`.

- [ ] **Step 1: Environment and code**

```bash
cd AutoResearcher && mkdir -p third_party
git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2 && git -C third_party/Wan2.2 rev-parse HEAD
export PIP_CACHE_DIR=$PWD/.cache/pip
conda create -y -n gen-wan22 python=3.10
conda run --no-capture-output -n gen-wan22 pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n gen-wan22 pip install -r third_party/Wan2.2/requirements.txt   # flash_attn: use a prebuilt wheel matching torch 2.7.1/cu12/py310 if the build fails
conda run --no-capture-output -n gen-wan22 pip check
```

Record in `docs/PORTABILITY.md`:
- the exact commands;
- the Wan2.2 commit SHA;
- the env size (`du -sh ~/miniforge3/envs/gen-wan22`).

Pin that SHA in `configs/kernel.yaml` as `generators.wan22.commit`. The backend refuses to run when `git -C <repo> rev-parse HEAD` differs, so a silently updated clone cannot change outputs.

- [ ] **Step 2: Single-GPU fit spike (CLI)**

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n gen-wan22 python third_party/Wan2.2/generate.py \
  --task ti2v-5B --size 1280*704 --ckpt_dir weights/wan2.2-ti2v-5b --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "A slow walk through a sunlit forest path" --base_seed 1 --save_file <scratchpad>/wan_t2v.mp4
```

Run it with cwd `AutoResearcher`. Then run the same command with `--image <frame.png>`.

Measure:
- wall time;
- peak GPU memory (`nvidia-smi --query-gpu=memory.used -lms 500` in the background);
- peak host RSS.

Try without `--t5_cpu` or without `--offload_model` if memory allows, and choose the fastest setting that stays under 22 GB. Record everything in `verification-log.md`.

- [ ] **Step 3: Bridge `wan22_generate.py`**

- **Arguments:** the bridge protocol plus `--repo --ckpt-dir --frames --offload-model/--no-offload-model --t5-cpu/--no-t5-cpu`.
- **Imports:** `sys.path.insert(0, repo)`, then `import wan` and `from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS`.
- **Setup:** build `wan.WanTI2V(config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=…, device_id=0, rank=0, t5_cpu=…, convert_model_dtype=True)` **once**. Copy the exact constructor and `generate(...)` arguments from `generate.py`'s ti2v branch at the pinned commit.
- **Per item:** call `generate(prompt, img=PIL image or None, size=SIZE_CONFIGS["1280*704"], max_area=MAX_AREA_CONFIGS["1280*704"], frame_num=frames, shift=cfg.sample_shift, sample_solver="unipc", sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale, seed=item["seed"], offload_model=…)`. Save with Wan's `save_video` helper at 24 fps to `<out>/<index>.mp4`, then write the status `{"ok": true, "seconds": t}`.

- [ ] **Step 4: `Wan22Backend`**

- **`produce`:** `run_workers` on `split_gpus(gpus, gpus_per_worker, workers)`, with cwd = the AutoResearcher repo root.
- **`finish`:** center-crop to 1248×704 with ffmpeg (`-vf crop=1248:704:16:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -an`). Probe the result and assert fps == 24 and the aspect is within 2 % of 16:9. Write the caption JSON.
- **`check_args`:** `frames % 4 == 1`, 1 < frames ≤ max, and `prompt` non-empty in every item.

Unit tests with a fake bridge (reuse `fake_gen_worker.py` via a `produce` override) that emits a 1280×704 mp4 made with `make_mp4`. Assert the published video is 1248×704 at 24 fps, and the candidate has no `pose` and no `camera_motion`.

- [ ] **Step 5: GPU smoke**

Add a `gpu` test with 4 items (2 T2V, 2 I2V) on 4 GPUs, one worker each. Assert:
- all 4 items succeed;
- each candidate, with `camera_motion: static`, passes `Ingestor` as `video_caption_static`;
- one candidate passed through `annotate_camera` and then ingested as `moving` passes `video_caption_camera`.

Record the wall time per clip. If it does not fit, leave it disabled and report.

- [ ] **Step 6: Enable, record, commit**

Commit the config (`ti2v-5b.enabled: true`, `extra_args` from the spike, `commit` SHA), the bridge, the backend, the tests, `PORTABILITY.md` and the verification log.

`git commit -m "feat(tools): rollout_wan22 (Wan 2.2 TI2V-5B, official code, own env)"`

---

### Task 7: `rollout_ltx25` (distilled, then dev)

**Files:**
- Create: `kernel/ar_kernel/bridges/ltx25_generate.py`; `Ltx25Backend` in `tools/rollouts.py`; tests
- Modify: `gpu_jobs.py`, `configs/kernel.yaml`, `docs/PORTABILITY.md`

**Interfaces:**
- Produces: tool `rollout_ltx25`.
  - **Job params:** `variant` (`distilled` | `dev`, enabled ones only), `frames` (8k+1 ≤ max), `height`/`width` (one of `resolutions`).
  - **Items:** `prompt`, optional `image`, `seed`.
  - **Candidate:** 24 fps 16:9 mp4 with the audio stripped, caption JSON, no pose. Provenance generator `ltx-2.5-<variant>`, license `LTX-2 Community License`.

- [ ] **Step 1: Environment and code**

```bash
cd AutoResearcher
git clone https://github.com/Lightricks/LTX-2.git third_party/LTX-2 && git -C third_party/LTX-2 rev-parse HEAD
export PIP_CACHE_DIR=$PWD/.cache/pip UV_CACHE_DIR=$PWD/.cache/uv
conda create -y -n gen-ltx25 python=3.12
```

Install `ltx-core` (with the `natten` extra that pins torch 2.13.0 cu132) and `ltx-pipelines` editable into the env. Use the package indexes from `packages/ltx-core/pyproject.toml`. Prefer `uv pip install --python $(conda run -n gen-ltx25 which python) -e "third_party/LTX-2/packages/ltx-core[<natten extra name>]" -e third_party/LTX-2/packages/ltx-pipelines`, reading the extra's exact name and index URLs from that pyproject. `ltx-kernels` is **not** needed: it is only for multi-GPU SP, which does not help on 24 GB cards (fact 7).

Then run `pip check`, `python -c "import ltx_pipelines.distilled"`, and record the commands, the SHA and the env size in `PORTABILITY.md`. Pin the SHA as `generators.ltx25.commit`, the same way as Task 6.

- [ ] **Step 2: Single-GPU fit spike, distilled**

```bash
W=weights/ltx-2.5
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run --no-capture-output -n gen-ltx25 \
  python -m ltx_pipelines.distilled \
  --transformer-path $W/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $W/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $W/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $W/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $W/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization fp8-cast --offload cpu --num-frames 121 --width 1024 --height 576 \
  --prompt "A slow walk through a sunlit forest path" --seed 1 --output-path <scratchpad>/ltx_distilled.mp4
```

Check the real flag names with `--help` first (for example `--offload` versus `--offload-mode`, and whether `--width`/`--height` apply to the final or stage-1 size). Measure:
- wall time, split into load and generation;
- peak GPU memory;
- **peak host RSS** (`/usr/bin/time -v` or sampling `ps`).

Also try the `vae/…-conv-bf16` VAE (lighter) if the DiffVAE is the memory peak. Record the results.

Then decide **workers**: `workers = min(len(gpus), floor((host_RAM_GB - 60) / peak_RSS_GB))`, keeping 60 GB free for the kernel and the OS. Also make sure the job never starts when `MemAvailable` is below `workers × peak_RSS`. `produce` checks `/proc/meminfo` and fails the job with a clear error instead of letting the OOM killer act (verification-log finding on host OOM).

- [ ] **Step 3: dev variant**

The dev variant is `ltx_pipelines.ti2vid_two_stages` with the dev transformer, `--distilled-lora ltx-2.5-22b-distilled-lora-450-bf16.safetensors 1.0` (read `--help` for the exact form) and the same flags otherwise. Measure it the same way. If it is more than 4× slower than distilled per clip, or does not fit, leave `dev` disabled and say so.

- [ ] **Step 4: Bridge `ltx25_generate.py`**

- **Arguments:** the bridge protocol plus `--weights --variant --frames --height --width --quantization fp8-cast --offload cpu`.
- **Setup:** build the pipeline **once** through the Python API (`ModelPaths.from_split(...)`, then `DistilledPipeline(...)` or the two-stage class). Copy the construction exactly from the module's `main()` at the pinned commit, including how the quantization policy and offload mode are passed.
- **Per item:** call the pipeline with `prompt`, `seed`, `num_frames`, the size and `images=[(item["image"], 0, 1.0)]` when an image is given. Encode at 24 fps to `<out>/<index>.mp4` with the package's `encode_video`, then write the status.

- [ ] **Step 5: `Ltx25Backend` + unit tests**

- **`check_args`:** frames `% 8 == 1` and ≤ max; `[height, width]` is in `resolutions`; the variant is enabled.
- **`produce`:** check the RAM, then `run_workers` over `split_gpus(gpus, 1, workers)`.
- **`finish`:** strip the audio (`ffmpeg -i in -an -c:v copy out`), probe the result for 24 fps and 16:9 within 2 %, and write the caption JSON.

Unit tests use a fake bridge. Also test that a RAM shortfall fails the job with a message: monkeypatch the meminfo reader.

- [ ] **Step 6: GPU smoke, enable, record, commit**

Add a `gpu` test with 2 items per enabled variant (1 T2V, 1 I2V) on the chosen workers. Assert the candidates pass `Ingestor` as `video_caption_static`, and record the timings. Enable what passed, then commit.

`git commit -m "feat(tools): rollout_ltx25 (LTX-2.5, fp8-cast + CPU offload, one worker per GPU)"`

---

### Task 8: Integration, agent knowledge, docs, full verification, merge

**Files:**
- Modify: `kernel/ar_kernel/tools/gpu_jobs.py` (`build_gpu_backends` complete), `seed_agent/knowledge/data_building.md`, spec §10/§5.4/§16.3/§17, the Plan 2 plan's Plan 4 contract table, `verification-log.md`
- Test: `tests/test_gpu_jobs.py` (the build test), `tests/test_tool_server.py` if it snapshots the tool list

**Interfaces:**
- Produces, for Plan 4: `build_gpu_backends(cfg, run_dir, gpus, registry, recorder) -> list[GpuJob | CaptionBackend]`, covering every enabled data tool including the captioner. The loop registers each on the run's `JobQueue`, then calls `register_gpu_tools(mcp, kit, q)` and `register_caption_tool(mcp, kit, q)`.

- [ ] **Step 1: `build_gpu_backends` test**

With a config where each tool and variant is toggled, the function returns exactly the enabled backends. A backend whose variants are all disabled is absent. The captioner is always present.

A tool-server test lists tools over MCP: disabled tools are absent, and each enabled tool's schema lists only the enabled variants (for example as an enum in the description, or a `Literal` type if the MCP SDK renders it; check how `tests/test_tool_server.py` inspects schemas).

- [ ] **Step 2: Agent knowledge (keep it short: simplicity rule)**

Add at most about 15 lines to `seed_agent/knowledge/data_building.md`:
- which rollout tools exist;
- that they are slow GPU jobs, so batch items;
- that AlayaWorld rollouts come back with poses and per-round segments (eligible for `per_chunk`);
- that Wan/LTX clips need `annotate_camera` before ingesting as `moving`, or are ingested as `static` only when the camera truly does not move;
- that the result items carry a ready `candidate` for `data_ingest`.

Do not add a new tool wrapper in `seed_agent/tools.py`; the MCP tools are already exposed.

Run `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py tests/test_contract_package.py -q`.

- [ ] **Step 3: Spec and contract amendments**

- **§10:**
  - the rollout and annotate rows get the as-built signatures (batched `items`, job-level params, output locations, candidate shape, which variants passed);
  - V2V for LTX is dropped (YAGNI; no agent flow needs it);
  - `annotate.camera`'s backend is ViGeo.
- **§5.4:** provenance `seed` is the item seed (the job seed for AlayaWorld).
- **§16.3 items 5–6:** a pointer to the verification-log results.
- **§17:** the new `annotate:` and `generators:` blocks.
- **Plan 2 plan:** the "Plan sequence" LTX sentence is corrected per fact 7, and in the Plan 4 contract `build_gpu_backends` becomes the way services are built.

- [ ] **Step 4: Full verification**

1. Run the unit suite: `conda run --no-capture-output -n autoresearcher python -m pytest -q`.
2. Run the docker suite alone: `-m docker`.
3. Run every `gpu` test of this plan together, once, on `AR_TEST_GPUS`, with nothing else running on those GPUs. The captioner `gpu` test is included; it now loads from `.cache/`.

All must pass, and the outputs are pasted into the report.

- [ ] **Step 5: Reviews and merge**

The whole-branch review happens after this task (controller).

Then:
1. Fast-forward AutoResearcher `main` to `feat/gpu-data-sources` and push.
2. Push WorldModel `main` (the Task 2 commit).
3. Confirm WBench has nothing new.
4. Tell the user which out-of-project copies are now redundant and could be removed at their discretion: `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8` (29 GB). Do not remove it.

---

## Self-review

- **Spec coverage.**
  - §10 `rollout.alayaworld` → Task 5; `rollout.wan22` → Task 6; `rollout.ltx25` → Task 7; `annotate.camera` → Task 4.
  - "Disabled variants omitted from schemas" → Tasks 3 and 8.
  - §5.4 rollout provenance → Task 3.
  - §6.2 standard-format rules → the `finish` steps plus ingest checks in each GPU smoke.
  - §16.3 item 5 → the smokes in Tasks 5–7; item 6 → Task 4 Step 5.
  - §17 → Task 3 Step 1.
  - The user's cache rule → Task 1.
  - The WorldModel patch → Task 2.
  - Intentionally dropped: LTX V2V (amended in Task 8).
- **Placeholder scan.**
  - The GPU tasks deliberately measure before they code: the AlayaWorld frame layout and offset, the LTX flag names and RAM-based worker count, and the Wan memory settings. Each is a concrete measurement step with a written decision rule, not a TBD.
  - The exact library call signatures (Wan `generate`, LTX pipeline constructors, ViGeo focal normalization) are read from the pinned code in a named step, because guessing them is how bugs slip in.
- **Type consistency.**
  - `GpuJob.submit(q, caller, args)` → `{job_id}`.
  - Result items are `{index, candidate | error, worker}` for rollouts and `{index, video, pose, frames, intrinsics | error}` for annotation.
  - The bridge protocol is the same for the three bridges and the fake worker.
  - `split_gpus` and `run_workers` signatures match across Tasks 3–7.
- **Review Focus.** Each of the five lines has a test in its owning task (Task 3: links, cancel, crash; Tasks 4/5: frame-count mismatch; Task 4: `max_frames`).
