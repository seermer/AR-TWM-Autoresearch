# AutoResearcher GPU Data Sources Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agents can request generated first-frame images (`generate_images`), generated training clips (`rollout_alayaworld`, `rollout_wan22`, `rollout_ltx25`) and per-frame camera poses (`annotate_camera`) as asynchronous GPU jobs. Every output lands in the agent's staging directory as a ready-to-ingest candidate, and every variant is smoke-verified on this machine before it is enabled.

**Architecture:** Each tool is a `JobQueue` backend (Plan 2, Task 9) built on one shared base, `GpuJob` in `kernel/ar_kernel/tools/gpu_jobs.py`, which does four things:
- It stages the agent's input files race-free into a kernel-private job directory.
- It runs one worker process per GPU group, in the generator's own conda env.
- It post-processes outputs to the standard layout (spec §6).
- It publishes them into `/workspace/staging/{rollouts,annotations,images}/<job_id>/` without following links the agent planted.

The workers are small bridge scripts in `kernel/ar_kernel/bridges/`. Each loads its model once and loops over its shard of items.

AlayaWorld is the one exception to the bridge scripts. It renders agent-written, WBench-style cases (image, perspective, per-turn navigation actions and interaction prompts) through the eval's own `scripts/tools/run_wbench.py` path, so the rollouts follow the same code as the eval. A small WorldModel patch saves the camera path that render generates. *(Amended 2026-09-26, as built: that path is what the actions commanded, not what the video shows; AlayaWorld follows translation reliably but rotation only weakly (Task 6 ViGeo check). By user decision the candidate carries no pose: the commanded path is published as metadata (`commanded_camera`), and agents take the pose from `annotate_camera` (ViGeo), then ingest the clip as `moving`.)*

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
  - Z-Image: `gen-zimage` (new).
  - Do not use or modify the `gen-alaya` env: it is not ours.
- **Large files stay inside the project** (user rule, 2026-09-25): models, datasets, compile caches and package caches.
  - Kernel subprocesses get `HF_HOME`, `XDG_CACHE_HOME` and the other cache variables pointing at `AutoResearcher/.cache/` (Task 1).
  - Env installs set `PIP_CACHE_DIR`/`UV_CACHE_DIR` there too.
  - `autoresearcher` and `alayaworld` stay named envs in `~/miniforge3/envs`. (As built, 2026-09-26: `/` ran short of space, 27 GB free, so the new generator envs — `gen-zimage`, `gen-wan22`, `gen-ltx25` — are instead conda-prefix envs under `AutoResearcher/.envs/<name>`, created with `conda create -p` and run with `conda run -p` via `ar_kernel.subproc.run_in_env`, which accepts repo-relative prefix paths. See `docs/PORTABILITY.md`.)
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
5. **AlayaWorld has no action input.** Action conditioning is off in the released configs (`control.candidates: [[]]`). Its only motion signal is a per-frame camera path (`cam_c2w`). For each round (4 latents = 32 frames), the trainer renders its ViGeo 3D memory into each target frame's camera and VAE-encodes the result (`_build_validation_vigeo_bank_spatial_context`, `rollout_trainer.py:3809`). Prompts switch per round (`_validation_prompt_schedule`).
6. **The WBench render turns actions into that camera path.**
   - `_build_wbench_camera_trajectory` (`rollout_trainer.py:1764`) holds each turn's action for `wbench_chunks_per_turn` rounds (3 in `configs/wbench_full.yaml`, i.e. 96 frames), using fixed steps: forward 0.16 per latent, yaw/pitch 6° per latent.
   - First person moves the camera; third person orbits a subject pivot found from the subject mask and depth.
   - Per-turn prompts come from the case's interactions (`navigation`, `subject_action`, `event_edit`, `perspective_switch`; counts over the 289 cases: 601 / 213 / 183 / 61).
   - The render writes `case_<id>_combined.mp4` plus a sidecar JSON with `turn_segments` (per-turn frame ranges) and `prompt_schedule`, but **not** the camera path. Task 2 adds it.
   - `scripts/tools/run_wbench.py --config --gpus --cases` drives it, as `eval/render.py` already does.
   - WBench's data has no `prompts_training_style.jsonl`, so the eval uses the fallback prompt builder.
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
| `kernel/ar_kernel/tools/images.py` (create), `kernel/ar_kernel/bridges/zimage_generate.py` (create) | `ImageBackend` (`generate_images`) and its worker in `gen-zimage` (Z-Image-Turbo) |
| `kernel/ar_kernel/tools/annotate.py` (create) | `AnnotateBackend` (`annotate_camera`) |
| `kernel/ar_kernel/tools/rollouts.py` (create) | `AlayaWorldBackend`, `Wan22Backend`, `Ltx25Backend` |
| `kernel/ar_kernel/bridges/vigeo_poses.py` (create) | Worker in `alayaworld`: video → `cam_c2w [N,4,4]` + pixel intrinsics |
| `kernel/ar_kernel/bridges/wan22_generate.py` (create) | Worker in `gen-wan22`: loads WanTI2V once, renders its shard |
| `kernel/ar_kernel/bridges/ltx25_generate.py` (create) | Worker in `gen-ltx25`: loads an LTX-2.5 pipeline once, renders its shard |
| `configs/kernel.yaml` (modify) | `generators:` reshaped, `annotate:` added |
| `tests/test_cache_env.py`, `tests/test_gpu_jobs.py`, `tests/test_annotate.py`, `tests/test_rollouts.py` (create) | Unit tests (fake workers) plus one `gpu` test per real backend |
| `tests/fixtures/fake_gen_worker.py` (create) | Fake worker implementing the bridge protocol |
| `WorldModel/alaya/trainer/rollout_trainer.py` (modify), `WorldModel/tests/test_wbench_camera_sidecar.py` (create) | The WBench-mode render also saves the camera path it used (`case_<id>_combined_camera.npz`) |
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

### Task 2: WorldModel: save the generated camera path next to each WBench-mode video

**Files:**
- Modify: `WorldModel/alaya/trainer/rollout_trainer.py` (`_save_wbench_output_video`)
- Test: `WorldModel/tests/test_wbench_camera_sidecar.py`

**Interfaces:**
- Produces: beside every `case_<id>_combined.mp4` / `.json`, the WBench-mode render now also writes `case_<id>_combined_camera.npz` containing:
  - `cam_c2w [F,4,4]` float32: the camera trajectory the render used (`metadata["cam_c2w"]`, built from the case's navigation actions by `_build_wbench_camera_trajectory`), sliced to exactly the F frames written to the mp4, and re-expressed relative to the first written frame (frame 0 = identity);
  - `intrinsic [3,3]`: the normalized intrinsics the render used for those frames. For uncalibrated sources this is the ViGeo-fitted prefix intrinsic (see `_validation_vigeo_target_cameras`), not the 0.5 placeholder. If the fitted value is not reachable from `_save_wbench_output_video` without restructuring, save only `cam_c2w` and say so in the report.

  The sidecar JSON gains `"camera_file": "<name>.npz"`. Nothing else changes, so WBench scoring is unaffected. WBench reads only the mp4 and its own inputs; confirm with `grep -rn "_camera.npz\|combined.json" ../WBench/src ../WBench/main.py`, which must find nothing that would pick up the new file.

- [ ] **Step 1: Find the frame mapping (read, no code)**

`_save_wbench_output_video` decodes `[prefix latents] + pred latents` and drops `prefix_latents * temporal_stride` leading frames, so mp4 frame 0 is the first generated frame. Work out which index of `metadata["cam_c2w"]` that frame corresponds to. The trajectory's own time base starts at pixel 0 of the render's timeline, and actions start at `action_start_pixel` (`_build_wbench_camera_trajectory`); see also `_vigeo_target_prefix_pixel_frames` and `target_base_start` in `_prepare_wbench_validation_metadata`.

Write the formula into the report and into a comment in the code. Task 6's GPU smoke checks it independently with ViGeo.

- [ ] **Step 2: Write the failing test**

The test builds a minimal fake of what `_save_wbench_output_video` needs:
- a `RolloutTrainer`-like object via `object.__new__(RolloutTrainer)`, with a `cfg` stub (`sample.temporal_stride=8`, `sample.fps=24`, `layout.sink_latent_frames`, `validation.video_history_latent_frames`);
- `_decode_latent_to_video_frames` monkeypatched to return `8 * latents` frames of zeros;
- `_write_video` monkeypatched to record the frame count.

It calls the method with a known `metadata["cam_c2w"]` (a per-frame translation ramp) and asserts:
- the `.npz` exists and its `cam_c2w` has as many frames as were written;
- frame 0 is the identity;
- the translations equal the ramp slice from Step 1's formula, re-expressed relative to that frame;
- the sidecar JSON names the file.

If building the fake needs more than ~40 lines, factor the slicing into a small pure helper `_wbench_written_camera(metadata, frames_written, prefix_frames) -> np.ndarray` and unit-test that helper directly instead. That is also acceptable.

- [ ] **Step 3: Run it to verify it fails**

Run: `cd WorldModel && conda run --no-capture-output -n alayaworld python -m pytest tests/test_wbench_camera_sidecar.py -q`
Expected: FAIL (no npz written).

- [ ] **Step 4: Implement, keeping it minimal**

Right after `self._write_video(output_path, frames)`, save the sliced, re-based trajectory with `np.savez(output_path.with_name(output_path.stem + "_camera.npz"), cam_c2w=..., intrinsic=...)` and add `"camera_file"` to `sidecar`.

- [ ] **Step 5: Run the WorldModel tests**

Run: `cd WorldModel && conda run --no-capture-output -n alayaworld python -m pytest tests/test_wbench_camera_sidecar.py tests/test_run_wbench_config_path.py -q` → PASS.

- [ ] **Step 6: Commit (WorldModel `main`)**

```bash
cd WorldModel
git add alaya/trainer/rollout_trainer.py tests/test_wbench_camera_sidecar.py
git commit -m "feat(wbench render): save the generated camera trajectory beside each video"
```

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
  - `register_gpu_tools(mcp, kit, q)`: registers `annotate_camera` / `generate_images` (Task 5) / `rollout_alayaworld` / `rollout_wan22` / `rollout_ltx25` only for backends registered on `q`.
  - `build_gpu_backends(cfg, run_dir, gpus, registry, recorder) -> list`: the enabled backends, filled in by Tasks 4–8.

- [ ] **Step 1: Config shape**

Replace the `generators:` block in `configs/kernel.yaml` with the block below. Everything stays disabled; Tasks 4–8 enable what their smokes verify.

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
    variants: {dmd4: {enabled: false}, ar30: {enabled: false}}   # dmd4 = configs/wbench_full.yaml as the eval renders
    max_turns: 9
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
    extra_args: {}               # set by Task 7 from the fit measurements
    max_items: 16
    timeout_s: 43200
    license: Apache-2.0
  ltx25:
    env: gen-ltx25
    repo: third_party/LTX-2
    weights: weights/ltx-2.5
    variants: {distilled: {enabled: false}, dev: {enabled: false}}
    gpus_per_worker: 1
    workers: null                # Task 8 sets it from host-RAM measurements
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

Write out all five registrations explicitly, with typed parameters, so the MCP schema is precise (each is added by the task that builds its backend):
- `rollout_alayaworld(items, variant, rounds_per_turn, seed)`
- `generate_images(items, width, height)`
- `rollout_wan22(items, frames)`
- `rollout_ltx25(items, variant, frames, height, width)`

Each backend exposes a `description` string that states:
- the item fields;
- that it returns `{job_id}` at once, with the result collected via `job_wait`;
- where outputs land and that each result item carries a ready `candidate` for `data_ingest`;
- that rollouts without poses need `annotate_camera` (or `camera_motion: static`) before ingest.

`build_gpu_backends(cfg, run_dir, gpus, registry, recorder)` returns `[]` for now. Tasks 4–8 each add their backend when its config says enabled.

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

### Task 5: `generate_images` (Z-Image-Turbo)

**Why (user, 2026-09-25):** AlayaWorld, Wan I2V and LTX I2V all start from an image, so agents need a way to make first frames.

**Files:**
- Create: `kernel/ar_kernel/bridges/zimage_generate.py`, `kernel/ar_kernel/tools/images.py` (`ImageBackend`), `tests/test_images.py`
- Modify: `gpu_jobs.py` (`_collect` gets an `image` kind; `register_gpu_tools`; `build_gpu_backends`), `configs/kernel.yaml`, `docs/PORTABILITY.md`

**Interfaces:**
- Produces: tool `generate_images`, backend `ImageBackend(GpuJob)` with `kind = "image"`.
  - **Job params:** `width`, `height` (multiples of 16, each 256..1920; default 1280×720, 16:9).
  - **Items:** `{"prompt": str, "seed": int}`, with up to `images.max_items` (64) per job.
  - **Result items:** `{"index", "image": "/workspace/staging/images/<job>/<i>.png", "prompt", "seed", "generator": "z-image-turbo", "license": "Apache-2.0"}` or `{"index", "error"}`.
  
  Images are inputs to rollouts, not training clips, so they carry no ingest candidate. A clip made from one is a rollout whose `inputs_hash` includes the image's hash (Task 3), and ingest's leakage check still compares its frames with WBench.

**Facts (checked 2026-09-25):**
- `Tongyi-MAI/Z-Image-Turbo` @ `f332072aa78be7aecdf3ee76d5c247082da564a6`, Apache-2.0, `diffusers:ZImagePipeline`.
- Contents: `transformer/` 24.6 GB (6B parameters stored in fp32), `text_encoder/` 8.0 GB (Qwen3-4B, bf16), `vae/` 0.17 GB, tokenizer and scheduler.
- Loaded in bf16, the model is about 12 GB (transformer) plus 8 GB (text encoder), which is tight on 24 GB but fits with the text encoder offloaded after encoding.
- Turbo runs 8 DiT steps (`num_inference_steps=9`, `guidance_scale=0.0`, as on the model card; re-read the card at the pinned revision).

- [ ] **Step 1: Weights and env**

```bash
cd AutoResearcher
HF_HOME=$PWD/.cache/huggingface conda run --no-capture-output -n autoresearcher \
  hf download Tongyi-MAI/Z-Image-Turbo --revision f332072aa78be7aecdf3ee76d5c247082da564a6 \
  --exclude "assets/*" --local-dir weights/z-image-turbo
du -sh weights/z-image-turbo
export PIP_CACHE_DIR=$PWD/.cache/pip
conda create -y -n gen-zimage python=3.11
conda run --no-capture-output -n gen-zimage pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n gen-zimage pip install "diffusers>=0.36" transformers accelerate safetensors
conda run --no-capture-output -n gen-zimage python -c "from diffusers import ZImagePipeline"
```

(as built: prefix env `.envs/gen-zimage`, `conda create -p`/`conda run -p`, see docs/PORTABILITY.md)

If `hf` is not available in `autoresearcher`, use `huggingface_hub.snapshot_download` with the same arguments. Record the exact versions installed (`pip freeze | grep -iE "diffusers|transformers|torch"`), the env size and the commands in `PORTABILITY.md`. Pin `images.revision` in the config.

- [ ] **Step 2: Fit spike (one GPU)**

A 10-line script in the scratchpad:
1. `ZImagePipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)`.
2. Try `.to("cuda")` first; if it OOMs, use `enable_model_cpu_offload()`.
3. Render 4 prompts at 1280×720, then one at 960×544 and one at 1920×1080.

Record the load time, the seconds per image and peak GPU memory, and look at the images: they should show plausible scenes with no NaN or black frames. Put the result in `verification-log.md`. Choose `offload: none | model` for the config from this measurement.

- [ ] **Step 3: Bridge `zimage_generate.py`**

- **Arguments:** the bridge protocol plus `--weights --width --height --steps --offload`.
- **Setup:** load the pipeline once.
- **Per item:** `pipe(prompt=item["prompt"], width, height, num_inference_steps=steps, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(item["seed"])).images[0].save(<out>/<index>.png)`, then write the status `{"ok": true, "seconds": t}`.

- [ ] **Step 4: `ImageBackend` + `_collect` image kind + unit tests**

- **`check_args`:** width and height are multiples of 16 within 256..1920; every item has a non-empty `prompt` and an int `seed`.
- **`produce`:** `run_workers(images.env, …, split_gpus(gpus, 1, None))`.
- **`finish`:** returns `{"image": out/<i>.png}` after checking that PIL opens it at the requested size.
- **`_collect`:** publishes the `image` role like the others. For `kind == "image"` it returns `{"index", "image", "prompt", "seed", "generator", "license"}`.

Unit tests with the fake worker (extend `fake_gen_worker.py` with `--kind image`, which writes a PNG of the requested size via PIL) check:
- the publish path and size;
- a size that is not a multiple of 16 is refused at submit;
- the tool is registered only when enabled.

- [ ] **Step 5: GPU smoke, enable, commit**

Add a `gpu` test: 8 prompts on 4 GPUs at 1280×720. All succeed and the sizes are right. One image is then used as the `image` input of a Wan or LTX item and as an AlayaWorld case image in those tasks' smoke tests, so the pipelines are shown to chain.

Config:

```yaml
images:                          # generate_images GPU job
  enabled: false                 # set true after the smoke
  env: gen-zimage
  weights: weights/z-image-turbo
  revision: f332072aa78be7aecdf3ee76d5c247082da564a6
  steps: 9
  offload: none                  # from the fit spike
  max_items: 64
  timeout_s: 7200
  license: Apache-2.0
```

`git commit -m "feat(tools): generate_images (Z-Image-Turbo) for first frames"`

---

### Task 6: `rollout_alayaworld`: WBench-style cases through the eval's own render path

*(Amended 2026-09-26, as built: the "exact poses" below were dropped. The commanded camera path does not match the rendered motion (rotation is followed only weakly), so the candidate has no `pose` and no `camera_motion`; the saved path is published as metadata `commanded_camera`, and agents run `annotate_camera` (ViGeo poses) before ingesting as `moving`. No trim was needed: F = 32·rounds − 7 already puts round boundaries on the `25 + 32k` grid. See the verification-log section "Plan 3 — rollout_alayaworld".)*

**Decision (user, 2026-09-25):** render exactly as the WBench eval does, covering the full WBench interaction set, not only navigation.
- **Per-turn inputs:** each turn has a prompt and an action.
- **Turn length:** a turn lasts `rounds_per_turn` rounds of 32 frames. The eval uses 3; `rounds_per_turn: 1` gives per-round granularity with no code change, since it is only `wbench_chunks_per_turn`.
- **No per-latent actions:** the eval holds each action for a whole turn.
- **No custom camera paths.** The model has no action input; WorldModel turns the actions into a per-frame camera path, and Task 2 saves that path.

**Files:**
- Create: `kernel/ar_kernel/tools/rollouts.py` (`AlayaWorldBackend`), `tests/test_rollouts.py`
- Modify: `gpu_jobs.py` (`build_gpu_backends`), `configs/kernel.yaml`

**Interfaces:**
- Consumes: Task 2 (`case_<id>_combined_camera.npz`), Task 3 (`GpuJob`), Task 4 (`AnnotateBackend`, only in the GPU check), `eval/render.py`'s `build_render_config` pattern.
- Produces: `AlayaWorldBackend(GpuJob)`, tool `rollout_alayaworld`.
  - **Job params:**
    - `variant`: `dmd4` (the eval's own setting, `configs/wbench_full.yaml`) or `ar30` (the same config with `paths.dmd_resume: null`, `validation.sampling_steps: 30`, `validation.scheduler: shift`, `validation.cfg_scale: 3.0`, taken from `configs/infer_i2v_camera_ar.yaml`'s differences). Enabled variants only.
    - `rounds_per_turn`: 1..3, default 3.
    - `seed`: int.
  - **Item = one WBench-style case:**
    - `image`: the first frame, any size (resized as in the eval);
    - `perspective`: `first_person` | `third_person`;
    - `environment_prompt`, `character_prompt`, `perspective_prompt`: strings; the last two may be empty;
    - `subject_mask`: optional image path, used by third-person orbit and subject anchoring as in the eval;
    - `turns`: a list of 1..`max_turns` objects `{"action": "<WBench navigation action, e.g. W, A, S, D, left, right, up, down, W+left, stop>", "subject_action"?: str, "event_edit"?: str, "perspective_switch"?: str}`.
  - **The kernel writes each item as a case file**, `case_<index>.json`, in the WBench schema: `interactions` entries of types `navigation`, `subject_action`, `event_edit` and `perspective_switch` per turn, exactly as WBench cases use them (see `WBench/data/cases/case_100.json` and the type counts in the task report). `settings.initial_image` and `settings.subject_mask` point at the staged files, and `metric_list: []`.
  - **Candidate:**
    - a 24 fps mp4;
    - `pose`: npz with `cam_c2w` for every published frame, plus pixel `intrinsics` if Task 2 provides them;
    - caption JSON: `{"caption": <the round-0 prompt the render used>, "segments": one per round, from the sidecar's prompt_schedule, adjacent equal prompts merged}`;
    - `camera_motion: "moving"`;
    - provenance generator `alayaworld-<variant>`.
    
    The result item also carries the sidecar's `actions` and `turn_segments`.

- [ ] **Step 1: Case writer, config writer, submit checks (unit-tested, no GPU)**

`produce` writes into `work/data/`:
- `cases/case_<index>.json`;
- `images/case_<index>.<ext>`;
- `masks/case_<index>_mask.png`, when a mask is given.

It builds the render config the way `eval/render.py:build_render_config` does (read that function first): `configs/wbench_full.yaml`, then:
- `mode.dataset.root = work/data`;
- `mode.dataset.case_ids = [indices]`;
- `mode.wbench_output_dir = work/videos`;
- `mode.wbench_chunks_per_turn = rounds_per_turn`;
- `run.output_dir` / `run.log_dir` under `work/`;
- `run.seed = seed`;
- `validation.per_sample_seed: true`;
- `paths.*` absolute;
- `validation.save_joystick: false`;
- the ar30 overrides for `ar30`.

It launches `scripts/tools/run_wbench.py --config <cfg> --gpus <node gpus> --cases <indices>` in `alayaworld` (cwd WorldModel) through `run_workers` with a single group of all node GPUs, and `MASTER_PORT` set to a free port if `run_wbench.py`/`train.sh` honor it (check).

Do **not** set `WBENCH_REWRITE_JSONL`. WBench's data has no rewrite file, so the eval uses the fallback prompt builder, and synthetic cases must too. After the run, map each `case_<index>_combined.{mp4,json}` plus the camera npz to `out/<index>.*` and the status JSON, per the bridge protocol.

Unit tests in `tests/test_rollouts.py`:
- The case writer emits valid WBench-schema JSON. Load it with WorldModel's own loader in the `alayaworld` env through a tiny script (`WBenchNaviDataset(root=…, case_ids=[…], include_non_navigation=True, …)`, reading `alaya/data/wbench.py` for the class name and args), and assert that loading yields the expected per-turn prompt schedule and actions.
- The config has the listed fields.
- ar30 differs from dmd4 only in the four listed keys.
- Submit refuses: an unknown action token (validate against `_wbench_action_to_nav`'s vocabulary, copied as a constant with a source comment); an empty `turns`; more than `max_turns`; `rounds_per_turn` outside 1..3; `third_person` with a mask that is not an image file.

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_rollouts.py -q` → PASS.

- [ ] **Step 2: GPU spike: output layout and pose check (dmd4)**

Render 2 items: one first person and one third person with a mask, 2 turns each, `rounds_per_turn: 3`. Use images that are not WBench's: frame 0 of a `WorldModel/data/examples` clip, and for third person a scene with a person plus a hand-made mask. Record:
- the mp4 frame count F;
- the sidecar's `turn_segments` and `prompt_schedule`;
- the Task 2 npz length, which must equal F;
- the rounds' frame ranges.

Then run `annotate_camera` on both outputs and compare with the saved `cam_c2w` under the Task 4 thresholds. This independently checks Task 2's index formula, and whether the camera has any effect in round 0 (memory is empty then; `memory_start_round: 1`). Write the findings into `verification-log.md`.

Repeat one item with `rounds_per_turn: 1` and confirm the actions and prompts switch every round.

- [ ] **Step 3: `finish`: align rounds to the training grid, poses, segments**

With round r occupying mp4 frames `[s + 32r, s + 32(r+1))` (s measured in Step 2; the sidecar's `turn_segments.frame_start` gives it directly):
- **Trim.** Remove `trim = (s - 25) % 32` leading frames, so that every round boundary lands at `25 + 32k` (spec §6.2 `per_chunk`). Use ffmpeg: `-vf select='gte(n\,trim)',setpts=N/24/TB -r 24 -c:v libx264 -crf 18 -pix_fmt yuv420p -an`.
- **Poses.** `cam_c2w = saved[trim:]`, re-based so frame 0 is the identity. Probe the trimmed mp4 and assert `len == frames` (Review Focus 4).
- **Intrinsics.** Normalized to pixels of the mp4, if present.
- **Segments.** One per round, in seconds of the trimmed clip, from the sidecar's `prompt_schedule` (a round's prompt is the one the render actually used, after the eval's prompt assembly). Adjacent equal prompts are merged, the first segment starts at 0 and the last ends at the duration.
- **Caption.** The round-0 prompt.

Unit-test `finish` with a synthetic mp4 (`make_mp4`) of the measured layout, a fake sidecar and a fake npz. Assert:
- the trimmed frame count equals the pose length;
- every internal boundary satisfies `abs(t*24 - (25 + 32k)) <= 0.5`;
- the candidate passes the real `Ingestor` as `video_timed_prompts_camera:per_chunk` (as `tests/test_ingest.py` does; CPU only).

- [ ] **Step 4: GPU smoke, both variants**

Add a `gpu` test: for each of dmd4 and ar30, 2 items (first and third person) with 2 turns each, where one turn carries a `subject_action` and one an `event_edit`. Assert:
- the job is `done`;
- both candidates pass `Ingestor` as `video_timed_prompts_camera:per_chunk`;
- ViGeo agrees with the published poses under the Task 4 thresholds;
- GPU memory is released.

Record the load time, the time per item and the peak memory. If a variant fails, leave it disabled and report why.

- [ ] **Step 5: Enable, record, commit**

- Enable the variants that passed.
- Add the `generators.alayaworld` config keys `max_turns: 9` (the longest WBench case) and `max_items: 16`, and remove the `config:` keys of the old sketch.
- Add the backend to `build_gpu_backends`.
- Write the verification-log section.
- Commit:

```bash
git add kernel/ar_kernel/tools/rollouts.py kernel/ar_kernel/tools/gpu_jobs.py configs/kernel.yaml tests/test_rollouts.py docs/superpowers/plans/verification-log.md
git commit -m "feat(tools): rollout_alayaworld renders WBench-style cases through the eval path, with exact poses"
```

---

### Task 7: `rollout_wan22` (TI2V-5B)

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

(as built: prefix env `.envs/gen-wan22`, `conda create -p`/`conda run -p`, see docs/PORTABILITY.md)

Record in `docs/PORTABILITY.md`:
- the exact commands;
- the Wan2.2 commit SHA;
- the env size (`du -sh .envs/gen-wan22`; as built, not `~/miniforge3/envs/gen-wan22`).

Pin that SHA in `configs/kernel.yaml` as `generators.wan22.commit`. The backend refuses to run when `git -C <repo> rev-parse HEAD` differs, so a silently updated clone cannot change outputs.

- [ ] **Step 2: Single-GPU fit spike (CLI)**

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n gen-wan22 python third_party/Wan2.2/generate.py \
  --task ti2v-5B --size 1280*704 --ckpt_dir weights/wan2.2-ti2v-5b --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "A slow walk through a sunlit forest path" --base_seed 1 --save_file <scratchpad>/wan_t2v.mp4
```

(as built: `-n gen-wan22` is `-p .envs/gen-wan22`, see docs/PORTABILITY.md)

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

### Task 8: `rollout_ltx25` (distilled, then dev)

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

(as built: prefix env `.envs/gen-ltx25`, `conda create -p`/`conda run -p`, see docs/PORTABILITY.md)

Install `ltx-core` (with the `natten` extra that pins torch 2.13.0 cu132) and `ltx-pipelines` editable into the env. Use the package indexes from `packages/ltx-core/pyproject.toml`. Prefer `uv pip install --python $(conda run -p .envs/gen-ltx25 which python) -e "third_party/LTX-2/packages/ltx-core[<natten extra name>]" -e third_party/LTX-2/packages/ltx-pipelines`, reading the extra's exact name and index URLs from that pyproject. `ltx-kernels` is **not** needed: it is only for multi-GPU SP, which does not help on 24 GB cards (fact 7).

Then run `pip check`, `python -c "import ltx_pipelines.distilled"`, and record the commands, the SHA and the env size in `PORTABILITY.md`. Pin the SHA as `generators.ltx25.commit`, the same way as Task 7.

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

(as built: `-n gen-ltx25` is `-p .envs/gen-ltx25`, see docs/PORTABILITY.md)

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

### Task 9: Integration, agent knowledge, docs, full verification, merge

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
- which rollout tools exist, and `generate_images` for making first frames (images for AlayaWorld cases and I2V items);
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
  - §10 `rollout.alayaworld` → Task 6; `rollout.wan22` → Task 7; `rollout.ltx25` → Task 8; `annotate.camera` → Task 4; `generate_images` (new, user request) → Task 5.
  - "Disabled variants omitted from schemas" → Tasks 3 and 8.
  - §5.4 rollout provenance → Task 3.
  - §6.2 standard-format rules → the `finish` steps plus ingest checks in each GPU smoke.
  - §16.3 item 5 → the smokes in Tasks 6–8; item 6 → Task 4 Step 5.
  - §17 → Task 3 Step 1.
  - The user's cache rule → Task 1.
  - The WorldModel patch → Task 2.
  - Intentionally dropped: LTX V2V (amended in Task 9).
- **Placeholder scan.**
  - The GPU tasks deliberately measure before they code: the AlayaWorld frame layout and offset, the LTX flag names and RAM-based worker count, and the Wan memory settings. Each is a concrete measurement step with a written decision rule, not a TBD.
  - The exact library call signatures (Wan `generate`, LTX pipeline constructors, ViGeo focal normalization) are read from the pinned code in a named step, because guessing them is how bugs slip in.
- **Type consistency.**
  - `GpuJob.submit(q, caller, args)` → `{job_id}`.
  - Result items are `{index, candidate | error, worker}` for rollouts and `{index, video, pose, frames, intrinsics | error}` for annotation.
  - The bridge protocol is the same for the three bridges and the fake worker.
  - `split_gpus` and `run_workers` signatures match across Tasks 3–8.
- **Review Focus.** Each of the five lines has a test in its owning task (Task 3: links, cancel, crash; Tasks 4/6: frame-count mismatch; Task 4: `max_frames`).
