# Moving this tree to another path or machine

Run `ar doctor` after any move. It checks everything below and exits non-zero on
a real problem (`--strict` also fails on warnings).

## The contract

**The three repos must sit side by side**, because `configs/kernel.yaml` locates
the other two relatively:

```
<anywhere>/
  AutoResearcher/     <- kernel, configs, runs
  WorldModel/         <- training, rendering, data checks
  WBench/             <- benchmark and metrics
```

`paths.worldmodel: ../WorldModel`, `paths.wbench: ../WBench`, `paths.runs_dir: runs`.
Keep them relative. An absolute path here pins the tree to one machine, and
`ar doctor` reports it as a failure.

## What is path-independent already

- **Tracked source.** WorldModel has no hardcoded absolute paths; AutoResearcher
  has them only inside `tests/fixtures/real_successful_train.log`, where they are
  captured data and must stay verbatim. WBench's only hits are upstream
  leaderboard scripts (`tools/add_three_models_*.py`, `tools/swap_cosmos3_*.py`)
  that this project never invokes.
- **Caches.** The prompt-embedding cache is keyed by SHA-1 of the prompt text and
  the DA3 cache by case id — neither encodes a path, so both survive a move.
- **`.env`.** `WorldModel/.env` and `WBench/.env` are *relative* symlinks to
  `../AutoResearcher/.env`. Keep them relative; `ar doctor` fails an absolute one.
- **Generated run configs.** Written per run under `runs/`, regenerated each time.

## Caches

Large caches stay inside the project (user rule, 2026-09-25): `AutoResearcher/.cache/`
(gitignored) holds model downloads, compile caches and package caches instead of the
usual dotfiles under `$HOME`. `ar_kernel.subproc.cache_env()` sets `HF_HOME`,
`XDG_CACHE_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `PIP_CACHE_DIR` and `UV_CACHE_DIR`, all rooted at
`cache_dir()` (`AutoResearcher/.cache` by default). Every `run_in_env` child gets these
(a caller's `extra_env` can still override one), and `ar`'s own `main()` applies them to
its own process too, so in-process `huggingface_hub` calls (e.g. HfTools) also use the
project cache. Set `AR_CACHE_DIR` to relocate the whole cache tree elsewhere (a symlink
farm on a bigger disk, say) without touching any of the variable names above.

On a new machine, `AutoResearcher/.cache/` starts empty, so the captioner's model
(`Qwen/Qwen3.8-27B-FP8`, ~29 GB) needs to be copied from an existing `~/.cache/huggingface/hub/`
or downloaded fresh into `AutoResearcher/.cache/huggingface/hub/` before the captioner can
start; the first captioner start after that also rebuilds the vLLM compile cache under
`.cache/vllm` and is slower than subsequent ones.

## Prefix envs (envs that don't fit `~/miniforge3`)

Named envs live in `~/miniforge3/envs` (disk budget: `/home` has limited free space, each
named env is ~8-12 GB). A generator env that would tip that over instead lives inside the
project as a **conda-prefix** env, `AutoResearcher/.envs/<name>` (gitignored). `ar_kernel.subproc`
tells the two apart by the env value: a plain name (`"autoresearcher"`) runs `conda run -n
<name>`; a value containing "/" (`".envs/gen-zimage"`) is a prefix env instead — repo-relative
unless absolute — and runs `conda run -p <path>` (`conda_command()` in `subproc.py`, tested in
`tests/test_subproc.py` without needing conda). `configs/kernel.yaml`'s `<block>.env` is just
this string, so a backend's `produce()` never needs to know which kind it got.

**`gen-zimage`** (Z-Image-Turbo, `generate_images`): created at
`AutoResearcher/.envs/gen-zimage`, python 3.11.

```bash
cd AutoResearcher
export CONDA_PKGS_DIRS=$PWD/.cache/conda/pkgs PIP_CACHE_DIR=$PWD/.cache/pip
conda create -y -p .envs/gen-zimage python=3.11
conda run --no-capture-output -p .envs/gen-zimage pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -p .envs/gen-zimage pip install "diffusers>=0.36" transformers accelerate safetensors
```

Pinned/measured versions (Task 5): `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`,
`diffusers==0.40.0`, `transformers==5.17.0`, `accelerate==1.15.0`, `safetensors==0.8.0`. Env
size on disk: ~5.7 GB. Weights: `Tongyi-MAI/Z-Image-Turbo` @
`f332072aa78be7aecdf3ee76d5c247082da564a6` downloaded with `hf download ... --exclude
"assets/*" --local-dir weights/z-image-turbo` (31 GB, `HF_HOME` under `AutoResearcher/.cache`).

**`gen-wan22`** (Wan2.2 TI2V-5B, `rollout_wan22`): a prefix env at `AutoResearcher/.envs/gen-wan22`,
python 3.10. The official Wan2.2 code is a separate, gitignored clone at `third_party/Wan2.2`
(put on `sys.path` by the bridge, not pip-installed), pinned as `generators.wan22.commit`;
`Wan22Backend` refuses to run when `git -C third_party/Wan2.2 rev-parse HEAD` differs.

```bash
cd AutoResearcher
export CONDA_PKGS_DIRS=$PWD/.cache/conda/pkgs PIP_CACHE_DIR=$PWD/.cache/pip
mkdir -p third_party
git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2
git -C third_party/Wan2.2 checkout 1ea34ff48f87168174e12956e200b1d908b1c5ff   # the pin
conda create -y -p .envs/gen-wan22 python=3.10
conda run --no-capture-output -p .envs/gen-wan22 pip install torch==2.7.1 torchvision==0.22.1 \
  torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# requirements.txt without flash_attn (a plain `pip install flash_attn` builds from source; see below)
grep -v flash_attn third_party/Wan2.2/requirements.txt > .cache/wan22_reqs_noflash.txt
conda run --no-capture-output -p .envs/gen-wan22 pip install -r .cache/wan22_reqs_noflash.txt
# flash_attn: the prebuilt wheel matching torch 2.7 / CUDA 12 / cp310 / cxx11abi TRUE
# (check the ABI with: python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
mkdir -p .cache/wheels
curl -sL -o .cache/wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
conda run --no-capture-output -p .envs/gen-wan22 pip install \
  .cache/wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# `import wan` eagerly imports every task (S2V, Animate), so it also needs decord, librosa and
# peft (from requirements_s2v.txt / requirements_animate.txt; their heavier deps are not needed).
# decord comes from the eva-decord fork (same `import decord`), from PyPI, with its dependencies:
conda run --no-capture-output -p .envs/gen-wan22 pip install eva-decord==0.6.1 librosa==0.11.0 peft==0.21.0
conda run --no-capture-output -p .envs/gen-wan22 pip check
```

`decord` is the `eva-decord` fork (same `import decord`): the PyPI `decord==0.6.0` wheel is
tagged `cp36-cp36m`, so `pip check` fails on it ("not supported on this platform"); with
`eva-decord` the check is clean ("No broken requirements found."). Every distribution that
`import wan` loads (checked against `sys.modules` after the import) comes from torch, the trimmed
requirements.txt, flash_attn or the line above, or their dependencies.

Versions (Task 7): Wan2.2 @ `1ea34ff48f87168174e12956e200b1d908b1c5ff` (2026-09-21);
`torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `torchaudio==2.7.1+cu126`,
`flash_attn==2.8.3` (wheel above; Wan uses FlashAttention-2, FA3 is not installed),
`diffusers==0.39.0`, `transformers==4.51.3`, `tokenizers==0.21.4`, `accelerate==1.15.0`,
`opencv-python==4.11.0.86`, `numpy==1.26.4` (requirements.txt caps `numpy<2`), `eva-decord==0.6.1`,
`librosa==0.11.0`, `peft==0.21.0`. Env size: ~7.2 GB. Weights: `Wan-AI/Wan2.2-TI2V-5B` at
`weights/wan2.2-ti2v-5b` (32 GB: DiT shards, `models_t5_umt5-xxl-enc-bf16.pth`,
`Wan2.2_VAE.pth`). Kernel launches set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(without it the 24 GB card OOMs in the VAE decode; see verification-log, Task 7).

## What needs doing on a new machine

1. **Create the conda environments** — `alayaworld`, `wbench-main`, `wbench-vp`,
   `autoresearcher`, plus the prefix envs above (`.envs/gen-zimage`, ...). Never use `base` or
   the system Python. `ar doctor` lists any named env that is missing.
2. **System tools on PATH:** `ffmpeg`, `ffprobe`, `conda`, `git`.
3. **Weights are not in git.** Re-download or copy:
   `WorldModel/weights/` (LTX-2.3 transformer, `alaya-world-ar`, VAE, Gemma),
   `WBench/weights/` (SAM2, DA3, MegaSAM, HPSv3, DINOv2 torch hub).
4. **Fill `AutoResearcher/.env`** if API-based metrics are wanted. The `ar`
   CLI loads it at startup into its own environment, so the keys also reach the
   WBench subprocesses. Variables already set in the shell win, and empty values
   are ignored, so the shipped placeholder enables nothing. Without
   `VLM_API_KEY` the six VLM metrics are excluded by design; the run records the
   exclusion and uses that fixed metric set for every node, so scores stay
   comparable. GPU-metric weights are different: if any is missing, a run
   refuses to start rather than silently scoring on fewer metrics.
5. **Run `ar doctor`,** then the unit suites.

## The failure that motivated this

`WBench/weights/hub/torchhub` and `.../hub/checkpoints` were symlinks left
pointing at a previous checkout location (`/home/...`) after the tree moved to
`/mnt/biometrics`. `Path.exists()` follows symlinks and returns `False` for a
dangling one, so the setup code believed the links were missing, tried to create
them, and hit `FileExistsError` — on **every** case.

It then failed quietly three times over: `run_megasam.py`'s workers exited 0
after failing every case; its `main()` ignored worker exit codes; and WBench's
`main.py` ignored the tool's return code. The phase printed
`MegaSAM done in 0s`, the run continued, and the report came out **missing five
metrics** (`spatial_consistency`, `gated_spatial_consistency`,
`navigation_trajectory`, `navigation_accuracy`, `navigation_consistency`) — a
wrong result that looked like a clean one.

All four faults are fixed: links are repaired in place rather than recreated
blindly, and failures now propagate at each level. `ar doctor` checks for broken
symlinks anywhere in the three repos, because this is how a move breaks this tree.
