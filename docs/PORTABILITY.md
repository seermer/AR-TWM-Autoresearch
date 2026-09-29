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
- **Node paths in the archive.** *(2026-09-27, Plan 4 as built.)* Every path the loop stores
  in a node's or attempt's record (checkpoint, recipe, resolved config, rationale) is written
  run-relative (`archive/nodes.py:run_rel`) and resolved against the run directory at read
  time (`run_abs`), so a run directory moved to a new path — even on another machine — still
  resolves without touching the database.

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

## Environments (all in `AutoResearcher/.envs/`)

*(Rewritten 2026-09-28.)* Each conda env is a **prefix** env, `AutoResearcher/.envs/<name>` (gitignored):
`autoresearcher`, `alayaworld`, `wbench-main`, `wbench-vp`, `vllm`, `gen-zimage`, `gen-wan22`,
`gen-ltx25`, `panel`. Only the `conda` executable itself stays outside (a prerequisite, like `git`).

**Building them: `scripts/setup_envs.sh [name ...]`** (all nine by default; the README has the walkthrough).
Per env it reads two tracked files in `envs/`:

- `<name>.yml`: the conda packages (python, libraries), `conda env create -p .envs/<name>`.
- `<name>.pip.txt`: every pip package at its exact version, installed with `pip install --no-deps -r`.
  `--no-deps` is deliberate: the pins already contain every dependency, and some exported sets do not
  resolve under pip's resolver (`alayaworld`: diffusers and transformers disagree on huggingface-hub),
  which the original envs tolerated because they were built step by step.

The pin files carry what a plain PyPI install cannot: the PyTorch index lines (`--extra-index-url
https://download.pytorch.org/whl/cu126` etc.), the prebuilt flash-attn wheel URLs (`alayaworld`,
`gen-wan22`; a plain install builds from source), OpenAI's CLIP by commit (`wbench-main`), and the
NATTEN wheel list (`--find-links https://whl.natten.org`, wheels hosted on SHI-Labs/NATTEN's GitHub releases; `gen-ltx25`).
The script then does what a pin file cannot:

- `autoresearcher`: `pip install --no-deps -e .` (the kernel itself).
- `gen-wan22`: clones `third_party/Wan2.2` at `generators.wan22.commit`. `Wan22Backend` refuses another commit.
- `gen-ltx25`: clones `third_party/LTX-2` at `generators.ltx25.commit` and installs `ltx-core` and `ltx-pipelines` editable.
  `pip check` reports one conflict by design: torch 2.13.0+cu132 asks for `nvidia-cudnn-cu13==9.20.0.48`, the env has
  9.24.0.43 (LTX-2's own cuDNN override; 9.20 lacks `libcudnn_engines_tensor_ir`). `ltx-kernels` is not installed: it only
  serves multi-GPU sequence parallelism and NVFP4 (Blackwell).
- `wbench-main`: builds MegaSAM's CUDA extensions (`lietorch`, `droid_backends`) from `../WBench/third_party/mega-sam/base`
  with `setup.py install`; needs `nvcc` on PATH (CUDA 12.x). `import torch` must come before `import droid_backends`.

**Verified 2026-09-28:** all nine rebuilt in a fresh clone of the three repos with an empty package cache; each env's
`pip freeze` is identical to the working env's (`gen-wan22` differs only in how flash-attn and eva-decord are spelled).

`ar_kernel.subproc.conda_command()` picks how an env value is run: a value containing "/" (`".envs/gen-zimage"`) is a
prefix path, repo-relative unless absolute (`conda run -p`); a plain name (`"alayaworld"`) runs `conda run -p
.envs/alayaworld` when that folder is a conda env (`project_env()`), otherwise `conda run -n <name>`. `ar doctor` reports
`ok` for an env in `.envs/`, `warn` for a named env outside the project, `fail` for a missing one.

**Refreshing the pin files after changing an env:** `conda env export -p .envs/<name> --no-builds` gives the conda part;
`.envs/<name>/bin/pip freeze` the pip part (drop the `@ file:` lines, keep the index lines that were added by hand).

**Weights** for the three generators, from the pins above: `Tongyi-MAI/Z-Image-Turbo` at
`f332072aa78be7aecdf3ee76d5c247082da564a6` (`--exclude "assets/*"`, 31 GB), `Wan-AI/Wan2.2-TI2V-5B` (32 GB) and
`Lightricks/LTX-2.5` (only the six files `bridges/ltx25_generate.py` opens, about 80 GB of the 188 GB repo). Kernel launches
of Wan and LTX set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the 24 GB card otherwise runs out of memory in the VAE decode).

## What needs doing on a new machine

1. **Create the conda environments in `.envs/`** with `scripts/setup_envs.sh` (see
   "Environments" above). Never use `base` or the system Python. `ar doctor` lists any env that is missing or lives outside `.envs/`.
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
   refuses to start rather than silently scoring on fewer metrics. *(2026-09-27, Plan 4 as
   built)* `ar run` for a **new** run additionally requires `OPENAI_API_KEY` and
   `OPENAI_MODEL` to be non-empty, checked before the run directory is created; `ar
   init-run`, `ar run --resume` and `ar status` do not need them until an agent phase
   actually runs. With `--git-remote URL`, the shell that runs `ar run` needs git access to
   URL without a prompt (e.g. an SSH key loaded in `ssh-agent`); pushes never prompt.
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
