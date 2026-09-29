# AutoResearcher: setup and run guide

Follow the steps in order, in one terminal, copying the commands as written.
Every step ends with a check; do not continue if it fails.

## 0. What you need

| | |
|---|---|
| OS | Linux x86_64 |
| GPUs | 4 or more NVIDIA 24 GB cards (tested: RTX 4090), all idle |
| Driver | `nvidia-smi` shows `CUDA Version: 13.2` or higher |
| CUDA toolkit | `nvcc --version` works and says 12.x (usually `/usr/local/cuda-12.8`) |
| RAM | 128 GB minimum, 256 GB recommended |
| Disk | 600 GB free on one disk (downloads about 400 GB) |
| Software | `conda` (Miniforge), `git`, `ffmpeg`, `docker` (your user can run `docker ps` without `sudo`) |
| Accounts | GitHub (SSH key), Hugging Face, an LLM API key (see step 3) |

Check:

```bash
nvidia-smi
nvcc --version
conda --version && git --version && ffmpeg -version | head -1 && docker ps
ssh -T git@github.com          # must say "Hi <your-user>!"
df -h .
```

If `nvcc` is not found: `export PATH=/usr/local/cuda-12.8/bin:$PATH` (add it to `~/.bashrc`).

## 1. Get the code

The three folders must sit side by side, with exactly these names.

```bash
mkdir WM-AutoResearch && cd WM-AutoResearch
git clone git@github.com:seermer/AR-TWM-Autoresearch.git AutoResearcher
git clone git@github.com:seermer/AlayaWorld-TWM-Autoresearch.git WorldModel
git clone --recurse-submodules git@github.com:seermer/WBench-TWM-Autoresearch.git WBench
git clone https://github.com/aigc3d/ViGeo WorldModel/third_party/ViGeo
git -C WorldModel/third_party/ViGeo checkout 87ee8fb
cd AutoResearcher
```

All later commands run in `WM-AutoResearch/AutoResearcher`.
Check: `ls ../WBench/third_party/mega-sam/base/setup.py` prints the path.

## 2. Create the environments

Nine conda environments, all created inside `AutoResearcher/.envs/`. Takes 30 to 60 minutes and about 50 GB.

```bash
scripts/setup_envs.sh
```

Check: `ls .envs` lists `alayaworld autoresearcher gen-ltx25 gen-wan22 gen-zimage panel vllm wbench-main wbench-vp`.
If it stops on an error, fix the cause and run it again: environments that already exist are skipped.
To rebuild one, delete `.envs/<name>` and run `scripts/setup_envs.sh <name>`.

From here on, every terminal starts with:

```bash
cd WM-AutoResearch/AutoResearcher
conda activate $PWD/.envs/autoresearcher
export HF_HOME=$PWD/.cache/huggingface
```

## 3. Keys and `.env`

| What | Where to get it | Used for |
|---|---|---|
| LLM API key | your provider (OpenAI, OpenRouter, DeepSeek, ...) | the agents that improve the model. Must speak the OpenAI Chat Completions API |
| Hugging Face token | https://huggingface.co/settings/tokens (type: Read) | downloading weights |
| GitHub repo for the run | create an empty private repo, e.g. `<you>/AR-run-1` | backup of every agent branch |

Accept the licences on these Hugging Face pages while logged in (gemma is approved manually and can take a while):
- https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
- https://huggingface.co/Lightricks/LTX-2.5

Log in once:

```bash
hf auth login            # paste the Hugging Face token
```

Create `.env` (never commit it) and the two links the other repos need:

```bash
cat > .env <<'EOF'
# Agent LLM (required)
OPENAI_API_KEY=<your LLM API key>
OPENAI_BASE_URL=<provider base URL, e.g. https://api.openai.com/v1>
OPENAI_MODEL=<model name as the provider spells it>
OPENAI_EFFORT=<reasoning effort the provider accepts, e.g. high; empty = not set>

# Leave empty: the six VLM metrics are switched off on purpose so scores stay comparable
VLM_API_KEY=

# Login for the monitoring panel (step 8)
PANEL_USER=<choose a user name>
PANEL_PASSWORD=<choose a password>
EOF
chmod 600 .env
ln -s ../AutoResearcher/.env ../WorldModel/.env
ln -s ../AutoResearcher/.env ../WBench/.env
```

Check: `ls -l ../WorldModel/.env ../WBench/.env` both show `-> ../AutoResearcher/.env`.

Optional model context size: if your model's context window is not 128000 tokens, edit `agents.context_window_tokens` in `configs/kernel.yaml`.

## 4. Download weights and data (about 400 GB)

Run all of it, from `AutoResearcher`. Each command can be re-run; it skips finished files.

```bash
# AlayaWorld (the model being improved) and its base model
hf download AlayaLab/AlayaWorld-v1.1-stage2b --local-dir ../WorldModel/weights/alaya-world-ar
hf download AlayaLab/AlayaWorld-v1.1-stage3 --local-dir ../WorldModel/weights/alaya-world-dmd
hf download Lightricks/LTX-2.3 --include "ltx-2.3-22b-dev.safetensors" --local-dir ../WorldModel/weights/ltx-2.3
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir ../WorldModel/weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized
hf download pkqbajng/ViGeo1.1 --local-dir ../WorldModel/third_party/ViGeo/checkpoints/ViGeo1.1

# WBench (the benchmark): data and metric weights.
# Keep the exclude: the visual-plausibility weights must NOT be present.
hf download meituan-longcat/WBench --repo-type dataset --exclude "splits/*" --local-dir ../WBench/data
hf download meituan-longcat/WBench-weights --exclude "qwen3vl*" --local-dir ../WBench/weights

# Tools the agents can use
hf download Tongyi-MAI/Z-Image-Turbo --revision f332072aa78be7aecdf3ee76d5c247082da564a6 \
  --exclude "assets/*" --local-dir weights/z-image-turbo
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/wan2.2-ti2v-5b
hf download Lightricks/LTX-2.5 --local-dir weights/ltx-2.5 \
  --include "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
            "vae/ltx-2.5-video-vae-bf16.safetensors" \
            "vae/ltx-2.5-audio-vae-bf16.safetensors" \
            "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
            "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

# Video captioning model (lands in .cache/huggingface)
hf download Qwen/Qwen3.8-27B-FP8
```

`401` or `403` errors mean the token is wrong or a licence in step 3 was not accepted.

## 5. Build the text cache (once, about 4 minutes)

The benchmark prompts are encoded once and reused by every evaluation.

```bash
cd ../WorldModel
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
  ../AutoResearcher/.envs/alayaworld/bin/python -m scripts.tools.precache_wbench_text_embeds \
  --config configs/wbench_full.yaml --device-map auto
cd ../AutoResearcher
```

Check: the last line says `[Precache] done` and `ls ../WorldModel/cache/text_embed_wbench | wc -l` prints `245`.

## 6. Pre-flight checks

```bash
ar doctor                       # must end with "0 failed"
docker run --rm python:3.12-slim true && echo docker ok
nvidia-smi --query-gpu=index,memory.used --format=csv     # every GPU near 0 MiB
pytest tests -q                 # optional: unit tests, no GPU needed
```

`ar doctor` names anything missing (weights, environments, broken links). Fix what it names and run it again.

## 7. Start the experiment

1. Choose a run name you have not used before (`ar` refuses to overwrite one), and the number of new nodes to try.
2. Start it in the background, so it survives closing the terminal:

```bash
export RUN=exp1                                  # your run name
export REMOTE=git@github.com:<you>/AR-run-1.git  # the empty repo from step 3
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ar run --run-id $RUN --max-nodes 6 --git-remote $REMOTE \
  > runs/$RUN.console.log 2>&1 &
```

The loop first measures the starting model (the "root", about 2 hours), then improves it node by node
(about 3 hours per node measured on 4 GPUs). The root score should be close to `0.787`.

Run only one loop at a time on a machine.

## 8. Monitor

```bash
ar status --run-id $RUN                 # nodes, scores, spend, alerts
tail -f runs/$RUN.console.log           # raw log
nvidia-smi                              # GPU use
```

Web panel (every conversation, tool call, code edit, training curve, video and score), on a second terminal:

```bash
.envs/panel/bin/python -m panel --run-id $RUN --no-share --port 7860
```

Open http://127.0.0.1:7860 and log in with `PANEL_USER` / `PANEL_PASSWORD`.
From another computer: `ssh -L 7860:127.0.0.1:7860 <user>@<server>`, then open the same address locally.

## 9. Stop and resume

```bash
ar stop --run-id $RUN                   # stops after the current step finishes
ar stop --run-id $RUN --force           # stops now, waits until the loop has exited
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ar run --run-id $RUN --resume >> runs/$RUN.console.log 2>&1 &
```

Resume keeps every finished node; a node that was still in progress is dropped and a new one is started.
Do not delete files under `runs/$RUN` by hand.

## 10. Results

- `ar status --run-id $RUN`: every node with its score and the best one.
- Code of each node: the branches `$RUN/node/<id>` in the GitHub repo from `$REMOTE`.
- Everything else (logs, conversations, checkpoints, videos): `runs/$RUN/`. Keep this folder; it is the record of the experiment.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is empty` | Fill `.env` (step 3). Run `ar` from `AutoResearcher`. |
| `run ... exists; use --resume` | Use a new `--run-id`, or add `--resume` to continue it. |
| `need at least 4 GPUs` / GPU out of memory at start | Set `CUDA_VISIBLE_DEVICES` to 4 or more idle GPUs. Check `nvidia-smi` for other processes. |
| `permission denied (publickey)` for the git remote | `ssh-add` your key in the shell that runs `ar`; check with `ssh -T git@github.com`. It shows up as a warning in `ar status`. |
| `docker: permission denied` | Add your user to the `docker` group, log in again. |
| Hugging Face `401` / `403` | Re-run `hf auth login` and accept the licences in step 3. |
| `ar doctor` reports missing weights or a broken link | Re-run the download command for that folder (step 4). |
| Evaluation fails with a missing text embedding | Redo step 5. |
| Env build fails at MegaSAM (`nvcc not found`) | Fix `PATH` (step 0), delete `.envs/wbench-main`, run `scripts/setup_envs.sh wbench-main`. |
| Disk fills up | Alerts start below 50 GB free. Free space, then `--resume`. |
