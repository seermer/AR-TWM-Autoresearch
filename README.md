# AutoResearcher: setup and run guide

Follow the steps in order, in one terminal, copying the commands as written.
Every step ends with a check; do not continue if it fails.

## 0. What you need

| | |
|---|---|
| OS | Linux x86_64 |
| GPUs | 4 NVIDIA cards of one model, at least 24 GB each: RTX 4090, A100 or H100 (tested: RTX 4090), all idle. A run takes the first `gpus.num_gpus` idle cards (`configs/kernel.yaml`: 4, a multiple of 4). Or 8 H100 80 GB cards with `configs/h100.yaml` (every step below names it where it applies) |
| Driver | `nvidia-smi` shows `CUDA Version: 13.2` or higher |
| RAM | 128 GB minimum, 256 GB recommended; 8 x H100: 768 GB, and 128 GB of `/dev/shm` |
| Disk | 850 GB free on one disk (downloads about 600 GB); 8 x H100: 1.5 TB (downloads about 900 GB) |
| Software | `conda` (Miniforge), `git`, `ffmpeg`, `bwrap` (bubblewrap, with unprivileged user namespaces allowed: agents run in bwrap sandboxes), and `curl`, `wget`, `unzip`, `gcc` (the sandboxes use them from the host) |
| Accounts | GitHub (SSH key), Hugging Face, an LLM API key (see step 3) |

Check:

```bash
nvidia-smi
conda --version && git --version && ffmpeg -version | head -1 && which curl wget unzip gcc
bwrap --unshare-user --ro-bind / / true && echo bwrap ok
ssh -T git@github.com          # must say "Hi <your-user>!"
df -h .
```

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

Ten conda environments, all created inside `AutoResearcher/.envs/`. Takes 30 to 60 minutes and about 62 GB.

```bash
scripts/setup_envs.sh
```

Check: `ls .envs` lists `agent alayaworld autoresearcher gen-lightx2v gen-ltx25 gen-wan22 gen-zimage vllm wbench-main wbench-vp`.
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

# Optional. With a key, the six VLM metrics are answered by that API (VLM_API_URL, VLM_MODEL_NAME).
# Empty: a local server of the captioner's model answers them instead (step 4). Both work, but
# scores from the two judges are not comparable, and a run refuses to resume with the other one.
VLM_API_KEY=

# Login for the monitoring panel (step 8)
PANEL_USER=<choose a user name>
PANEL_PASSWORD=<choose a password>

# Optional. A Hugging Face token with write access: after every phase the run uploads itself to a private
# dataset repo <token's user>/<run id> (step 8), and agents use it for gated repos. Empty: no upload.
HFK=
EOF
chmod 600 .env
ln -s ../AutoResearcher/.env ../WorldModel/.env
ln -s ../AutoResearcher/.env ../WBench/.env
```

Check: `ls -l ../WorldModel/.env ../WBench/.env` both show `-> ../AutoResearcher/.env`.

Optional model context size: `agents.context_window_tokens` in `configs/kernel.yaml` is set to 1000000 tokens, and the agent compacts its conversation at `agents.compact_at` (0.6) of it. Lower it if your model's context window is smaller.

## 4. Download weights and data (about 600 GB; 8 x H100: about 900 GB)

The script downloads what the config needs: the AlayaWorld weights, WBench's data and metric weights,
the tools' models (generators switched off in the config are skipped) and the captioner's model
(Qwen3.8-27B-FP8; with `configs/h100.yaml`, Qwen3.8-Flash-Next in bf16, into `.cache/huggingface`).
It can be re-run: what it finished is skipped, and an interrupted download resumes.

```bash
python scripts/download_weights.py                              # 4 x 24 GB
python scripts/download_weights.py --config configs/h100.yaml   # 8 x H100
python scripts/download_weights.py --check [--config ...]       # lists anything missing
```

Then, once on one GPU, build rollout_h3's AdaLN cache (again after changing infer_steps in configs/h3_lightx2v.json):

```bash
CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 conda run -p .envs/gen-lightx2v python \
  third_party/LightX2V/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py --model_path weights/minimax-h3 \
  --config_json configs/h3_lightx2v.json --model-variant fl2av       # writes weights/minimax-h3-adaln
```

`401` or `403` errors mean the token is wrong or a licence in step 3 was not accepted.

## 5. Build the text cache (optional, about 4 minutes)

Every evaluation first encodes the prompts of its cases that are not in the cache yet, so this step only
saves the first evaluation that time.

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
ar doctor --gpu                 # must end with "0 failed"
nvidia-smi --query-gpu=index,memory.used --format=csv     # every GPU near 0 MiB
pytest tests -q                 # optional: unit tests, no GPU needed, about 12 minutes
```

`ar doctor` names anything missing (weights, environments, broken links); `--gpu` also runs a small GPU op in every environment (cards, driver, compiled packages). Fix what it names and run it again.

On a machine the setup has not run on before (the first 8 x H100 run, say), also run the riskiest GPU paths
once before starting a real run: one captioner server answering caption, locate_events and judge requests, a
few training steps, the render of the longest proxy cases, and LTX-2.5 at its longest clip. It needs the three
repos committed; `--steps` also runs the other tools (images, annotate, alayaworld, h3).

```bash
python scripts/preflight.py --config configs/h100.yaml     # must end with "all 6 steps passed"
```

Each step prints its peak GPU memory per card and the lowest free host memory (`runs/preflight-<time>/preflight.json`).

## 7. Start the experiment

1. Choose a run name you have not used before (`ar` refuses to overwrite one), and the number of new nodes to try. Commit any changes in the three repos first: a new run refuses to start with uncommitted changes, so that its recorded commits fully name the code it ran.
2. Start it in the background, so it survives closing the terminal:

```bash
export RUN=exp1                                  # your run name
export REMOTE=git@github.com:<you>/AR-run-1.git  # the empty repo from step 3
nohup ar run --run-id $RUN --max-nodes 6 --git-remote $REMOTE \
  > runs/$RUN.console.log 2>&1 &
```

The run takes the first `gpus.num_gpus` cards (`configs/kernel.yaml`, 4) that have no memory in use, and refuses to start while `CUDA_VISIBLE_DEVICES` is set.
On 8 x H100 add `--config configs/h100.yaml` (eight cards, the captioner and local judge on Qwen3.8-Flash-Next, the
80 GB training recipe `configs/base_recipe_h100.yaml` and render settings); a skill config goes after it,
e.g. `--config configs/h100.yaml configs/rain_start.yaml`. The files apply in that order.

The loop first measures the starting model (the "root", about 2 hours), then improves it node by node
(about 9-14 hours per node measured on 4 GPUs). The root is measured once: it is kept in `root_cache/`
and reused by every later run with the same cases, metrics, judge and WorldModel/WBench commits. A run
scores the first `eval.proxy_size` (50) cases of `configs/proxy_cases.txt`, and the score is a weighted
mean of the metrics those cases list, all 22 here (`eval.score_weights` in `configs/kernel.yaml`).

To score another benchmark, add `--config <file>` to `ar run`. The file holds only the `kernel.yaml`
keys that differ, and the run keeps it, so `--resume` takes no `--config`. Two are provided, one per
skill benchmark (built by `../SkillWBench`, kept in `../WBench/data_<benchmark>`):

| Config | Scores | Cases | Skill metric |
|---|---|---|---|
| `configs/rain_start.yaml` | rain starts | 60 | `event_edit_adherence`, half the score |
| `configs/animal_appears.yaml` | a land animal appears | 60 | `event_edit_adherence`, half the score |

Each sets the benchmark folder (`eval.data`), its case list (`eval.cases`, `eval.proxy_size`), the
skill's weight, and how the data roles are told the skill the run aims to improve (`eval.target_skill`). The
other 16 metrics these cases list share the other half; a listed metric missing from an eval fails it.

Run only one loop at a time on a machine.

## 8. Monitor

```bash
ar status --run-id $RUN                 # nodes, scores, spend, alerts
tail -f runs/$RUN.console.log           # raw log
nvidia-smi                              # GPU use
```

Web panel (every conversation, tool call, code edit, training curve, video and score), on a second terminal:

```bash
.envs/autoresearcher/bin/python -m panel --run-id $RUN --no-share --port 7860
```

Open http://127.0.0.1:7860 and log in with `PANEL_USER` / `PANEL_PASSWORD`.
From another computer: `ssh -L 7860:127.0.0.1:7860 <user>@<server>`, then open the same address locally.

With `HFK` set, another machine with this setup (step 1, the `autoresearcher` env and `.env`) can download the run
and open the panel on it; run it again to catch up with later phases:

```bash
ar fetch --run-id $RUN     # into runs/$RUN, from <HFK's user>/$RUN
```

## 9. Stop and resume

```bash
ar stop --run-id $RUN                   # stops after the current node finishes (can take hours)
ar stop --run-id $RUN --force           # stops now, waits until the loop has exited
nohup ar run --run-id $RUN --resume >> runs/$RUN.console.log 2>&1 &
```

Resume keeps every finished node; a node that was still in progress starts again at the phase it stopped in (that phase from scratch, with the same attempt number).
If WorldModel, WBench or AutoResearcher changed since the run started, resume prints a warning and records a `code_changed` alert: later nodes are then not built and scored like the earlier ones.
Do not delete files under `runs/$RUN` by hand.

## 10. Results

- `ar status --run-id $RUN`: every node with its score and the best one.
- `ar score-node --run-id $RUN --node <id>`: scores a node of the run again. It reads the run and writes only under `scores/`, and is refused while a loop is running (the loop holds the GPUs); besides `ar run` and `ar stop`, no command changes a run.
- Code of each node: the branches `$RUN/node/<id>` in the GitHub repo from `$REMOTE`.
- Everything else (logs, conversations, checkpoints, videos): `runs/$RUN/`. Keep this folder; it is the record of the experiment.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is empty` | Fill `.env` (step 3). Run `ar` from `AutoResearcher`. |
| `run ... exists; use --resume` | Use a new `--run-id`, or add `--resume` to continue it. |
| `need 4 free GPUs` / GPU out of memory at start | Fewer idle cards than `gpus.num_gpus`. Check `nvidia-smi` for other processes. |
| `unset CUDA_VISIBLE_DEVICES` | The run picks its own cards: `unset CUDA_VISIBLE_DEVICES` in that shell. |
| `phase_failed` alert in `ar status`, and the run stopped | A phase failed twice for a reason that is not the agents' (machine, kernel or LLM provider); the alert carries the error. Fix it, then `--resume`: the phase starts again. |
| `node_faults` alert in `ar status`, and the run stopped | Two nodes in a row ended because their training or evaluation failed twice (`train_failed` / `eval_failed`); the alert carries the last error. Fix it, then `--resume`. |
| `enabled tools do not fit GPUs` | A tool needs more GPU memory in total than the run's cards have (`min_total_gpu_gib` in `configs/kernel.yaml`). Give the run more or larger GPUs, or switch that generator off before starting a new run. |
| `tool_does_not_fit` alert in `ar status`, and the run stopped | A GPU tool found the machine too small while running; the alert names the worker log. Give the run more or larger GPUs, then `--resume`. |
| `permission denied (publickey)` for the git remote | `ssh-add` your key in the shell that runs `ar`; check with `ssh -T git@github.com`. It shows up as a warning in `ar status`. |
| `ar doctor` fails at `bwrap` | Install bubblewrap; if a sandbox does not start, an administrator must allow unprivileged user namespaces. |
| Hugging Face `401` / `403` | Re-run `hf auth login` and accept the licences in step 3. |
| `ar doctor` reports missing weights or a broken link | Re-run the download command for that folder (step 4). |
| Evaluation fails with a missing text embedding | Redo step 5. |
| Env build stops with `unsupported or mixed GPUs` | Only one kind of RTX 4090, A100 or H100 per machine is supported. |
| `ar doctor --gpu` says torch is built for a newer CUDA than the driver supports | Update the NVIDIA driver (step 0). |
| `ar doctor --gpu` reports `no kernel image` for an env | It was built on another kind of card: delete `.envs/<name>` and run `scripts/setup_envs.sh <name>`. |
| Disk fills up | Alerts start below 50 GB free. Free space, then `--resume`. |
