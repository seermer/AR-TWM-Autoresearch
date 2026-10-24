#!/usr/bin/env bash
#SBATCH --job-name=ar-run
#SBATCH --account=slurm_grp_vision
#SBATCH --partition=gpu-mid
#SBATCH --qos=slurm_grp_vision_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=768G
#SBATCH --gres=gpu:h200:8
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm-%j.log
# Slurm job for one `ar run`; starts it, or resumes it when runs/$RUN exists (also after a preemption requeue).
# Submit from AutoResearcher/; options on the command line override the lines above:
#   RUN=exp1 MAX_NODES=6 sbatch -J exp1 scripts/sbatch_run.sh
# CONDA_EXE=<conda install>/bin/conda when conda is only a shell function in the submitting shell.
# Optional: CONFIG="configs/h100.yaml configs/rain_start.yaml" (new runs only; files applied in order).
# SIGTERM stops the run: on preemption the job is requeued and resumes it; 15 min before the time limit,
# resubmit the same RUN to resume it.
# Host for 8 GPUs: >= 128 GB /dev/shm, an NVIDIA driver supporting CUDA >= 13.0, and 1.5 TB disk for the
# whole tree (weights, envs, caches, runs).
set -euo pipefail
[ -f configs/kernel.yaml ] || { echo "submit from AutoResearcher/ (cwd is $PWD)" >&2; exit 1; }
: "${RUN:?set RUN}"

# conda: $CONDA_EXE (exported by an initialized shell), else a conda executable on PATH
CONDA_EXE=${CONDA_EXE:-$(type -P conda || true)}
[ -x "$CONDA_EXE" ] || { echo "conda not found: submit with CONDA_EXE=<conda install>/bin/conda" >&2; exit 1; }
set +u      # conda's shell scripts read unset variables
source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
conda activate "$PWD/.envs/autoresearcher"
set -u
export HF_HOME=$PWD/.cache/huggingface
unset CUDA_VISIBLE_DEVICES

ar doctor --gpu ${CONFIG:+--config $CONFIG}

if [ -f "runs/$RUN/config/run.json" ]; then
  exec ar run --run-id "$RUN" --resume
fi
: "${MAX_NODES:?set MAX_NODES}"
exec ar run --run-id "$RUN" --max-nodes "$MAX_NODES" ${CONFIG:+--config $CONFIG}
