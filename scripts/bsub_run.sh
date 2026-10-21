#!/usr/bin/env bash
# LSF job for one `ar run`; starts it, or resumes it when runs/$RUN exists.
# Submit from AutoResearcher/, with the bsub options on the command line:
#   RUN=exp1 MAX_NODES=6 bsub -J exp1 -x -gpu "num=4" -o runs/lsf.%J.log -wa TERM -wt 15 scripts/bsub_run.sh
# Optional: CONFIG=configs/rain_start.yaml (new runs only).
set -euo pipefail
[ -f configs/kernel.yaml ] || { echo "submit from AutoResearcher/ (cwd is $PWD)" >&2; exit 1; }
: "${RUN:?set RUN}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.envs/autoresearcher"
export HF_HOME=$PWD/.cache/huggingface
unset CUDA_VISIBLE_DEVICES

ar doctor --gpu

if [ -f "runs/$RUN/config/run.json" ]; then
  exec ar run --run-id "$RUN" --resume
fi
: "${MAX_NODES:?set MAX_NODES}"
exec ar run --run-id "$RUN" --max-nodes "$MAX_NODES" ${CONFIG:+--config "$CONFIG"}
