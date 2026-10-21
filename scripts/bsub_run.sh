#!/usr/bin/env bash
# LSF job for one `ar run`; starts it, or resumes it when runs/$RUN exists.
# Submit from AutoResearcher/:
#   RUN=exp1 MAX_NODES=6 REMOTE=git@github.com:<you>/AR-run-1.git bsub -J exp1 < scripts/bsub_run.sh
# Optional: CONFIG=configs/rain_start.yaml (new runs only).
#BSUB -x
#BSUB -gpu "num=4"
#BSUB -o runs/lsf.%J.log
#BSUB -wa TERM
#BSUB -wt 15
##BSUB -q <queue>
##BSUB -W <hours>:00
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
: "${MAX_NODES:?set MAX_NODES}" "${REMOTE:?set REMOTE}"
exec ar run --run-id "$RUN" --max-nodes "$MAX_NODES" --git-remote "$REMOTE" ${CONFIG:+--config "$CONFIG"}
