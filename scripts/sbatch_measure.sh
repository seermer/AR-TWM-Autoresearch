#!/usr/bin/env bash
#SBATCH --job-name=ar-measure
#SBATCH --account=slurm_grp_vision
#SBATCH --partition=gpu-mid
#SBATCH --qos=slurm_grp_vision_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=768G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/slurm-measure-%j.log
# Measures the GPU tools of configs/h100.yaml for its host-RAM and worker values, with scripts/preflight.py:
#   0. ar doctor --gpu: stops the job when an env or check fails (e.g. gen-lightx2v without FlashAttention 3, which
#      configs/h100.yaml's H3 needs: on an H100 node, delete .envs/gen-lightx2v and run scripts/setup_envs.sh gen-lightx2v)
#   1. the overlay as it is: weights, vllm (captioner), ltx25 (bf16, no offload), images, annotate, h3 (FlashAttention 3)
#   2. ltx25 with CPU offload (4 workers): the output 1 must reproduce
#   3. h3 with its DiT on the GPU (configs/h3_lightx2v_h100.json): the output must reproduce 1's
# Submit from AutoResearcher/ (same environment as scripts/sbatch_run.sh):  sbatch scripts/sbatch_measure.sh
# Results in runs/measure-<job>/: summary.json (per step: seconds, items, GPU peak per card, the job's host memory
# at start and peak; PSNR of each clip against its counterpart, inf = identical) and each preflight's log.
set -euo pipefail
[ -f configs/kernel.yaml ] || { echo "submit from AutoResearcher/ (cwd is $PWD)" >&2; exit 1; }

CONDA_EXE=${CONDA_EXE:-$(type -P conda || true)}
[ -x "$CONDA_EXE" ] || { echo "conda not found: submit with CONDA_EXE=<conda install>/bin/conda" >&2; exit 1; }
set +u
source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
conda activate "$PWD/.envs/autoresearcher"
set -u
export HF_HOME=$PWD/.cache/huggingface
unset CUDA_VISIBLE_DEVICES
ulimit -u "$(ulimit -Hu)"

out=runs/measure-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
cat > "$out/ltx_cpu.yaml" <<'EOF'
generators:
  ltx25: {offload: cpu, workers: 4, peak_rss_gib: 80}
EOF
cat > "$out/h3_gpu.yaml" <<'EOF'
generators:
  h3: {config: configs/h3_lightx2v_h100.json}
EOF

ar doctor --gpu --config configs/h100.yaml

preflight() {   # name, preflight args; records the preflight run's name; a failed step does not stop the job
  local name=$1
  shift
  python scripts/preflight.py "$@" > "$out/$name.log" 2>&1 || true
  grep -m1 -o 'preflight run preflight-[0-9-]*' "$out/$name.log" | cut -d' ' -f3 > "$out/$name.run" || true
}
preflight overlay --config configs/h100.yaml --steps weights vllm ltx25 images annotate h3
preflight ltx_cpu --config configs/h100.yaml "$out/ltx_cpu.yaml" --steps ltx25
preflight h3_gpu --config configs/h100.yaml "$out/h3_gpu.yaml" --steps h3

python - "$out" <<'EOF'
import json, re, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1])
def report(name):
    run = (out / f"{name}.run").read_text().strip() if (out / f"{name}.run").exists() else ""
    path = Path("runs") / run / "preflight.json"
    return json.loads(path.read_text()) if run and path.exists() else {}

def psnr(a, b):
    r = subprocess.run(["ffmpeg", "-i", a, "-i", b, "-lavfi", "psnr", "-f", "null", "-"], capture_output=True, text=True)
    found = re.search(r"average:(\S+)", r.stderr)
    return found.group(1) if found else f"ffmpeg failed: {r.stderr.strip()[-300:]}"

def videos(rep, step):
    return (rep.get(step, {}).get("detail") or {}).get("videos", [])

reports = {name: report(name) for name in ("overlay", "ltx_cpu", "h3_gpu")}
summary = {name: {step: {k: v for k, v in r.items() if k != "traceback"} for step, r in rep.items()}
           for name, rep in reports.items()}
for name, step in (("ltx_cpu", "ltx25"), ("h3_gpu", "h3")):
    a, b = videos(reports["overlay"], step), videos(reports[name], step)
    summary[f"psnr_{name}"] = [psnr(x, y) for x, y in zip(a, b)] if a and len(a) == len(b) else \
        f"not compared: {len(a)} and {len(b)} clips"
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
EOF
