#!/usr/bin/env bash
# Builds the project's conda envs into AutoResearcher/.envs/<name>.
#   scripts/setup_envs.sh              # all eleven, in order
#   scripts/setup_envs.sh vllm alayaworld   # only these
# Candidate generator backends, built only when named: gen-vllm-omni gen-sglang
# gen-sglang compiles kernels on first use; launch it with `conda run -p .envs/gen-sglang` and
#   CUDA_HOME=<env> CPATH=<env>/targets/x86_64-linux/include SGLANG_CACHE_DIR=.cache/sglang
#   LIBRARY_PATH=<env>/targets/x86_64-linux/lib:<env>/targets/x86_64-linux/lib/stubs
# An env that already exists is skipped; delete .envs/<name> to rebuild it.
# Needs: conda, git and an NVIDIA driver; no CUDA toolkit (the envs that compile bring their own nvcc).
# CUDA code is compiled for the detected card only: RTX 4090 (8.9), A100 (8.0) or H100 (9.0), one kind per machine.
# Run from anywhere; run one copy at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/tmp third_party
# TMPDIR: the same short symlink to .cache/tmp as ar_kernel.subproc.tmp_link (Unix socket paths are capped at 107 bytes)
TMP_TARGET=$(cd .cache/tmp && pwd -P)
TMPDIR=/tmp/ar-$(id -un)-$(printf %s "$TMP_TARGET" | sha1sum | cut -c1-8)
[ -L "$TMPDIR" ] || ln -s "$TMP_TARGET" "$TMPDIR"
[ "$(readlink -f "$TMPDIR")" = "$TMP_TARGET" ] || { echo "$TMPDIR is not a symlink to $TMP_TARGET: remove it" >&2; exit 1; }
export CONDA_PKGS_DIRS=$PWD/.cache/conda/pkgs PIP_CACHE_DIR=$PWD/.cache/pip TMPDIR CONDA_REGISTER_ENVS=false
ROOT=$PWD

ALL="autoresearcher bwrap agent vllm alayaworld wbench-main wbench-vp gen-zimage gen-wan22 gen-ltx25 gen-lightx2v"
run() { conda run --no-capture-output -p ".envs/$1" "${@:2}"; }
# `configs/kernel.yaml` pins the Wan2.2, LTX-2 and LightX2V revisions; read them with the autoresearcher env's yaml
pin() { .envs/autoresearcher/bin/python -c "import yaml; print(yaml.safe_load(open('configs/kernel.yaml'))['generators']['$1']['commit'])"; }

cuda_arch() {   # the cards' compute capability, for TORCH_CUDA_ARCH_LIST
  local a
  a=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)
  case $a in
    8.0|8.9|9.0) echo "$a" ;;
    *) echo "unsupported or mixed GPUs (compute capability: $(echo $a)); supported: 8.0, 8.9, 9.0" >&2; return 1 ;;
  esac
}

cuda_run() {   # env command...: runs command in env, compiling CUDA with the env's own nvcc and g++ for this card
  local e=$ROOT/.envs/$1 arch
  arch=$(cuda_arch) || return 1
  PATH=$e/bin:$PATH CC=$e/bin/x86_64-conda-linux-gnu-gcc CXX=$e/bin/x86_64-conda-linux-gnu-g++ CUDA_HOME=$e \
    CPATH=$e/targets/x86_64-linux/include LIBRARY_PATH=$e/targets/x86_64-linux/lib TORCH_CUDA_ARCH_LIST=$arch \
    conda run --no-capture-output -p "$e" "${@:2}"
}

clone_pinned() {   # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}

build() {
  local n=$1
  if [ -d ".envs/$n" ]; then echo "== $n: exists, skipped"; return; fi
  echo "== $n"
  conda env create -y -q -p ".envs/$n" -f "envs/$n.yml"
  [ ! -f "envs/$n.pip.txt" ] || run "$n" pip install -q --no-deps -r "envs/$n.pip.txt"   # exact pins; --no-deps because the pins already contain every dependency
  case $n in
    autoresearcher) run "$n" pip install -q --no-deps -e . ;;
    gen-wan22) clone_pinned https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2 "$(pin wan22)" ;;
    gen-ltx25)
      clone_pinned https://github.com/Lightricks/LTX-2.git third_party/LTX-2 "$(pin ltx25)"
      run "$n" pip install -q --no-deps -e third_party/LTX-2/packages/ltx-core -e third_party/LTX-2/packages/ltx-pipelines ;;
    gen-lightx2v)   # LightX2V plus SageAttention (and FlashAttention 3 on H100), compiled with the env's own g++ 13 and CUDA 13.0
      clone_pinned https://github.com/ModelTC/LightX2V.git third_party/LightX2V "$(pin h3)"
      run "$n" pip install -q --no-deps -e third_party/LightX2V
      clone_pinned https://github.com/thu-ml/SageAttention.git third_party/SageAttention d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
      sed -i 's/-std=c++17/-std=c++20/g' third_party/SageAttention/setup.py   # torch 2.14 headers need C++20
      rm -rf third_party/SageAttention/build   # objects from an earlier build would be linked unchanged
      cuda_run "$n" pip install -q --no-deps --no-build-isolation third_party/SageAttention
      if [ "$(cuda_arch)" = 9.0 ]; then   # FlashAttention 3: rollout_h3's attention on H100 (configs/h100.yaml)
        clone_pinned https://github.com/Dao-AILab/flash-attention.git third_party/flash-attention bb25ba3ce4b1c8df5d4ef6a94807f0d3910599ca
        git -C third_party/flash-attention submodule update --init csrc/cutlass
        rm -rf third_party/flash-attention/hopper/build
        # only H3's case (forward, bf16/fp16, head dim 128); MAX_JOBS bounds nvcc's host RAM
        MAX_JOBS=16 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
          FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
          FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
          FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE cuda_run "$n" pip install -q --no-deps --no-build-isolation third_party/flash-attention/hopper
      fi ;;
    wbench-main)   # MegaSAM's CUDA extensions (lietorch, droid_backends), from a copy whose fixed arch list is removed
      rm -rf .cache/tmp/mega-sam
      cp -r ../WBench/third_party/mega-sam/base .cache/tmp/mega-sam
      sed -i "/-gencode=arch=/d" .cache/tmp/mega-sam/setup.py
      (cd .cache/tmp/mega-sam && cuda_run "$n" python setup.py install)
      rm -rf .cache/tmp/mega-sam ;;
  esac
}

for n in ${@:-$ALL}; do build "$n"; done
echo "envs done: $(ls .envs | tr '\n' ' ')"
