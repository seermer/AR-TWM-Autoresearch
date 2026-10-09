#!/usr/bin/env bash
# Builds the project's conda envs into AutoResearcher/.envs/<name>.
#   scripts/setup_envs.sh              # all nine, in order
#   scripts/setup_envs.sh vllm panel   # only these
# An env that already exists is skipped; delete .envs/<name> to rebuild it.
# Needs: conda, git, and a CUDA toolkit (nvcc) for wbench-main. Run from anywhere; run one copy at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export CONDA_PKGS_DIRS=$PWD/.cache/conda/pkgs PIP_CACHE_DIR=$PWD/.cache/pip
ROOT=$PWD
mkdir -p .cache third_party

ALL="autoresearcher panel vllm alayaworld wbench-main wbench-vp gen-zimage gen-wan22 gen-ltx25 gen-lightx2v"
run() { conda run --no-capture-output -p ".envs/$1" "${@:2}"; }
# `configs/kernel.yaml` pins the Wan2.2, LTX-2 and LightX2V revisions; read them with the autoresearcher env's yaml
pin() { .envs/autoresearcher/bin/python -c "import yaml; print(yaml.safe_load(open('configs/kernel.yaml'))['generators']['$1']['commit'])"; }

clone_pinned() {   # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}

build() {
  local n=$1
  if [ -d ".envs/$n" ]; then echo "== $n: exists, skipped"; return; fi
  echo "== $n"
  conda env create -y -q -p ".envs/$n" -f "envs/$n.yml"
  run "$n" pip install -q --no-deps -r "envs/$n.pip.txt"   # exact pins; --no-deps because the pins already contain every dependency
  case $n in
    autoresearcher) run "$n" pip install -q --no-deps -e . ;;
    gen-wan22) clone_pinned https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2 "$(pin wan22)" ;;
    gen-ltx25)
      clone_pinned https://github.com/Lightricks/LTX-2.git third_party/LTX-2 "$(pin ltx25)"
      run "$n" pip install -q --no-deps -e third_party/LTX-2/packages/ltx-core -e third_party/LTX-2/packages/ltx-pipelines ;;
    gen-lightx2v)   # LightX2V plus SageAttention, which is compiled with the env's own g++ 13 and CUDA 13.2
      clone_pinned https://github.com/ModelTC/LightX2V.git third_party/LightX2V "$(pin h3)"
      run "$n" pip install -q --no-deps -e third_party/LightX2V
      clone_pinned https://github.com/thu-ml/SageAttention.git third_party/SageAttention d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
      sed -i 's/-std=c++17/-std=c++20/g' third_party/SageAttention/setup.py   # torch 2.14 headers need C++20
      local e=$ROOT/.envs/$n
      PATH=$e/bin:$PATH CC=$e/bin/x86_64-conda-linux-gnu-gcc CXX=$e/bin/x86_64-conda-linux-gnu-g++ CUDA_HOME=$e \
        CPATH=$e/targets/x86_64-linux/include LIBRARY_PATH=$e/targets/x86_64-linux/lib CUDA_ARCHITECTURES=8.9 \
        run "$n" pip install -q --no-deps --no-build-isolation third_party/SageAttention ;;
    wbench-main)   # MegaSAM's CUDA extensions (lietorch, droid_backends)
      (cd ../WBench/third_party/mega-sam/base && LD_LIBRARY_PATH="$ROOT/.envs/$n/lib:${LD_LIBRARY_PATH:-}" \
        conda run --no-capture-output -p "$ROOT/.envs/$n" python setup.py install) ;;
  esac
}

for n in ${@:-$ALL}; do build "$n"; done
echo "envs done: $(ls .envs | tr '\n' ' ')"
