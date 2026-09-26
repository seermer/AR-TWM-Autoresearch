"""Large caches stay inside the project (user rule 2026-09-25): every kernel subprocess gets
cache variables pointing at AutoResearcher/.cache unless AR_CACHE_DIR relocates it."""
import json
from pathlib import Path

from ar_kernel.config import REPO_ROOT
from ar_kernel.subproc import cache_dir, cache_env, run_in_env

VARS = ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "PIP_CACHE_DIR", "UV_CACHE_DIR")


def test_default_cache_dir_is_inside_the_repo(monkeypatch):
    monkeypatch.delenv("AR_CACHE_DIR", raising=False)
    assert cache_dir() == REPO_ROOT / ".cache"
    env = cache_env()
    assert set(env) == set(VARS)
    assert all(Path(v).is_relative_to(REPO_ROOT / ".cache") for v in env.values())
    assert env["HF_HOME"] == str(REPO_ROOT / ".cache" / "huggingface")


def test_ar_cache_dir_relocates(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    assert cache_dir() == tmp_path
    assert cache_env()["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm")


def test_children_get_cache_env_and_extra_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/else")          # a user-level setting must not leak through
    code = "import json, os; print(json.dumps({k: os.environ.get(k) for k in ('HF_HOME', 'TORCH_HOME')}))"
    proc = run_in_env("autoresearcher", ["python", "-c", code], cwd=tmp_path,
                      extra_env={"TORCH_HOME": str(tmp_path / "mine")})
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen == {"HF_HOME": str(tmp_path / "huggingface"), "TORCH_HOME": str(tmp_path / "mine")}
