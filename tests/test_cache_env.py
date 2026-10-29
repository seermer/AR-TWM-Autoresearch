"""Large caches stay inside the project: every kernel subprocess gets
cache variables pointing at AutoResearcher/.cache unless AR_CACHE_DIR relocates it."""
import json
import os
from pathlib import Path

import pytest

from ar_kernel import subproc
from ar_kernel.config import REPO_ROOT
from ar_kernel.subproc import cache_dir, cache_env, run_in_env

VARS = ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "PIP_CACHE_DIR", "UV_CACHE_DIR", "VLLM_CONFIG_ROOT",
        "FLASHINFER_WORKSPACE_BASE", "CUPY_CACHE_DIR", "MPLCONFIGDIR", "TMPDIR")


def test_default_cache_dir_is_inside_the_repo(monkeypatch):
    monkeypatch.delenv("AR_CACHE_DIR", raising=False)
    assert cache_dir() == REPO_ROOT / ".cache"
    env = cache_env()
    assert env.pop("VLLM_NO_USAGE_STATS") == "1" and set(env) == set(VARS)
    assert all(Path(v).resolve().is_relative_to(REPO_ROOT / ".cache") for v in env.values())
    assert env["HF_HOME"] == str(REPO_ROOT / ".cache" / "huggingface")
    tmp = Path(env["TMPDIR"])
    assert tmp.parent == Path("/tmp") and len(str(tmp)) <= 30
    assert tmp.is_symlink() and tmp.resolve() == (REPO_ROOT / ".cache" / "tmp").resolve()


@pytest.fixture
def links(monkeypatch, tmp_path):
    """Symlinks for a relocated cache go under tmp_path, not /tmp."""
    monkeypatch.setattr(subproc, "TMP_LINK_DIR", tmp_path / "links")
    (tmp_path / "links").mkdir()
    return tmp_path / "links"


def test_cache_env_creates_tmpdir(monkeypatch, tmp_path, links):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    tmp = Path(cache_env()["TMPDIR"])
    assert tmp.parent == links and tmp.is_dir() and tmp.resolve() == tmp_path / "tmp"
    assert Path(cache_env()["TMPDIR"]) == tmp                 # an existing right link is reused


def test_foreign_path_at_the_link_name_raises(monkeypatch, tmp_path, links):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    name = Path(cache_env()["TMPDIR"]).name
    os.unlink(links / name)
    (links / name).mkdir()
    with pytest.raises(RuntimeError, match="not a symlink"):
        cache_env()


def test_ar_cache_dir_relocates(monkeypatch, tmp_path, links):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    assert cache_dir() == tmp_path
    assert cache_env()["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm")


def test_children_get_cache_env_and_extra_env_wins(monkeypatch, tmp_path, links):
    monkeypatch.setenv("AR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/else")          # a user-level setting must not leak through
    code = "import json, os; print(json.dumps({k: os.environ.get(k) for k in ('HF_HOME', 'TORCH_HOME')}))"
    proc = run_in_env("autoresearcher", ["python", "-c", code], cwd=tmp_path,
                      extra_env={"TORCH_HOME": str(tmp_path / "mine")})
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen == {"HF_HOME": str(tmp_path / "huggingface"), "TORCH_HOME": str(tmp_path / "mine")}


def test_free_port_is_bindable():
    import socket
    from ar_kernel.subproc import free_port
    port = free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
    assert 1024 <= port <= 65535
