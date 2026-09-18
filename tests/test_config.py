import pytest
from ar_kernel.config import KernelConfig, GpuPolicyError, resolve_gpus

def test_loads_defaults_and_resolves_paths():
    cfg = KernelConfig.load()
    assert cfg.get("gpus.min_count") == 4
    assert cfg.worldmodel.is_dir() and cfg.wbench.is_dir()
    assert cfg.get("train.lora_allowlist") == [[16, 16], [32, 32], [64, 64]]

def test_gpu_list_defaults_when_env_unset():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {}) == [0, 1, 2, 3]

def test_gpu_list_uses_env_verbatim_including_gpu5():
    cfg = KernelConfig.load()
    assert resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,2,4,5"}) == [0, 2, 4, 5]

def test_gpu_list_below_min_count_is_refused():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="at least 4"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2"})

def test_gpu_list_rejects_non_integer_entries():
    cfg = KernelConfig.load()
    with pytest.raises(GpuPolicyError, match="not an integer"):
        resolve_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1,2,gpu3"})
