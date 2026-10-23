import importlib.util
from pathlib import Path

from ar_kernel.config import KernelConfig

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_weights", REPO / "scripts" / "download_weights.py")
dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)


def test_each_config_downloads_its_own_captioner_and_skips_switched_off_generators():
    names = lambda cfg: {e["name"]: e["repo"] for e in dw.entries(cfg)}
    base, h100 = names(KernelConfig.load()), names(KernelConfig.load(overlays=[REPO / "configs" / "h100.yaml"]))
    assert base["captioner"] == "Qwen/Qwen3.8-27B-FP8" and h100["captioner"] == "Qwen/Qwen3.8-Flash-Next"
    assert "wan2.2" not in base and "ltx-2.5" in base


def test_a_finished_download_is_skipped(tmp_path, monkeypatch):
    import huggingface_hub
    raw = KernelConfig.load().raw
    monkeypatch.setattr(dw.KernelConfig, "load", lambda overlays: KernelConfig(raw=raw, repo_root=tmp_path))
    monkeypatch.setattr(dw, "cache_env", lambda: {})
    monkeypatch.setattr(dw, "entries", lambda cfg: [dict(name="a", repo="org/a", dest=tmp_path / "a"),
                                                    dict(name="b", repo="org/b", dest=None)])
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, local_dir=None, **kw: (calls.append(repo), local_dir and local_dir.mkdir()))
    assert dw.main(["--check"]) == 1
    assert dw.main([]) == 0 and calls == ["org/a", "org/b"]
    assert dw.main([]) == 0 and calls == ["org/a", "org/b"]          # nothing downloaded again
    assert dw.main(["--check"]) == 0
    (tmp_path / "a").rmdir()                                         # a marker without its folder is not done
    assert dw.main(["--check"]) == 1
