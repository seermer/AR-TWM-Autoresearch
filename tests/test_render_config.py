import yaml
from ar_kernel.config import KernelConfig
from ar_kernel.eval.render import build_render_config

CFG = KernelConfig.load()
STUDENT_RANK = yaml.safe_load((CFG.worldmodel / "configs" / "wbench_full.yaml").read_text())["lora"]["rank"]


def _config(tmp_path, eval_lora, node_rank):
    path = build_render_config(CFG, eval_lora, node_rank, tmp_path / "history_encoder.pt",
                               tmp_path / "videos", ["1", "2"], tmp_path / "node")
    return yaml.safe_load(path.read_text())


def test_a_node_renders_the_released_base_with_the_concatenated_adapter(tmp_path):
    config = _config(tmp_path, tmp_path / "node" / "eval" / "lora", 64)
    assert config["paths"]["resume_checkpoint"] == str(CFG.worldmodel / "weights/alaya-world-ar")
    assert config["paths"]["dmd_resume"] == str(tmp_path / "node" / "eval" / "lora")
    assert config["lora"]["rank"] == config["lora"]["alpha"] == STUDENT_RANK + 64


def test_the_root_renders_the_released_model(tmp_path):
    config = _config(tmp_path, None, 0)
    assert config["paths"]["dmd_resume"] == str(CFG.worldmodel / "weights/alaya-world-dmd")
    assert config["lora"]["rank"] == STUDENT_RANK


def test_the_cases_prompts_are_encoded_before_the_render(tmp_path, monkeypatch):
    import types
    from ar_kernel.eval import render
    from ar_kernel.telemetry.recorder import Recorder
    calls = []

    def fake(env, args, **kwargs):
        calls.append((args, kwargs.get("extra_env")))
        if "scripts/tools/run_wbench.py" in args:
            for i in ("1", "2"):
                (tmp_path / "videos" / f"case_{i}_combined.mp4").touch()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(render, "run_in_env", fake)
    path = build_render_config(CFG, None, 0, tmp_path / "history_encoder.pt", tmp_path / "videos", ["1", "2"],
                               tmp_path / "node")
    render.render_proxy(CFG, path, [0, 1, 2, 3], "root", Recorder(tmp_path / "run"), ["1", "2"])
    (precache, env), (rollout, _) = calls
    assert precache[1:3] == ["-m", "scripts.tools.precache_wbench_text_embeds"] and str(path) in precache
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and "scripts/tools/run_wbench.py" in rollout
