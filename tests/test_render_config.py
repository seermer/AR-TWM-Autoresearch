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
