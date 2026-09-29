from ar_kernel.config import KernelConfig
from ar_kernel.eval.judge import judge_env, resolve_judge

CFG = KernelConfig.load()


def test_api_judge_when_key_present():
    j = resolve_judge(CFG, {"VLM_API_KEY": "k", "VLM_MODEL_NAME": "Doubao-Seed-2.0-lite"})
    assert (j.kind, j.model) == ("api", "Doubao-Seed-2.0-lite")
    assert judge_env(CFG, j, None) == {}


def test_local_judge_when_key_blank():
    j = resolve_judge(CFG, {"VLM_API_KEY": "  "})
    assert j.kind == "local" and j.model == CFG.get("captioner.model")
    env = judge_env(CFG, j, "http://127.0.0.1:8123")
    assert env["VLM_API_URL"] == "http://127.0.0.1:8123/v1/chat/completions"
    assert env["VLM_API_KEY"] == "local" and env["VLM_MODEL_NAME"] == j.model
    assert '"enable_thinking": false' in env["VLM_EXTRA_BODY"]


def test_local_judge_server_allows_images_and_serves_under_the_judge_model(tmp_path):
    from ar_kernel.eval.judge import judge_server
    j = resolve_judge(CFG, {})
    cmd = judge_server(CFG, j, [0, 1, 2, 3], tmp_path).command
    assert cmd[cmd.index("--served-model-name") + 1] == j.model
    assert cmd[cmd.index("--limit-mm-per-prompt") + 1] == '{"image": 64, "video": 1}'
    assert "--allowed-local-media-path" not in cmd
