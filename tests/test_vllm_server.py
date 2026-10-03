import sys
import time
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.tools.vllm_server import VllmServer, parallelism, serve_command

CFG = KernelConfig.load().get("captioner")

FAKE = """
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if sys.argv[2] == "exit":
    print("CUDA out of memory", flush=True); sys.exit(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


@pytest.mark.parametrize("n, tp, dp", [(2, 2, 1), (4, 4, 1), (8, 8, 1), (6, 4, 1), (1, 1, 1)])
def test_parallelism_follows_the_gpu_list(n, tp, dp):
    assert parallelism(list(range(n))) == (tp, dp)


def test_configured_tensor_parallel_fills_the_rest_with_data_parallel():
    assert parallelism([0, 1, 2, 3], 2) == (2, 2)
    cmd = serve_command({**CFG, "tensor_parallel": 2}, [0, 1, 2, 3], 8000, "judge")
    assert cmd[cmd.index("--data-parallel-size") + 1] == "2"


def test_a_non_power_of_two_count_warns_about_idle_gpus(caplog):
    with caplog.at_level("WARNING"):
        parallelism([0, 1, 2, 3, 4, 5])
    assert "4 of the 6" in caplog.text


def test_serve_command_names_the_model_and_omits_the_media_path_when_none():
    cmd = serve_command(CFG, [0, 1, 2, 3], 8000, "judge-name")
    assert cmd[cmd.index("--served-model-name") + 1] == "judge-name"
    assert "--allowed-local-media-path" not in cmd
    assert "--data-parallel-size" not in cmd


def _server(tmp_path, mode, gpus=(0, 1)):
    from ar_kernel.subproc import free_port
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    port = free_port()
    return VllmServer("autoresearcher", ["python", str(script), str(port), mode], list(gpus), port,
                      cwd=tmp_path, log_path=tmp_path / "s.log", poll_s=0.1)


def test_start_waits_for_health_and_stop_ends_the_process(tmp_path):
    s = _server(tmp_path, "ok")
    load_s = s.start(60)
    assert load_s is not None and load_s > 0
    s.stop()
    import httpx
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{s.base_url}/health", timeout=2)


def test_a_server_that_exits_early_raises_with_the_log_tail(tmp_path):
    s = _server(tmp_path, "exit")
    try:
        with pytest.raises(RuntimeError, match="(?s)exited during startup.*CUDA out of memory"):
            s.start(60)
    finally:
        s.stop()


def test_a_server_that_is_never_ready_times_out(tmp_path):
    s = _server(tmp_path, "ok")
    s.command = [sys.executable, "-c", "import time; time.sleep(600)"]
    try:
        with pytest.raises(RuntimeError, match="not ready within 3 s"):
            s.start(3)
    finally:
        s.stop()


def test_gpus_holding_more_than_the_limit_are_named_and_a_display_server_is_let_through():
    from ar_kernel.tools.vllm_server import require_free_gpus
    require_free_gpus([0, 1], 512, gpu_memory=lambda gpus: {0: 9, 1: 300})          # e.g. Xorg
    require_free_gpus([0, 1], 512, gpu_memory=lambda gpus: None)                    # unreadable: not refused
    with pytest.raises(RuntimeError, match=r"GPU 1 has 3200 MiB in use \(limit 512 MiB\).*nvidia-smi"):
        require_free_gpus([0, 1], 512, gpu_memory=lambda gpus: {0: 9, 1: 3200})


def test_the_free_check_waits_with_backoff_for_a_job_that_is_still_giving_memory_back():
    from ar_kernel.tools.vllm_server import FREE_WAITS_S, require_free_gpus
    readings, slept = iter([9000, 4000, 600, 20]), []
    require_free_gpus([0], 512, gpu_memory=lambda gpus: {0: next(readings)}, waits=FREE_WAITS_S, sleep=slept.append)
    assert slept == [4, 8, 16]
    slept.clear()
    with pytest.raises(RuntimeError, match="GPU 0 has 9000 MiB"):
        require_free_gpus([0], 512, gpu_memory=lambda gpus: {0: 9000}, waits=FREE_WAITS_S, sleep=slept.append)
    assert slept == [4, 8, 16, 32, 64]                 # 124 s in all, then the refusal
