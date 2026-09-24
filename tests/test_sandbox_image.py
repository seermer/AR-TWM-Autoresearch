import subprocess

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.image import base_tag, ensure_image, image_tag

CFG = KernelConfig.load()


def test_tags_are_deterministic_and_requirements_sensitive():
    assert image_tag(CFG, "requests==2.0\n") == image_tag(CFG, "requests==2.0\n")
    assert image_tag(CFG, "requests==2.0\n") != image_tag(CFG, "requests==2.1\n")
    assert base_tag(CFG).startswith("ar-agent-base:")


def test_requirement_order_and_blank_lines_do_not_change_the_tag():
    assert image_tag(CFG, "b==1\na==1\n") == image_tag(CFG, "\na==1\n\nb==1")


@pytest.mark.docker
def test_built_image_has_the_runtime_stack_and_no_network_needed():
    tag = ensure_image(CFG, "")
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", tag, "python", "-c",
         "import mcp, httpx, httpx2, langgraph, langchain_core, langchain_openai, cv2, numpy, PIL; "
         "from langgraph.types import Send; print('ok')"],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    ff = subprocess.run(["docker", "run", "--rm", "--network", "none", tag, "ffprobe", "-version"],
                        capture_output=True, text=True, timeout=120)
    assert ff.returncode == 0
