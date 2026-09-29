"""Who answers the VLM metrics: the API named in the environment, or a local Qwen server."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import KernelConfig
from ..subproc import free_port
from ..tools.vllm_server import VllmServer, serve_command

DEFAULT_API_MODEL = "doubao-seed-2-0-lite-260215"
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3"


@dataclass(frozen=True)
class Judge:
    kind: str            # "api" | "local"
    model: str
    url: str | None      # the API endpoint; None for a local server (its port is chosen at start)


def resolve_judge(cfg: KernelConfig, env: Mapping[str, str]) -> Judge:
    """The API judge when `VLM_API_KEY` is set, else the captioner's model served locally."""
    if env.get("VLM_API_KEY", "").strip():
        return Judge("api", env.get("VLM_MODEL_NAME") or DEFAULT_API_MODEL,
                     env.get("VLM_API_URL") or DEFAULT_API_URL)
    return Judge("local", cfg.get("captioner.model"), None)


def judge_env(cfg: KernelConfig, judge: Judge, base_url: str | None) -> dict[str, str]:
    """Extra environment for WBench's vlm phase. An API judge uses the caller's own VLM_* variables."""
    if judge.kind == "api":
        return {}
    return {"VLM_API_URL": f"{base_url}/v1/chat/completions", "VLM_API_KEY": "local",
            "VLM_MODEL_NAME": judge.model,
            "VLM_EXTRA_BODY": json.dumps(cfg.get("eval.judge.extra_body") or {})}


def judge_server(cfg: KernelConfig, judge: Judge, gpus: list[int], cwd: Path, recorder=None,
                 node: str = "run") -> VllmServer:
    """An unstarted vLLM server for a local judge, on all of `gpus`. WBench sends each video inline
    (base64) and the interaction and causal metrics send frames as images, so no media directory is
    exposed and up to `eval.judge.max_images` images are allowed per prompt."""
    c = cfg.get("captioner")
    port = free_port()
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    return VllmServer(c["env"], serve_command(c, gpus, port, judge.model,
                                        mm_limits={"image": cfg.get("eval.judge.max_images"), "video": 1}), gpus, port, cwd=cwd,
                      log_path=cwd / "judge_vllm.log", recorder=recorder, node=node, phase="judge",
                      label="judge server")
