"""Agent images: a pinned base plus a per-requirements layer (spec 9.4.1)."""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path


class ImageBuildError(RuntimeError):
    """The agent image could not be built; carries the build log tail."""


def _dockerfile(cfg) -> Path:
    return cfg.repo_root / "docker" / "agent.Dockerfile"


def _norm_requirements(requirements: str) -> str:
    return "\n".join(sorted(line.strip() for line in requirements.splitlines() if line.strip())) + "\n"


def base_tag(cfg) -> str:
    digest = hashlib.sha256(_dockerfile(cfg).read_bytes()).hexdigest()[:12]
    return f"{cfg.get('sandbox.image')}-base:{digest}"


def image_tag(cfg, requirements: str) -> str:
    digest = hashlib.sha256((base_tag(cfg) + _norm_requirements(requirements)).encode()).hexdigest()[:12]
    return f"{cfg.get('sandbox.image')}:{digest}"


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def _build(tag: str, dockerfile_text: str, context: Path, recorder, node: str) -> None:
    proc = subprocess.run(["docker", "build", "-t", tag, "-f", "-", str(context)],
                          input=dockerfile_text, capture_output=True, text=True, timeout=3600)
    if recorder is not None:
        recorder.event("sandbox.image_build", node=node, component="sandbox", tag=tag,
                       returncode=proc.returncode,
                       payload={"dockerfile": dockerfile_text, "stdout": proc.stdout, "stderr": proc.stderr})
    if proc.returncode != 0:
        raise ImageBuildError(f"docker build {tag} failed:\n{(proc.stdout + proc.stderr)[-4000:]}")


def ensure_image(cfg, requirements: str, *, recorder=None, node: str = "run") -> str:
    base = base_tag(cfg)
    if not image_exists(base):
        _build(base, _dockerfile(cfg).read_text(), cfg.repo_root / "docker", recorder, node)
    tag = image_tag(cfg, requirements)
    if image_exists(tag):
        return tag
    reqs = _norm_requirements(requirements)
    with tempfile.TemporaryDirectory() as ctx:
        (Path(ctx) / "requirements.txt").write_text(reqs)
        install = ("RUN pip install --no-cache-dir -r /tmp/requirements.txt\n"
                   if reqs.strip() else "")
        _build(tag, f"FROM {base}\nCOPY requirements.txt /tmp/requirements.txt\n{install}",
               Path(ctx), recorder, node)
    return tag
