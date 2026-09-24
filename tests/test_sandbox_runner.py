import json
import os
import subprocess

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.image import ensure_image
from ar_kernel.sandbox.runner import Mounts, container_name, diff, run_container, snapshot
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()


def test_container_name_is_prefixed_and_docker_safe():
    name = container_name("20260921_1200", "n/7 x", "improve_recipe", 2)
    assert name.startswith("ar-20260921_1200-n-7-x-improve_recipe-2-")
    assert all(c.isalnum() or c in "_.-" for c in name)


def test_snapshot_diff_reports_changes(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    before = snapshot(tmp_path, hash_files=True)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    assert diff(before, snapshot(tmp_path, hash_files=True)) == {
        "added": ["c.txt"], "removed": ["b.txt"], "changed": ["a.txt"]}


@pytest.fixture
def mounts(tmp_path):
    dirs = {k: tmp_path / k for k in ("agent", "workspace", "context", "store", "sockets")}
    for d in dirs.values():
        d.mkdir()
    staging = dirs["workspace"] / "staging"
    staging.mkdir()
    (dirs["store"] / "blob.bin").write_bytes(b"x")
    return Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=staging,
                  context=dirs["context"], store=dirs["store"],
                  contract=CFG.repo_root / "contract", sockets=dirs["sockets"])


def _run(tmp_path, mounts, script, timeout_s=120, env=None):
    return run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "test", 1),
                         mounts=mounts, command=["python", "-c", script], env=env or {},
                         cpus=2, memory_gb=2, timeout_s=timeout_s, recorder=Recorder(tmp_path / "run"),
                         node="n1", phase="test", attempt=1, stats_every_s=1)


def test_sandbox_start_never_records_the_container_token(tmp_path, mounts, monkeypatch):
    """Controller ruling: the runner must not record the AR_TOKEN value in
    telemetry, regardless of which Recorder is used (no add_redaction call here).
    Fail the docker launch immediately so no real container is needed."""
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="docker not invoked")

    monkeypatch.setattr(subprocess, "run", fake_run)
    secret = "ar-super-secret-token-value"
    recorder = Recorder(tmp_path / "run")
    run_container(image="unused:tag", name=container_name("t", "n1", "test", 1), mounts=mounts,
                  command=["python", "-c", "pass"], env={"AR_TOKEN": secret},
                  cpus=2, memory_gb=2, timeout_s=5, recorder=recorder,
                  node="n1", phase="test", attempt=1, stats_every_s=1)

    events = recorder.read_events("n1")
    start_events = [e for e in events if e["type"] == "sandbox.start"]
    assert start_events, "expected a sandbox.start event"
    payload = recorder.load_payload(start_events[0]["payload"])
    dumped = json.dumps(payload)
    assert secret not in dumped
    assert "AR_TOKEN=<redacted>" in dumped
    raw_events_text = recorder.events_path("n1").read_text()
    assert secret not in raw_events_text


@pytest.mark.docker
def test_isolation_holds_from_inside(tmp_path, mounts):
    """Spec 16.3 item 4: no internet, no host services, no kernel/WorldModel/WBench/.env,
    no writes to /store or /context; files written are owned by the host user.

    The host paths checked below are derived from KernelConfig (project/repo
    roots), never hardcoded, so the test is portable across checkouts/machines.
    """
    host_paths = [str(CFG.project_root), str(CFG.repo_root / "kernel"),
                  str(CFG.worldmodel), str(CFG.wbench), str(CFG.repo_root / ".env")]
    script = r"""
import os, socket, urllib.request, json, sys
host_paths = json.loads(sys.argv[1])
out = {}
try: urllib.request.urlopen("https://pypi.org", timeout=4); out["internet"] = "open"
except Exception: out["internet"] = "blocked"
s = socket.socket(); s.settimeout(2)
try: out["host_ssh"] = "open" if s.connect_ex(("172.17.0.1", 22)) == 0 else "closed"
except OSError: out["host_ssh"] = "unreachable"
out["visible"] = [p for p in host_paths if os.path.exists(p)]
for target in ("/store/new.bin", "/context/new.json", "/etc/new"):
    try: open(target, "w").write("x"); out[target] = "writable"
    except OSError: out[target] = "denied"
open("/workspace/staging/made.txt", "w").write("x")
print(json.dumps(out))
"""
    res = run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "test", 1),
                        mounts=mounts, command=["python", "-c", script, json.dumps(host_paths)],
                        env={}, cpus=2, memory_gb=2, timeout_s=120,
                        recorder=Recorder(tmp_path / "run"), node="n1", phase="test", attempt=1,
                        stats_every_s=1)
    assert res.exit_code == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["internet"] == "blocked"
    assert out["host_ssh"] in ("closed", "unreachable")
    assert out["visible"] == []
    assert out["/store/new.bin"] == out["/context/new.json"] == out["/etc/new"] == "denied"
    assert os.stat(mounts.staging / "made.txt").st_uid == os.getuid()


@pytest.mark.docker
def test_timeout_kills_and_removes_the_container(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import time; print('started', flush=True); time.sleep(600)", timeout_s=8)
    assert res.timed_out and "started" in res.stdout
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={res.container}", "-q"],
                            capture_output=True, text=True).stdout.strip()
    assert listed == ""


@pytest.mark.docker
def test_exit_code_and_streams_are_captured(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert res.exit_code == 3 and "out" in res.stdout and "err" in res.stderr and not res.timed_out
