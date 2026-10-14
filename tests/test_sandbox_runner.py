import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import docker
import pytest
import requests

from ar_kernel.config import KernelConfig
from ar_kernel.liveness import Liveness
from ar_kernel.sandbox.image import ensure_image
from ar_kernel.sandbox.runner import (Mounts, container_name, container_prefix, diff,
                                       kill_run_containers, run_container, snapshot)
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()


def test_container_name_is_prefixed_and_docker_safe():
    name = container_name("20260921_1200", "n/7 x", "improve_recipe", 2)
    assert name.startswith("ar-20260921_1200-n-7-x-improve_recipe-2-")
    assert all(c.isalnum() or c in "_.-" for c in name)


def test_container_prefix_matches_container_name():
    assert container_name("r:1", "n2", "edit_self", 1).startswith(container_prefix("r:1", "n2"))
    assert container_prefix("r:1") == "ar-r-1-"


class _FakeContainer:
    """`wait` answers from `waits` in order: an exception is raised, anything else returned."""
    def __init__(self, waits=({"StatusCode": 0},)):
        self.waits, self.killed = list(waits), False

    def wait(self, timeout=None):
        w = self.waits.pop(0)
        if isinstance(w, BaseException):
            raise w
        return w

    def stats(self, stream=False):
        raise requests.ConnectionError("no stats in tests")

    def logs(self, stdout=True, stderr=False):
        return b"out" if stdout else b"err"

    def kill(self):
        self.killed = True


class _FakeClient:
    def __init__(self, container=None, launch_error=None, listed=()):
        self.container, self.launch_error, self.ran, self.removed = container, launch_error, [], []
        self.containers = SimpleNamespace(run=self._run)
        self.api = SimpleNamespace(remove_container=lambda name, force: self.removed.append(name),
                                   containers=lambda all, filters: [{"Names": [f"/{n}"]} for n in listed])

    def _run(self, **kw):
        self.ran.append(kw)
        if self.launch_error is not None:
            raise self.launch_error
        return self.container


def test_kill_run_containers_removes_only_the_prefix(monkeypatch):
    client = _FakeClient(listed=["ar-r1-n2-edit_self-1-abc", "ar-r10-n2-edit_self-1-def", "other"])
    monkeypatch.setattr(docker, "from_env", lambda: client)
    assert kill_run_containers("r1", "n2") == ["ar-r1-n2-edit_self-1-abc"]
    assert client.removed == ["ar-r1-n2-edit_self-1-abc"]


def test_finished_nodes_are_mounted_read_only_with_eval_hidden(tmp_path):
    from ar_kernel.sandbox.runner import _run_kwargs
    n1, root = tmp_path / "n1", tmp_path / "root"
    (n1 / "eval").mkdir(parents=True)
    root.mkdir()
    m = Mounts(agent=tmp_path, workspace=tmp_path, staging=tmp_path, context=tmp_path,
               contract=tmp_path, sockets=tmp_path, nodes={"n1": n1, "root": root})
    kw = _run_kwargs("img", "c", m, ["true"], {}, 1, 1)
    assert f"{n1}:/nodes/n1:ro" in kw["volumes"] and f"{root}:/nodes/root:ro" in kw["volumes"]
    assert kw["tmpfs"]["/nodes/n1/eval"] == "ro,size=4k" and "/nodes/root/eval" not in kw["tmpfs"]


def test_the_run_gpus_reach_the_container_through_the_nvidia_runtime(tmp_path):
    from ar_kernel.sandbox.runner import _run_kwargs
    m = Mounts(agent=tmp_path, workspace=tmp_path, staging=tmp_path, context=tmp_path,
               contract=tmp_path, sockets=tmp_path)
    kw = _run_kwargs("img", "c", m, ["true"], {}, 1, 1, gpus=[0, 2, 3])
    assert kw["runtime"] == "nvidia" and kw["environment"]["NVIDIA_VISIBLE_DEVICES"] == "0,2,3"
    plain = _run_kwargs("img", "c", m, ["true"], {}, 1, 1)
    assert "runtime" not in plain and plain["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_separate_running_code_is_mounted_read_only_and_named_in_the_env(tmp_path):
    from ar_kernel.sandbox.runner import _run_kwargs
    m = Mounts(agent=tmp_path, workspace=tmp_path, staging=tmp_path, context=tmp_path,
               contract=tmp_path, sockets=tmp_path, code=tmp_path / "code")
    kw = _run_kwargs("img", "c", m, ["true"], {}, 1, 1)
    assert f"{tmp_path / 'code'}:/code:ro" in kw["volumes"] and kw["environment"]["AR_CODE_DIR"] == "/code"
    m.code = None
    assert "AR_CODE_DIR" not in _run_kwargs("img", "c", m, ["true"], {}, 1, 1)["environment"]


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
    dirs = {k: tmp_path / k for k in ("agent", "workspace", "context", "sockets")}
    for d in dirs.values():
        d.mkdir()
    staging = dirs["workspace"] / "staging"
    staging.mkdir()
    return Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=staging,
                  context=dirs["context"],
                  contract=CFG.repo_root / "contract", sockets=dirs["sockets"])


def _run(tmp_path, mounts, script, timeout_s=120, env=None, liveness=None, poll_s=5.0):
    return run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "test", 1),
                         mounts=mounts, command=["python", "-c", script], env=env or {},
                         cpus=2, memory_gb=2, timeout_s=timeout_s, recorder=Recorder(tmp_path / "run"),
                         node="n1", phase="test", attempt=1, stats_every_s=1, liveness=liveness, poll_s=poll_s)


def test_failed_launch_is_still_removed_and_only_the_recorded_settings_are_redacted(tmp_path, mounts, monkeypatch):
    """A launch can fail *after* creating
    the container (OCI runtime error, bad bind source, missing executable),
    leaving a `Created` container behind. The removal must still run on that
    path -- "the container is removed on every exit path" wins over the code
    that skipped it via an early return outside the `finally`.

    Also: the runner must not record the
    AR_TOKEN value in telemetry, regardless of which Recorder is used (no
    add_redaction call here) -- but that redaction must only touch the
    recorded copy, never the real settings handed to docker.
    """
    client = _FakeClient(launch_error=docker.errors.APIError("oci runtime error"))
    monkeypatch.setattr(docker, "from_env", lambda: client)
    secret = "ar-super-secret-token-value"
    recorder = Recorder(tmp_path / "run")
    name = container_name("t", "n1", "test", 1)
    res = run_container(image="unused:tag", name=name, mounts=mounts,
                        command=["python", "-c", "pass"], env={"AR_TOKEN": secret},
                        cpus=2, memory_gb=2, timeout_s=5, recorder=recorder,
                        node="n1", phase="test", attempt=1, stats_every_s=1)
    assert res.exit_code is None and "oci runtime error" in res.stderr

    assert client.ran[0]["environment"]["AR_TOKEN"] == secret, "the real launch must still get the real token"
    assert client.removed == [name], "a failed launch must still be removed"

    events = recorder.read_events("n1")
    start_events = [e for e in events if e["type"] == "sandbox.start"]
    assert start_events, "expected a sandbox.start event"
    payload = recorder.load_payload(start_events[0]["payload"])
    dumped = json.dumps(payload)
    assert secret not in dumped
    assert payload["kwargs"]["environment"]["AR_TOKEN"] == "<redacted>"
    raw_events_text = recorder.events_path("n1").read_text()
    assert secret not in raw_events_text


@pytest.mark.docker
def test_isolation_holds_from_inside(tmp_path, mounts):
    """With network `none` (contract containers; agent phases use `sandbox.network`,
    bridge by default): no internet, no host services, no kernel/WorldModel/WBench/.env,
    no writes to /context; files written are owned by the host user. The socket dir is
    mounted read-only: the gateway socket cannot be deleted, but connecting to it still works.

    The host paths checked below are derived from KernelConfig (project/repo
    roots), never hardcoded, so the test is portable across checkouts/machines.
    """
    host_paths = [str(CFG.repo_root.parent), str(CFG.repo_root / "kernel"),
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
for target in ("/context/new.json", "/etc/new"):
    try: open(target, "w").write("x"); out[target] = "writable"
    except OSError: out[target] = "denied"
try: os.remove("/run/ar/gateway.sock"); out["sock_rm"] = "removed"
except OSError: out["sock_rm"] = "denied"
c = socket.socket(socket.AF_UNIX); c.settimeout(10); c.connect("/run/ar/gateway.sock")
out["sock_reply"] = c.recv(16).decode()
open("/workspace/staging/made.txt", "w").write("x")
print(json.dumps(out))
"""
    server = socket.socket(socket.AF_UNIX)          # stands in for the gateway's socket
    server.bind(str(mounts.sockets / "gateway.sock"))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        conn.sendall(b"hello")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    res = run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "test", 1),
                        mounts=mounts, command=["python", "-c", script, json.dumps(host_paths)],
                        env={}, cpus=2, memory_gb=2, timeout_s=120,
                        recorder=Recorder(tmp_path / "run"), node="n1", phase="test", attempt=1,
                        stats_every_s=1, network="none")
    assert res.exit_code == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["internet"] == "blocked"
    assert out["host_ssh"] in ("closed", "unreachable")
    assert out["visible"] == []
    assert out["/context/new.json"] == out["/etc/new"] == "denied"
    assert out["sock_rm"] == "denied" and out["sock_reply"] == "hello"
    assert (mounts.sockets / "gateway.sock").exists()
    server.close()
    assert os.stat(mounts.staging / "made.txt").st_uid == os.getuid()


@pytest.mark.docker
def test_timeout_kills_and_removes_the_container(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import time; print('started', flush=True); time.sleep(600)", timeout_s=8)
    assert res.timed_out and "started" in res.stdout
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={res.container}", "-q"],
                            capture_output=True, text=True).stdout.strip()
    assert listed == ""


@pytest.mark.docker
def test_liveness_ends_a_stalled_container_well_before_the_hard_cap(tmp_path, mounts):
    """`sleep` uses no CPU, so the liveness signal (container CPU%, added inside
    run_container) never changes; the probe window must end the container long
    before the 600s hard cap."""
    started = time.monotonic()
    lv = Liveness(3, probe_window_s=3, extension_frac=0.25, signals=[lambda: 0])
    res = _run(tmp_path, mounts, "print('go', flush=True); import time; time.sleep(600)",
               timeout_s=600, liveness=lv, poll_s=1.0)
    assert res.timed_out and time.monotonic() - started < 60
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={res.container}", "-q"],
                            capture_output=True, text=True).stdout.strip()
    assert listed == ""


@pytest.mark.docker
def test_exit_code_and_streams_are_captured(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert res.exit_code == 3 and "out" in res.stdout and "err" in res.stderr and not res.timed_out


def test_snapshot_skips_fifos_and_marks_unreadable_files(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    (tmp_path / "locked.txt").write_text("secret")
    (tmp_path / "locked.txt").chmod(0)
    os.mkfifo(tmp_path / "pipe")
    try:
        snap = snapshot(tmp_path, True)
    finally:
        (tmp_path / "locked.txt").chmod(0o644)
    assert snap["locked.txt"] == "unreadable" and "pipe" not in snap and len(snap["ok.txt"]) == 64


def _lock_up(root):
    """What an agent can leave behind: a chmod-000 file and a chmod-000 directory holding a file."""
    (root / "locked.txt").write_text("x")
    (root / "locked.txt").chmod(0)
    (root / "sealed").mkdir()
    (root / "sealed" / "inner.txt").write_text("y")
    (root / "sealed").chmod(0)


def test_restored_access_lets_a_retry_copy_and_delete_the_tree(tmp_path):
    from ar_kernel.sandbox.runner import restore_owner_access

    ws = tmp_path / "ws"
    ws.mkdir()
    _lock_up(ws)
    (ws / "link").symlink_to(tmp_path / "elsewhere")          # dangling link: must not be followed
    restore_owner_access(ws)
    shutil.copytree(ws, tmp_path / "retry", symlinks=True)    # the retry's workspace copy
    shutil.rmtree(ws)                                         # re-running an attempt number
    assert (tmp_path / "retry" / "sealed" / "inner.txt").read_text() == "y"


@pytest.mark.docker
def test_container_runs_restore_owner_access_on_every_writable_mount(tmp_path, mounts):
    script = ("import os\n"
              "for root in ('/agent', '/workspace', '/workspace/staging'):\n"
              "    open(root + '/locked.txt', 'w').write('x'); os.chmod(root + '/locked.txt', 0)\n"
              "    os.mkdir(root + '/sealed'); open(root + '/sealed/inner.txt', 'w').write('y')\n"
              "    os.chmod(root + '/sealed', 0)\n")
    res = _run(tmp_path, mounts, script)
    assert res.exit_code == 0, res.stderr
    for root in (mounts.agent, mounts.workspace, mounts.staging):
        assert (root / "sealed" / "inner.txt").read_text() == "y"
        assert (root / "locked.txt").read_text() == "x"
    shutil.copytree(mounts.workspace, tmp_path / "retry", symlinks=True)
    shutil.rmtree(mounts.workspace)


def test_a_wait_that_fails_without_an_exit_code_waits_again(tmp_path, mounts, monkeypatch):
    container = _FakeContainer([requests.ConnectionError("read timed out"), {"StatusCode": 0}])
    client = _FakeClient(container)
    monkeypatch.setattr(docker, "from_env", lambda: client)
    res = run_container(image="unused:tag", name="ar-t-n1-test-1-abc", mounts=mounts, command=["true"], env={},
                        cpus=2, memory_gb=2, timeout_s=60, recorder=Recorder(tmp_path / "run"),
                        node="n1", phase="test", attempt=1, stats_every_s=60, poll_s=0.01)
    assert res.exit_code == 0 and not res.timed_out and (res.stdout, res.stderr) == ("out", "err")
    assert container.waits == [] and not container.killed and client.removed == ["ar-t-n1-test-1-abc"]


_SIGINT_CHILD = r"""
import json, signal, sys
from pathlib import Path
from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.image import ensure_image
from ar_kernel.sandbox.runner import Mounts, run_container
from ar_kernel.telemetry.recorder import Recorder
signal.signal(signal.SIGINT, lambda *a: None)       # what drive() does on the first Ctrl-C: carry on
cfg, d = KernelConfig.load(), json.loads(sys.argv[1])
res = run_container(image=ensure_image(cfg, ""), name=d["name"], mounts=Mounts(**d["mounts"]),
                    command=["python", "-c", "import time; print('started', flush=True); time.sleep(8); print('done')"],
                    env={}, cpus=2, memory_gb=2, timeout_s=120, recorder=Recorder(Path(d["run"])),
                    node="n1", phase="test", attempt=1, stats_every_s=1, poll_s=1.0)
print(json.dumps({"exit_code": res.exit_code, "stdout": res.stdout}))
"""


@pytest.mark.docker
def test_a_ctrl_c_to_the_process_group_does_not_end_the_container(tmp_path, mounts):
    """The first Ctrl-C is graceful. The terminal sends SIGINT to the whole foreground
    process group; the wait for the container must survive it."""
    name = container_name("sigint", "n1", "test", 1)
    arg = json.dumps({"name": name, "run": str(tmp_path / "run"),
                      "mounts": {k: str(v) for k, v in vars(mounts).items() if k not in ("agent_readonly", "nodes")}})
    child = subprocess.Popen([os.sys.executable, "-c", _SIGINT_CHILD, arg], stdout=subprocess.PIPE, text=True,
                             cwd=CFG.repo_root, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": str(CFG.repo_root / "kernel")})
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout
            if "started" in logs:
                break
            time.sleep(0.5)
        time.sleep(1.0)                                     # docker wait is running by now
        os.killpg(child.pid, signal.SIGINT)                 # as the terminal delivers a Ctrl-C
        out, _ = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    result = json.loads(out.strip().splitlines()[-1])
    assert result["exit_code"] == 0 and "done" in result["stdout"]


def test_every_phase_sees_the_same_node_files_and_only_eval_is_hidden(tmp_path):
    """edit_self reads what improve_recipe reads: only prompts and tools differ."""
    from ar_kernel.sandbox.runner import Mounts, _run_kwargs
    node = tmp_path / "n1"
    for folder in ("eval", "attempts/improve_recipe-1/context", "contract/attempt-1"):
        (node / folder).mkdir(parents=True)
    kw = _run_kwargs("img", "c", Mounts(agent=tmp_path, workspace=tmp_path, staging=tmp_path, context=tmp_path,
                                        contract=tmp_path, sockets=tmp_path, nodes={"n1": node}), ["true"], {}, 1, 1)
    assert [v for v in kw["volumes"] if ":/nodes/" in v] == [f"{node}:/nodes/n1:ro"]
    assert [k for k in kw["tmpfs"] if k.startswith("/nodes/")] == ["/nodes/n1/eval"]
    assert kw["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void" and not any(":/store" in v for v in kw["volumes"])


@pytest.mark.gpu
def test_torch_in_the_container_computes_on_exactly_the_passed_gpus(tmp_path, mounts):
    """Run with the GPUs to pass in AR_TEST_GPUS (default 0,1,2,3)."""
    import os
    gpus = [int(g) for g in os.environ.get("AR_TEST_GPUS", "0,1,2,3").split(",")]
    res = run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "gpu", 1), mounts=mounts,
                        command=["python", "-c", "import torch; n = torch.cuda.device_count(); "
                                 "print(n, sum(torch.ones(3, device=f'cuda:{i}').sum().item() for i in range(n)))"],
                        env={}, cpus=2, memory_gb=8, timeout_s=300, recorder=Recorder(tmp_path / "run"),
                        node="n1", phase="gpu", attempt=1, gpus=gpus)
    assert res.exit_code == 0, res.stderr
    assert res.stdout.split() == [str(len(gpus)), str(3.0 * len(gpus))]
