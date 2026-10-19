"""Real bwrap sandboxes with .envs/agent; run with -m sandbox."""
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.runner import Mounts, container_name, run_container
from ar_kernel.telemetry.recorder import Recorder

pytestmark = pytest.mark.sandbox
CFG = KernelConfig.load()
ENV_DIR = CFG.repo_root / ".envs" / "agent"


@pytest.fixture
def mounts(tmp_path):
    dirs = {k: tmp_path / k for k in ("agent", "workspace", "staging", "context", "sockets")}
    for d in dirs.values():
        d.mkdir()
    return Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"],
                  context=dirs["context"], contract=CFG.repo_root / "contract", sockets=dirs["sockets"])


def _run(tmp_path, mounts, script, *args, env=None, gpus=()):
    return run_container(name=container_name("t", "n1", "test", 1), mounts=mounts,
                         command=["python", "-c", script, *args], env=env or {}, env_dir=ENV_DIR, shm_gb=2,
                         recorder=Recorder(tmp_path / "run"), node="n1", phase="test", attempt=1, gpus=gpus)


def test_isolation_holds_from_inside(tmp_path, mounts):
    """Read-only mounts refuse writes, /workspace and its staging mount accept them, a hard link
    between the two fails, the socket folder connects but refuses unlink, the environment holds
    only the given variables, a finished node's eval is empty, and nothing of the project but
    the agent env is visible. Files written belong to the host user."""
    node = tmp_path / "n1"
    (node / "eval").mkdir(parents=True)
    (node / "eval" / "case_1.json").write_text("{}")
    (node / "train.log").write_text("x")
    mounts.nodes = {"n1": node}
    hidden = [str(CFG.repo_root / "kernel"), str(CFG.worldmodel), str(CFG.wbench), str(CFG.repo_root / ".env"),
              str(CFG.repo_root / "runs")]
    script = r"""
import os, socket, json, sys
hidden, project = json.loads(sys.argv[1]), sys.argv[2]
out = {"visible": [p for p in hidden if os.path.exists(p)], "project": os.listdir(project),
       "env": sorted(set(os.environ) - {"LC_CTYPE"}), "eval": os.listdir("/nodes/n1/eval"), "node": sorted(os.listdir("/nodes/n1"))}
for target in ("/context/new.json", "/etc/new", "/usr/new", "/agent/new", "/nodes/n1/new", "/ar_contract/new",
               "/workspace/new", "/workspace/staging/made.txt", "/tmp/new", "/dev/shm/new"):
    try: open(target, "w").write("x"); out[target] = "writable"
    except OSError: out[target] = "denied"
try: os.link("/workspace/new", "/workspace/staging/linked"); out["hard_link"] = "made"
except OSError: out["hard_link"] = "refused"
try: os.remove("/run/ar/gateway.sock"); out["sock_rm"] = "removed"
except OSError: out["sock_rm"] = "denied"
c = socket.socket(socket.AF_UNIX); c.settimeout(10); c.connect("/run/ar/gateway.sock")
out["sock_reply"] = c.recv(16).decode()
print(json.dumps(out))
"""
    server = socket.socket(socket.AF_UNIX)          # stands in for the gateway's socket
    fd = os.open(mounts.sockets, os.O_DIRECTORY)    # the real path is over the socket path limit
    server.bind(f"/proc/self/fd/{fd}/gateway.sock")
    os.close(fd)
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        conn.sendall(b"hello")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    mounts.agent_readonly = True
    res = _run(tmp_path, mounts, script, json.dumps(hidden), str(CFG.repo_root.parent), env={"AR_TOKEN": "t"})
    server.close()
    assert res.exit_code == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["visible"] == [] and out["project"] == [CFG.repo_root.name]
    # LC_CTYPE is left out above: Python's own C-locale coercion (PEP 538) sets it; bwrap sets PWD
    assert out["env"] == sorted(["PATH", "HOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                                 "AR_SOCKET_DIR", "AR_AGENT_DIR", "AR_CONTEXT_DIR", "AR_WORKSPACE", "AR_TOKEN", "PWD"])
    assert out["eval"] == [] and out["node"] == ["eval", "train.log"]
    for target in ("/context/new.json", "/etc/new", "/usr/new", "/agent/new", "/nodes/n1/new", "/ar_contract/new"):
        assert out[target] == "denied", target
    for target in ("/workspace/new", "/workspace/staging/made.txt", "/tmp/new", "/dev/shm/new"):
        assert out[target] == "writable", target
    assert out["hard_link"] == "refused"
    assert out["sock_rm"] == "denied" and out["sock_reply"] == "hello" and (mounts.sockets / "gateway.sock").exists()
    assert os.stat(mounts.staging / "made.txt").st_uid == os.getuid()
    assert not (mounts.agent / "new").exists() and (node / "eval" / "case_1.json").exists()


def test_exit_code_and_streams_are_captured(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert res.exit_code == 3 and res.stdout == "out\n" and res.stderr == "err\n"


def test_pip_install_lands_in_the_workspace_home(tmp_path, mounts):
    """The env is read-only: pip falls back to a user install under /workspace/.home."""
    script = ("import subprocess, sys; r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
              "'--no-index', '--no-deps', '--no-build-isolation', sys.argv[1]], capture_output=True, text=True); "
              "print(r.returncode, r.stderr[-500:]); subprocess.run([sys.executable, '-c', "
              "'import ar_test_pkg; print(ar_test_pkg.__file__)'], check=True)")   # a new interpreter sees the user site
    pkg = mounts.workspace / "pkg"
    (pkg / "ar_test_pkg").mkdir(parents=True)
    (pkg / "ar_test_pkg" / "__init__.py").write_text("")
    (pkg / "pyproject.toml").write_text('[project]\nname = "ar-test-pkg"\nversion = "0.1"\n'
                                        '[build-system]\nrequires = ["setuptools"]\n'
                                        'build-backend = "setuptools.build_meta"\n')
    res = _run(tmp_path, mounts, script, "/workspace/pkg")
    assert res.exit_code == 0, res.stdout + res.stderr
    assert res.stdout.strip().splitlines()[-1].startswith("/workspace/.home/.local/lib/python3.12/site-packages/")


def test_requirements_installed_in_one_sandbox_are_importable_in_the_next(tmp_path, mounts):
    from ar_kernel.agent_phase import install_requirements
    import zipfile
    wheel = "ar_req_pkg-0.1-py3-none-any.whl"                 # a local wheel: no network needed
    with zipfile.ZipFile(mounts.workspace / wheel, "w") as z:
        z.writestr("ar_req_pkg/__init__.py", "")
        z.writestr("ar_req_pkg-0.1.dist-info/METADATA", "Metadata-Version: 2.1\nName: ar-req-pkg\nVersion: 0.1\n")
        z.writestr("ar_req_pkg-0.1.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                                                     "Tag: py3-none-any\n")
        z.writestr("ar_req_pkg-0.1.dist-info/RECORD", "")
    (mounts.agent / "agent").mkdir()
    (mounts.agent / "agent" / "requirements.txt").write_text(f"--no-index\n/workspace/{wheel}\n")
    mounts.agent_readonly = True
    rec = Recorder(tmp_path / "run")
    installed = install_requirements(run_container, name="ar-t-n1-install-1", mounts=mounts,
                                     path="/agent/agent/requirements.txt", cfg=CFG, recorder=rec, node="n1",
                                     phase="test", attempt=1)
    assert installed.exit_code == 0, installed.stdout + installed.stderr
    assert (tmp_path / "install.stdout").exists()
    res = _run(tmp_path, mounts, "import ar_req_pkg; print(ar_req_pkg.__file__)")
    assert res.exit_code == 0 and res.stdout.startswith("/workspace/.home/.local/"), res.stderr


def test_the_sandbox_runs_restore_owner_access_on_every_writable_mount(tmp_path, mounts):
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
    shutil.copytree(mounts.workspace, tmp_path / "copy", symlinks=True)
    shutil.rmtree(mounts.workspace)


_CHILD = r"""
import json, signal, sys
from pathlib import Path
from ar_kernel.sandbox.runner import Mounts, run_container
from ar_kernel.telemetry.recorder import Recorder
signal.signal(signal.SIGINT, lambda *a: None)       # what drive() does on the first Ctrl-C: carry on
d = json.loads(sys.argv[1])
res = run_container(name="ar-t-n1-test-1", mounts=Mounts(**d["mounts"]), command=["bash", "-c", d["command"]],
                    env={}, env_dir=Path(d["env_dir"]), shm_gb=1, recorder=Recorder(Path(d["run"])),
                    node="n1", phase="test", attempt=1)
print(json.dumps({"exit_code": res.exit_code, "stdout": res.stdout}))
"""


def _child(tmp_path, mounts, command):
    arg = json.dumps({"run": str(tmp_path / "run"), "env_dir": str(ENV_DIR), "command": command,
                      "mounts": {k: str(v) for k, v in vars(mounts).items() if k not in ("agent_readonly", "nodes", "code")}})
    return subprocess.Popen([os.sys.executable, "-c", _CHILD, arg], stdout=subprocess.PIPE, text=True,
                            cwd=CFG.repo_root, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(CFG.repo_root / "kernel")})


def _wait_for(path, seconds=60):
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.2)


def test_a_ctrl_c_to_the_kernels_process_group_does_not_end_the_sandbox(tmp_path, mounts):
    """The first Ctrl-C is graceful. The terminal sends SIGINT to the whole foreground process group."""
    child = _child(tmp_path, mounts, "touch /workspace/started; sleep 4; echo done")
    try:
        _wait_for(mounts.workspace / "started")
        os.killpg(child.pid, signal.SIGINT)
        out, _ = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
    result = json.loads(out.strip().splitlines()[-1])
    assert result["exit_code"] == 0 and "done" in result["stdout"]


def test_a_dead_kernel_ends_every_process_in_the_sandbox(tmp_path, mounts):
    """A detached `setsid nohup` process does not outlive the kernel that started the sandbox."""
    beat = mounts.workspace / "beat"
    child = _child(tmp_path, mounts, "setsid nohup bash -c 'while true; do date +%s%N > /workspace/beat; "
                                     "sleep 0.1; done' >/dev/null 2>&1 & sleep 600")
    try:
        _wait_for(beat)
        os.killpg(child.pid, signal.SIGKILL)               # the kernel process dies without cleanup
        child.wait()
        time.sleep(1.0)
        last = beat.read_text()
        time.sleep(1.0)
        assert beat.read_text() == last
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)


@pytest.mark.gpu
def test_torch_in_the_sandbox_computes_on_exactly_the_chosen_gpus(tmp_path, mounts):
    from ar_kernel.guards import choose_gpus, device_minors
    gpus = choose_gpus(CFG, {})
    script = ("import torch, subprocess; n = torch.cuda.device_count(); "
              "print(n, sum(torch.ones(3, device=f'cuda:{i}').sum().item() for i in range(n))); "
              "print(subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], "
              "capture_output=True, text=True).stdout)")
    res = _run(tmp_path, mounts, script, gpus=gpus)
    assert res.exit_code == 0, res.stderr
    lines = res.stdout.split("\n")
    assert lines[0].split() == [str(len(gpus)), str(3.0 * len(gpus))]
    host = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout
    uuids = {int(i): u.strip() for i, u in (line.split(",") for line in host.splitlines())}
    assert sorted(line.strip() for line in lines[1:] if line.strip()) == sorted(uuids[g] for g in gpus)
    assert device_minors(gpus)


def test_cancel_ends_a_real_sandbox(tmp_path, mounts):
    cancel = threading.Event()
    threading.Timer(2.0, cancel.set).start()
    started = time.monotonic()
    res = run_container(name=container_name("t", "n1", "test", 1), mounts=mounts, command=["sleep", "600"], env={},
                        env_dir=ENV_DIR, shm_gb=2, recorder=Recorder(tmp_path / "run"), node="n1", phase="test",
                        attempt=1, cancel=cancel)
    assert res.exit_code == -9 and time.monotonic() - started < 30


def test_a_missing_command_is_a_kernel_fault(tmp_path, mounts):
    from ar_kernel.faults import KernelFault
    with pytest.raises(KernelFault, match="bwrap: execvp"):
        run_container(name=container_name("t", "n1", "test", 1), mounts=mounts, command=["/no/such/command"],
                      env={}, env_dir=ENV_DIR, shm_gb=2, recorder=Recorder(tmp_path / "run"), node="n1",
                      phase="test", attempt=1)
