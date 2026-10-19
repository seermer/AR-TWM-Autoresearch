import json
import os
import shutil
import subprocess
import threading

import pytest

from ar_kernel.faults import KernelFault
from ar_kernel.sandbox import runner
from ar_kernel.sandbox.runner import Mounts, bwrap_args, container_name, diff, run_container, snapshot
from ar_kernel.telemetry.recorder import Recorder

ENV_DIR = "/envs/agent"


def test_container_name_is_a_safe_label():
    name = container_name("20260921_1200", "n/7 x", "improve_recipe", 2)
    assert name.startswith("ar-20260921_1200-n-7-x-improve_recipe-2-")
    assert all(c.isalnum() or c in "_.-" for c in name)


def _mounts(tmp_path, **kw):
    return Mounts(agent=tmp_path / "agent", workspace=tmp_path / "ws", staging=tmp_path / "st",
                  context=tmp_path / "ctx", contract=tmp_path / "contract", sockets=tmp_path / "sock", **kw)


def _binds(argv) -> dict[str, tuple[str, str]]:
    """sandbox path -> (flag, host path) of every bind."""
    out = {}
    for i, a in enumerate(argv):
        if a in ("--bind", "--ro-bind", "--dev-bind"):
            out[argv[i + 2]] = (a, argv[i + 1])
    return out


def test_mounts_keep_todays_layout_and_modes(tmp_path):
    m = _mounts(tmp_path)
    argv, _ = bwrap_args(m, ["python", "-m", "x"], {}, ENV_DIR, 64)
    b = _binds(argv)
    assert b["/agent"] == ("--bind", str(m.agent)) and b["/workspace"] == ("--bind", str(m.workspace))
    assert b["/workspace/staging"] == ("--bind", str(m.staging))
    for path, host in (("/context", m.context), ("/ar_contract", m.contract), ("/run/ar", m.sockets),
                       (ENV_DIR, ENV_DIR), ("/usr", "/usr")):
        assert b[path] == ("--ro-bind", str(host))
    assert argv.index("/workspace") < argv.index("/workspace/staging")          # staging over the workspace
    assert argv[-8:] == ["--remount-ro", "/", "--chdir", "/workspace", "--", "python", "-m", "x"]
    assert {"--unshare-user", "--unshare-pid", "--disable-userns", "--die-with-parent", "--new-session"} <= set(argv)
    assert "--unshare-net" not in argv and "--clearenv" not in argv
    ro = bwrap_args(_mounts(tmp_path, agent_readonly=True), ["true"], {}, ENV_DIR, 64)[0]
    assert _binds(ro)["/agent"] == ("--ro-bind", str(m.agent))


def test_bwrap_is_found_on_the_kernels_path(tmp_path, monkeypatch):
    """Popen would search the sandbox's PATH, which has no /usr/local/bin."""
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/opt/bin/" + name)
    assert bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)[0][0] == "/opt/bin/bwrap"
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="install bubblewrap"):
        bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)


def test_tmp_and_shm_are_sized_tmpfs(tmp_path):
    argv, _ = bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 8)
    i, j = argv.index("/tmp"), argv.index("/dev/shm")
    assert argv[i - 3:i] == ["--size", str(4 * 2**30), "--tmpfs"] and argv[j - 3:j] == ["--size", str(8 * 2**30), "--tmpfs"]


def test_no_gpu_device_and_no_sys_without_gpus(tmp_path):
    argv, _ = bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)
    assert not any(a.startswith("/dev/nvidia") for a in argv) and "/sys" not in argv


def test_exactly_the_given_minors_with_gpus(tmp_path):
    argv, _ = bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64, gpu_minors=[1, 0])
    devs = sorted(p for p, (flag, _) in _binds(argv).items() if flag == "--dev-bind")
    assert devs == ["/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl"]
    assert _binds(argv)["/sys"] == ("--ro-bind", "/sys")


def test_the_environment_holds_only_the_given_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host")
    _, env = bwrap_args(_mounts(tmp_path), ["true"], {"AR_TOKEN": "t", "AR_NODE": "n1"}, ENV_DIR, 64)
    assert env == {"PATH": f"{ENV_DIR}/bin:/usr/bin:/bin", "HOME": "/workspace/.home", "PYTHONPATH": "/ar_contract",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "AR_SOCKET_DIR": "/run/ar",
                   "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context", "AR_WORKSPACE": "/workspace",
                   "AR_TOKEN": "t", "AR_NODE": "n1"}


def test_separate_running_code_is_mounted_read_only_and_named_in_the_env(tmp_path):
    argv, env = bwrap_args(_mounts(tmp_path, code=tmp_path / "code"), ["true"], {}, ENV_DIR, 64)
    assert _binds(argv)["/code"] == ("--ro-bind", str(tmp_path / "code")) and env["AR_CODE_DIR"] == "/code"
    assert "AR_CODE_DIR" not in bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)[1]


def test_every_phase_sees_the_same_node_files_and_only_eval_is_hidden(tmp_path):
    """edit_self reads what improve_recipe reads: only prompts and tools differ."""
    n1, root = tmp_path / "n1", tmp_path / "root"
    for folder in ("eval", "attempts/improve_recipe-1/context", "contract/attempt-1"):
        (n1 / folder).mkdir(parents=True)
    root.mkdir()
    argv, _ = bwrap_args(_mounts(tmp_path, nodes={"n1": n1, "root": root}), ["true"], {}, ENV_DIR, 64)
    b = _binds(argv)
    assert b["/nodes/n1"] == ("--ro-bind", str(n1)) and b["/nodes/root"] == ("--ro-bind", str(root))
    assert [a for a in argv if a.startswith("/nodes/") and a.endswith("/eval")] == ["/nodes/n1/eval"] * 2
    i = argv.index("/nodes/n1/eval")
    assert argv[i - 1] == "--tmpfs" and argv[i + 1:i + 3] == ["--remount-ro", "/nodes/n1/eval"]
    assert not any(a.startswith("/store") for a in argv)


def test_snapshot_diff_reports_changes(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    before = snapshot(tmp_path, hash_files=True)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    assert diff(before, snapshot(tmp_path, hash_files=True)) == {
        "added": ["c.txt"], "removed": ["b.txt"], "changed": ["a.txt"]}


class _FakePopen:
    """Writes `out`/`err` to the files it was given; `wait` raises `raises` once, if set."""
    def __init__(self, out="out", err="err", code=0, raises=None):
        self.out, self.err, self.code, self.raises, self.killed, self.calls = out, err, code, raises, False, []

    def __call__(self, argv, env, stdin, stdout, stderr, start_new_session):
        self.calls.append({"argv": argv, "env": env, "start_new_session": start_new_session})
        stdout.write(self.out), stderr.write(self.err)
        self.returncode = None
        return self

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.raises is not None:
            raises, self.raises = self.raises, None
            raise raises
        self.returncode = -9 if self.killed else self.code
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def mounts(tmp_path):
    m = _mounts(tmp_path)
    for d in (m.agent, m.workspace, m.staging, m.context, m.sockets):
        d.mkdir()
    return m


def _run(tmp_path, mounts, env=None, gpus=(), recorder=None):
    return run_container(name="ar-t-n1-test-1-abc", mounts=mounts, command=["true"], env=env or {},
                         env_dir=tmp_path / "env", shm_gb=8, recorder=recorder or Recorder(tmp_path / "run"),
                         node="n1", phase="test", attempt=1, gpus=gpus)


def test_exit_code_and_streams_come_from_the_output_files(tmp_path, mounts, monkeypatch):
    fake = _FakePopen(out="hello\n", err="oops\n", code=3)
    monkeypatch.setattr(runner.subprocess, "Popen", fake)
    recorder = Recorder(tmp_path / "run")
    res = _run(tmp_path, mounts, recorder=recorder)
    assert (res.exit_code, res.stdout, res.stderr) == (3, "hello\n", "oops\n")
    assert (tmp_path / "sandbox.stdout").read_text() == "hello\n" and not fake.killed
    assert fake.calls[0]["start_new_session"]                 # a Ctrl-C to the kernel's group misses it
    (end,) = [e for e in recorder.read_events("n1") if e["type"] == "sandbox.end"]
    assert end["exit_code"] == 3 and recorder.load_payload(end["payload"])["stderr"] == "oops\n"


def test_an_exception_during_the_wait_kills_the_sandbox_and_restores_access(tmp_path, mounts, monkeypatch):
    from ar_kernel.control import ForceStop
    fake = _FakePopen(raises=ForceStop("SIGTERM"))
    monkeypatch.setattr(runner.subprocess, "Popen", fake)
    restored = []
    monkeypatch.setattr(runner, "restore_owner_access", restored.append)
    with pytest.raises(ForceStop):
        _run(tmp_path, mounts)
    assert fake.killed and restored == [mounts.agent, mounts.workspace, mounts.staging]


def test_a_failed_launch_is_recorded_and_only_the_recorded_settings_are_redacted(tmp_path, mounts, monkeypatch):
    """The runner never records the AR_TOKEN value, whichever Recorder is used (no add_redaction
    here), but the sandbox itself still gets the real token."""
    seen = []

    def no_bwrap(argv, env, **kw):
        seen.append(env)
        raise FileNotFoundError(2, "No such file or directory", "bwrap")
    monkeypatch.setattr(runner.subprocess, "Popen", no_bwrap)
    secret = "ar-super-secret-token-value"
    recorder = Recorder(tmp_path / "run")
    with pytest.raises(KernelFault, match="bwrap"):
        _run(tmp_path, mounts, env={"AR_TOKEN": secret}, recorder=recorder)
    assert seen[0]["AR_TOKEN"] == secret
    events = recorder.read_events("n1")
    (start,) = [e for e in events if e["type"] == "sandbox.start"]
    payload = recorder.load_payload(start["payload"])
    assert payload["settings"]["environment"]["AR_TOKEN"] == "<redacted>" and secret not in json.dumps(payload)
    assert secret not in " ".join(payload["settings"]["argv"])
    assert [e for e in events if e["type"] == "sandbox.error"]
    assert secret not in recorder.events_path("n1").read_text()


def test_a_card_without_a_minor_is_a_kernel_fault(tmp_path, mounts, monkeypatch):
    from ar_kernel.config import GpuPolicyError

    def no_minor(gpus):
        raise GpuPolicyError("nvidia-smi -q -i 9 shows no Minor Number")
    monkeypatch.setattr(runner, "device_minors", no_minor)
    recorder = Recorder(tmp_path / "run")
    with pytest.raises(KernelFault, match="Minor Number"):
        _run(tmp_path, mounts, gpus=[9], recorder=recorder)
    assert [e for e in recorder.read_events("n1") if e["type"] == "sandbox.error"]


def test_an_unreadable_output_file_is_recorded(tmp_path, mounts, monkeypatch):
    class Unreadable(_FakePopen):
        def wait(self, timeout=None):
            (tmp_path / "sandbox.stderr").unlink()
            (tmp_path / "sandbox.stderr").mkdir()
            return super().wait()
    monkeypatch.setattr(runner.subprocess, "Popen", Unreadable())
    recorder = Recorder(tmp_path / "run")
    res = _run(tmp_path, mounts, recorder=recorder)
    assert res.stdout == "out" and res.stderr.startswith("could not read the sandbox's stderr:")
    errors = [recorder.load_payload(e["payload"]) for e in recorder.read_events("n1") if e["type"] == "sandbox.error"]
    assert errors and "stderr" in errors[0]["stderr"]


def test_bwraps_own_error_is_a_kernel_fault_and_the_agents_exit_is_not(tmp_path, mounts, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen(err="bwrap: Can't mount tmpfs on /tmp\n", code=1))
    with pytest.raises(KernelFault, match="Can't mount tmpfs"):
        _run(tmp_path, mounts)
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen(err="Traceback ...\n", code=137))
    assert _run(tmp_path, mounts).exit_code == 137


def test_on_start_gets_the_sandbox_pid(tmp_path, mounts, monkeypatch):
    fake, seen = _FakePopen(), []
    fake.pid = 4242
    monkeypatch.setattr(runner.subprocess, "Popen", fake)
    run_container(name="c", mounts=mounts, command=["true"], env={}, env_dir=tmp_path / "env", shm_gb=8,
                  recorder=Recorder(tmp_path / "run"), node="n1", phase="test", attempt=1, on_start=seen.append)
    assert seen == [4242]


def test_cancel_kills_a_running_sandbox(tmp_path, mounts, monkeypatch):
    class Sleeping(_FakePopen):
        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("bwrap", timeout)
            return super().wait()
    fake, cancel = Sleeping(), threading.Event()
    monkeypatch.setattr(runner.subprocess, "Popen", fake)
    threading.Timer(0.3, cancel.set).start()
    res = run_container(name="c", mounts=mounts, command=["true"], env={}, env_dir=tmp_path / "env", shm_gb=8,
                        recorder=Recorder(tmp_path / "run"), node="n1", phase="test", attempt=1,
                        cancel=cancel, poll_s=0.05)
    assert fake.killed and res.exit_code == -9


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


def test_restored_access_lets_the_kernel_copy_and_delete_the_tree(tmp_path):
    from ar_kernel.sandbox.runner import restore_owner_access

    ws = tmp_path / "ws"
    ws.mkdir()
    _lock_up(ws)
    (ws / "link").symlink_to(tmp_path / "elsewhere")          # dangling link: must not be followed
    restore_owner_access(ws)
    shutil.copytree(ws, tmp_path / "copy", symlinks=True)
    shutil.rmtree(ws)                                         # re-running an attempt number
    assert (tmp_path / "copy" / "sealed" / "inner.txt").read_text() == "y"
