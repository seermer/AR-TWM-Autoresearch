import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import telemetry_payload, telemetry_records

from ar_kernel import subproc, telemetry
from ar_kernel.faults import KernelFault
from ar_kernel.sandbox import runner
from ar_kernel.sandbox.runner import Mounts, bwrap_args, container_name, diff, run_container, snapshot
from ar_kernel.telemetry import tree

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


def test_bwrap_comes_from_the_project_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_env", lambda name: Path("/p/.envs") / name)
    assert bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)[0][0] == "/p/.envs/bwrap/bin/bwrap"
    monkeypatch.setattr(runner, "project_env", lambda name: None)
    with pytest.raises(FileNotFoundError, match="setup_envs.sh bwrap"):
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


def test_earlier_subphases_are_mounted_read_only_in_order(tmp_path):
    argv, _ = bwrap_args(_mounts(tmp_path, subphases=[tmp_path / "p1", tmp_path / "p3"]), ["true"], {}, ENV_DIR, 64)
    b = _binds(argv)
    assert b["/earlier/1"] == ("--ro-bind", str(tmp_path / "p1")) and b["/earlier/2"] == ("--ro-bind", str(tmp_path / "p3"))
    assert argv.index("/earlier/2") < argv.index("--remount-ro")
    assert not any(a.startswith("/earlier") for a in bwrap_args(_mounts(tmp_path), ["true"], {}, ENV_DIR, 64)[0])


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


_POPEN = subprocess.Popen


class _FakePopen:
    """Starts a shell in place of bwrap: it writes `out`/`err` and exits with `code`, or sleeps if `code` is None."""
    def __init__(self, out="out", err="err", code=0):
        self.out, self.err, self.code, self.calls, self.proc = out, err, code, [], None

    def __call__(self, argv, env, stdin, stdout, stderr, start_new_session):
        self.calls.append({"argv": argv, "env": env, "start_new_session": start_new_session})
        script = 'printf %s "$OUT"; printf %s "$ERR" >&2; ' + ("exec sleep 600" if self.code is None else f"exit {self.code}")
        self.proc = _POPEN(["sh", "-c", script], env={"OUT": self.out, "ERR": self.err, "PATH": os.environ["PATH"]},
                           stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session)
        return self.proc


@pytest.fixture
def mounts(tmp_path):
    m = _mounts(tmp_path)
    for d in (m.agent, m.workspace, m.staging, m.context, m.sockets):
        d.mkdir()
    return m


def _run(tmp_path, mounts, env=None, gpus=()):
    return run_container(name="ar-t-n1-test-1-abc", mounts=mounts, command=["true"], env=env or {},
                         env_dir=tmp_path / "env", shm_gb=8, gpus=gpus)


def test_exit_code_and_streams_come_from_the_output_files(tmp_path, mounts, monkeypatch):
    fake = _FakePopen(out="hello\n", err="oops\n", code=3)
    monkeypatch.setattr(subproc.subprocess, "Popen", fake)
    res = _run(tmp_path, mounts)
    assert (res.exit_code, res.stdout, res.stderr) == (3, "hello\n", "oops\n")
    assert (tmp_path / "sandbox.stdout").read_text() == "hello\n"
    assert fake.calls[0]["start_new_session"]                 # a Ctrl-C to the kernel's group misses it


def test_an_exception_during_the_wait_kills_the_sandbox_and_restores_access(tmp_path, mounts, monkeypatch):
    from ar_kernel.control import ForceStop
    fake = _FakePopen(code=None)
    monkeypatch.setattr(subproc.subprocess, "Popen", fake)

    def stopped(*a, **k):
        raise ForceStop("SIGTERM")
    monkeypatch.setattr(subproc, "wait_leader", stopped)
    restored = []
    monkeypatch.setattr(runner, "restore_owner_access", restored.append)
    with pytest.raises(ForceStop):
        _run(tmp_path, mounts)
    assert fake.proc.returncode == -9 and restored == [mounts.agent, mounts.workspace, mounts.staging]


def test_a_failed_launch_is_a_kernel_fault_and_the_sandbox_gets_the_real_token(tmp_path, mounts, monkeypatch):
    seen = []

    def no_bwrap(argv, env, **kw):
        seen.append(env)
        raise FileNotFoundError(2, "No such file or directory", "bwrap")
    monkeypatch.setattr(subproc.subprocess, "Popen", no_bwrap)
    secret = "ar-super-secret-token-value"
    with pytest.raises(KernelFault, match="bwrap"):
        _run(tmp_path, mounts, env={"AR_TOKEN": secret, "HF_TOKEN": "hf-secret"})
    assert seen[0]["AR_TOKEN"] == secret and seen[0]["HF_TOKEN"] == "hf-secret"


def test_a_card_without_a_minor_is_a_kernel_fault(tmp_path, mounts, monkeypatch):
    from ar_kernel.config import GpuPolicyError

    def no_minor(gpus):
        raise GpuPolicyError("nvidia-smi -q -i 9 shows no Minor Number")
    monkeypatch.setattr(runner, "device_minors", no_minor)
    with pytest.raises(KernelFault, match="Minor Number"):
        _run(tmp_path, mounts, gpus=[9])


def test_an_unreadable_output_file_is_reported(tmp_path, mounts, monkeypatch):
    real_wait = subproc.wait_leader

    def unreadable(*a, **k):
        (tmp_path / "sandbox.stderr").unlink()
        (tmp_path / "sandbox.stderr").mkdir()
        return real_wait(*a, **k)
    monkeypatch.setattr(subproc.subprocess, "Popen", _FakePopen())
    monkeypatch.setattr(subproc, "wait_leader", unreadable)
    res = _run(tmp_path, mounts)
    assert res.stdout == "out" and res.stderr.startswith("could not read the sandbox's stderr:")


def test_bwraps_own_error_is_a_kernel_fault_and_the_agents_exit_is_not(tmp_path, mounts, monkeypatch):
    monkeypatch.setattr(subproc.subprocess, "Popen", _FakePopen(err="bwrap: Can't mount tmpfs on /tmp\n", code=1))
    with pytest.raises(KernelFault, match="Can't mount tmpfs"):
        _run(tmp_path, mounts)
    monkeypatch.setattr(subproc.subprocess, "Popen", _FakePopen(err="Traceback ...\n", code=137))
    assert _run(tmp_path, mounts).exit_code == 137


def test_on_start_gets_the_sandbox_pid(tmp_path, mounts, monkeypatch):
    fake, seen = _FakePopen(), []
    monkeypatch.setattr(subproc.subprocess, "Popen", fake)
    run_container(name="c", mounts=mounts, command=["true"], env={}, env_dir=tmp_path / "env", shm_gb=8, on_start=seen.append)
    assert seen == [fake.proc.pid]


def test_cancel_kills_a_running_sandbox(tmp_path, mounts, monkeypatch):
    fake, cancel = _FakePopen(code=None), threading.Event()
    monkeypatch.setattr(subproc.subprocess, "Popen", fake)
    threading.Timer(0.3, cancel.set).start()
    res = run_container(name="c", mounts=mounts, command=["true"], env={}, env_dir=tmp_path / "env", shm_gb=8,
                        cancel=cancel, poll_s=0.05)
    assert res.exit_code == -9


def test_snapshot_skips_fifos_and_marks_unreadable_files(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    (tmp_path / "locked.txt").write_text("secret")
    (tmp_path / "locked.txt").chmod(0)
    os.mkfifo(tmp_path / "pipe")
    try:
        snap = snapshot(tmp_path, True)
    finally:
        (tmp_path / "locked.txt").chmod(0o644)
    assert snap["locked.txt"] == (6, "unreadable") and "pipe" not in snap and len(snap["ok.txt"][1]) == 64


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


def _traced(tmp_path):
    run = tmp_path / "run"
    telemetry.open(run)
    telemetry.start("ar.run")
    return run


def test_a_sandbox_is_a_span_with_its_inputs_and_output_lines(tmp_path, mounts, monkeypatch):
    run = _traced(tmp_path)
    node = tmp_path / "node0001"
    (node / "eval").mkdir(parents=True)
    (node / "eval" / "case.mp4").write_text("hidden")
    (node / "recipe.yaml").write_text("lr: 1")
    mounts.nodes = {"node0001": node}
    monkeypatch.setattr(subproc.subprocess, "Popen", _FakePopen(out="one\ntwo\n", err="oops\n", code=3))
    run_container(name="c", mounts=mounts, command=["true"], env={"AR_TOKEN": "tok-123"}, env_dir=tmp_path / "env",
                  shm_gb=8)
    records = telemetry_records(run)
    [start] = [r for r in records if r["kind"] == "start" and r["name"] == "sandbox"]
    inputs = telemetry_payload(run, start["inputs"])
    assert inputs["name"] == "c" and inputs["argv"][-1] == "true" and inputs["env"]["AR_TOKEN"] == "tok-123"
    assert inputs["nodes"] == {"node0001": [["recipe.yaml", 5]]} and inputs["gpus"] == [] and inputs["pid"]
    lines = [(r["stream"], r["text"]) for r in records if r["kind"] == "log" and r["span"] == start["span"]]
    assert sorted(lines) == [("stderr", "oops"), ("stdout", "one"), ("stdout", "two")]
    [end] = [r for r in records if r["kind"] == "end" and r["span"] == start["span"]]
    assert end["status"] == "error" and telemetry_payload(run, end["output"])["exit_code"] == 3
    assert tree.check(tree.load(run)) == []


def test_the_changes_in_the_writable_folders_are_recorded_with_sizes(tmp_path, mounts):
    run = _traced(tmp_path)
    (mounts.workspace / "gone.txt").write_text("abc")
    (mounts.workspace / "kept.txt").write_text("k")
    before = runner.files(mounts)
    (mounts.workspace / "gone.txt").unlink()
    (mounts.workspace / "kept.txt").write_text("longer")
    (mounts.staging / "new.bin").write_bytes(b"12345")
    (mounts.agent / "a.py").write_text("x = 1")
    runner.record_changes(mounts, before)
    [changes] = [r for r in telemetry_records(run) if r["kind"] == "end" and r["name"] == "sandbox.changes"]
    assert telemetry_payload(run, changes["output"]) == {
        "/workspace": {"added": [], "removed": [["gone.txt", 3]], "changed": [["kept.txt", 6]]},
        "/workspace/staging": {"added": [["new.bin", 5]], "removed": [], "changed": []},
        "/agent": {"added": [["a.py", 5]], "removed": [], "changed": []}}
    assert tree.check(tree.load(run)) == []


def test_a_cancelled_sandbox_ends_killed_with_a_kill_why_cancel(tmp_path, mounts, monkeypatch):
    run = _traced(tmp_path)
    cancel = threading.Event()
    monkeypatch.setattr(subproc.subprocess, "Popen", _FakePopen(code=None))
    run_container(name="c", mounts=mounts, command=["true"], env={}, env_dir=tmp_path / "env", shm_gb=8,
                  cancel=cancel, poll_s=0.05, on_start=lambda pid: cancel.set())
    records = telemetry_records(run)
    [start] = [r for r in records if r["kind"] == "start" and r["name"] == "sandbox"]
    [end] = [r for r in records if r["kind"] == "end" and r["span"] == start["span"]]
    assert end["status"] == "killed" and telemetry_payload(run, end["output"])["signal"] == "SIGKILL"
    [kill] = [r for r in records if r["kind"] == "start" and r["name"] == "process.kill"]
    assert kill["parent"] == start["span"] and telemetry_payload(run, kill["inputs"])["why"] == "cancel"
    assert tree.check(tree.load(run)) == []


def test_restoring_access_records_only_the_modes_it_changes(tmp_path):
    run = _traced(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "fine.txt").write_text("x")
    _lock_up(ws)
    runner.restore_owner_access(ws)
    chmods = [telemetry_payload(run, r["inputs"])["path"] for r in telemetry_records(run)
              if r["kind"] == "start" and r["name"] == "fs.chmod"]
    assert sorted(chmods) == [str(ws / "locked.txt"), str(ws / "sealed")]
