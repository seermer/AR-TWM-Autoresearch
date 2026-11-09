"""Processes as spans: launch, output lines, exit, kills."""
import subprocess
import threading
import time

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import subproc, telemetry
from ar_kernel.subproc import run_in_env

ENV = "autoresearcher"


@pytest.fixture
def run(tmp_path):
    telemetry.open(tmp_path / "run")
    telemetry.start("ar.run")
    return tmp_path / "run"


def _spans(run, name):
    """(start, end or None, inputs, output) of each span named `name`."""
    records = telemetry_records(run)
    ends = {r["span"]: r for r in records if r["kind"] == "end"}
    out = []
    for s in (r for r in records if r["kind"] == "start" and r["name"] == name):
        e = ends.get(s["span"])
        out.append((s, e, telemetry_payload(run, s["inputs"]),
                    telemetry_payload(run, e["output"]) if e and e.get("output") else None))
    return out


def test_output_lines_arrive_while_the_process_runs_and_the_exit_is_recorded(run, tmp_path):
    flag = tmp_path / "go"
    script = f"print('one', flush=True)\nimport os, time, sys\nwhile not os.path.exists({str(flag)!r}): time.sleep(0.1)\n" \
             "print('two'); print('err line', file=sys.stderr); print('no newline', end=''); sys.exit(3)"

    def release():
        deadline = time.time() + 60
        while time.time() < deadline:
            if any(r.get("text") == "one" for r in telemetry_records(run)):
                flag.touch()
                return
            time.sleep(0.1)
    t = threading.Thread(target=release)
    t.start()
    proc = run_in_env(ENV, ["python", "-c", script], cwd=tmp_path, poll_s=0.2)
    t.join()
    assert proc.returncode == 3 and flag.exists()                   # "one" was recorded before the exit
    [(start, end, inputs, output)] = _spans(run, "process")
    logs = [(r["stream"], r["text"]) for r in telemetry_records(run) if r["kind"] == "log"]
    assert {("stdout", "one"), ("stdout", "two"), ("stderr", "err line"), ("stdout", "no newline")} <= set(logs)
    assert all(r["span"] == start["span"] for r in telemetry_records(run) if r["kind"] == "log")
    assert inputs["pid"] == inputs["pgid"] and inputs["command"][:2] == ["conda", "run"] and inputs["cwd"] == str(tmp_path)
    assert end["status"] == "error" and output["exit_code"] == 3 and output["rusage"]["ru_maxrss"] > 0
    [(kill, _, kill_in, kill_out)] = _spans(run, "process.kill")
    assert kill["parent"] == start["span"] and kill_in["why"] == "leftover" and kill_out["already_gone"] is True


def test_a_cancel_is_a_kill_span_and_the_end_names_the_signal(run, tmp_path):
    cancel = threading.Event()
    threading.Timer(2.0, cancel.set).start()
    proc = run_in_env(ENV, ["bash", "-c", "echo started; sleep 600"], cwd=tmp_path, cancel=cancel, poll_s=0.2)
    assert proc.cancelled
    [(start, end, _, output)] = _spans(run, "process")
    kills = _spans(run, "process.kill")
    assert [(k[2]["why"], k[3]["signals"]) for k in kills][0] == ("cancel", ["SIGTERM"])
    assert all(k[0]["parent"] == start["span"] for k in kills)
    assert end["status"] == "killed" and output["signal"] == "SIGTERM"


def test_a_failed_launch_records_no_process_span(run, tmp_path):
    with pytest.raises(OSError):
        run_in_env(ENV, ["true"], cwd=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        subproc.run(["/no/such/program"])
    assert _spans(run, "process") == []


def test_the_recorded_environment_is_redacted(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-SECRET-VALUE\n")
    telemetry.open(tmp_path / "run", [env_file], {})
    telemetry.start("ar.run")
    run_in_env(ENV, ["true"], cwd=tmp_path, extra_env={"OPENAI_API_KEY": "sk-SECRET-VALUE", "OTHER": "x"})
    [(_, _, inputs, _)] = _spans(tmp_path / "run", "process")
    assert inputs["env"]["OPENAI_API_KEY"] == "<redacted>" and inputs["env"]["OTHER"] == "x"
    assert "sk-SECRET-VALUE" not in "".join(p.read_text() for p in (tmp_path / "run" / "telemetry").rglob("*.jsonl"))


def test_run_records_a_failing_command_with_its_output(run, tmp_path):
    proc = subproc.run(["sh", "-c", "echo out; echo err >&2; exit 4"], cwd=tmp_path)
    assert (proc.returncode, proc.stdout, proc.stderr) == (4, "out\n", "err\n")
    [(start, end, inputs, output)] = _spans(run, "process")
    assert end["status"] == "error" and output == {"exit_code": 4, "stdout": "out\n", "stderr": "err\n"}
    with pytest.raises(subprocess.CalledProcessError):
        subproc.run(["false"], check=True)


def test_run_with_a_timeout_kills_and_ends_timeout(run):
    with pytest.raises(subprocess.TimeoutExpired):
        subproc.run(["sleep", "30"], timeout=0.5)
    [(start, end, _, output)] = _spans(run, "process")
    [(kill, _, kill_in, kill_out)] = _spans(run, "process.kill")
    assert end["status"] == "timeout" and output["signal"] == "SIGKILL"
    assert kill["parent"] == start["span"] and kill_in["why"] == "timeout" and kill_out["signals"] == ["SIGKILL"]
