"""Failed subprocess errors include stderr and traceback details."""
from types import SimpleNamespace
from ar_kernel.subproc import output_tail


def test_tail_includes_stderr_traceback():
    proc = SimpleNamespace(
        stdout="[run_wbench] 40 cases in 6 turn-count buckets\n",
        stderr="Traceback (most recent call last):\nValueError: not in the subpath of\n")
    out = output_tail(proc)
    assert "ValueError: not in the subpath of" in out
    assert "[run_wbench] 40 cases" in out
    assert "stdout" in out and "stderr" in out


def test_tail_truncates_each_stream_to_limit():
    proc = SimpleNamespace(stdout="a" * 9000, stderr="b" * 9000)
    out = output_tail(proc, 100)
    # the "(last 100)" headers contain letters too, so assert on the runs themselves
    assert "a" * 100 in out and "a" * 101 not in out
    assert "b" * 100 in out and "b" * 101 not in out


def test_tail_omits_empty_streams():
    assert "stderr" not in output_tail(SimpleNamespace(stdout="only out", stderr="   "))
    assert "stdout" not in output_tail(SimpleNamespace(stdout="", stderr="only err"))


def test_tail_reports_when_nothing_was_captured():
    assert output_tail(SimpleNamespace(stdout="", stderr="")) == "(no output captured)"
