"""`ar trace`: trees from records, open spans, and the `--check` findings."""
import json
import threading

import pytest

from ar_kernel import cli, telemetry
from ar_kernel.telemetry import registry, tree
from ar_kernel.telemetry.trace import render


@pytest.fixture
def run(tmp_path, monkeypatch):
    for name in ("node", "phase.attempt", "t.job"):
        monkeypatch.setitem(registry.SPANS, name, ("loop", "test"))
    monkeypatch.setitem(registry.DECISIONS, "t.pick", ("loop", "test"))
    run = tmp_path / "r1"
    (run / "config").mkdir(parents=True)
    (run / "config" / "run.json").write_text("{}")
    telemetry.open(run)
    return run


def _scripted(run):
    """ar.run > node n1 > phase.attempt 1 (two jobs at once, one never ends) and attempt 2; a decision."""
    root = telemetry.start("ar.run", {"resume": False})
    with telemetry.span("node", {"node": "n1"}) as node:
        telemetry.decision("t.pick", {"parent": "root"}, ["root"], "root", "the only node")
        for attempt in (1, 2):
            with telemetry.span("phase.attempt", {"attempt": attempt}) as sp:
                if attempt == 1:
                    a, b = telemetry.start("t.job", {"job": "a"}), telemetry.start("t.job", {"job": "b"})
                    telemetry.log("stdout", "both running")
                    t = threading.Thread(target=a.end, args=("ok", {"frames": 3}))
                    t.start()
                    t.join()
                    del b                                                  # never ends
                sp.output = {"ok": attempt == 2}
        node.output = "done"
    root.end("ok", {"exit_code": 0})


def test_a_tree_prints_in_order_indented_with_concurrent_spans_and_open_ones_marked(run):
    _scripted(run)
    text = render(tree.load(run))
    names = [line.split()[1:3] for line in text.splitlines() if line[:7].strip().isdigit()]
    assert [n[0] for n in names] == ["ar.run", "node", "·", "phase.attempt", "t.job", "∥", "·", "phase.attempt"]
    assert "∥ t.job" in text and "no end recorded; the writer's last record at" in text
    assert "process alive" in text and '"exit_code": 0' in text and 'decision t.pick: root (the only node)' in text
    lines = text.splitlines()
    assert lines[0].startswith("writer ") and lines[1].startswith("      2 ") and "  ar.run ok" in lines[1]


def test_node_attempt_from_and_depth_select_what_is_printed(run):
    _scripted(run)
    trace = tree.load(run)
    second = render(trace, node="n1", attempt=2)
    assert "phase.attempt ok" in second and "t.job" not in second and '"attempt": 2' in second
    shallow = render(trace, node="n1", depth=2)
    assert "phase.attempt" in shallow and "t.job" not in shallow and "phase" not in render(trace, node="n1", depth=1)
    seqs = [int(line[:7]) for line in render(trace, since=6).splitlines() if line[:7].strip().isdigit()]
    assert seqs and min(seqs) >= 6
    full = render(trace, full=True)
    assert '{\n' in full and '"node": "n1"' in full


def test_an_open_span_of_another_host_is_not_checkable_here(run):
    _scripted(run)
    trace = tree.load(run)
    trace.writers[0]["host"] = "elsewhere"
    assert "process not checkable here" in render(trace)


def test_a_clean_run_has_no_findings_even_with_records_after_their_span_ended(run):
    _scripted(run)
    late = telemetry.start("t.job", {"job": "late"})                     # under ar.run, after it ended
    late.end("ok", {"x": 1})
    telemetry.log("logging", "a server's line after its span ended")
    assert tree.check(tree.load(run)) == []


def test_check_lists_each_kind_of_telemetry_bug_with_where_it_is(run):
    _scripted(run)
    w = telemetry.writer()
    sp = telemetry.start("t.job", {"job": "c"})
    w.write("end", "loop", sp.id, name="t.job", status="ok")                          # neither output nor error
    sp.end("ok", {"again": True})                                                     # a second end
    w.write("start", "loop", "orphan", name="t.job", parent=None)                     # no parent
    w.write("start", "loop", "lost", name="t.job", parent="missing")                  # parent never started
    w.write("decision", "loop", None, name="no.such", options=[], chosen=None, reason="")
    w.write("end", "loop", "ghost", payloads={"output": 1}, name="t.job", status="ok")
    path = next((run / "telemetry" / "records").iterdir())
    with path.open("a") as f:
        f.write('{"v": 1, "writer": "x", "se')                                        # a torn last line
    findings = tree.check(tree.load(run))
    wid = w.id
    seq = {r["span"]: r["seq"] for r in tree.load(run).records if r["kind"] == "start"}
    assert findings == [
        f"unreadable line: telemetry/records/{path.name}:{len(path.read_text().splitlines())}",
        f"end with neither output nor error: {wid}#{seq[sp.id] + 1} end t.job",
        f"no parent: {wid}#{seq['orphan']} start t.job",
        f"parent never started: {wid}#{seq['lost']} start t.job",
        f"unregistered name: {wid}#{seq['lost'] + 1} decision no.such",
        f"no parent: {wid}#{seq['lost'] + 1} decision no.such",
        f"end without a start: {wid}#{seq['lost'] + 2} end t.job",
        f"2 ends: {wid}#{seq[sp.id]} t.job"]


def test_ar_trace_prints_and_checks_without_writing(run, monkeypatch, capsys):
    _scripted(run)
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: run.parent))
    before = sorted(p for p in run.rglob("*"))
    assert cli.main(["trace", "--run-id", "r1", "--node", "n1"]) == 0
    assert "node ok" in capsys.readouterr().out
    assert cli.main(["trace", "--run-id", "r1", "--check"]) == 0
    assert capsys.readouterr().out.strip() == "no findings"
    w = telemetry.writer()
    w.write("start", "loop", "orphan", name="t.job", parent=None)
    assert cli.main(["trace", "--run-id", "r1", "--check"]) == 1
    assert "no parent" in capsys.readouterr().out
    assert sorted(p for p in run.rglob("*")) == before
    assert json.loads((run / "config" / "run.json").read_text()) == {}
