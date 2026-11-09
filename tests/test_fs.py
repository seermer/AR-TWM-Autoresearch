"""File changes as spans: what changes, the state before and after."""
import os
import shutil

import pytest
from conftest import telemetry_payload, telemetry_records

from ar_kernel import fs, telemetry


@pytest.fixture
def run(tmp_path):
    telemetry.open(tmp_path / "run")
    telemetry.start("ar.run")
    return tmp_path / "run"


def _spans(run):
    """[(name, status, inputs, output)] of every span but the root, in start order."""
    records = telemetry_records(run)
    ends = {r["span"]: r for r in records if r["kind"] == "end"}
    out = []
    for s in records:
        if s["kind"] == "start" and s["name"] != "ar.run":
            e = ends[s["span"]]
            out.append((s["name"], e["status"], telemetry_payload(run, s["inputs"]),
                        telemetry_payload(run, e["output"]) if e.get("output") else None))
    return out


def test_write_text_records_the_text_and_the_sizes_before_and_after(run, tmp_path):
    path = tmp_path / "a.json"
    fs.write_text(path, "first")
    fs.write_text(path, "second, longer", atomic=True)
    assert path.read_text() == "second, longer" and not list(tmp_path.glob(".a.json.*"))
    assert _spans(run) == [
        ("fs.write", "ok", {"path": str(path), "size_before": None, "text": "first"}, {"size": 5}),
        ("fs.write", "ok", {"path": str(path), "size_before": 5, "text": "second, longer"}, {"size": 14})]


def test_write_text_can_record_the_texts_hash_instead(run, tmp_path):
    import hashlib
    fs.write_text(tmp_path / "t.jsonl", "line\n", atomic=True, record_text=False)
    assert _spans(run) == [("fs.write", "ok", {"path": str(tmp_path / "t.jsonl"), "size_before": None,
                                               "sha256": hashlib.sha256(b"line\n").hexdigest()}, {"size": 5})]


def test_write_bytes_records_sizes_not_bytes(run, tmp_path):
    fs.write_bytes(tmp_path / "b.bin", b"\x00" * 10)
    assert _spans(run) == [("fs.write", "ok", {"path": str(tmp_path / "b.bin"), "size_before": None, "bytes": 10},
                            {"size": 10})]


def test_remove_and_rmtree_name_each_file_with_its_size_and_skip_what_is_gone(run, tmp_path):
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "x").write_text("12")
    (tmp_path / "d" / "sub" / "y").write_text("345")
    (tmp_path / "f").write_text("6789")
    fs.remove(tmp_path / "f")
    fs.rmtree(tmp_path / "d")
    fs.remove(tmp_path / "f")
    fs.rmtree(tmp_path / "d")
    assert _spans(run) == [
        ("fs.remove", "ok", {"path": str(tmp_path / "f"), "size": 4}, {"removed": True}),
        ("fs.rmtree", "ok", {"path": str(tmp_path / "d"), "files": [["sub/y", 3], ["x", 2]]}, {"removed": True})]


def test_an_ignored_failure_is_an_error_record_with_the_reason(run, tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()

    def denied(path):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(shutil, "rmtree", denied)
    fs.rmtree(tmp_path / "d", ignore_errors="a scratch folder")
    with pytest.raises(PermissionError):
        fs.rmtree(tmp_path / "d")
    [(_, status, _, output), (_, failed, _, _)] = _spans(run)
    [err] = [r for r in telemetry_records(run) if r["kind"] == "error"]
    assert status == "ok" and output == {"removed": False} and failed == "error"
    assert err["handled_as"] == "ignored: a scratch folder"


def test_rename_link_copy_chmod_mkdir_and_copytree(run, tmp_path):
    src = tmp_path / "src"
    fs.mkdir(src / "deep")
    fs.mkdir(src / "deep")                                              # exists: nothing changes
    (src / "deep" / "f").write_text("abc")
    fs.copytree(src, tmp_path / "linked", link=True)
    fs.link(src / "deep" / "f", tmp_path / "hard")
    fs.copy(src / "deep" / "f", tmp_path / "copied")
    fs.chmod(tmp_path / "copied", 0o600)
    fs.rename(tmp_path / "copied", tmp_path / "moved")
    assert os.path.samefile(src / "deep" / "f", tmp_path / "linked" / "deep" / "f")
    names = [(n, s) for n, s, _, _ in _spans(run)]
    assert names == [("fs.mkdir", "ok"), ("fs.copytree", "ok"), ("fs.link", "ok"), ("fs.copy", "ok"),
                     ("fs.chmod", "ok"), ("fs.rename", "ok")]
    spans = {n: (i, o) for n, _, i, o in _spans(run)}
    assert spans["fs.copytree"] == ({"src": str(src), "dst": str(tmp_path / "linked"), "link": True,
                                     "files": [["deep/f", 3]]}, {"files": 1})
    assert spans["fs.chmod"][1] == {"mode": "0o600"} and spans["fs.rename"][1] == {"size": 3}


def test_a_failed_change_ends_its_span_with_the_error(run, tmp_path):
    with pytest.raises(FileNotFoundError):
        fs.write_text(tmp_path / "missing" / "x", "y")
    [(name, status, _, _)] = _spans(run)
    assert (name, status) == ("fs.write", "error")
