"""The run folder's upload to Hugging Face after every phase, and its fetch."""
import io
import sqlite3
import tarfile

import pytest

from ar_kernel import hf_upload
from ar_kernel.hf_upload import HfUpload, fetch, run_files
from ar_kernel.telemetry.recorder import Recorder


class FakeHub:
    """whoami / create_repo / create_commit / list_repo_files / hf_hub_download over a dict of uploaded files."""
    def __init__(self, fail=False):
        self.files, self.repos, self.fail = {}, [], fail

    def whoami(self):
        return {"name": "me"}

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.repos.append((repo, repo_type, private))

    def create_commit(self, repo_id, operations, repo_type, commit_message):
        if self.fail:
            raise ConnectionError("hub down")
        for op in operations:
            self.files[op.path_in_repo] = open(op.path_or_fileobj, "rb").read()

    def list_repo_files(self, repo, repo_type):
        return [".gitattributes", *self.files]

    def download(self, repo_id, filename, repo_type, local_dir, token):
        path = f"{local_dir}/x.tar"
        open(path, "wb").write(self.files[filename])
        return path


def _run(tmp_path):
    run = tmp_path / "myrun"
    for rel in ("config/run.json", "nodes/n1/eval/a.mp4", "nodes/n1/attempts/improve_recipe-1/workspace/.home/big",
                "nodes/n1/attempts/improve_recipe-1/workspace/result.json", "staging/n1/improve_recipe-1/hf/raw",
                "staging/n1/improve_recipe-1/pool/c.mp4", "cache/text_embed/e", "control/state.json"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(rel)
    with sqlite3.connect(run / "archive.db") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE t (x)")
        db.execute("INSERT INTO t VALUES (1)")
    return run


def _members(data: bytes) -> list[str]:
    return sorted(tarfile.open(fileobj=io.BytesIO(data)).getnames())


def test_each_upload_holds_the_changed_files_a_database_copy_and_the_file_list(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    up("root", "eval", 0)
    assert hub.repos == [("me/myrun", "dataset", True)]
    assert _members(hub.files["uploads/00001/0000.tar"]) == [
        ".hf_manifest.json", "archive.db", "config/run.json", "control/state.json",
        "nodes/n1/attempts/improve_recipe-1/workspace/result.json", "nodes/n1/eval/a.mp4",
        "staging/n1/improve_recipe-1/pool/c.mp4"]
    (run / "nodes/n1/eval/b.mp4").write_text("new")
    up("n1", "eval", 0)
    # the first upload's own event is in the second
    assert [m for m in _members(hub.files["uploads/00002/0000.tar"]) if not m.startswith("telemetry/payloads/")] == [".hf_manifest.json", "archive.db", "nodes/n1/eval/b.mp4",
                                                       "telemetry/events/root.jsonl"]
    assert [e["type"] for e in Recorder(run).read_events("n1")] == ["hf_upload"]


def test_a_failed_upload_is_an_alert_and_its_files_go_with_the_next(tmp_path):
    run, hub = _run(tmp_path), FakeHub(fail=True)
    rec = Recorder(run)
    up = HfUpload(run, rec, "tok", api=hub)
    up("root", "eval", 0)
    alerts = [e for e in rec.read_events() if e["type"] == "alert"]
    assert alerts[-1]["kind"] == "hf_upload_failed" and "hub down" in alerts[-1]["message"]
    hub.fail = False
    up("n1", "edit_self", 1)
    assert list(hub.files) == ["uploads/00001/0000.tar"] and "config/run.json" in _members(hub.files["uploads/00001/0000.tar"])


def test_fetch_rebuilds_the_folder_and_drops_what_the_run_removed(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    up("root", "eval", 0)
    (run / "nodes/n1/eval/a.mp4").unlink()
    (run / "nodes/n1/eval/b.mp4").write_text("b")
    up("n1", "eval", 0)
    there = tmp_path / "elsewhere" / "myrun"
    assert fetch(there, None, "tok", api=hub, download=hub.download) == f"fetched 2 uploads of me/myrun into {there}"
    assert sorted(f for f in run_files(there) if not f.startswith("telemetry/payloads/")) == [
        "config/run.json", "control/state.json", "nodes/n1/attempts/improve_recipe-1/workspace/result.json",
        "nodes/n1/eval/b.mp4", "staging/n1/improve_recipe-1/pool/c.mp4", "telemetry/events/root.jsonl"]
    assert not (there / "nodes/n1/eval/a.mp4").exists() and (there / "nodes/n1/eval/b.mp4").read_text() == "b"
    assert sqlite3.connect(there / "archive.db").execute("SELECT x FROM t").fetchall() == [(1,)]
    assert fetch(there, None, "tok", api=hub, download=hub.download) == f"{there} is up to date with me/myrun"
    with pytest.raises(FileExistsError, match="was not made by fetch"):
        fetch(run, None, "tok", api=hub, download=hub.download)


def test_a_large_upload_is_split_into_parts_of_one_commit(tmp_path, monkeypatch):
    run, hub = _run(tmp_path), FakeHub()
    monkeypatch.setattr(hf_upload, "PART_BYTES", 40)
    monkeypatch.chdir(tmp_path)
    HfUpload(run.relative_to(tmp_path), Recorder(run), "tok", api=hub)("root", "eval", 0)
    parts = sorted(hub.files)
    assert len(parts) > 1 and all(p.startswith("uploads/00001/") for p in parts)
    assert ".hf_manifest.json" in _members(hub.files[parts[-1]])
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, None, "tok", api=hub, download=hub.download)
    assert (there / "staging/n1/improve_recipe-1/pool/c.mp4").read_text() == "staging/n1/improve_recipe-1/pool/c.mp4"


def test_recorded_files_go_up_except_those_of_hf_download(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    for rel in ("telemetry/files/aaa", "telemetry/files_hf/bbb"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(rel)
    HfUpload(run, Recorder(run), "tok", api=hub)("root", "eval", 0)
    members = _members(hub.files["uploads/00001/0000.tar"])
    assert "telemetry/files/aaa" in members and "telemetry/files_hf/bbb" not in members
