"""The run folder's upload to Hugging Face after every phase, and its fetch."""
import hashlib
import io
import json
import sqlite3
import tarfile
from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationDelete
from huggingface_hub.hf_api import RepoFile

from ar_kernel import hf_upload
from ar_kernel.hf_upload import HfUpload, fetch, run_files
from ar_kernel.telemetry.recorder import Recorder


class FakeHub:
    """whoami / create_repo / create_commit / list_datasets / list_repo_tree / delete_repo / hf_hub_download over
    a dict of repos, each a dict of uploaded files. `lost_reply`: the commit lands, the caller sees an error."""
    def __init__(self, fail=False):
        self.repos, self.created, self.fail, self.lost_reply = {}, [], fail, False

    def whoami(self):
        return {"name": "me"}

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.created.append((repo, repo_type, private))
        self.repos.setdefault(repo, {})

    def create_commit(self, repo_id, operations, repo_type, commit_message):
        if self.fail:
            raise ConnectionError("hub down")
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                del self.repos[repo_id][op.path_in_repo]
            else:
                self.repos[repo_id][op.path_in_repo] = open(op.path_or_fileobj, "rb").read()
        if self.lost_reply:
            raise TimeoutError("no reply")

    def list_datasets(self, author, search):
        return [SimpleNamespace(id=r) for r in self.repos if r.startswith(f"{author}/") and search in r]

    def list_repo_tree(self, repo, repo_type, recursive):
        out = [RepoFile(path=".gitattributes", size=1, oid="g")]
        for path, data in self.repos[repo].items():
            lfs = {"size": len(data), "oid": hashlib.sha256(data).hexdigest(), "pointerSize": 130}
            out.append(RepoFile(path=path, size=len(data), oid="pointer", lfs=lfs))
        return out

    def delete_repo(self, repo, repo_type):
        del self.repos[repo]

    def download(self, repo_id, filename, repo_type, local_dir, token):
        path = f"{local_dir}/x.tar"
        open(path, "wb").write(self.repos[repo_id][filename])
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
    assert hub.created == [("me/myrun-00001", "dataset", True)]
    assert _members(hub.repos["me/myrun-00001"]["0000.tar"]) == [
        ".hf_manifest.json", "archive.db", "config/run.json", "control/state.json",
        "nodes/n1/attempts/improve_recipe-1/workspace/result.json", "nodes/n1/eval/a.mp4",
        "staging/n1/improve_recipe-1/pool/c.mp4"]
    (run / "nodes/n1/eval/b.mp4").write_text("new")
    up("n1", "eval", 0)
    # the first upload's own event is in the second
    assert [m for m in _members(hub.repos["me/myrun-00002"]["0000.tar"]) if not m.startswith("telemetry/payloads/")] == [
        ".hf_manifest.json", "archive.db", "nodes/n1/eval/b.mp4", "telemetry/events/root.jsonl"]
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
    assert list(hub.repos) == ["me/myrun-00001"] and "config/run.json" in _members(hub.repos["me/myrun-00001"]["0000.tar"])


def test_fetch_rebuilds_the_folder_and_drops_what_the_run_removed(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    up("root", "eval", 0)
    (run / "nodes/n1/eval/a.mp4").unlink()
    (run / "nodes/n1/eval/b.mp4").write_text("b")
    up("n1", "eval", 0)
    there = tmp_path / "elsewhere" / "myrun"
    assert fetch(there, "tok", api=hub, download=hub.download) == f"fetched 2 uploads of myrun into {there}"
    assert sorted(f for f in run_files(there) if not f.startswith("telemetry/payloads/")) == [
        "config/run.json", "control/state.json", "nodes/n1/attempts/improve_recipe-1/workspace/result.json",
        "nodes/n1/eval/b.mp4", "staging/n1/improve_recipe-1/pool/c.mp4", "telemetry/events/root.jsonl"]
    assert not (there / "nodes/n1/eval/a.mp4").exists() and (there / "nodes/n1/eval/b.mp4").read_text() == "b"
    assert sqlite3.connect(there / "archive.db").execute("SELECT x FROM t").fetchall() == [(1,)]
    assert fetch(there, "tok", api=hub, download=hub.download) == f"{there} is up to date"
    with pytest.raises(FileExistsError, match="was not made by fetch"):
        fetch(run, "tok", api=hub, download=hub.download)


def test_a_large_upload_is_split_into_parts_of_one_commit(tmp_path, monkeypatch):
    run, hub = _run(tmp_path), FakeHub()
    monkeypatch.setattr(hf_upload, "PART_BYTES", 40)
    monkeypatch.chdir(tmp_path)
    HfUpload(run.relative_to(tmp_path), Recorder(run), "tok", api=hub)("root", "eval", 0)
    parts = sorted(hub.repos["me/myrun-00001"])
    assert len(parts) > 1 and list(hub.repos) == ["me/myrun-00001"]
    assert ".hf_manifest.json" in _members(hub.repos["me/myrun-00001"][parts[-1]])
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, "tok", api=hub, download=hub.download)
    assert (there / "staging/n1/improve_recipe-1/pool/c.mp4").read_text() == "staging/n1/improve_recipe-1/pool/c.mp4"


def test_recorded_files_go_up_except_those_of_hf_download(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    for rel in ("telemetry/files/aaa", "telemetry/files_hf/bbb"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(rel)
    HfUpload(run, Recorder(run), "tok", api=hub)("root", "eval", 0)
    members = _members(hub.repos["me/myrun-00001"]["0000.tar"])
    assert "telemetry/files/aaa" in members and "telemetry/files_hf/bbb" not in members


def test_ar_upload_sends_a_stopped_runs_changes_and_refuses_a_live_loop(tmp_path, monkeypatch):
    from ar_kernel import cli
    from ar_kernel.control import Control
    run, hub = _run(tmp_path), FakeHub()
    (run / "control" / "state.json").write_text('{"node": "n1", "phase": "train", "attempt": 2}')
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(cli, "load_dotenv", lambda path, env: [])
    monkeypatch.setenv("HFK", "tok")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: hub)
    Control(run).claim()                                         # this process stands in for a live loop
    with pytest.raises(RuntimeError, match="still running"):
        cli.main(["upload", "--run-id", "myrun"])
    Control(run).release()
    assert cli.main(["upload", "--run-id", "myrun"]) == 0
    assert "config/run.json" in _members(hub.repos["me/myrun-00001"]["0000.tar"])
    (event,) = [e for e in Recorder(run).read_events("n1") if e["type"] == "hf_upload"]
    assert (event["phase"], event["attempt"]) == ("manual", 2)
    hub.fail = True
    (run / "nodes/n1/eval/b.mp4").write_text("new")
    with pytest.raises(ConnectionError):                         # not an alert: the caller sees it
        cli.main(["upload", "--run-id", "myrun"])
    monkeypatch.setenv("HFK", "")
    with pytest.raises(ValueError, match="HFK"):
        cli.main(["upload", "--run-id", "myrun"])


def _uploaded(tmp_path, hub, n):
    run = _run(tmp_path)
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    for i in range(n):
        (run / f"nodes/n1/eval/{i}.mp4").write_text(str(i))
        up("n1", "eval", 0)
    return run, up


def test_fetch_refuses_a_gap_instead_of_building_an_incomplete_run(tmp_path):
    hub = FakeHub()
    _uploaded(tmp_path, hub, 3)
    del hub.repos["me/myrun-00002"]
    there = tmp_path / "elsewhere" / "myrun"
    with pytest.raises(RuntimeError, match="upload 2 of myrun is not on Hugging Face"):
        fetch(there, "tok", api=hub, download=hub.download)
    assert (there / "nodes/n1/eval/0.mp4").exists() and not (there / "nodes/n1/eval/2.mp4").exists()


def test_fetch_waits_for_an_upload_whose_commit_has_not_landed(tmp_path):
    hub = FakeHub()
    run, up = _uploaded(tmp_path, hub, 1)
    hub.create_repo("me/myrun-00002", "dataset", True, True)        # created, its commit still on the way
    there = tmp_path / "elsewhere" / "myrun"
    assert fetch(there, "tok", api=hub, download=hub.download) == f"fetched 1 uploads of myrun into {there}"
    up("n1", "eval", 0)
    assert fetch(there, "tok", api=hub, download=hub.download) == f"fetched 1 uploads of myrun into {there}"


def test_delete_after_fetch_then_the_resumed_run_goes_on_from_the_next_upload(tmp_path):
    """The run keeps its own upload record: with its repos deleted, the next upload holds only what is new."""
    hub = FakeHub()
    run, up = _uploaded(tmp_path, hub, 2)
    there = tmp_path / "elsewhere" / "myrun"
    assert fetch(there, "tok", delete=True, api=hub, download=hub.download) == (
        f"fetched 2 uploads of myrun into {there}; deleted 2 upload repos from Hugging Face")
    assert hub.repos == {}
    (run / "nodes/n1/eval/new.mp4").write_text("new")
    up("n1", "eval", 0)
    assert list(hub.repos) == ["me/myrun-00003"]
    assert "nodes/n1/eval/0.mp4" not in _members(hub.repos["me/myrun-00003"]["0000.tar"])
    fetch(there, "tok", delete=True, api=hub, download=hub.download)
    assert (there / "nodes/n1/eval/new.mp4").read_text() == "new" and hub.repos == {}


def test_delete_refuses_and_deletes_nothing_when_a_repo_is_not_what_this_copy_applied(tmp_path):
    hub = FakeHub()
    _uploaded(tmp_path, hub, 2)
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, "tok", api=hub, download=hub.download)
    listed = hub.list_repo_tree

    def change_after_the_fetch(repo, repo_type, recursive):      # the repo changes between fetch and delete
        out = listed(repo, repo_type, recursive)
        if repo.endswith("00002"):
            hub.repos[repo]["0000.tar"] += b"changed"
        return out
    hub.list_repo_tree = change_after_the_fetch
    with pytest.raises(RuntimeError, match="myrun-00002 differs"):
        fetch(there, "tok", delete=True, api=hub, download=hub.download)
    assert sorted(hub.repos) == ["me/myrun-00001", "me/myrun-00002"]
    hub.list_repo_tree = listed
    state = json.loads((there / "control/hf_fetch.json").read_text())
    del state["applied"]["1"]
    (there / "control/hf_fetch.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="myrun-00001: .* has no record of applying it"):
        fetch(there, "tok", delete=True, api=hub, download=hub.download)
    assert sorted(hub.repos) == ["me/myrun-00001", "me/myrun-00002"]


def test_fetch_refuses_a_copy_made_from_a_single_run_repo(tmp_path):
    there = tmp_path / "myrun"
    (there / "control").mkdir(parents=True)
    (there / "control/hf_fetch.json").write_text('{"repo": "me/myrun", "last": "uploads/00015/0000.tar"}')
    with pytest.raises(ValueError, match="older `ar fetch`"):
        fetch(there, "tok", api=FakeHub(), download=None)


def test_the_upload_record_is_fsynced_with_its_folder(tmp_path, monkeypatch):
    run, hub = _run(tmp_path), FakeHub()
    synced = []
    real = hf_upload.os.fsync
    monkeypatch.setattr(hf_upload.os, "fsync", lambda fd: synced.append(hf_upload.os.readlink(f"/proc/self/fd/{fd}"))
                        or real(fd))
    HfUpload(run, Recorder(run), "tok", api=hub)("root", "eval", 0)
    assert [f for f in synced if "/control" in f] == [str(run / "control/hf_upload.tmp"), str(run / "control")]
    assert json.loads((run / "control/hf_upload.json").read_text())["seq"] == 1


def test_a_retry_after_a_commit_that_landed_unnoticed_replaces_it_and_fetch_applies_it_again(tmp_path, monkeypatch):
    run, hub = _run(tmp_path), FakeHub()
    monkeypatch.setattr(hf_upload, "PART_BYTES", 40)
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    gone = [f"nodes/n1/eval/x{i}.mp4" for i in range(10)]
    for rel in gone:
        (run / rel).write_text("x" * 50)
    hub.lost_reply = True
    up("root", "eval", 0)                                          # landed in 00001; the run does not know
    first = set(hub.repos["me/myrun-00001"])
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, "tok", api=hub, download=hub.download)            # a copy of the lost attempt
    hub.lost_reply = False
    for rel in gone:
        (run / rel).unlink()
    (run / "nodes/n1/eval/b.mp4").write_text("b")
    up("n1", "eval", 0)                                            # the retry: fewer parts, into 00001 again
    assert list(hub.repos) == ["me/myrun-00001"] and len(hub.repos["me/myrun-00001"]) < len(first)
    assert fetch(there, "tok", api=hub, download=hub.download) == (
        f"fetched 0 uploads of myrun into {there} (1 applied again: changed on Hugging Face since)")
    assert (there / "nodes/n1/eval/b.mp4").read_text() == "b" and not (there / "nodes/n1/eval/x0.mp4").exists()
    assert fetch(there, "tok", delete=True, api=hub, download=hub.download).endswith("deleted 1 upload repos from Hugging Face")


def test_a_retry_into_a_repo_the_copy_already_deleted_is_applied_again(tmp_path):
    run, hub = _run(tmp_path), FakeHub()
    up = HfUpload(run, Recorder(run), "tok", api=hub)
    hub.lost_reply = True
    up("root", "eval", 0)
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, "tok", delete=True, api=hub, download=hub.download)
    assert hub.repos == {}
    hub.lost_reply = False
    (run / "nodes/n1/eval/b.mp4").write_text("b")
    up("n1", "eval", 0)                                            # re-creates 00001
    fetch(there, "tok", api=hub, download=hub.download)
    assert (there / "nodes/n1/eval/b.mp4").read_text() == "b"


def test_an_older_upload_rewritten_after_its_record_reverted_is_applied_again_in_order(tmp_path):
    hub = FakeHub()
    run, up = _uploaded(tmp_path, hub, 3)
    there = tmp_path / "elsewhere" / "myrun"
    fetch(there, "tok", api=hub, download=hub.download)
    state = json.loads((run / "control/hf_upload.json").read_text())
    (run / "control/hf_upload.json").write_text(json.dumps({**state, "seq": 1}))   # the record as after upload 1
    (run / "nodes/n1/eval/new.mp4").write_text("new")
    up("n1", "eval", 0)                                            # into 00002 again, while 00003 exists
    assert fetch(there, "tok", api=hub, download=hub.download).endswith("(1 applied again: changed on Hugging Face since)")
    assert (there / "nodes/n1/eval/new.mp4").read_text() == "new" and (there / "nodes/n1/eval/2.mp4").read_text() == "2"
    assert fetch(there, "tok", api=hub, download=hub.download) == f"{there} is up to date"
