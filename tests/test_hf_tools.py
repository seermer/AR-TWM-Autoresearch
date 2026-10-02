"""No network: HfApi and snapshot_download are replaced by fakes."""
from types import SimpleNamespace

import httpx
import huggingface_hub.errors as hf_errors
import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.hf_tools import HfTools
from ar_kernel.tools.server import ToolError

CFG = KernelConfig.load()
SHA = "a" * 40


class FakeApi:
    datasets = None
    no_access = False

    def list_datasets(self, search, limit, full=True):
        self.last_search = search
        if self.datasets is not None:
            return [SimpleNamespace(**{"tags": [], "downloads": 0, "last_modified": None, "card_data": {}, **d})
                    for d in self.datasets]
        return [SimpleNamespace(id="org/walks", tags=["license:cc-by-4.0", "task:video"],
                                downloads=12, last_modified=None, card_data={"license": "cc-by-4.0"})]

    def auth_check(self, repo_id, repo_type=None):
        if self.no_access:
            raise hf_errors.GatedRepoError("gated", response=httpx.Response(403, request=httpx.Request("GET", "http://x")))

    def dataset_info(self, repo_id, revision=None, files_metadata=False):
        return SimpleNamespace(id=repo_id, sha=SHA, card_data={"license": "cc-by-4.0"},
                               siblings=[SimpleNamespace(rfilename="videos/a.mp4", size=1000),
                                         SimpleNamespace(rfilename="videos/b.mp4", size=3000),
                                         SimpleNamespace(rfilename="README.md", size=10)])


def fake_snapshot(repo_id, repo_type, revision, allow_patterns, local_dir, **_):
    from pathlib import Path
    for name in ("videos/a.mp4", "videos/b.mp4"):
        p = Path(local_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return local_dir


@pytest.fixture
def env(tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    return HfTools(CFG, tmp_path / "hf_tmp", api=FakeApi(), snapshot=fake_snapshot), caller


def test_search_reports_license(env):
    tools, caller = env
    hits = tools.search(caller, "walks", "dataset", 5)
    assert hits[0]["id"] == "org/walks" and hits[0]["license"] == "cc-by-4.0"


def test_download_pins_revision_and_lands_in_staging(env):
    tools, caller = env
    out = tools.download(caller, "org/walks", "main", ["videos/*.mp4"])
    assert out["revision"] == SHA and out["bytes"] == 4000
    assert out["files"] == [f"/workspace/staging/hf/org__walks/{SHA}/videos/a.mp4",
                            f"/workspace/staging/hf/org__walks/{SHA}/videos/b.mp4"]
    assert out["provenance"] == {"kind": "hf_dataset", "repo": "org/walks", "revision": SHA,
                                 "files": ["videos/a.mp4", "videos/b.mp4"]}
    assert out["license"] == "cc-by-4.0"


def test_download_over_the_cap_is_refused_before_transfer(env):
    tools, caller = env
    with pytest.raises(ToolError, match="4000 bytes"):
        tools.download(caller, "org/walks", "main", ["videos/*.mp4"], max_bytes=3500)
    assert not (caller.staging_host / "hf").exists()


def test_patterns_matching_nothing_is_an_error(env):
    tools, caller = env
    with pytest.raises(ToolError, match="no files match"):
        tools.download(caller, "org/walks", "main", ["*.parquet"])


def test_repo_ids_cannot_escape_staging(env):
    tools, caller = env
    with pytest.raises(ToolError):
        tools.download(caller, "../../etc", "main", ["*"])


class EscapeApi(FakeApi):
    """A dataset whose sibling listing itself tries to escape the staging dir."""

    def dataset_info(self, repo_id, revision=None, files_metadata=False):
        return SimpleNamespace(id=repo_id, sha=SHA, card_data={"license": "cc-by-4.0"},
                               siblings=[SimpleNamespace(rfilename="../escape.mp4", size=10),
                                         SimpleNamespace(rfilename="/etc/passwd", size=10)])


def test_repo_reported_paths_cannot_escape_dest(tmp_path):
    calls = []

    def spy_snapshot(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    tools = HfTools(CFG, tmp_path / "hf_tmp", api=EscapeApi(), snapshot=spy_snapshot)

    with pytest.raises(ToolError, match="unsafe"):
        tools.download(caller, "org/walks", "main", ["*"])
    assert not calls
    assert not (caller.staging_host / "hf").exists()
    assert not (tmp_path / "escape.mp4").exists()


def test_download_refuses_a_staging_symlink_that_leaves_staging(tmp_path):
    """The agent owns /workspace/staging: a planted staging/hf -> outside link must not be written through."""
    calls = []

    def spy_snapshot(**kwargs):
        calls.append(kwargs)
        return fake_snapshot(**kwargs)

    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "staging" / "hf").symlink_to(outside)
    tools = HfTools(CFG, tmp_path / "hf_tmp", api=FakeApi(), snapshot=spy_snapshot)

    with pytest.raises(ToolError, match="outside staging"):
        tools.download(caller, "org/walks", "main", ["videos/*.mp4"])
    assert not calls
    assert list(outside.iterdir()) == []


def _hub_error(cls, message):
    resp = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/datasets/org/x"))
    return cls(message, response=resp)


class DatasetInfoRaisesApi(FakeApi):
    def __init__(self, exc):
        self.exc = exc

    def dataset_info(self, repo_id, revision=None, files_metadata=False):
        raise self.exc


@pytest.mark.parametrize("cls, message", [
    (hf_errors.GatedRepoError, "403 Client Error: gated"),
    (hf_errors.RepositoryNotFoundError, "was not found on the Hub"),
    (hf_errors.RevisionNotFoundError, "404 Client Error: revision not found"),
])
def test_dataset_info_hub_errors_become_tool_errors(cls, message, tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    exc = _hub_error(cls, message)
    tools = HfTools(CFG, tmp_path / "hf_tmp", api=DatasetInfoRaisesApi(exc), snapshot=fake_snapshot)

    with pytest.raises(ToolError, match=message):
        tools.download(caller, "org/walks", "main", ["*"])


def test_snapshot_download_hub_error_becomes_tool_error(env):
    tools, caller = env
    message = "404 Client Error: entry not found"

    def raising_snapshot(**kwargs):
        # RemoteEntryNotFoundError, not EntryNotFoundError: in the installed huggingface_hub only the
        # former subclasses HfHubHTTPError (the latter is a plain Exception, used for local lookups).
        raise _hub_error(hf_errors.RemoteEntryNotFoundError, message)

    tools.snapshot = raising_snapshot
    with pytest.raises(ToolError, match=message):
        tools.download(caller, "org/walks", "main", ["videos/*.mp4"])


def test_links_planted_inside_an_earlier_download_are_not_written_through(env, tmp_path):
    """snapshot_download writes dest/<file> and dest/.cache/...: after a first download the agent
    can replace dest/videos or plant dest/.cache as links to the host. The kernel downloads into a
    private dir and never follows a link when moving files into staging."""
    tools, caller = env
    first = tools.download(caller, "org/walks", "main", ["videos/*.mp4"])
    dest = caller.staging_host / "hf" / "org__walks" / first["revision"]
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ("videos/a.mp4", "videos/b.mp4"):
        (dest / name).unlink()
    (dest / "videos").rmdir()
    (dest / "videos").symlink_to(outside)
    (dest / ".cache").symlink_to(outside)

    with pytest.raises(ToolError, match="staging"):
        tools.download(caller, "org/walks", "main", ["videos/*.mp4"])
    assert list(outside.iterdir()) == []
    assert list((tmp_path / "hf_tmp").iterdir()) == []          # the private dir is cleaned on failure


# ---- hf_list_files, and refusals that say what the repo holds ----

class ManyFilesApi(FakeApi):
    """A repo like the ones n1/n2 probed: thousands of files across folders."""
    def dataset_info(self, repo_id, revision=None, files_metadata=False):
        sib = [SimpleNamespace(rfilename=f"clips/part_{i:04d}.tar", size=1000 + i) for i in range(300)]
        sib += [SimpleNamespace(rfilename=f"meta/labels_{i}.json", size=10) for i in range(3)]
        sib += [SimpleNamespace(rfilename="README.md", size=5), SimpleNamespace(rfilename="big.zip", size=10**9)]
        return SimpleNamespace(id=repo_id, sha=SHA, card_data={"license": "cc-by-4.0"}, siblings=sib)


@pytest.fixture
def many(tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    return HfTools(CFG, tmp_path / "hf_tmp", api=ManyFilesApi(), snapshot=fake_snapshot), caller


def test_list_files_pages_through_matching_paths_with_sizes(many):
    tools, caller = many
    out = tools.list_files(caller, "org/big", "main", "clips/*", limit=100, offset=100)
    assert out["revision"] == SHA and out["license"] == "cc-by-4.0"
    assert out["matching_files"] == 300 and out["matching_bytes"] == sum(1000 + i for i in range(300))
    assert out["offset"] == 100 and len(out["files"]) == 100
    assert out["files"][0] == {"path": "clips/part_0100.tar", "size": 1100}
    assert out["repo_files"] == 305


def test_list_files_defaults_to_everything_and_caps_the_page(many):
    tools, caller = many
    out = tools.list_files(caller, "org/big", "main")
    assert out["matching_files"] == 305 and len(out["files"]) == 200      # default page
    assert len(tools.list_files(caller, "org/big", "main", limit=10**6)["files"]) == 305   # capped at 1000
    assert tools.list_files(caller, "org/big", "main", "*.parquet")["files"] == []


def test_list_files_refuses_a_bad_repo_id(many):
    tools, caller = many
    with pytest.raises(ToolError, match="invalid dataset repo id"):
        tools.list_files(caller, "../x", "main")


def test_over_the_cap_names_the_largest_matching_files(many):
    tools, caller = many
    with pytest.raises(ToolError) as exc:
        tools.download(caller, "org/big", "main", ["*"], max_bytes=1000)
    msg = str(exc.value)
    assert "over the 1000-byte cap" in msg and "big.zip (1000000000 bytes)" in msg
    assert "clips/part_0299.tar" in msg and "hf_list_files" in msg


def test_no_match_says_what_the_repo_holds(many):
    tools, caller = many
    with pytest.raises(ToolError) as exc:
        tools.download(caller, "org/big", "main", ["*.mp4"])
    msg = str(exc.value)
    assert "no files match" in msg and "hf_list_files" in msg
    assert "clips/ (300 files)" in msg and "meta/ (3 files)" in msg
    assert ".tar: 300" in msg and ".json: 3" in msg


def test_search_matches_all_words(env):
    tools, caller = env
    tools.api.datasets = [{"id": "a/TartanAir-videos"}, {"id": "b/TartanAir"}]
    rows = tools.search(caller, "TartanAir videos")
    assert [r["id"] for r in rows] == ["a/TartanAir-videos"]
    assert tools.api.last_search == "tartanair"              # the longest word goes to the Hub


def test_search_blank_query_is_an_error(env):
    tools, caller = env
    with pytest.raises(ToolError, match="empty"):
        tools.search(caller, "  ")


def test_list_files_reports_gated_and_access(env):
    tools, caller = env
    assert tools.list_files(caller, "org/walks", "main")["accessible"] is True
    tools.api.no_access = True
    out = tools.list_files(caller, "org/walks", "main")
    assert out["accessible"] is False and "gated" in out


def test_a_blocked_repo_is_absent_from_search(env):
    tools, caller = env
    tools.api.datasets = [{"id": "org/walks"}, {"id": "meituan-longcat/WBench"},
                          {"id": "org/mirror", "tags": ["wbench"]}]
    assert [h["id"] for h in tools.search(caller, "org", "dataset", 5)] == ["org/walks"]


def test_a_blocked_repo_reads_exactly_like_a_missing_one(env, monkeypatch):
    tools, caller = env

    def missing(repo_id, revision=None, files_metadata=False):
        raise hf_errors.RepositoryNotFoundError(
            "404", response=httpx.Response(404, request=httpx.Request("GET", "http://x")))
    monkeypatch.setattr(tools.api, "dataset_info", missing)
    messages = []
    for repo in ("org/gone", "meituan-longcat/WBench"):
        for call in (lambda: tools.list_files(caller, repo, "main"),
                     lambda: tools.download(caller, repo, "main", ["*"])):
            with pytest.raises(ToolError) as err:
                call()
            messages.append(str(err.value).replace(repo, "<repo>"))
    assert set(messages) == {"dataset '<repo>' was not found on the Hub"}
