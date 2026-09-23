import pytest

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import PathError, TokenRegistry, bearer, to_container, to_host


@pytest.fixture
def caller(tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    ws = tmp_path / "nodes" / "n1" / "attempts" / "recipe-1" / "workspace"
    st = tmp_path / "staging" / "n1" / "recipe-1"
    ws.mkdir(parents=True), st.mkdir(parents=True)
    return reg, reg.issue(node="n1", phase="improve_recipe", attempt=1,
                          workspace_host=ws, staging_host=st)


def test_issued_token_is_unguessable_and_resolves(caller):
    reg, c = caller
    assert len(c.token) >= 40 and c.token.startswith("ar-")
    assert reg.lookup(c.token) == c
    assert reg.lookup("ar-guess") is None


def test_revoked_token_no_longer_resolves(caller):
    reg, c = caller
    reg.revoke(c.token)
    assert reg.lookup(c.token) is None


def test_issued_tokens_are_redacted_from_telemetry(tmp_path):
    rec = Recorder(tmp_path)
    c = TokenRegistry(rec).issue(node="n", phase="p", attempt=1,
                                 workspace_host=tmp_path, staging_host=tmp_path)
    assert c.token in rec.redact


def test_bearer_parsing():
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer abc") == "abc"
    assert bearer(None) is None and bearer("Basic x") is None


def test_staging_paths_map_to_the_staging_mount(caller):
    _, c = caller
    assert to_host(c, "/workspace/staging/hf/clip.mp4") == c.staging_host / "hf" / "clip.mp4"
    assert to_host(c, "/workspace/notes.txt") == c.workspace_host / "notes.txt"


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "/store/blobs/video/x.mp4", "/workspace/../../etc/passwd",
    "/workspace/staging/../../../../kernel", "relative/path.mp4", "",
])
def test_paths_outside_the_workspace_are_refused(caller, bad):
    _, c = caller
    with pytest.raises(PathError):
        to_host(c, bad)


def test_symlink_escaping_the_workspace_is_refused(caller, tmp_path):
    """The agent controls /workspace; a link inside it must not reach the host."""
    _, c = caller
    (c.staging_host / "evil").symlink_to(tmp_path.parent)
    with pytest.raises(PathError):
        to_host(c, "/workspace/staging/evil/anything")


def test_round_trip_to_container(caller):
    _, c = caller
    host = c.staging_host / "a" / "b.mp4"
    assert to_container(c, host) == "/workspace/staging/a/b.mp4"


def test_on_revoke_fires_once_per_revoke(caller):
    reg, c = caller
    calls = []
    reg.on_revoke(calls.append)
    reg.revoke(c.token)
    assert calls == [c.token]
    # revoking an unknown/already-revoked token does not fire the callback again
    reg.revoke(c.token)
    assert calls == [c.token]
