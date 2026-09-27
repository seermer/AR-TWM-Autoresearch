import pytest

from ar_kernel.vcs.agents_repo import AgentsRepo


@pytest.fixture
def seed(tmp_path):
    s = tmp_path / "seed"
    (s / "agent").mkdir(parents=True)
    (s / "agent" / "entry.py").write_text("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    (s / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    return s


def test_init_commits_the_seed_on_the_root_branch(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    assert repo.resolve(repo.branch_ref("root")) == root
    assert "def edit_self" in repo.read_file(root, "agent/entry.py")


def test_checkout_then_commit_attempt_records_parent_and_ref(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    work = tmp_path / "work"
    repo.checkout(root, work)
    (work / "agent" / "entry.py").write_text("# edited\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    (work / "agent" / "__pycache__").mkdir()
    (work / "agent" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    child = repo.commit_tree(work, root, "edit_self attempt 1")
    repo.set_ref(repo.attempt_ref("n1", "edit_self", 1), child)
    assert repo.resolve("refs/attempts/n1/edit_self-1") == child
    assert repo.read_file(child, "agent/__pycache__/x.pyc") is None       # .gitignore honored
    stats = repo.diff_stats(root, child)
    assert stats == [{"path": "agent/entry.py", "added": 1, "removed": 0}]


def test_failed_attempts_keep_their_refs(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    for k in (1, 2):
        work = tmp_path / f"w{k}"
        repo.checkout(root, work)
        (work / "agent" / f"try{k}.py").write_text("x")
        repo.set_ref(repo.attempt_ref("n1", "edit_self", k), repo.commit_tree(work, root, f"attempt {k}"))
    assert repo.resolve("refs/attempts/n1/edit_self-1") and repo.resolve("refs/attempts/n1/edit_self-2")


def test_checkout_refuses_a_non_empty_destination(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(FileExistsError):
        repo.checkout(root, tmp_path / "busy")


def test_resolve_unknown_ref_is_none(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    repo.init(seed)
    assert repo.resolve("refs/heads/node/nope") is None


def test_checkout_of_a_symlink_leaving_the_tree_raises_checkout_error(tmp_path, seed):
    from ar_kernel.vcs.agents_repo import CheckoutError

    (seed / "agent" / "ctx").symlink_to("/context/x")      # committed by an agent, points outside the tree
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    with pytest.raises(CheckoutError, match="ctx"):
        repo.checkout(root, tmp_path / "work")


def _remote_refs(remote) -> dict:
    import subprocess
    out = subprocess.run(["git", "--git-dir", str(remote), "for-each-ref", "--format=%(refname) %(objectname)"],
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split() for line in out.splitlines())


def test_every_ref_is_pushed_under_the_run_name(tmp_path, seed):
    import subprocess
    remote = tmp_path / "github.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    assert AgentsRepo.check_remote(str(remote)) is None
    repo = AgentsRepo(tmp_path / "agents.git", remote=str(remote), namespace="run1")
    root = repo.init(seed)
    assert _remote_refs(remote) == {"refs/heads/run1/node/root": root}
    work = tmp_path / "work"
    repo.checkout(root, work)
    (work / "agent" / "new.py").write_text("x = 1\n")
    child = repo.commit_tree(work, root, "n1 edit_self attempt 1")
    repo.set_ref(repo.attempt_ref("n1", "edit_self", 1), child)
    repo.set_ref(repo.branch_ref("n1"), child)
    assert _remote_refs(remote) == {"refs/heads/run1/node/root": root, "refs/heads/run1/node/n1": child,
                                    "refs/heads/run1/attempts/n1/edit_self-1": child}


def test_a_failed_push_is_reported_and_caught_up_later(tmp_path, seed):
    import subprocess
    remote = tmp_path / "github.git"
    errors = []
    repo = AgentsRepo(tmp_path / "agents.git", remote=str(remote), namespace="run1", on_push_error=errors.append)
    root = repo.init(seed)                                   # the remote does not exist yet
    assert len(errors) == 1 and AgentsRepo.check_remote(str(remote)) is not None
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    repo.push()
    assert _remote_refs(remote) == {"refs/heads/run1/node/root": root}


def test_without_a_remote_nothing_is_pushed(tmp_path, seed, monkeypatch):
    repo = AgentsRepo(tmp_path / "agents.git")
    calls = []
    real = repo._git
    monkeypatch.setattr(repo, "_git", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    repo.init(seed)
    assert "push" not in calls
