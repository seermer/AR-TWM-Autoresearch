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
