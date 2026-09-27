from ar_kernel import cli


def test_run_refuses_an_existing_run_without_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["run", "--run-id", "r1", "--max-nodes", "3"]) == 2
    assert "--resume" in capsys.readouterr().err


def test_stop_writes_the_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.KernelConfig, "runs_dir", property(lambda self: tmp_path))
    (tmp_path / "r1" / "config").mkdir(parents=True)
    (tmp_path / "r1" / "config" / "run.json").write_text("{}")
    assert cli.main(["stop", "--run-id", "r1"]) == 0
    assert (tmp_path / "r1" / "control" / "stop").exists()
