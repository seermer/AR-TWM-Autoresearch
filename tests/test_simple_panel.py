import getpass
import json
import os

import pytest

from fixtures.panel_run import make_run
from simple_panel.__main__ import chat, fence, read_auth
from simple_panel.export import PROJECT, Scrubber, _split, export


def test_scrubber_hides_host_details_and_keeps_sandbox_paths():
    scrub = Scrubber()
    text = (f"{PROJECT}/AutoResearcher/runs/r1 /home/someone/x.py /workspace/data/a.mp4 /agent/entry.py "
            f"key sk-abcdefghijklmnopqrstuv hf_ABCDEFGHIJKLMNOPQRSTUVWX Bearer abc.def-123 "
            f"token AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCd mail a.b@c.org at 10.1.2.3 by {getpass.getuser()} "
            f"commit 555053705215c0526e80918108de9a8b30a13360 https://x.org/a/b --- /dev/null")
    out = scrub(text)
    assert "<project>/AutoResearcher/runs/r1" in out and "/home/someone" not in out and "<path>" in out
    assert "/workspace/data/a.mp4" in out and "/agent/entry.py" in out and "/dev/null" in out
    assert "sk-" not in out and "hf_" not in out and "abc.def-123" not in out and "AbCdEf" not in out
    assert "<email>" in out and "<ip>" in out and getpass.getuser() not in out
    assert "555053705215c0526e80918108de9a8b30a13360" in out and "https://x.org/a/b" in out


def test_read_auth(tmp_path):
    f = tmp_path / "auth"
    with pytest.raises(SystemExit, match="no login file"):
        read_auth(f)
    f.write_text("u:p:w\n")
    os.chmod(f, 0o644)
    with pytest.raises(SystemExit, match="readable by others"):
        read_auth(f)
    os.chmod(f, 0o600)
    assert read_auth(f) == ("u", "p:w")
    f.write_text("u:\n")
    with pytest.raises(SystemExit, match="one `user:password` line"):
        read_auth(f)


def test_export_keeps_only_the_snapshot(tmp_path):
    run = make_run(tmp_path, "r1")
    snap = export(run, tmp_path / "snap" / "r1")
    export(run, snap)                                    # a refresh replaces the snapshot in place
    assert not (tmp_path / "snap" / "r1.new").exists() and not (tmp_path / "snap" / "r1.old").exists()
    s = json.loads((snap / "summary.json").read_text())
    assert [n["node"] for n in s["nodes"]] == ["root", "n1", "n2"]
    assert s["alerts"][0]["name"] == "monitor.stall" and s["scores"][1]["m1"] == 0.7
    n1 = json.loads((snap / s["nodes"][1]["file"]).read_text())
    assert (snap / n1["training_videos"][0]["file"]).is_file() and n1["eval_videos"][0]["scores"]["m1"] == 0.7
    assert len(n1["edits"]["attempts"]) == 2 and "+X = 3" in n1["edits"]["node_diff"]
    kinds = {i["kind"] for c in n1["conversations"] for i in c["items"]}
    assert "system" in kinds and "tool_call" in kinds and "usage" not in kinds
    lines = [i["text"] for c in n1["conversations"] for i in c["items"] if i["kind"] == "tokens"]
    assert lines and all(t.startswith("turn ") and "$" not in t and " s" not in t.split("output")[-1] for t in lines)
    assert s["nodes"][1]["output"] == sum(c["tokens"]["output"] for c in n1["conversations"]) > 0
    assert s["tokens"]["uncached"] == sum(n["uncached"] for n in s["nodes"]) > 0
    assert str(tmp_path) not in "".join(p.read_text() for p in snap.glob("*.json"))


def test_training_video_export_keeps_its_budget_across_datasets(tmp_path, monkeypatch):
    from panel import media, views
    from simple_panel.export import MAX_VIDEOS, _training_videos
    run = views.Run(make_run(tmp_path))
    manifest = {"datasets": {"a": {"clips": ["clip1"] * 2},
                             "b": {"clips": ["missing"] + ["clip1"] * (MAX_VIDEOS + 5)},
                             "c": {"clips": ["clip1"] * (MAX_VIDEOS + 5)}}}
    monkeypatch.setattr(media, "_manifest", lambda *args: (None, manifest))
    snapshot = tmp_path / "snapshot"
    videos = _training_videos(run, "n1", snapshot, "node", lambda text: text)
    assert len(videos) == MAX_VIDEOS
    assert [v["label"].split(" · ")[0] for v in videos] == ["a"] * 2 + ["b"] * (MAX_VIDEOS - 3) + ["c"]
    assert all((snapshot / v["file"]).is_file() for v in videos)
    assert len(list(snapshot.rglob("*.mp4"))) == MAX_VIDEOS


def test_token_split():
    assert _split({"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}) \
        == {"cached": 60, "uncached": 40, "output": 7}
    assert _split({}) == {"cached": 0, "uncached": 0, "output": 0}


def test_chat_and_fence():
    assert fence("a ```` b").startswith("`````\n")
    msgs = chat([{"kind": "system", "title": "System prompt", "text": "s"},
                 {"kind": "tool_output", "title": "Tool output: x", "text": "o"},
                 {"kind": "user", "title": None, "text": ""},
                 {"kind": "tokens", "title": None, "text": "turn 1: 5 cached · 2 uncached · 1 output"}])
    assert msgs[0] == {"role": "assistant", "content": "s", "metadata": {"title": "System prompt"}}
    assert msgs[1]["content"] == "```\no\n```" and msgs[2] == {"role": "user", "content": "(empty)"}
    assert msgs[3] == {"role": "assistant", "content": "*turn 1: 5 cached · 2 uncached · 1 output*"}
