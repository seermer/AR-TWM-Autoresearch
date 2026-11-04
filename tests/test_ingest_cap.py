"""data_ingest's per-part cap (improve_recipe.clips_per_part), with a stand-in for the clip checks."""
import contextvars
import threading
import time

import pytest

from ar_kernel.archive.clips import ClipStore
from ar_kernel.archive.db import open_db
from ar_kernel.config import KernelConfig, merged
from ar_kernel.data.ingest import IngestResult
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import data_tools
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import DataTools
from ar_kernel.tools.server import ToolError

BASE = KernelConfig.load()
CFG = KernelConfig(raw=merged(BASE.raw, {"improve_recipe": {"clips_per_part": 3}}), repo_root=BASE.repo_root)


class FakeIngestor:
    """Accepts a candidate unless its folder starts with `bad`; the clip id is the folder name."""
    def __init__(self, cfg, run_dir, conn, recorder, leakage):
        self.clips = ClipStore(conn)

    def ingest(self, candidates, node_id, attempt):
        time.sleep(0.2)
        out = []
        for c in candidates:
            name = c.video.parent.name
            if name.startswith("bad"):
                out.append(IngestResult(accepted=False, reasons=["no"]))
                continue
            self.clips.add({"clip_id": name, "video_digest": "v", "caption_digest": "c", "annotation_digest": "a",
                            "pose_digest": "p", "metadata": {}, "warnings": [], "derived_from": {},
                            "ingested_by": node_id, "attempt": attempt})
            out.append(IngestResult(accepted=True, clip_id=name))
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(data_tools, "Ingestor", FakeIngestor)
    open_db(tmp_path).close()
    rec = Recorder(tmp_path)
    tools = DataTools(CFG, tmp_path, rec, [0], threading.Lock())
    tools._leakage = object()
    reg = TokenRegistry(rec)

    def caller(attempt):
        ws = tmp_path / "nodes" / "n1" / "attempts" / f"improve_recipe-{attempt}" / "workspace"
        st = tmp_path / "staging" / "n1" / f"improve_recipe-{attempt}"
        ws.mkdir(parents=True, exist_ok=True), st.mkdir(parents=True, exist_ok=True)
        return reg.issue(node="n1", phase="improve_recipe", attempt=attempt, workspace_host=ws, staging_host=st)

    def cands(c, *names):
        for name in names:
            (c.staging_host / name).mkdir(exist_ok=True)
            for f in ("v.mp4", "a.json", "p.npz"):
                (c.staging_host / name / f).write_text("x")
        return [{"video": f"/workspace/staging/{n}/v.mp4", "annotation": f"/workspace/staging/{n}/a.json",
                 "pose": f"/workspace/staging/{n}/p.npz", "derived_from": {"kind": "model", "name": "x"}}
                for n in names]
    return tools, caller, cands


def test_a_part_may_fill_its_cap_and_no_more(env):
    tools, caller, cands = env
    c = caller(1)
    assert tools.ingest(c, cands(c, "a", "b"))["accepted"] == 2
    with pytest.raises(ToolError, match="2 of the 3 clips allowed until the result is submitted are added; send at most 1 candidates"):
        tools.ingest(c, cands(c, "c", "d"))
    assert tools.ingest(c, cands(c, "c"))["accepted"] == 1
    with pytest.raises(ToolError, match="3 of the 3 clips allowed until the result is submitted are added; none are left"):
        tools.ingest(c, cands(c, "d"))


def test_rejected_candidates_and_clips_already_in_the_pool_take_none_of_the_cap(env):
    tools, caller, cands = env
    c = caller(1)
    assert tools.ingest(c, cands(c, "bad1", "bad2", "a"))["accepted"] == 1
    assert tools.ingest(c, cands(c, "a", "b"))["accepted"] == 2          # `a` again: already in the pool
    assert tools.ingest(c, cands(c, "c"))["accepted"] == 1


def test_two_calls_at_once_cannot_pass_the_cap_together(env):
    tools, caller, cands = env
    c = caller(1)
    batches, errors = [cands(c, "a", "b"), cands(c, "c", "d")], []

    def send(batch):
        try:
            tools.ingest(c, batch)
        except ToolError as exc:
            errors.append(exc)
    threads = [threading.Thread(target=contextvars.copy_context().run, args=(send, b)) for b in batches]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 1


def test_a_new_attempt_starts_with_its_whole_cap(env):
    tools, caller, cands = env
    first = caller(1)
    tools.ingest(first, cands(first, "a", "b", "c"))
    second = caller(2)
    assert tools.ingest(second, cands(second, "d", "e", "f"))["accepted"] == 3
