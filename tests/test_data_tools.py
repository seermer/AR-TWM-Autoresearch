import subprocess
import json
import threading
from pathlib import Path

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import DataTools
from ar_kernel.tools.server import ToolError
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Build once: ingest runs WorldModel's checker (~5 s per clip)."""
    run = tmp_path_factory.mktemp("run")
    rec = Recorder(run)
    nodes = NodeStore(open_db(run))
    nodes.create("root", None, 0)
    nodes.create("n1", "root", 1)
    tools = DataTools(CFG, run, rec, [0, 1, 2, 3], threading.Lock())
    reg = TokenRegistry(rec)
    ws = run / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    st = run / "staging" / "n1" / "improve_recipe-1"
    ws.mkdir(parents=True), st.mkdir(parents=True)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=st)
    cands = []
    for i in range(4):
        seconds = 4.0 + 0.5 * i                      # distinct content -> distinct clips
        d = st / f"c{i}"
        make_mp4(d / "v.mp4", seconds=seconds)
        write_caption(d / "c.json")
        write_poses(d / "p.npz", n_frames=int(seconds * 30))
        cands.append({"video": f"/workspace/staging/c{i}/v.mp4",
                      "caption": f"/workspace/staging/c{i}/c.json",
                      "pose": f"/workspace/staging/c{i}/p.npz",
                      "camera_motion": "moving", "provenance": PROV})
    (ws / "cands.json").write_text(json.dumps(cands[2:]))        # a list may also come from a file
    results = _rows(caller, tools.ingest(caller, cands[:2])) + _rows(caller, tools.ingest(caller, "cands.json"))
    return tools, caller, results, run


def _rows(caller, out: dict) -> list[dict]:
    """The per-candidate rows of a data_ingest result: they are in its result file."""
    assert out["result_file"].startswith("/workspace/staging/results/data_ingest-")
    rows = json.loads((caller.staging_host / "results" / Path(out["result_file"]).name).read_text())
    assert (out["accepted"], out["rejected"]) == (sum(r["accepted"] for r in rows), sum(not r["accepted"] for r in rows))
    return rows


def test_ingest_accepts_staged_candidates_by_container_path(env):
    _, _, results, _ = env
    assert all(r["accepted"] for r in results), results
    assert len({r["clip_id"] for r in results}) == 4
    assert all("video_caption_camera" in r["formats"] for r in results)


def test_ingest_refuses_paths_outside_the_workspace(env):
    tools, caller, _, _ = env
    with pytest.raises(ToolError, match="outside /workspace"):
        tools.ingest(caller, [{"video": "/etc/passwd", "caption": "/workspace/staging/x.json",
                               "camera_motion": "moving", "provenance": PROV}])


def test_ingest_rejects_a_malformed_candidate_as_a_tool_error(env):
    tools, caller, _, _ = env
    with pytest.raises(ToolError, match="provenance"):
        tools.ingest(caller, [{"video": "/workspace/staging/x.mp4",
                               "caption": "/workspace/staging/x.json", "camera_motion": "moving"}])


def test_every_bad_candidate_is_named_in_one_refusal(env):
    tools, caller, _, _ = env
    with pytest.raises(ToolError) as refused:
        tools.ingest(caller, [{"video": "/workspace/staging/no1.mp4", "caption": "/workspace/staging/no1.json",
                               "camera_motion": "moving", "provenance": PROV},
                              {"video": "/workspace/staging/no2.mp4", "camera_motion": "moving", "provenance": PROV}])
    text = str(refused.value)
    assert "item 0: video /workspace/staging/no1.mp4 does not exist" in text
    assert "item 1: caption is required" in text and "nothing was submitted" in text
    assert "/workspace/staging/results/data_ingest-refused-" in text


def test_rejections_are_summarised_with_counts_and_commit_takes_clip_ids_from_a_file(env):
    tools, caller, results, _ = env
    for name in ("r1", "r2"):
        (caller.staging_host / f"{name}.mp4").write_bytes(b"x"), (caller.staging_host / f"{name}.json").write_text("{}")
    out = tools.ingest(caller, [{"video": f"/workspace/staging/{n}.mp4", "caption": f"/workspace/staging/{n}.json",
                                 "camera_motion": "sideways", "provenance": PROV} for n in ("r1", "r2")])
    assert (out["accepted"], out["rejected"]) == (0, 2)
    assert out["rejected_for"] == [{"count": 2, "example": "camera_motion must be moving or static, got 'sideways'"}]
    (caller.workspace_host / "ids.json").write_text(json.dumps([r["clip_id"] for r in results]))
    done = tools.commit(caller, None, {"from_file": {"format": "video_caption_camera", "weight": 1.0,
                                                    "clips": "ids.json"}}, "ids from a file")
    assert done["datasets"]["from_file"]["clips"] == 4
    assert tools.query(caller, {"clip_ids": "/workspace/ids.json"})["total"] == 4


def test_probe_reports_display_geometry(env):
    tools, caller, _, run = env
    make_mp4(caller.staging_host / "probe_me.mp4", seconds=3.0)
    info = tools.probe(caller, "/workspace/staging/probe_me.mp4")
    assert info["width"] == 736 and info["rotation"] == 0 and abs(info["display_aspect"] - 736 / 414) < 1e-6


def test_query_returns_the_archive_wide_pool_with_provenance(env):
    tools, caller, results, _ = env
    out = tools.query(caller, {"format": "video_caption_camera"})
    ids = {c["clip_id"] for c in out["clips"]}
    assert {r["clip_id"] for r in results} <= ids
    clip = next(c for c in out["clips"] if c["clip_id"] == results[0]["clip_id"])
    assert clip["provenance"] == PROV and clip["ingested_by"] == "n1" and clip["used_by_scores"] == []


def test_commit_returns_id_and_per_dataset_stats(env):
    tools, caller, results, _ = env
    out = tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                              "weight": 1.0, "clips": [r["clip_id"] for r in results]}},
                       "four clips")
    assert len(out["commit_id"]) == 64
    stats = out["datasets"]["cam"]
    assert sum(stats.pop("sources").values()) == 4
    assert stats == {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": 4}


def test_a_commit_takes_over_the_datasets_of_included_commits_and_a_query_reads_a_commit_back(env):
    from ar_kernel.tools.server import ToolError
    tools, caller, results, _ = env
    ids = [r["clip_id"] for r in results]
    cam = {"format": "video_caption_camera", "prompt_mode": None, "weight": 2.0, "clips": ids[:3]}
    first = tools.commit(caller, None, {"cam": cam}, "three clips")["commit_id"]
    extra = {"format": "video_caption_camera", "prompt_mode": None, "weight": 1.0, "clips": ids[3:]}
    second = tools.commit(caller, first, {"extra": extra}, "the parent's data and one clip", include=[first])
    assert {n: (d["clips"], d["weight"]) for n, d in second["datasets"].items()} == {"cam": (3, 2.0), "extra": (1, 1.0)}
    replaced = tools.commit(caller, first, {"cam": {**cam, "clips": ids[:2]}}, "one dropped", include=[first])
    assert replaced["datasets"]["cam"]["clips"] == 2
    out = tools.query(caller, {"data_commit": second["commit_id"]})
    assert {c["clip_id"]: c["datasets"] for c in out["clips"]} == {**{i: ["cam"] for i in ids[:3]}, ids[3]: ["extra"]}
    with pytest.raises(ToolError, match="unknown data commit"):
        tools.commit(caller, None, {}, "m", include=["f" * 64])
    with pytest.raises(ToolError, match="unknown data commit"):
        tools.query(caller, {"data_commit": "f" * 64})
    with pytest.raises(ToolError, match="more than one included commit"):
        tools.commit(caller, None, {}, "m", include=[first, replaced["commit_id"]])


def test_a_clips_source_is_its_generator_or_repo_and_derived_clips_inherit_it():
    from ar_kernel.tools.data_tools import clip_source, dataset_stats
    clips = [{"clip_id": "a", "provenance": {"kind": "rollout", "generator": "alayaworld-dmd4"}},
             {"clip_id": "b", "provenance": {"kind": "hf_dataset", "repo": "org/set"}},
             {"clip_id": "c", "provenance": {"kind": "derived"}, "derived_from": ["a"]},
             {"clip_id": "d", "provenance": {"kind": "derived"}, "derived_from": ["c", "b"]},
             {"clip_id": "e", "provenance": {"kind": "derived"}, "derived_from": []}]
    by_id = {c["clip_id"]: c for c in clips}
    assert [clip_source(c, by_id) for c in clips] == [
        "rollout:alayaworld-dmd4", "hf:org/set", "rollout:alayaworld-dmd4",
        "hf:org/set+rollout:alayaworld-dmd4", "derived"]
    manifest = {"datasets": {"x": {"format": "f", "prompt_mode": None, "weight": 1.0, "clips": ["a", "c", "e"]}}}
    assert dataset_stats(manifest, clips)["x"]["sources"] == {"rollout:alayaworld-dmd4": 2, "derived": 1}


def test_query_pages_through_the_pool_with_offset(env):
    tools, caller, results, _ = env
    everything = tools.query(caller, {})
    first = tools.query(caller, {"limit": 3})
    rest = tools.query(caller, {"limit": 3, "offset": 3})
    assert first["total"] == rest["total"] == everything["total"] and first["returned"] == 3
    assert [c["clip_id"] for c in first["clips"] + rest["clips"]] == [c["clip_id"] for c in everything["clips"]][:6]


def test_commit_validation_errors_become_tool_errors(env):
    tools, caller, results, _ = env
    with pytest.raises(ToolError, match="more than once"):
        tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                            "weight": 1.0, "clips": [results[0]["clip_id"]] * 2}}, "dup")
    with pytest.raises(ToolError, match="weight must be a number"):            # one line, no file: not a list
        tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                            "weight": "x", "clips": [results[0]["clip_id"]]}}, "w")


def test_every_bad_clip_of_a_commit_is_named_in_one_refusal_with_a_file(env):
    """n4 of live-10-03 was told a 12-character prefix of one clip id, which data_query could not find."""
    tools, caller, results, _ = env
    ids = [r["clip_id"] for r in results]
    with pytest.raises(ToolError) as refused:
        tools.commit(caller, None, {"still": {"format": "video_caption_static", "prompt_mode": None, "weight": 1.0,
                                              "clips": [*ids, "nope"]}}, "bad")
    text = str(refused.value)
    assert f"{len(ids) + 1} of {len(ids) + 1} items are bad" in text and "item 0: not eligible" in text
    [file] = (caller.staging_host / "results").glob("data_commit-still-refused-*.json")
    rows = json.loads(file.read_text())
    assert [r["item"] for r in rows] == [*ids, "nope"] and rows[-1]["error"] == "unknown clip"


def test_recipe_check_reports_gate_failures_without_leaving_a_view(env, monkeypatch):
    """An earlier version of this test was vacuous -- its recipe
    fails the "not tunable" check before Gate.check ever reaches materialize(), so
    the "no leftover view" assertion passed even with cleanup code deleted.

    This version uses a recipe that clears every pre-materialize gate check (see
    tests/test_gate.py for the same arithmetic), so the gate genuinely builds a
    view under the scratch directory, and monkeypatches only the GPU-job steps
    (check_dataset / precache_dry_run / describe, all routed through
    ar_kernel.train.gate.run_in_env) so the test needs no GPU or alayaworld env.
    The fake run_in_env itself asserts the view was populated with real hardlinked
    clip files -- proving recipe_check's cleanup has something real to remove.
    """
    tools, caller, results, _ = env
    commit = tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                                 "weight": 1.0, "clips": [r["clip_id"] for r in results]}},
                          "for check")["commit_id"]

    seen_populated_view = []

    def fake_run_in_env(env_name, args, **kwargs):
        views = list((caller.workspace_host.parent / "recipe_check").glob("*/view"))
        seen_populated_view.append(bool(views) and any(any(v.rglob("*.mp4")) for v in views))
        return subprocess.CompletedProcess(args, 1, "", "monkeypatched: no GPU job in unit tests")

    monkeypatch.setattr("ar_kernel.train.gate.run_in_env", fake_run_in_env)

    # Clears tunable/value checks, resolution/lora defaults (base recipe values
    # are in the allowlists), the 4-clips->4-GPUs check, and steps_per_epoch:
    # epoch_windows=4, per_rank=4//4=1, per_epoch=1//1=1, 1*1 >= max_steps=1.
    recipe = {"optimizer.max_steps": 1, "optimizer.epochs": 1, "optimizer.grad_accum_steps": 1}
    out = tools.recipe_check(caller, recipe, commit)

    assert seen_populated_view and all(seen_populated_view), \
        "the gate must have actually built and populated a view before the GPU-job steps ran"
    assert out["ok"] is False
    assert any("failed" in f for f in out["failures"]), out["failures"]
    leftovers = list((caller.workspace_host.parent / "recipe_check").glob("*"))
    assert leftovers == []


def test_recipe_check_is_refused_at_once_while_a_gpu_job_holds_the_gpus(env):
    tools, caller, results, _ = env
    commit = tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                                 "weight": 1.0, "clips": [r["clip_id"] for r in results]}},
                          "for check")["commit_id"]
    with tools.gpu_lock, pytest.raises(ToolError, match="a GPU job is running"):
        tools.recipe_check(caller, {}, commit)
    assert tools.gpu_lock.acquire(blocking=False)          # and the refusal left the lock alone


def test_register_names_are_openai_safe():
    import asyncio
    import re
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.server import ToolKit, new_mcp
    mcp = new_mcp()
    register_data_tools(mcp, ToolKit(None, None), None)
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert len(names) == 5 and all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)


def test_leakage_checker_is_built_once_across_concurrent_ingests(tmp_path, monkeypatch):
    """Building it hashes every WBench case image (7.5 s warm, 72 s cold); it was rebuilt per call."""
    import time
    built = []

    class CountingChecker:
        def __init__(self, cfg):
            built.append(cfg)
            time.sleep(0.2)                  # widen the race window between the two callers

    monkeypatch.setattr("ar_kernel.data.ingest.LeakageChecker", CountingChecker)
    monkeypatch.setattr("ar_kernel.tools.data_tools.LeakageChecker", CountingChecker)
    rec = Recorder(tmp_path)
    NodeStore(open_db(tmp_path)).create("n1", None, 0)
    tools = DataTools(CFG, tmp_path, rec, [0, 1, 2, 3], threading.Lock())
    ws, st = tmp_path / "ws", tmp_path / "staging" / "n1" / "a1"
    ws.mkdir(), st.mkdir(parents=True)
    caller = TokenRegistry(rec).issue(node="n1", phase="improve_recipe", attempt=1,
                                      workspace_host=ws, staging_host=st)
    (st / "v.mp4").write_bytes(b"x"), (st / "c.json").write_text("{}")
    # An invalid camera_motion is rejected before any file is touched: this exercises only setup.
    # Distinct values keep the two rejection payloads distinct (the recorder writes identical
    # payloads through one shared tmp name, a separate race).
    def ingest(motion):
        cand = {"video": "/workspace/staging/v.mp4", "caption": "/workspace/staging/c.json",
                "camera_motion": motion, "provenance": PROV}
        results.extend(_rows(caller, tools.ingest(caller, [cand])))
    results = []
    threads = [threading.Thread(target=ingest, args=(m,)) for m in ("sideways", "upways")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["accepted"] for r in results] == [False, False]
    assert len(built) == 1


def test_a_staged_file_stays_and_ingesting_it_again_gives_the_same_clip(env):
    tools, caller, results, run = env
    out = tools.ingest(caller, [{"video": "/workspace/staging/c0/v.mp4", "caption": "/workspace/staging/c0/c.json",
                                 "pose": "/workspace/staging/c0/p.npz", "camera_motion": "moving", "provenance": PROV}])
    assert out["accepted"] == 1
    [row] = json.loads((caller.staging_host / "results" / Path(out["result_file"]).name).read_text())
    assert row["clip_id"] == results[0]["clip_id"]
    with pytest.raises(ToolError, match=r"/workspace/staging/gone/v.mp4 does not exist") as exc:
        tools.ingest(caller, [{"video": "/workspace/staging/gone/v.mp4", "caption": "/workspace/staging/c0/c.json",
                               "camera_motion": "moving", "provenance": PROV}])
    assert str(run) not in str(exc.value)


def test_a_path_that_is_not_text_or_not_staged_is_that_candidates_error(env):
    """A caption given as the object itself crashed the call; a file outside staging was named by its host path."""
    tools, caller, _, _ = env
    (caller.workspace_host / "loose.mp4").write_bytes(b"x"), (caller.staging_host / "a.mp4").write_bytes(b"x")
    with pytest.raises(ToolError) as refused:
        tools.ingest(caller, [{"video": "/workspace/staging/a.mp4", "caption": {"caption": "a room"},
                               "camera_motion": "static", "provenance": PROV},
                              {"video": "/workspace/loose.mp4", "caption": "/workspace/staging/a.json",
                               "camera_motion": "static", "provenance": PROV}])
    text = str(refused.value)
    assert "item 0: caption must be the path of a file" in text
    assert "item 1: video /workspace/loose.mp4 is not under /workspace/staging" in text
    assert str(caller.workspace_host) not in text


def test_messages_that_differ_in_quoted_text_and_numbers_count_as_one():
    from ar_kernel.tools.data_tools import _counted
    texts = [f"the caption file: segment {t!r} lasts {d} s < one window (2.375 s), so segment mode never trains on it"
             for t, d in (("A person reaches", 1.04), ("The knife chops", 1.33))] + ["fps 12.0 is below 24"]
    assert [row["count"] for row in _counted(texts)] == [2, 1]


def test_recipe_and_dataset_numbers_are_typed():
    """A real agent's numbers inside an untyped map arrived as strings ("lora.rank": "64", "weight": "1.0")."""
    import asyncio
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.server import ToolKit, new_mcp
    mcp = new_mcp()
    register_data_tools(mcp, ToolKit(None, None), None)
    listed = {t.name: t.input_schema for t in asyncio.run(mcp.list_tools())}
    assert listed["recipe_check"]["properties"]["recipe"]["additionalProperties"] == {
        "anyOf": [{"type": "integer"}, {"type": "number"}]}
    assert "number" in json.dumps(listed["data_commit"]["$defs"]["Dataset"]["properties"]["weight"])
    model = mcp._tool_manager.get_tool("recipe_check").fn_metadata.arg_model
    assert model.model_validate({"recipe": {"lora.rank": "64", "optimizer.lr": "1e-4", "sample.width": 736},
                                 "data_commit": "c"}).recipe == {"lora.rank": 64, "optimizer.lr": 1e-4, "sample.width": 736}
