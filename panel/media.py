"""Training data, evaluation, and file views."""
from __future__ import annotations

import json
import re
import zipfile

import numpy as np

from .runfiles import fmt_ts, loads
from .views import Run

TEXT_LIMIT = 2_000_000
VIDEO = {".mp4", ".webm", ".mov"}
IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT = {".txt", ".log", ".json", ".jsonl", ".yaml", ".yml", ".md", ".py", ".csv", ".toml", ".sh", ".cfg",
        ".ini", ".html"}


def _manifest(run: Run, node_id: str) -> tuple[dict | None, dict]:
    n = run.node(node_id)
    if not n or not n["data_commit"]:
        return None, {}
    rows = run.files.query("SELECT * FROM data_commits WHERE commit_id = ?", (n["data_commit"],))
    return (rows[0], loads(rows[0]["manifest"], {}) or {}) if rows else (None, {})


def training_data(run: Run, node_id: str) -> dict:
    commit, manifest = _manifest(run, node_id)
    return {"commit": commit,
            "datasets": [{"dataset": k, "clips": len(v.get("clips", []))}
                         for k, v in (manifest.get("datasets") or {}).items()],
            "ingests": ingest_calls(run, node_id), "staging": staging_files(run, node_id)}


def clips(run: Run, node_id: str, dataset: str, limit: int | None = None) -> list[dict]:
    _, manifest = _manifest(run, node_id)
    ids = ((manifest.get("datasets") or {}).get(dataset) or {}).get("clips", [])
    leakage = leakage_index(run)
    rows = [clip_detail(run, cid, leakage) for cid in ids[:limit]]
    return [r for r in rows if r]


def clip_detail(run: Run, clip_id: str, leakage: dict | None = None) -> dict | None:
    rows = run.files.query("SELECT * FROM clips WHERE clip_id = ?", (clip_id,))
    if not rows:
        return None
    c = rows[0]
    leakage = leakage_index(run) if leakage is None else leakage
    return {"clip_id": clip_id, "video": f"store/blobs/video/{c['video_digest']}.mp4",
            "caption": run.files.read_json(f"store/blobs/caption/{c['caption_digest']}.json"),
            "annotation": run.files.read_json(f"store/blobs/annotation/{c['annotation_digest']}.json"),
            "pose": f"store/blobs/pose/{c['pose_digest']}.npz",
            "metadata": loads(c["metadata"], {}), "warnings": loads(c["warnings"], []),
            "derived_from": loads(c["derived_from"], {}), "ingested_by": c["ingested_by"],
            "leakage": leakage.get(clip_id)}


def _ingest_calls(run: Run) -> list[dict]:
    """Each data_ingest call: its place, time, the rows of the result file it handed back (in the file store),
    and per staged video (as the agent named it) its candidate span's inputs and leakage evidence."""
    out = []
    records, load = run.records, run.records.payload
    for n in records.named("tool.data_ingest"):
        rows, candidates = [], {}
        for c in n.children:
            if not hasattr(c, "start"):
                continue
            inputs = load(c.start.get("inputs")) or {}
            if c.start["name"] == "file.kept" and inputs.get("how") == "handed" and c.end \
                    and "/data_ingest-" in inputs.get("path", ""):
                rows = records.kept((load(c.end.get("output")) or {}).get("sha256")) or []
            elif c.start["name"] == "ingest.candidate":
                leak = next(((load(d.get("inputs")) or {}).get("leakage") for d in c.children if isinstance(d, dict)
                             and d["kind"] == "decision" and d["name"] == "ingest.check"
                             and (load(d.get("inputs")) or {}).get("check") == "held-out cases"), None)
                candidates[(inputs.get("named") or {}).get("video")] = {**inputs, "leakage": leak}
        out.append({"where": records.where(n.id), "ts": n.start["ts"], "rows": rows, "candidates": candidates})
    return out


def leakage_index(run: Run) -> dict[str, dict]:
    """Clip id -> the leakage evidence of the candidate it was ingested from."""
    out = {}
    for call in _ingest_calls(run):
        for r in call["rows"]:
            leak = (call["candidates"].get(r.get("video")) or {}).get("leakage")
            if r.get("clip_id") and leak is not None:
                out[r["clip_id"]] = {"matches": leak.get("matches"), "near": leak.get("near_matches")}
    return out


def origin(derived_from) -> str:
    """A clip's `derived_from` in one cell: `<kind>: <name>`."""
    return f"{derived_from.get('kind')}: {derived_from.get('name')}" if isinstance(derived_from, dict) else ""


def ingest_calls(run: Run, node_id: str) -> list[dict]:
    """One row per candidate of every data_ingest call of the node."""
    rows = []
    for call in _ingest_calls(run):
        node, phase, attempt = call["where"]
        if node != node_id:
            continue
        for r in call["rows"]:
            video = r.get("video") or ""
            cand = call["candidates"].get(video) or {}
            named = cand.get("named") or {}
            rows.append({"time": fmt_ts(call["ts"]), "phase": phase, "attempt": attempt,
                         "video": video, "annotation": named.get("annotation"), "pose": named.get("pose"),
                         "derived_from": origin(cand.get("derived_from")),
                         "outcome": "accepted" if r.get("accepted") else "rejected", "clip_id": r.get("clip_id"),
                         "reasons": "; ".join(r.get("reasons") or []),
                         "host_video": run.files.host_path(node_id, phase, attempt, video)})
    return rows


def staging_files(run: Run, node_id: str, cap: int = 2000) -> list[dict]:
    folder = run.files.path(f"staging/{node_id}")
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out.append({"path": str(p.relative_to(run.files.root)), "bytes": p.stat().st_size})
            if len(out) >= cap:
                break
    return out


def camera_path(run: Run, rel: str) -> dict:
    with np.load(run.files.path(rel)) as arrays:
        t = np.asarray(arrays["cam_c2w"])[:, :3, 3]
    return {"x": t[:, 0].tolist(), "y": t[:, 1].tolist(), "z": t[:, 2].tolist()}


def _case_key(name: str) -> int:
    m = re.match(r"case_(\d+)", name)
    return int(m.group(1)) if m else 0


# The metrics a judge model grades, shown first; the rest follow by name.
JUDGED = ("event_edit_adherence", "subject_action_adherence", "perspective_switch_adherence", "causal_fidelity")
# A judge file saves only each question's id. These are short forms of the questions as the judge is asked them.
QUESTIONS = {
    "event_edit_adherence": {"Q1": "Scene unchanged?", "Q2": "Event visible?", "Q3": "Event completed?",
                             "Q4": "Details correct?", "Q5": "Anything unrelated appears?"},
    "subject_action_adherence": {"Q1": "Subject idle?", "Q2": "Action visible?", "Q3": "Action completed?",
                                 "Q4": "Details correct?", "Q5": "Unnatural movement?"},
    "perspective_switch_adherence": {"Q1": "Viewpoint changes?", "Q3": "Ends in the asked view?",
                                     "Q4": "End view valid?", "Q_cam": "Changed by the camera? (not scored)"},
}


def metric_order(names) -> list[str]:
    return sorted(names, key=lambda m: (JUDGED.index(m) if m in JUDGED else len(JUDGED), m))


def _eval_root(run: Run, node_id: str):
    work = run.files.path(f"nodes/{node_id}/eval/work_dirs")
    roots = sorted(p.parent for p in work.glob("*/videos")) if work.is_dir() else []
    return roots[0].relative_to(run.files.root) if roots else None


def judge_view(run: Run, node_id: str, case: str) -> list[dict]:
    """The judge's answers for one case: per turn and question, the answer and whether it is the expected
    one. causal_fidelity has grades instead: `passed` is None."""
    rel = _eval_root(run, node_id)
    rows = []
    for metric in JUDGED if rel else ():
        d = run.files.read_json(f"{rel}/evaluation/{metric}/case_{case}.json")
        if not d:
            continue
        if metric == "causal_fidelity":
            grades = [("overall", d["track1"]), *[(g["dim_name"], g) for g in d["track2"]["details"]]]
            rows += [{"turn": "", "question": f"Causal fidelity: {name}", "answer": f"{g['score']}/{g['max']}",
                      "passed": None} for name, g in grades]
            continue
        for t in d["turn_details"]:
            if metric == "perspective_switch_adherence":
                answers = [(q, "yes" if ok else "no", ok) for q, ok in t["q_results"].items()]
            else:
                answers = [(q["sub_id"], q["answer"], q["correct"]) for q in t["questions"]]
            rows += [{"turn": f"{t['turn']}: {t['action']}", "question": QUESTIONS[metric][q], "answer": answer,
                      "passed": ok} for q, answer, ok in answers]
    return rows


def eval_view(run: Run, node_id: str) -> dict:
    aggregates = run.files.read_json(f"nodes/{node_id}/eval/aggregates.json")
    rel = _eval_root(run, node_id)
    if rel is None:
        return {"cases": [], "aggregates": aggregates}
    root = run.files.path(rel)
    per_case = (run.files.read_json(f"{rel}/evaluation/report.json") or {}).get("per_case") or {}
    cases = []
    for mp4 in sorted((root / "videos").glob("case_*_combined.mp4"), key=lambda p: _case_key(p.name)):
        cid = mp4.name[len("case_"):-len("_combined.mp4")]
        meta = run.files.read_json(f"{rel}/videos/case_{cid}_combined.json") or {}
        cases.append({"case": cid, "video": f"{rel}/videos/{mp4.name}", "perspective": meta.get("perspective"),
                      "actions": meta.get("actions"), "prompt_schedule": meta.get("prompt_schedule"),
                      "scores": {name: per_case[cid][name] for name in metric_order(per_case.get(cid, {}))}})
    return {"cases": cases, "aggregates": aggregates}


def list_dir(run: Run, rel: str = "", cap: int = 5000) -> list[dict]:
    folder = run.files.path(rel)
    if not folder.is_dir():
        return []
    entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name))[:cap]
    out = []
    for p in entries:
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "type": "dir" if p.is_dir() else "file",
                    "bytes": None if p.is_dir() else st.st_size, "modified": fmt_ts(st.st_mtime)})
    return out


def _npz_arrays(path) -> list[dict]:
    """Array names, shapes and dtypes from the headers only; no array is loaded."""
    out = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(handle)
            out.append({"name": name.removesuffix(".npy"), "shape": list(shape), "dtype": str(dtype)})
    return out


def _looks_textual(path) -> bool:
    with path.open("rb") as handle:
        return b"\x00" not in handle.read(8192)


def preview(run: Run, rel: str) -> dict:
    p = run.files.path(rel)
    if p.is_dir():
        return {"kind": "dir", "path": rel}
    st = p.stat()
    base = {"path": rel, "bytes": st.st_size, "modified": fmt_ts(st.st_mtime)}
    suffix = p.suffix.lower()
    if suffix in VIDEO:
        return {"kind": "video", **base}
    if suffix in IMAGE:
        return {"kind": "image", **base}
    if suffix == ".npz":
        try:
            return {"kind": "npz", "arrays": _npz_arrays(p), **base}
        except (zipfile.BadZipFile, ValueError, OSError):
            return {"kind": "binary", **base}
    if suffix in TEXT or (suffix not in {".db", ".pt", ".safetensors", ".bin", ".zst"} and _looks_textual(p)):
        text = run.files.read_text(rel, limit=TEXT_LIMIT)
        if suffix == ".json" and st.st_size <= TEXT_LIMIT:
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except ValueError:
                pass
        return {"kind": "text", "text": text, **base}
    return {"kind": "binary", **base}
