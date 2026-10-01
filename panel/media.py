"""Training data, eval and file views (tabs 6, 8 and 11)."""
from __future__ import annotations

import json
import math
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


def clips(run: Run, node_id: str, dataset: str, page: int = 0, page_size: int = 24) -> dict:
    _, manifest = _manifest(run, node_id)
    ids = ((manifest.get("datasets") or {}).get(dataset) or {}).get("clips", [])
    pages = max(1, math.ceil(len(ids) / page_size))
    page = min(max(0, int(page or 0)), pages - 1)
    leakage = leakage_index(run)
    rows = [clip_detail(run, cid, leakage) for cid in ids[page * page_size:(page + 1) * page_size]]
    return {"rows": [r for r in rows if r], "total": len(ids), "pages": pages}


def clip_detail(run: Run, clip_id: str, leakage: dict | None = None) -> dict | None:
    rows = run.files.query("SELECT * FROM clips WHERE clip_id = ?", (clip_id,))
    if not rows:
        return None
    c = rows[0]
    leakage = leakage_index(run) if leakage is None else leakage
    return {"clip_id": clip_id, "video": f"store/blobs/video/{c['video_digest']}.mp4",
            "caption": run.files.read_json(f"store/blobs/caption/{c['caption_digest']}.json"),
            "pose": f"store/blobs/pose/{c['pose_digest']}.npz" if c["pose_digest"] else None,
            "camera_motion": c["camera_motion"], "metadata": loads(c["metadata"], {}),
            "formats": loads(c["formats"], []), "warnings": loads(c["warnings"], []),
            "provenance": loads(c["provenance"], {}), "license": c["license"],
            "derived_from": loads(c["derived_from"], []), "ingested_by": c["ingested_by"],
            "leakage": leakage.get(clip_id)}


def leakage_index(run: Run) -> dict[str, dict]:
    """Clip id -> its leakage verdict, paired with its accept by the candidate's staged video."""
    out: dict[str, dict] = {}
    for name in sorted({e["_file"] for e in run.log.events()} - {"run"}):
        by_video: dict[str, dict] = {}
        for e in run.log.events(files=[name]):
            kind = e.get("type")
            if kind not in ("ingest.leakage", "ingest.accepted"):
                continue
            p = run.log.payload(e.get("payload")) or {}
            video = (p.get("candidate") or {}).get("video")
            if kind == "ingest.leakage":
                by_video[video] = p
            elif p.get("clip_id") and video in by_video:
                found = by_video[video]
                out[p["clip_id"]] = {"matches": found.get("matches"), "near": found.get("near")}
    return out


def ingest_calls(run: Run, node_id: str) -> list[dict]:
    """One row per candidate: the data_ingest call lists the candidates and its result has
    one entry per candidate, in order."""
    events = run.log.events(files=[node_id])
    done = {e.get("parent_span_id"): e for e in events
            if e.get("type") in ("tool.result", "tool.error") and e.get("tool") == "data_ingest"}
    rows = []
    for e in events:
        if e.get("type") != "tool.call" or e.get("tool") != "data_ingest":
            continue
        candidates = ((run.log.payload(e.get("payload")) or {}).get("args") or {}).get("candidates") or []
        end = done.get(e.get("span_id"))
        body = (run.log.payload(end.get("payload")) or {}) if end else {}
        results = body.get("result") if end and end.get("type") == "tool.result" else None
        error = body.get("error") if end and end.get("type") == "tool.error" else None
        for i, c in enumerate(candidates):
            r = results[i] if isinstance(results, list) and i < len(results) else {}
            outcome = "accepted" if r.get("accepted") else "rejected" if r else "error" if error else "pending"
            video = c.get("video") or ""
            rows.append({"time": fmt_ts(e["ts_wall"]), "phase": e.get("phase"), "attempt": e.get("attempt"),
                         "video": video, "caption": c.get("caption"), "pose": c.get("pose"),
                         "camera_motion": c.get("camera_motion"), "outcome": outcome, "clip_id": r.get("clip_id"),
                         "reasons": "; ".join(r.get("reasons") or []) or (error or ""),
                         "host_video": run.files.host_path(node_id, e.get("phase"), e.get("attempt"), video)})
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


def eval_view(run: Run, node_id: str) -> dict:
    aggregates = run.files.read_json(f"nodes/{node_id}/eval/aggregates.json")
    work = run.files.path(f"nodes/{node_id}/eval/work_dirs")
    roots = sorted(p.parent for p in work.glob("*/videos")) if work.is_dir() else []
    if not roots:
        return {"cases": [], "aggregates": aggregates}
    root = roots[0]
    rel = root.relative_to(run.files.root)
    per_case: dict[str, dict] = {}
    for f in sorted((root / "evaluation").glob("*/case_*.json")):
        d = loads(f.read_text(errors="replace"), {}) or {}
        per_case.setdefault(str(d.get("case_id") or f.stem[5:]), {}).update(d.get("summary") or {})
    cases = []
    for mp4 in sorted((root / "videos").glob("case_*_combined.mp4"), key=lambda p: _case_key(p.name)):
        cid = mp4.name[len("case_"):-len("_combined.mp4")]
        meta = run.files.read_json(f"{rel}/videos/case_{cid}_combined.json") or {}
        cases.append({"case": cid, "video": f"{rel}/videos/{mp4.name}", "perspective": meta.get("perspective"),
                      "actions": meta.get("actions"), "prompt_schedule": meta.get("prompt_schedule"),
                      "scores": per_case.get(cid, {})})
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
