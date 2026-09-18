from __future__ import annotations
import hashlib, json, os, re, sqlite3, time
from pathlib import Path

FORMATS = {"video_caption_camera", "video_timed_prompts_camera", "video_caption_static"}
PROMPT_MODES = {"segment", "per_chunk"}
BUILTIN_NAMES = {"sekai_real_hq", "spatialvid_hq", "sekai_game_walking", "sekai_real_walking",
                 "mugen_v2", "RealEstate10K", "spatialvid", "veo3", "OpenVid", "mp4_frame_game_3",
                 "sekai_real_mini"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

class CommitError(ValueError):
    """The proposed data commit is invalid."""

class CommitStore:
    def __init__(self, conn: sqlite3.Connection, blobs, clips) -> None:
        self.conn = conn
        self.blobs = blobs
        self.clips = clips

    def _validate(self, datasets: dict[str, dict]) -> dict:
        if not datasets:
            raise CommitError("a commit needs at least one dataset")
        normalized: dict[str, dict] = {}
        usable = 0
        for name, entry in datasets.items():
            if not NAME_RE.match(name):
                raise CommitError(f"dataset name {name!r} must match {NAME_RE.pattern}")
            if name in BUILTIN_NAMES:
                raise CommitError(f"dataset name {name!r} is a built-in source name")
            fmt = entry.get("format")
            if fmt not in FORMATS:
                raise CommitError(f"{name}.format must be one of {sorted(FORMATS)}, got {fmt!r}")
            mode = entry.get("prompt_mode")
            if fmt == "video_timed_prompts_camera":
                if mode not in PROMPT_MODES:
                    raise CommitError(f"{name}.prompt_mode must be one of {sorted(PROMPT_MODES)}")
            elif mode is not None:
                raise CommitError(f"{name}.prompt_mode only applies to video_timed_prompts_camera")
            weight = float(entry.get("weight", 1.0))
            if weight < 0:
                raise CommitError(f"{name}.weight must be >= 0")
            clip_ids = list(entry.get("clips") or [])
            key = fmt if mode is None else f"{fmt}:{mode}"
            for clip_id in clip_ids:
                try:
                    clip = self.clips.get(clip_id)
                except KeyError:
                    raise CommitError(f"{name}: unknown clip {clip_id}") from None
                if key not in clip["formats"]:
                    raise CommitError(f"{name}: clip {clip_id[:12]} is not eligible for {key}")
            if weight > 0 and clip_ids:
                usable += 1
            normalized[name] = {"format": fmt, "prompt_mode": mode, "weight": weight,
                                "clips": sorted(clip_ids)}
        if usable == 0:
            raise CommitError("no dataset has both weight > 0 and clips")
        return {"datasets": dict(sorted(normalized.items()))}

    def commit(self, parent: str | None, datasets: dict[str, dict], message: str,
               node_id: str, attempt: int = 0) -> str:
        manifest = self._validate(datasets)
        blob = json.dumps(manifest, sort_keys=True)
        commit_id = hashlib.sha256(
            f"{parent or ''}|{hashlib.sha256(blob.encode()).hexdigest()}|{message}".encode()
        ).hexdigest()
        self.conn.execute(
            """INSERT OR IGNORE INTO data_commits
               (commit_id, parent_commit, node_id, attempt, manifest, message, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (commit_id, parent, node_id, attempt, blob, message, time.time()),
        )
        return commit_id

    def get(self, commit_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM data_commits WHERE commit_id=?", (commit_id,)).fetchone()
        if row is None:
            raise KeyError(commit_id)
        return dict(row)

    def manifest(self, commit_id: str) -> dict:
        return json.loads(self.get(commit_id)["manifest"])

    def materialize(self, commit_id: str, dest: Path) -> dict[str, Path]:
        dest = Path(dest)
        roots: dict[str, Path] = {}
        for name, entry in self.manifest(commit_id)["datasets"].items():
            if entry["weight"] <= 0 or not entry["clips"]:
                continue
            root = dest / name
            (root / "videos").mkdir(parents=True, exist_ok=True)
            (root / "captions").mkdir(parents=True, exist_ok=True)
            needs_poses = entry["format"] != "video_caption_static"
            if needs_poses:
                (root / "poses").mkdir(parents=True, exist_ok=True)
            for clip_id in entry["clips"]:
                clip = self.clips.get(clip_id)
                links = [(self.blobs.path(clip["video_digest"], "video"), root / "videos" / f"{clip_id}.mp4"),
                         (self.blobs.path(clip["caption_digest"], "caption"), root / "captions" / f"{clip_id}.json")]
                if needs_poses and clip["pose_digest"]:
                    links.append((self.blobs.path(clip["pose_digest"], "pose"),
                                  root / "poses" / f"{clip_id}.npz"))
                for source, target in links:
                    if target.exists():
                        target.unlink()
                    os.link(source, target)
            roots[name] = root
        return roots
