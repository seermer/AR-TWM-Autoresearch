from __future__ import annotations
import hashlib, json, math, os, re, shutil, sqlite3, time
from pathlib import Path

FORMATS = {"video_caption_camera", "video_timed_prompts_camera", "video_caption_static"}
PROMPT_MODES = {"segment", "per_chunk"}
BUILTIN_NAMES = {"sekai_real_hq", "spatialvid_hq", "sekai_game_walking", "sekai_real_walking",
                 "mugen_v2", "RealEstate10K", "spatialvid", "veo3", "OpenVid", "mp4_frame_game_3",
                 "sekai_real_mini"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Written into every view materialize() builds; the only directories it will replace.
VIEW_MARKER = ".ar_view"

class CommitError(ValueError):
    """The proposed data commit is invalid."""

class BadClips(CommitError):
    """Clips of one dataset cannot be committed. `bad`: index in `clips` -> what is wrong with that clip."""
    def __init__(self, dataset: str, clips: list, bad: dict[int, str]) -> None:
        self.dataset, self.clips, self.bad = dataset, clips, bad
        first = "; ".join(f"clip {clips[n]}: {error}" for n, error in list(bad.items())[:5])
        super().__init__(f"{dataset}: {len(bad)} of {len(clips)} clips are bad: {first}")

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
            raw_weight = entry.get("weight", 1.0)
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise CommitError(f"{name}.weight must be a number, got {raw_weight!r}")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                # inf/nan used to pass here and then crash steps_per_epoch in the gate.
                raise CommitError(f"{name}.weight must be finite and >= 0, got {raw_weight!r}")
            clip_ids = list(entry.get("clips") or [])
            key = fmt if mode is None else f"{fmt}:{mode}"
            bad, seen = {}, set()
            for n, clip_id in enumerate(clip_ids):
                if not isinstance(clip_id, str):
                    bad[n] = "not a clip id"
                    continue
                if clip_id in seen:
                    # A repeated clip inflates the clip count the gate checks against the
                    # GPU count and silently reweights sampling; weights exist for that.
                    bad[n] = "listed more than once: use the dataset weight to upsample instead"
                    continue
                seen.add(clip_id)
                try:
                    clip = self.clips.get(clip_id)
                except KeyError:
                    bad[n] = "unknown clip"
                    continue
                if key not in clip["formats"]:
                    bad[n] = f"not eligible for {key}; it is eligible for {', '.join(clip['formats'])}"
            if bad:
                raise BadClips(name, clip_ids, bad)
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
        """Build a hardlink view of `commit_id` at `dest`, REPLACING anything there.

        The view is assembled in a sibling temp directory and swapped in, so the
        result holds exactly the commit's clips. Adding into an existing view was
        wrong: WorldModel's loader lists videos/ directly, so a clip dropped
        between gate attempts stayed in the view and the node trained on data its
        recorded commit did not contain.

        Replacing deletes the old destination, so it only ever deletes a directory
        carrying VIEW_MARKER (written here) -- never an arbitrary path. Removing a
        view unlinks hardlinks only; the blob store keeps its own link.
        """
        dest = Path(dest)
        if dest.exists() and any(dest.iterdir()) and not (dest / VIEW_MARKER).is_file():
            raise CommitError(f"{dest} exists and is not a kernel-built view; refusing to replace it")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(f".{dest.name}.building-{os.getpid()}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        (staging / VIEW_MARKER).write_text(commit_id + "\n")
        roots = self._link_into(commit_id, staging)
        if dest.exists():
            retired = dest.with_name(f".{dest.name}.retired-{os.getpid()}")
            dest.rename(retired)
            staging.rename(dest)
            shutil.rmtree(retired)
        else:
            staging.rename(dest)
        return {name: dest / root.name for name, root in roots.items()}

    def _link_into(self, commit_id: str, dest: Path) -> dict[str, Path]:
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
                    os.link(source, target)
            roots[name] = root
        return roots
