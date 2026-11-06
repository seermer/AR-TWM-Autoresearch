from __future__ import annotations
import json, math, os, re, shutil, sqlite3, time
from pathlib import Path

from .clips import hidden
from .db import transaction

BUILTIN_NAMES = {"sekai_real_hq", "spatialvid_hq", "sekai_game_walking", "sekai_real_walking",
                 "mugen_v2", "RealEstate10K", "spatialvid", "veo3", "OpenVid", "mp4_frame_game_3",
                 "sekai_real_mini"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Written into every view materialize() builds; the only directories it will replace.
VIEW_MARKER = ".ar_view"

class CommitError(ValueError):
    """The proposed data commit is invalid."""


def limit_text(cap: int) -> str:
    return f"A data commit may hold at most {cap} new clips: clips that no kept data commit holds."


def over_limit(n: int, cap: int) -> str:
    return f"{n} new clips, more than the {cap} a data commit may hold; a clip is new unless a kept data commit holds it"


UNUSABLE = {"obsolete": "data commit {} can no longer be used: a result was submitted that did not name it. "
                        "Its clips are still in the pool",
            "failed": "data commit {} can no longer be used: the work that made it failed"}

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
            raw_weight = entry.get("weight", 1.0)
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise CommitError(f"{name}.weight must be a number, got {raw_weight!r}")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                # inf and nan would crash steps_per_epoch in the gate.
                raise CommitError(f"{name}.weight must be finite and >= 0, got {raw_weight!r}")
            clip_ids = list(entry.get("clips") or [])
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
                if hidden(clip["ingested_by"]):
                    bad[n] = "not in the pool"
            if bad:
                raise BadClips(name, clip_ids, bad)
            if weight > 0 and clip_ids:
                usable += 1
            normalized[name] = {"weight": weight, "clips": sorted(clip_ids)}
        if usable == 0:
            raise CommitError("no dataset has both weight > 0 and clips")
        return {"datasets": dict(sorted(normalized.items()))}

    def commit(self, datasets: dict[str, dict], message: str, node_id: str, attempt: int = 0,
               cap: int | None = None) -> tuple[str, int]:
        """A new draft `dc<n>` and its new clips; refused over `cap` new clips."""
        manifest = self._validate(datasets)
        new = self.new_clips(manifest)
        if cap is not None and new > cap:
            raise CommitError(over_limit(new, cap))
        with transaction(self.conn):
            row = self.conn.execute("SELECT seq FROM sqlite_sequence WHERE name='data_commits'").fetchone()
            n = (row[0] if row else 0) + 1
            self.conn.execute(
                """INSERT INTO data_commits (n, commit_id, node_id, attempt, status, manifest, message, created_at)
                   VALUES (?,?,?,?,'draft',?,?,?)""",
                (n, f"dc{n}", node_id, attempt, json.dumps(manifest, sort_keys=True), message, time.time()))
        return f"dc{n}", new

    def new_clips(self, manifest: dict) -> int:
        """Clips of the manifest that no accepted commit holds."""
        kept = set()
        for row in self.conn.execute("SELECT manifest FROM data_commits WHERE status='accepted'"):
            kept.update(c for d in json.loads(row[0])["datasets"].values() for c in d["clips"])
        return len({c for d in manifest["datasets"].values() for c in d["clips"]} - kept)

    def usable(self, commit_id: str) -> dict:
        """The commit's row; KeyError when unknown, CommitError when obsolete or failed."""
        row = self.get(commit_id)
        if row["status"] in UNUSABLE:
            raise CommitError(UNUSABLE[row["status"]].format(commit_id))
        return row

    def settle(self, node_id: str, attempt: int, accepted: str | None) -> dict[str, list[str]]:
        """An attempt's result named `accepted`: it is kept, the attempt's other drafts are obsolete."""
        changed = {"accepted": [], "obsolete": []}
        if accepted is not None and self.usable(accepted)["status"] != "accepted":
            self.conn.execute("UPDATE data_commits SET status='accepted' WHERE commit_id=?", (accepted,))
            changed["accepted"].append(accepted)
        changed["obsolete"] = self._mark("obsolete", "node_id=? AND attempt=? AND status='draft'", (node_id, attempt))
        return changed

    def fail(self, node_id: str, attempt: int | None = None) -> list[str]:
        """An attempt's drafts, or every commit of the node, are failed."""
        if attempt is not None:
            return self._mark("failed", "node_id=? AND attempt=? AND status='draft'", (node_id, attempt))
        return self._mark("failed", "node_id=? AND status!='failed'", (node_id,))

    def _mark(self, status: str, where: str, args: tuple) -> list[str]:
        ids = [r[0] for r in self.conn.execute(f"SELECT commit_id FROM data_commits WHERE {where} ORDER BY n", args)]
        self.conn.executemany("UPDATE data_commits SET status=? WHERE commit_id=?", [(status, i) for i in ids])
        return ids

    def remove(self, node_id: str, attempt: int) -> int:
        """Delete the commits a cleaned-up attempt made."""
        return self.conn.execute("DELETE FROM data_commits WHERE node_id=? AND attempt=?", (node_id, attempt)).rowcount

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
            for sub in ("videos", "captions", "poses"):
                (root / sub).mkdir(parents=True, exist_ok=True)
            for clip_id in entry["clips"]:
                clip = self.clips.get(clip_id)
                for source, target in (
                        (self.blobs.path(clip["video_digest"], "video"), root / "videos" / f"{clip_id}.mp4"),
                        (self.blobs.path(clip["caption_digest"], "caption"), root / "captions" / f"{clip_id}.json"),
                        (self.blobs.path(clip["pose_digest"], "pose"), root / "poses" / f"{clip_id}.npz")):
                    os.link(source, target)
            roots[name] = root
        return roots
