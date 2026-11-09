"""File changes as small spans: the start says what changes and the state before, the end the state after.
A call that changes nothing (a folder that exists, a path already gone) records nothing."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from . import telemetry


def _size(path) -> int | None:
    try:
        return os.lstat(path).st_size
    except FileNotFoundError:
        return None


def files(root: Path) -> list[list]:
    """[relative path, size] of every file under `root`."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return [[root.name, _size(root)]]
    return sorted([str(Path(d, f).relative_to(root)), _size(Path(d, f))] for d, _, files in os.walk(root) for f in files)


def _fsync(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_text(path, text: str, *, atomic: bool = False, sync: bool = False, encoding: str = "utf-8",
               errors: str = "strict", record_text: bool = True) -> None:
    """`atomic`: through a temporary file and a rename, so a reader never sees half of it. `sync` (with `atomic`):
    on disk with its folder before it returns, since a network filesystem can lose what was only written.
    Not `record_text`: the record holds the text's sha256 instead (its content is recorded elsewhere)."""
    path = Path(path)
    content = {"text": text} if record_text else \
        {"sha256": hashlib.sha256(text.encode(encoding, errors)).hexdigest()}
    with telemetry.span("fs.write", {"path": str(path), "size_before": _size(path), **content}) as sp:
        if atomic:
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(text, encoding=encoding, errors=errors)
            if sync:
                _fsync(tmp)
            os.replace(tmp, path)
            if sync:
                _fsync(path.parent)
        else:
            path.write_text(text, encoding=encoding, errors=errors)
        sp.output = {"size": _size(path)}


def write_bytes(path, data: bytes) -> None:
    path = Path(path)
    with telemetry.span("fs.write", {"path": str(path), "size_before": _size(path), "bytes": len(data)}) as sp:
        path.write_bytes(data)
        sp.output = {"size": _size(path)}


def _ignored(sp, exc: OSError, reason: str) -> None:
    telemetry.error(exc, f"ignored: {reason}")
    sp.output = {"removed": False}


def remove(path, *, ignore_errors: str | None = None) -> None:
    """Remove a file; with `ignore_errors` (the reason), a failure is recorded and ignored."""
    path = Path(path)
    if (size := _size(path)) is None:
        return
    with telemetry.span("fs.remove", {"path": str(path), "size": size}) as sp:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            if ignore_errors is None:
                raise
            return _ignored(sp, exc, ignore_errors)
        sp.output = {"removed": True}


def rmtree(path, *, ignore_errors: str | None = None) -> None:
    """Remove a folder and everything in it, each file named with its size; with `ignore_errors` (the reason),
    a failure is recorded and ignored."""
    path = Path(path)
    if _size(path) is None:
        return
    with telemetry.span("fs.rmtree", {"path": str(path), "files": files(path)}) as sp:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            if ignore_errors is None:
                raise
            return _ignored(sp, exc, ignore_errors)
        sp.output = {"removed": True}


def rename(src, dst) -> None:
    with telemetry.span("fs.rename", {"src": str(src), "dst": str(dst), "size": _size(src),
                                      "dst_size_before": _size(dst)}) as sp:
        os.replace(src, dst)
        sp.output = {"size": _size(dst)}


def link(src, dst, *, follow_symlinks: bool = True) -> None:
    with telemetry.span("fs.link", {"src": str(src), "dst": str(dst), "size": _size(src)}) as sp:
        os.link(src, dst, follow_symlinks=follow_symlinks)
        sp.output = {"size": _size(dst)}


def copy(src, dst) -> None:
    """A file's bytes and modes (shutil.copy2)."""
    with telemetry.span("fs.copy", {"src": str(src), "dst": str(dst), "size": _size(src),
                                    "dst_size_before": _size(dst)}) as sp:
        shutil.copy2(src, dst)
        sp.output = {"size": _size(dst)}


def chmod(path, mode: int) -> None:
    with telemetry.span("fs.chmod", {"path": str(path), "mode_before": oct(os.stat(path).st_mode & 0o7777),
                                     "mode": oct(mode)}) as sp:
        os.chmod(path, mode)
        sp.output = {"mode": oct(os.stat(path).st_mode & 0o7777)}


def mkdir(path) -> None:
    """A folder and its missing parents."""
    path = Path(path)
    if path.is_dir():
        return
    with telemetry.span("fs.mkdir", {"path": str(path)}) as sp:
        path.mkdir(parents=True, exist_ok=True)
        sp.output = {"created": True}


def _place(place, base: Path, rel: str) -> list[str]:
    """`place(name, dir_fd)` in base/rel's folder without following any link under `base`: each directory is
    opened with O_NOFOLLOW and the file is placed relative to that directory's fd, so a link an agent plants
    or swaps in (it owns staging, and its sandbox runs meanwhile) raises instead of redirecting. Returns the
    folders it made."""
    *dirs, name = PurePosixPath(rel).parts
    made = []
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in dirs:
            try:
                os.mkdir(part, dir_fd=fd)
                made.append(part)
            except FileExistsError:
                pass
            sub = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = sub
        place(name, fd)
    finally:
        os.close(fd)
    return made


def place(src, base, rel: str, *, link: bool = False) -> None:
    """Move `src` (or with `link`, hard-link it, keeping its own name) to base/rel (see `_place`)."""
    src = Path(src)
    with telemetry.span("fs.link" if link else "fs.rename", {"src": str(src), "dst": str(Path(base) / rel),
                                                             "size": _size(src)}) as sp:
        if link:
            made = _place(lambda name, fd: os.link(src, name, dst_dir_fd=fd, follow_symlinks=False), base, rel)
        else:
            made = _place(lambda name, fd: os.replace(src, name, dst_dir_fd=fd), base, rel)
        sp.output = {"size": _size(Path(base) / rel), "folders_made": made}


def place_bytes(data: bytes, base, rel: str) -> None:
    """`data` as base/rel (see `_place`), written through a new file's own handle first."""
    with telemetry.span("fs.write", {"path": str(Path(base) / rel), "bytes": len(data)}) as sp:
        with tempfile.NamedTemporaryFile(dir=base, suffix=".tmp", delete=False) as handle:
            handle.write(data)
        try:
            made = _place(lambda name, fd: os.replace(handle.name, name, dst_dir_fd=fd), base, rel)
        finally:
            Path(handle.name).unlink(missing_ok=True)
        sp.output = {"size": _size(Path(base) / rel), "folders_made": made}


def copytree(src, dst, *, link: bool = False, ignore=None, symlinks: bool = False) -> None:
    """`link`: hard links instead of copies; `symlinks`: links copied as links."""
    with telemetry.span("fs.copytree", {"src": str(src), "dst": str(dst), "link": link, "files": files(src)}) as sp:
        shutil.copytree(src, dst, copy_function=os.link if link else shutil.copy2, ignore=ignore, symlinks=symlinks)
        sp.output = {"files": len(files(dst))}
