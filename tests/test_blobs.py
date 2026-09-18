import hashlib, pytest
from ar_kernel.archive.blobs import BlobStore, BlobError
from ar_kernel.archive.db import open_db

def _clip(tmp_path, name, data=b"video-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return p

def test_put_returns_sha256_and_moves_file(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    src = _clip(tmp_path, "a.mp4")
    digest = store.put(src, "video")
    assert digest == hashlib.sha256(b"video-bytes").hexdigest()
    assert not src.exists()
    assert store.path(digest, "video").read_bytes() == b"video-bytes"

def test_identical_content_is_stored_once(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    d1 = store.put(_clip(tmp_path, "a.mp4"), "video")
    d2 = store.put(_clip(tmp_path, "b.mp4"), "video")
    assert d1 == d2
    assert len(list((tmp_path / "store" / "blobs" / "video").iterdir())) == 1

def test_stored_blob_is_read_only(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    digest = store.put(_clip(tmp_path, "a.mp4"), "video")
    assert store.path(digest, "video").stat().st_mode & 0o222 == 0

def test_unknown_kind_is_rejected(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    with pytest.raises(BlobError, match="unknown kind"):
        store.put(_clip(tmp_path, "a.bin"), "depth")
