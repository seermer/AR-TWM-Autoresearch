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


def test_putting_the_stored_blob_itself_is_refused_and_the_blob_survives(tmp_path):
    """Put(target) used to see target.exists() and unlink the source --
    which WAS the blob. The DB kept saying it existed; every commit using it broke."""
    store = BlobStore(tmp_path, open_db(tmp_path))
    digest = store.put(_clip(tmp_path, "v.mp4", b"clip bytes"), "video")
    blob = store.path(digest, "video")
    with pytest.raises(BlobError, match="inside the blob store"):
        store.put(blob, "video")
    assert blob.read_bytes() == b"clip bytes"


def test_any_path_inside_the_blob_store_is_refused(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    sneaky = store.root / "video" / "not-a-digest.mp4"
    sneaky.write_bytes(b"planted")
    with pytest.raises(BlobError, match="inside the blob store"):
        store.put(sneaky, "video")
    assert sneaky.exists()


def test_truncated_existing_blob_is_repaired_not_trusted(tmp_path):
    """A crash mid-copy used to leave a truncated file under the right
    hash name, and every later put of that content kept the corrupt copy forever."""
    store = BlobStore(tmp_path, open_db(tmp_path))
    good = b"x" * 4096
    digest = store.put(_clip(tmp_path, "a.mp4", good), "video")
    blob = store.path(digest, "video")
    blob.chmod(0o644)
    blob.write_bytes(good[:100])          # simulate a torn write under the final name
    store.put(_clip(tmp_path, "b.mp4", good), "video")
    assert blob.read_bytes() == good


def test_put_leaves_no_temp_files_behind(tmp_path):
    store = BlobStore(tmp_path, open_db(tmp_path))
    store.put(_clip(tmp_path, "a.mp4", b"payload"), "video")
    assert [p.name for p in (store.root / "video").iterdir() if ".tmp" in p.name] == []
