import hashlib
from pathlib import Path

import pytest

from services.storage import ImmutableObjectError, LocalDiskStorage, sha256_hex


def test_put_refuses_overwrite(tmp_path: Path):
    storage = LocalDiskStorage(tmp_path)
    storage.put("papers/a/r1.pdf", b"original")
    with pytest.raises(ImmutableObjectError, match="already stored"):
        storage.put("papers/a/r1.pdf", b"tampered")
    assert storage.get("papers/a/r1.pdf") == b"original"


def test_checksum_changes_when_bytes_change(tmp_path: Path):
    storage = LocalDiskStorage(tmp_path)
    storage.put("notes/n/r1.pdf", b"hello")
    assert storage.checksum("notes/n/r1.pdf") == sha256_hex(b"hello")
    dest = storage.path_for("notes/n/r1.pdf")
    dest.write_bytes(b"mutated")
    assert storage.checksum("notes/n/r1.pdf") != hashlib.sha256(b"hello").hexdigest()


def test_rejects_path_escape(tmp_path: Path):
    storage = LocalDiskStorage(tmp_path)
    with pytest.raises(ValueError, match="invalid storage key"):
        storage.path_for("../secret")
