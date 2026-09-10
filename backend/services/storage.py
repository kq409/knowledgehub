"""Preservation I/O. Local disk today; the protocol is what a prod S3 swap implements.

`put` refuses to overwrite an existing key. That is the whole point of this
layer: the original is immutable. A new version is a new key (`…/r{n}.ext`).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class ImmutableObjectError(FileExistsError):
    """Raised when a caller tries to replace an original already on disk."""


class Storage(Protocol):
    def put(self, key: str, data: bytes) -> Path: ...

    def get(self, key: str) -> bytes: ...

    def checksum(self, key: str) -> str: ...

    def exists(self, key: str) -> bool: ...

    def path_for(self, key: str) -> Path: ...


class LocalDiskStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        cleaned = key.replace("\\", "/").lstrip("/")
        dest = (self.root / cleaned).resolve()
        root = self.root.resolve()
        if dest != root and root not in dest.parents:
            raise ValueError(f"invalid storage key {key!r}")
        return dest

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def put(self, key: str, data: bytes) -> Path:
        dest = self.path_for(key)
        if dest.exists():
            raise ImmutableObjectError(f"Original already stored at {key}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def get(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def checksum(self, key: str) -> str:
        return sha256_hex(self.get(key))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_storage() -> LocalDiskStorage:
    configured = os.getenv("STORAGE_DIR", "data")
    path = Path(configured)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return LocalDiskStorage(path)
