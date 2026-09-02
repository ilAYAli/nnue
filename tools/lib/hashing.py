from __future__ import annotations

import hashlib
from pathlib import Path

COPY_BLOCK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
