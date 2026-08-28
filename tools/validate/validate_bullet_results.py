#!/usr/bin/env python3
"""Validate Bullet result labels and optionally atomically merge shards.

Bullet stores a categorical terminal result in every 32-byte record.  This is
separate from the score target.  In particular, a corpus of synthetic draws
can be structurally valid while being unusable for training, so this check is
mandatory before an LC0 corpus is accepted by the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable


RECORD_BYTES = 32
RESULT_OFFSET = 26
RESULT_NAMES = ("loss", "draw", "win")
COPY_BLOCK_BYTES = 8 * 1024 * 1024


def bullet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        paths = sorted(candidate for candidate in path.glob("*.bullet") if candidate.is_file())
        if paths:
            return paths
    raise ValueError(f"no Bullet input found at {path}")


def validate_and_merge(
    paths: Iterable[Path],
    *,
    merge_output: Path | None = None,
    replace: bool = False,
    require_win_loss: bool = False,
) -> dict[str, object]:
    paths = list(paths)
    if not paths:
        raise ValueError("no Bullet inputs")
    if merge_output is not None and merge_output.exists() and not replace:
        raise ValueError(f"{merge_output} exists; pass --replace to overwrite")

    temporary: Path | None = None
    destination = None
    digest = hashlib.sha256()
    counts = [0, 0, 0]
    records = 0
    total_bytes = 0
    try:
        if merge_output is not None:
            merge_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = merge_output.with_name(f".{merge_output.name}.partial.{os.getpid()}")
            temporary.unlink(missing_ok=True)
            destination = temporary.open("xb")

        for path in paths:
            size = path.stat().st_size
            if size == 0 or size % RECORD_BYTES:
                raise ValueError(f"{path}: invalid Bullet size {size}")
            with path.open("rb") as source:
                while block := source.read(COPY_BLOCK_BYTES):
                    if len(block) % RECORD_BYTES:
                        raise ValueError(f"{path}: unaligned read")
                    for result in range(3):
                        counts[result] += block[RESULT_OFFSET::RECORD_BYTES].count(result)
                    invalid = len(block) // RECORD_BYTES - sum(
                        block[RESULT_OFFSET::RECORD_BYTES].count(result)
                        for result in range(3)
                    )
                    if invalid:
                        raise ValueError(f"{path}: {invalid} invalid Bullet result labels")
                    records += len(block) // RECORD_BYTES
                    total_bytes += len(block)
                    digest.update(block)
                    if destination is not None:
                        destination.write(block)

        if require_win_loss and (counts[0] == 0 or counts[2] == 0):
            raise ValueError(
                "corpus has a missing decisive outcome: "
                f"loss={counts[0]} draw={counts[1]} win={counts[2]}"
            )
        if destination is not None and temporary is not None and merge_output is not None:
            destination.flush()
            os.fsync(destination.fileno())
            destination.close()
            destination = None
            if temporary.stat().st_size != total_bytes:
                raise ValueError("merged Bullet byte count mismatch")
            os.replace(temporary, merge_output)
            directory_fd = os.open(merge_output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

        return {
            "schema": "enyo.bullet-result-validation.v1",
            "inputs": [str(path) for path in paths],
            "merged_output": str(merge_output) if merge_output is not None else None,
            "bytes": total_bytes,
            "records": records,
            "result_counts": dict(zip(RESULT_NAMES, counts)),
            "sha256": digest.hexdigest(),
        }
    finally:
        if destination is not None:
            destination.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Bullet file or flat shard directory")
    parser.add_argument("--merge-output", type=Path, help="Atomically merge validated shards here")
    parser.add_argument("--replace", action="store_true", help="Allow replacement of --merge-output")
    parser.add_argument(
        "--require-win-loss",
        action="store_true",
        help="Reject an all-draw or one-sided corpus",
    )
    args = parser.parse_args()
    try:
        result = validate_and_merge(
            bullet_paths(args.input),
            merge_output=args.merge_output,
            replace=args.replace,
            require_win_loss=args.require_win_loss,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid Bullet corpus: {exc}") from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
