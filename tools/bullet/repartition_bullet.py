#!/usr/bin/env python3
"""Repartition fixed-width BulletFormat files without relabeling or reordering."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

RECORD_BYTES = 32
COPY_BYTES = 8 * 1024 * 1024


def input_files(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard-*.bullet"))
    if not paths:
        raise ValueError(f"no shard-*.bullet files in {directory}")
    for path in paths:
        if path.stat().st_size % RECORD_BYTES:
            raise ValueError(f"{path}: size is not a multiple of {RECORD_BYTES}")
    return paths


def repartition(sources: list[Path], output: Path, chunks: int) -> dict:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_records = [path.stat().st_size // RECORD_BYTES for path in sources]
    total = sum(source_records)
    if total < chunks:
        raise ValueError(f"{total} records cannot fill {chunks} chunks")

    source_index = 0
    source_handle = sources[0].open("rb")
    source_remaining = source_records[0] * RECORD_BYTES
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    written = 0
    outputs: list[dict[str, int | str]] = []
    try:
        for index in range(chunks):
            records = total // chunks + int(index < total % chunks)
            target = output / f"chunk-{index:04d}.bullet"
            partial = target.with_name(f".{target.name}.partial.{os.getpid()}")
            partial.unlink(missing_ok=True)
            with partial.open("wb") as destination:
                remaining = records * RECORD_BYTES
                while remaining:
                    if source_remaining == 0:
                        source_handle.close()
                        source_index += 1
                        source_handle = sources[source_index].open("rb")
                        source_remaining = source_records[source_index] * RECORD_BYTES
                    take = min(remaining, source_remaining, COPY_BYTES)
                    payload = source_handle.read(take)
                    if len(payload) != take:
                        raise ValueError(f"truncated source: {sources[source_index]}")
                    destination.write(payload)
                    source_hash.update(payload)
                    output_hash.update(payload)
                    remaining -= take
                    source_remaining -= take
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(partial, target)
            written += records
            outputs.append({"path": target.name, "records": records, "bytes": records * RECORD_BYTES})
    finally:
        source_handle.close()
    if written != total or source_index != len(sources) - 1 or source_remaining:
        raise ValueError("repartition did not consume every input record")
    if source_hash.digest() != output_hash.digest():
        raise ValueError("output stream hash differs from input stream hash")
    return {
        "schema": "enyo.bullet-repartition.v1",
        "record_bytes": RECORD_BYTES,
        "input": [{"path": str(path), "records": records} for path, records in zip(sources, source_records)],
        "total_records": total,
        "chunks": outputs,
        "stream_sha256": source_hash.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=int)
    parser.add_argument("--write-path-manifest", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if args.write_path_manifest and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = "".join(str((args.output_dir / item["path"]).resolve()) + "\n" for item in manifest["chunks"])
        (args.output_dir / "chunks.paths").write_text(paths, encoding="utf-8")
        print(json.dumps({"chunks": len(manifest["chunks"]), "path_manifest": str(args.output_dir / "chunks.paths")}))
        return 0
    manifest = repartition(input_files(args.input_dir), args.output_dir, args.chunks)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_path_manifest:
        paths = "".join(str((args.output_dir / item["path"]).resolve()) + "\n" for item in manifest["chunks"])
        (args.output_dir / "chunks.paths").write_text(paths, encoding="utf-8")
    print(json.dumps({"records": manifest["total_records"], "chunks": len(manifest["chunks"]), "sha256": manifest["stream_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
