#!/usr/bin/env python3
"""Bind a merged LC0 Bullet corpus to its measured calibration evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "score"))
import lc0_calibration  # noqa: E402

RECORD_BYTES = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attest(corpus: Path, calibration_path: Path, stats_dir: Path | None = None) -> dict:
    if not corpus.is_file() or corpus.stat().st_size == 0 or corpus.stat().st_size % RECORD_BYTES:
        raise ValueError(f"invalid Bullet corpus: {corpus}")
    _, calibration_sha256 = lc0_calibration.load(calibration_path)
    stats_paths = sorted(stats_dir.rglob("*.stats.json")) if stats_dir is not None else []
    if not stats_paths:
        raise ValueError("no label shard statistics supplied")
    shard_records = 0
    for path in stats_paths:
        stats = json.loads(path.read_text(encoding="utf-8"))
        calibration = stats.get("calibration") if isinstance(stats, dict) else None
        if not isinstance(calibration, dict) or calibration.get("sha256") != calibration_sha256:
            raise ValueError(f"{path}: missing or mismatched calibration provenance")
        if calibration.get("schema") != lc0_calibration.SCHEMA:
            raise ValueError(f"{path}: invalid calibration schema")
        shard_records += int(stats.get("written", -1))
    records = corpus.stat().st_size // RECORD_BYTES
    if records != shard_records:
        raise ValueError(f"merged corpus records={records} do not match shard statistics={shard_records}")
    return {
        "schema": "enyo.lc0-corpus-attestation.v1",
        "valid": True,
        "corpus": str(corpus), "corpus_sha256": sha256(corpus), "records": records,
        "calibration": {"path": str(calibration_path), "sha256": calibration_sha256},
        "label_stats": [{"path": str(path), "sha256": sha256(path)} for path in stats_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "enyo.lc0-corpus-attestation.v1" or manifest.get("valid") is not True:
            raise SystemExit("invalid LC0 calibration manifest")
        if manifest.get("corpus_sha256") != sha256(args.input):
            raise SystemExit("LC0 corpus changed after calibration attestation")
        calibration = manifest.get("calibration", {})
        if not isinstance(calibration, dict) or not calibration.get("path"):
            raise SystemExit("LC0 calibration manifest has no artifact")
        lc0_calibration.load(Path(str(calibration["path"])))
        print(json.dumps(manifest, sort_keys=True))
        return
    if args.calibration is None or args.stats_dir is None:
        raise SystemExit("--calibration and --stats-dir are required when creating an attestation")
    manifest = attest(args.input, args.calibration, args.stats_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(f".{args.manifest.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
