#!/usr/bin/env python3
"""Run the complete LC0 -> Stockfish -> Enyo-runtime -> Bullet Forge pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


DEFAULT_INPUT = Path.home() / "assets/training/lc0/test91-forge-input"
DEFAULT_OUTPUT = Path.home() / "assets/training/bullet/lc0-stockfish/test91-stockfish-enyo.bullet"
DEFAULT_ENGINE = Path.home() / "assets/engines/reference"
DEFAULT_NET = Path.home() / "assets/nets/nn-0ee0657fb25e.nnue"
LEGACY_OUTPUT_DIR = Path.home() / "assets/training/bullet/lc0/test91"
MIN_ARCHIVES = 100
MIN_BYTES = 100_000_000_000


def archive_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.tar")
        if path.is_file() and path.name.startswith(("training.", "training-"))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight_source(root: Path, *, allow_small: bool) -> tuple[int, int]:
    if not root.is_dir():
        raise SystemExit(f"LC0 input is not a directory: {root}")
    archives = archive_paths(root)
    total_bytes = sum(path.stat().st_size for path in archives)
    if not archives:
        raise SystemExit(f"LC0 input contains no training.*.tar archives: {root}")
    if not allow_small and (len(archives) < MIN_ARCHIVES or total_bytes < MIN_BYTES):
        raise SystemExit(
            "refusing undersized LC0 input (this catches the four-archive fixture): "
            f"archives={len(archives):,}, bytes={total_bytes:,}; "
            "pass --allow-small-input only for an intentional small test"
        )
    return len(archives), total_bytes


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")


def cleanup_old_outputs(root: Path, *, keep: Path | None = None) -> list[Path]:
    """Remove only named conversion products, never the LC0 source or static corpus."""
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    patterns = (
        "lc0-root-*.bullet",
        "lc0-root-*.bullet.*",
        "lc0-root-*.calibration.json",
        "lc0-root-*.validation.json",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if keep is not None and path == keep:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(path)
    for path in root.glob("lc0-root-*-chunk*"):
        if keep is not None and path == keep:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def build_command(args: argparse.Namespace, template: Path) -> list[str]:
    return [
        "forge", "run", str(template),
        "--input", str(args.input),
        "--output", str(args.output),
        "--engine", str(args.engine),
        "--net", str(args.net),
        "--depth", str(args.depth),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--max-records", "0",
        "--split-records", str(args.shards),
        "--min-ply", str(args.min_ply),
        "--shards", str(args.shards),
        "--wait",
    ] + (["--quiet-only"] if args.quiet_only else ["--no-quiet-only"])


def write_provenance(path: Path, *, args: argparse.Namespace, archive_count: int,
                     archive_bytes: int, validation: dict[str, object],
                     removed: Iterable[Path]) -> Path:
    manifest = path.with_name(path.name + ".manifest.json")
    payload = {
        "schema": "enyo.lc0-stockfish-enyo-bullet.v1",
        "pipeline": [
            "LC0 V6 decode",
            "Stockfish UCI search labels via EvalFile",
            "Enyo runtime clamp and phase normalization",
            "BulletFormat serialization",
        ],
        "input": str(args.input),
        "input_archive_count": archive_count,
        "input_archive_bytes": archive_bytes,
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "net": str(args.net),
        "net_sha256": sha256_file(args.net),
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "min_ply": args.min_ply,
        "quiet_only": args.quiet_only,
        "output": str(path),
        "output_sha256": sha256_file(path),
        "validation": validation,
        "cleaned_before_run": [str(item) for item in removed],
    }
    temporary = manifest.with_name(f".{manifest.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--shards", type=int, default=1600)
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--quiet-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-small-input", action="store_true")
    parser.add_argument("--clean-old", action="store_true", help="Remove old lc0-root conversion products before launch")
    parser.add_argument("--clean-only", action="store_true", help="Remove old conversion products and exit")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.engine = args.engine.expanduser().resolve()
    args.net = args.net.expanduser().resolve()
    archive_count, archive_bytes = preflight_source(args.input, allow_small=args.allow_small_input)
    require_file(args.engine, "Stockfish engine")
    require_file(args.net, "Stockfish net")
    if args.shards <= 0 or args.depth <= 0 or args.threads <= 0 or args.hash <= 0:
        raise SystemExit("depth, threads, hash, and shards must be positive")
    template = Path(__file__).resolve().with_name("label-lc0-stockfish-enyo.template.json")
    removed: list[Path] = []
    if args.clean_only:
        removed = cleanup_old_outputs(args.output.parent)
        if LEGACY_OUTPUT_DIR != args.output.parent:
            removed.extend(cleanup_old_outputs(LEGACY_OUTPUT_DIR))
        print(json.dumps({"cleaned": [str(path) for path in removed]}, indent=2))
        return 0
    if args.clean_old and not args.dry_run:
        removed = cleanup_old_outputs(args.output.parent)
        if LEGACY_OUTPUT_DIR != args.output.parent:
            removed.extend(cleanup_old_outputs(LEGACY_OUTPUT_DIR))
    if args.output.exists() and not args.dry_run:
        raise SystemExit(f"output already exists; refusing overwrite: {args.output}")
    command = build_command(args, template)
    print(json.dumps({
        "input": str(args.input),
        "input_archive_count": archive_count,
        "input_archive_bytes": archive_bytes,
        "output": str(args.output),
        "command": command,
        "cleaned": [str(path) for path in removed],
    }, indent=2), flush=True)
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    if not args.output.is_file():
        raise SystemExit(f"Forge completed without final output: {args.output}")
    validator = Path(__file__).resolve().parents[1] / "validate" / "validate_bullet_results.py"
    result = subprocess.run(
        [sys.executable, str(validator), "--input", str(args.output), "--require-win-loss"],
        check=True, capture_output=True, text=True,
    )
    validation = json.loads(result.stdout)
    manifest = write_provenance(
        args.output, args=args, archive_count=archive_count,
        archive_bytes=archive_bytes, validation=validation, removed=removed,
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), "validation": validation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
