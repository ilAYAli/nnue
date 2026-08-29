#!/usr/bin/env python3
"""Run the complete LC0 -> Stockfish -> Enyo-runtime -> Bullet Forge pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.validate.validate_bullet_results import validate_and_merge  # noqa: E402


DEFAULT_INPUT = Path.home() / "assets/training/lc0/test91-forge-input"
DEFAULT_OUTPUT = Path.home() / "assets/training/bullet/lc0-stockfish/test91-stockfish-enyo.bullet"
DEFAULT_ENGINE = Path.home() / "assets/engines/reference"
DEFAULT_NET = Path.home() / "assets/nets/nn-0ee0657fb25e.nnue"
LEGACY_OUTPUT_DIR = Path.home() / "assets/training/bullet/lc0/test91"
MIN_ARCHIVES = 100
MIN_BYTES = 100_000_000_000
DEFAULT_BATCH_BYTES = 20_000_000_000
FORGE_UNPACKED = Path.home() / ".cache/forge/unpacked-lc0"
FORGE_INPUTS = Path.home() / ".cache/forge/inputs"
FORGE_TASK_INPUTS = Path.home() / ".cache/forge/task-inputs"


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


def partition_archives(archives: list[Path], max_bytes: int) -> list[list[Path]]:
    if max_bytes <= 0:
        raise ValueError("batch byte limit must be positive")
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for archive in archives:
        size = archive.stat().st_size
        if current and current_bytes + size > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(archive)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def link_batch(source: Path, archives: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for index, archive in enumerate(archives):
        # Keep the training.* prefix required by Forge while making names
        # unique even if a future source directory contains duplicate basenames.
        name = f"training.{index:05d}-{archive.name.removeprefix('training.')}"
        (destination / name).symlink_to(archive)


def cleanup_new_cache_entries(before: set[Path], root: Path) -> None:
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry not in before and entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)


def balanced_shard_counts(batch_sizes: list[int], total: int) -> list[int]:
    """Allocate exactly ``total`` Forge tasks across batches by byte weight."""
    if not batch_sizes or total < len(batch_sizes):
        raise ValueError("cannot allocate fewer tasks than batches")
    overall = sum(batch_sizes)
    if overall <= 0:
        raise ValueError("batch sizes must have a positive total")
    raw = [total * size / overall for size in batch_sizes]
    counts = [max(1, int(value)) for value in raw]
    while sum(counts) > total:
        index = max(
            (i for i, count in enumerate(counts) if count > 1),
            key=lambda i: counts[i] - raw[i],
        )
        counts[index] -= 1
    while sum(counts) < total:
        index = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    return counts


def verify_forge_partition(command: list[str]) -> dict[str, object]:
    """Build a manifest without launching workers and reject overlapping tasks."""
    result = subprocess.run(
        [*command, "--print-manifest"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SystemExit(
            "Forge partition preflight failed: "
            + (detail[-1] if detail else f"rc={result.returncode}")
        )
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit("Forge partition preflight returned invalid JSON") from exc
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("Forge partition preflight produced no tasks")
    match = re.search(r"found\s+([0-9,]+)\s+files", result.stderr)
    if match is None:
        raise SystemExit("Forge partition preflight did not report its inventory size")
    expected_file_count = int(match.group(1).replace(",", ""))

    seen: set[tuple[str, str, int]] = set()
    seen_paths: set[str] = set()
    for task in tasks:
        inputs = [item for item in task.get("inputs", []) if item.get("tree") == "lc0-inventory"]
        if len(inputs) != 1:
            raise SystemExit(f"{task.get('id', '?')}: expected exactly one LC0 task inventory")
        item = inputs[0]
        task_inventory_path = Path(str(item.get("path", ""))).expanduser() / "inventory.json"
        try:
            payload = json.loads(task_inventory_path.read_text(encoding="utf-8"))
            entries = payload["files"]
            task_keys = {
                (str(entry["path"]), str(entry["sha256"]), int(entry["size"]))
                for entry in entries
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read task inventory: {task_inventory_path}") from exc
        if len(task_keys) != len(entries):
            raise SystemExit(f"{task.get('id', '?')}: duplicate entries inside task inventory")
        paths = {key[0] for key in task_keys}
        if seen_paths & paths:
            overlap = sorted(seen_paths & paths)[0]
            raise SystemExit(f"Forge task inventories overlap at {overlap}")
        seen.update(task_keys)
        seen_paths.update(paths)
        if int(item.get("files", -1)) != len(task_keys):
            raise SystemExit(f"{task.get('id', '?')}: manifest file count disagrees with inventory")
    if len(seen) != expected_file_count:
        raise SystemExit(
            "Forge task inventories do not exactly cover the coordinator inventory: "
            f"tasks={len(seen):,} expected={expected_file_count:,}"
        )
    return {
        "tasks": len(tasks),
        "files": len(seen),
        "inventory_files": expected_file_count,
    }


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
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES,
                        help="Maximum compressed archive bytes staged per sequential Forge run")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive_count, archive_bytes = preflight_source(args.input, allow_small=args.allow_small_input)
    archives = archive_paths(args.input)
    require_file(args.engine, "Stockfish engine")
    require_file(args.net, "Stockfish net")
    if args.shards <= 0 or args.depth <= 0 or args.threads <= 0 or args.hash <= 0 or args.batch_bytes <= 0:
        raise SystemExit("depth, threads, hash, shards, and batch-bytes must be positive")
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
    batches = partition_archives(archives, args.batch_bytes)
    batch_sizes = [sum(item.stat().st_size for item in batch) for batch in batches]
    batch_shards = balanced_shard_counts(batch_sizes, args.shards)
    print(json.dumps({
        "input": str(args.input),
        "input_archive_count": archive_count,
        "input_archive_bytes": archive_bytes,
        "output": str(args.output),
        "batch_count": len(batches),
        "batch_bytes_limit": args.batch_bytes,
        "batch_shards": batch_shards,
        "command": command,
        "cleaned": [str(path) for path in removed],
    }, indent=2), flush=True)
    if args.dry_run:
        return 0
    batch_root = args.output.parent / f".{args.output.stem}.batches.{os.getpid()}"
    batch_root.mkdir(parents=True, exist_ok=False)
    batch_outputs: list[Path] = []
    try:
        for index, (batch, shard_count) in enumerate(zip(batches, batch_shards, strict=True)):
            batch_input = batch_root / f"input-{index:04d}"
            batch_output = batch_root / f"output-{index:04d}.bullet"
            link_batch(args.input, batch, batch_input)
            batch_args = argparse.Namespace(**vars(args))
            batch_args.input = batch_input
            batch_args.output = batch_output
            batch_args.shards = shard_count
            batch_command = build_command(batch_args, template)
            print(json.dumps({
                "batch": index + 1,
                "batches": len(batches),
                "archives": len(batch),
                "archive_bytes": sum(item.stat().st_size for item in batch),
                "shards": shard_count,
                "command": batch_command,
            }), flush=True)
            unpacked_before = set(FORGE_UNPACKED.iterdir()) if FORGE_UNPACKED.is_dir() else set()
            inputs_before = set(FORGE_INPUTS.iterdir()) if FORGE_INPUTS.is_dir() else set()
            task_inputs_before = set(FORGE_TASK_INPUTS.iterdir()) if FORGE_TASK_INPUTS.is_dir() else set()
            try:
                partition = verify_forge_partition(batch_command)
                print(json.dumps({"batch": index + 1, "partition": partition}), flush=True)
                subprocess.run(batch_command, cwd=REPO_ROOT, check=True)
            finally:
                cleanup_new_cache_entries(unpacked_before, FORGE_UNPACKED)
                cleanup_new_cache_entries(inputs_before, FORGE_INPUTS)
                cleanup_new_cache_entries(task_inputs_before, FORGE_TASK_INPUTS)
            if not batch_output.is_file():
                raise SystemExit(f"Forge completed without batch output: {batch_output}")
            batch_outputs.append(batch_output)

        validation = validate_and_merge(
            batch_outputs,
            merge_output=args.output,
            require_win_loss=True,
        )
    finally:
        shutil.rmtree(batch_root, ignore_errors=True)
    manifest = write_provenance(
        args.output, args=args, archive_count=archive_count,
        archive_bytes=archive_bytes, validation=validation, removed=removed,
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), "validation": validation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
