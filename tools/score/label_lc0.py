#!/usr/bin/env python3
"""Stream LC0 V6 records through a UCI engine into a BulletFormat shard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import chess

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "posgen"))
sys.path.insert(0, str(TOOLS / "score"))

from lib import bullet_format, bullet_text  # noqa: E402
import lc0_to_jsonl  # noqa: E402
from label_with_uci import EngineTimeout, UciEngine  # noqa: E402


def is_quiet(row: dict) -> bool:
    board = chess.Board(str(row["fen"]))
    if board.is_check():
        return False
    played = next(
        (item for item in row.get("moves", []) if "played" in item.get("roles", [])),
        None,
    )
    if played is None or not played.get("legal"):
        return False
    move = chess.Move.from_uci(str(played["move"]))
    return (
        move.promotion is None
        and not board.is_capture(move)
        and not board.is_castling(move)
    )


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial.{os.getpid()}")


def fsync_file(handle: object) -> None:
    handle.flush()  # type: ignore[attr-defined]
    os.fsync(handle.fileno())  # type: ignore[attr-defined]


def label(args: argparse.Namespace, *, engine_type: type[UciEngine] = UciEngine) -> dict:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    output = args.output
    stats_path = args.stats
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = temporary_path(output)
    stats_tmp = temporary_path(stats_path)
    output_tmp.unlink(missing_ok=True)
    stats_tmp.unlink(missing_ok=True)

    decode = lc0_to_jsonl.Stats()
    stats: dict[str, object] = {
        "schema": "enyo.lc0-label-stats.v1",
        "input": str(args.input),
        "inventory": str(args.inventory),
        "output": str(output),
        "engine": args.engine,
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "max_records": args.max_records,
        "min_ply": args.min_ply,
        "quiet_only": args.quiet_only,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "read": 0,
        "selected": 0,
        "written": 0,
        "skipped_min_ply": 0,
        "skipped_non_quiet": 0,
        "skipped_mate": 0,
        "skipped_no_score": 0,
        "skipped_timeout": 0,
        "skipped_cp": 0,
    }
    start = time.monotonic()
    engine = engine_type(args.engine, threads=args.threads, hash_mb=args.hash)
    try:
        with output_tmp.open("wb") as dst:
            for row, ply in lc0_to_jsonl.iter_rows(
                args.input,
                inventory=args.inventory,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                max_records=args.max_records,
                top_policy=0,
                stats=decode,
            ):
                stats["read"] = decode.records
                if ply < args.min_ply:
                    stats["skipped_min_ply"] = int(stats["skipped_min_ply"]) + 1
                    continue
                if args.quiet_only and not is_quiet(row):
                    stats["skipped_non_quiet"] = int(stats["skipped_non_quiet"]) + 1
                    continue
                stats["selected"] = int(stats["selected"]) + 1
                try:
                    score, mate = engine.label(
                        row["fen"],
                        depth=args.depth,
                        timeout_s=args.engine_timeout_s,
                    )
                except EngineTimeout:
                    stats["skipped_timeout"] = int(stats["skipped_timeout"]) + 1
                    engine.restart()
                    continue
                if mate is not None:
                    stats["skipped_mate"] = int(stats["skipped_mate"]) + 1
                    continue
                if score is None:
                    stats["skipped_no_score"] = int(stats["skipped_no_score"]) + 1
                    continue
                row["score"] = score
                white_score = bullet_text.white_score_from_row(
                    row,
                    enyo_runtime_target=args.enyo_runtime_target,
                )
                if args.max_abs_cp > 0 and abs(white_score) > args.max_abs_cp:
                    stats["skipped_cp"] = int(stats["skipped_cp"]) + 1
                    continue
                bullet_format.write_row(
                    dst,
                    row,
                    enyo_runtime_target=args.enyo_runtime_target,
                )
                stats["written"] = int(stats["written"]) + 1
                if args.progress > 0 and int(stats["selected"]) % args.progress == 0:
                    elapsed = max(time.monotonic() - start, 1e-6)
                    print(
                        f"shard {args.shard_index}/{args.shard_count} "
                        f"read={decode.records} selected={stats['selected']} "
                        f"written={stats['written']} rate={int(stats['selected']) / elapsed:.1f}/s",
                        flush=True,
                    )
            stats["read"] = decode.records
            fsync_file(dst)

        size = output_tmp.stat().st_size
        written = int(stats["written"])
        if size == 0:
            raise ValueError("Bullet shard is empty")
        if size % bullet_format.RECORD_BYTES:
            raise ValueError(f"Bullet shard size {size} is not divisible by 32")
        if written != size // bullet_format.RECORD_BYTES:
            raise ValueError("Bullet shard record count does not match output size")
        stats["bytes"] = size
        stats["elapsed_s"] = round(time.monotonic() - start, 3)
        stats["decoder"] = {
            "files": decode.files,
            "invalid_records": decode.invalid_records,
            "unsupported_records": decode.unsupported_records,
            "invalid_boards": decode.invalid_boards,
        }
        with stats_tmp.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
            fsync_file(handle)
        os.replace(output_tmp, output)
        os.replace(stats_tmp, stats_path)
        print(json.dumps(stats, sort_keys=True), flush=True)
        return stats
    finally:
        engine.close()
        output_tmp.unlink(missing_ok=True)
        stats_tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=9_000_000)
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--quiet-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--engine-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-abs-cp", type=int, default=10_000)
    parser.add_argument("--progress", type=int, default=1000)
    parser.add_argument("--enyo-runtime-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    label(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
