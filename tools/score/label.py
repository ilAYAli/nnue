#!/usr/bin/env python3
"""Stream LC0 V6 records through a UCI engine into a BulletFormat shard."""
from __future__ import annotations

import argparse
import json
import math
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


def lc0_root_score(
    row: dict,
    *,
    eval_scale: float,
    value_epsilon: float,
) -> tuple[int, float, bool, bool, bool] | None:
    """Return an STM-relative CP score from an LC0 V6 root value head.

    LC0 stores root_q = P(win) - P(loss) and root_d = P(draw), both from the
    side to move. Bullet's search target is sigmoid(score / eval_scale), so
    its inverse maps the expected game score P(win) + P(draw) / 2 back to
    centipawns. Since P(win) + P(draw) / 2 = (1 + root_q) / 2, the draw head
    is audited but does not enter the scalar conversion.
    """
    try:
        lc0 = row["lc0"]
        root_q = float(lc0["root_q"])
        root_d = float(lc0["root_d"])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        not math.isfinite(root_q)
        or not math.isfinite(root_d)
        or root_q < -1.0
        or root_q > 1.0
        or root_d < 0.0
        or root_d > 1.0
    ):
        return None

    probability = (1.0 + root_q) / 2.0
    probability_out_of_range = not 0.0 <= probability <= 1.0
    # LC0's independently rounded q and draw heads can produce a value a few
    # nanounits beyond [0, 1] at the endpoints.  This mirrors the parser's
    # existing result-WDL handling and retains the position while auditing it.
    probability = min(1.0, max(0.0, probability))

    clamped_low = probability <= value_epsilon
    clamped_high = probability >= 1.0 - value_epsilon
    bounded = min(1.0 - value_epsilon, max(value_epsilon, probability))
    score = int(round(eval_scale * math.log(bounded / (1.0 - bounded))))
    return score, probability, clamped_low, clamped_high, probability_out_of_range


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

    score_source = getattr(args, "score_source", "uci")
    eval_scale = float(getattr(args, "eval_scale", 400.0))
    value_epsilon = float(getattr(args, "value_epsilon", 1e-6))
    if score_source not in {"uci", "lc0-root"}:
        raise ValueError(f"unsupported score source: {score_source}")
    if not math.isfinite(eval_scale) or eval_scale <= 0.0:
        raise ValueError("eval scale must be finite and positive")
    if not math.isfinite(value_epsilon) or not 0.0 < value_epsilon < 0.5:
        raise ValueError("value epsilon must be finite and in (0, 0.5)")
    if score_source == "lc0-root" and args.static:
        raise ValueError("--static is only valid with --score-source uci")

    decode = lc0_to_jsonl.Stats()
    stats: dict[str, object] = {
        "schema": "enyo.label-stats.v2",
        "input": str(args.input),
        "inventory": str(args.inventory) if args.inventory is not None else None,
        "output": str(output),
        "engine": args.engine if score_source == "uci" else None,
        "net": args.net,
        "net_option": args.net_option,
        "score_source": score_source,
        "eval_scale": eval_scale if score_source == "lc0-root" else None,
        "value_epsilon": value_epsilon if score_source == "lc0-root" else None,
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
        "skipped_invalid_lc0_value": 0,
        "lc0_root_value_count": 0,
        "lc0_root_probability_min": None,
        "lc0_root_probability_max": None,
        "lc0_root_probability_sum": 0.0,
        "lc0_root_score_cp_min": None,
        "lc0_root_score_cp_max": None,
        "lc0_root_score_cp_sum": 0,
        "lc0_root_clamped_low": 0,
        "lc0_root_clamped_high": 0,
        "lc0_root_probability_out_of_range": 0,
    }
    start = time.monotonic()
    engine = None
    if score_source == "uci":
        engine = engine_type(
            args.engine,
            threads=args.threads,
            hash_mb=args.hash,
            net=args.net,
            net_option=args.net_option,
        )
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
                if score_source == "lc0-root":
                    root_score = lc0_root_score(
                        row,
                        eval_scale=eval_scale,
                        value_epsilon=value_epsilon,
                    )
                    if root_score is None:
                        stats["skipped_invalid_lc0_value"] = (
                            int(stats["skipped_invalid_lc0_value"]) + 1
                        )
                        continue
                    score, probability, clamped_low, clamped_high, out_of_range = root_score
                    stats["lc0_root_value_count"] = int(stats["lc0_root_value_count"]) + 1
                    stats["lc0_root_probability_sum"] = (
                        float(stats["lc0_root_probability_sum"]) + probability
                    )
                    stats["lc0_root_score_cp_sum"] = (
                        int(stats["lc0_root_score_cp_sum"]) + score
                    )
                    for key, value in (
                        ("lc0_root_probability_min", probability),
                        ("lc0_root_score_cp_min", score),
                    ):
                        current = stats[key]
                        stats[key] = value if current is None else min(current, value)
                    for key, value in (
                        ("lc0_root_probability_max", probability),
                        ("lc0_root_score_cp_max", score),
                    ):
                        current = stats[key]
                        stats[key] = value if current is None else max(current, value)
                    if clamped_low:
                        stats["lc0_root_clamped_low"] = (
                            int(stats["lc0_root_clamped_low"]) + 1
                        )
                    if clamped_high:
                        stats["lc0_root_clamped_high"] = (
                            int(stats["lc0_root_clamped_high"]) + 1
                        )
                    if out_of_range:
                        stats["lc0_root_probability_out_of_range"] = (
                            int(stats["lc0_root_probability_out_of_range"]) + 1
                        )
                else:
                    assert engine is not None
                    try:
                        if args.static:
                            score = engine.static_eval(
                                row["fen"],
                                timeout_s=args.engine_timeout_s,
                            )
                            mate = None
                        else:
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
        stats["bytes"] = size
        stats["elapsed_s"] = round(time.monotonic() - start, 3)
        stats["decoder"] = {
            "files": decode.files,
            "invalid_records": decode.invalid_records,
            "unsupported_records": decode.unsupported_records,
            "invalid_boards": decode.invalid_boards,
        }
        error = None
        if size == 0:
            error = "Bullet shard is empty"
        elif size % bullet_format.RECORD_BYTES:
            error = f"Bullet shard size {size} is not divisible by 32"
        elif written != size // bullet_format.RECORD_BYTES:
            error = "Bullet shard record count does not match output size"
        if error is not None:
            stats["error"] = error
        with stats_tmp.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
            fsync_file(handle)
        os.replace(stats_tmp, stats_path)
        if error is not None:
            raise ValueError(error)
        os.replace(output_tmp, output)
        print(json.dumps(stats, sort_keys=True), flush=True)
        return stats
    finally:
        if engine is not None:
            engine.close()
        output_tmp.unlink(missing_ok=True)
        stats_tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Canonical inventory for distributed conversion; optional for one standalone archive.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--net", default=None)
    parser.add_argument("--net-option", default="nnue_file")
    parser.add_argument(
        "--score-source",
        choices=("uci", "lc0-root"),
        default="uci",
        help="Use a UCI evaluation (default) or the LC0 V6 root value head.",
    )
    parser.add_argument(
        "--eval-scale",
        type=float,
        default=400.0,
        help="Centipawn scale for --score-source lc0-root (must match architecture.json).",
    )
    parser.add_argument(
        "--value-epsilon",
        type=float,
        default=1e-6,
        help="Clamp LC0 root probabilities to [epsilon, 1-epsilon] before logit.",
    )
    parser.add_argument("--static", action="store_true")
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
