#!/usr/bin/env python3
"""Relabel pgn_to_jsonl rows using UCI search or static evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import chess.syzygy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))

from label_with_uci import EngineTimeout, UciEngine  # noqa: E402

TB_SCORE_CP = 2045


def wdl_to_cp(wdl: int) -> int:
    if wdl > 0:
        return TB_SCORE_CP
    if wdl < 0:
        return -TB_SCORE_CP
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="rows.jsonl from pgn_to_jsonl.py")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--engine", default=str(Path.home() / "assets/engines/candidate"))
    ap.add_argument("--net", default=str(Path.home() / "assets/nets/nn-1a298aa575a0.nnue"))
    ap.add_argument("--net-option", default="nnue_file")
    ap.add_argument("--mode", choices=("search", "static"), required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--max-abs-cp", type=int, default=10000)
    ap.add_argument(
        "--tb-pieces",
        type=int,
        default=0,
        help="Override engine labels with Syzygy WDL through this piece count; 0 disables it",
    )
    ap.add_argument("--tb-dir", action="append", default=[], help="Syzygy directory (repeatable)")
    ap.add_argument("--progress-every", type=int, default=500, help="Print a read=/selected=/written=/rate= progress line every N input rows")
    args = ap.parse_args()

    engine = UciEngine(
        args.engine,
        threads=args.threads,
        hash_mb=args.hash,
        net=args.net,
        net_option=args.net_option,
    )
    tablebase = None
    if args.tb_pieces:
        tb_dirs = args.tb_dir or [
            str(Path.home() / "assets/tablebases/6-wdl"),
            str(Path.home() / "assets/tablebases/3-4-5-wdl"),
        ]
        tablebase = chess.syzygy.open_tablebase(tb_dirs[0])
        for directory in tb_dirs[1:]:
            tablebase.add_directory(directory)

    read = 0
    written = 0
    skipped_cp = 0
    tb_hits = 0
    tb_misses = 0
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    board = chess.Board(row["fen"])
                    score = None
                    mate = None
                    score_source = f"uci-{args.mode}"
                    if tablebase is not None and len(board.piece_map()) <= args.tb_pieces:
                        try:
                            score = wdl_to_cp(tablebase.probe_wdl(board))
                            score_source = "syzygy-wdl"
                            tb_hits += 1
                        except (chess.syzygy.MissingTableError, KeyError):
                            tb_misses += 1
                    if score is None and args.mode == "static":
                        score = engine.static_eval(row["fen"], timeout_s=args.timeout_s)
                    elif score is None:
                        score, mate = engine.label(
                            row["fen"],
                            depth=args.depth,
                            timeout_s=args.timeout_s,
                        )
                except EngineTimeout:
                    engine.restart()
                    read += 1
                    continue
                read += 1
                if score is None or mate is not None or abs(score) > args.max_abs_cp:
                    skipped_cp += 1
                else:
                    row["enyo_score"] = row.get("score")
                    row["score"] = score
                    row["score_source"] = score_source
                    dst.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1
                if read % args.progress_every == 0:
                    elapsed = max(time.monotonic() - start, 0.001)
                    print(f"read={read} selected={read - skipped_cp} written={written} rate={read / elapsed:.1f}/s", flush=True)
    finally:
        engine.close()
        if tablebase is not None:
            tablebase.close()

    elapsed = max(time.monotonic() - start, 0.001)
    print(f"read={read} selected={read - skipped_cp} written={written} rate={read / elapsed:.1f}/s", flush=True)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "written": written,
        "skipped_cp": skipped_cp,
        "tb_hits": tb_hits,
        "tb_misses": tb_misses,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
