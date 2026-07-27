#!/usr/bin/env python3
"""Replace the score field in a pgn_to_jsonl.py rows JSONL with an exact
Syzygy WDL value for positions with <= --tb-pieces pieces, falling back to
a fresh Stockfish static eval (same mechanism as relabel_with_stockfish.py)
for every other position. Keeps every other field unchanged.

Reuses relabel_with_stockfish.py's Subject/UciEvalEngine for the fallback
path, and python-chess's syzygy module for tablebase probing, rather than
inventing new evaluation mechanisms.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validate"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from structural_net_audit import Subject, UciEvalEngine  # noqa: E402

import chess  # noqa: E402
import chess.syzygy  # noqa: E402

TB_SCORE_CP = 2045  # matches the runtime +/-2045 cp clamp used throughout the pipeline


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
    ap.add_argument("--net", default=str(Path.home() / "assets/nets/nn-0ee0657fb25e.nnue"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--max-abs-cp", type=int, default=10000)
    ap.add_argument("--tb-pieces", type=int, default=6, help="Probe Syzygy for positions with this many pieces or fewer")
    ap.add_argument(
        "--tb-dir", action="append", default=[],
        help="Syzygy directory (repeatable); defaults to ~/assets/tablebases/{6-wdl,3-4-5-wdl}",
    )
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    tb_dirs = args.tb_dir or [
        str(Path.home() / "assets/tablebases/6-wdl"),
        str(Path.home() / "assets/tablebases/3-4-5-wdl"),
    ]
    tablebase = chess.syzygy.open_tablebase(tb_dirs[0])
    for extra in tb_dirs[1:]:
        tablebase.add_directory(extra)

    subject = Subject(name="stockfish", engine=args.engine, net=args.net, command="eval")
    engine = UciEvalEngine(
        subject,
        threads=args.threads,
        hash_mb=args.hash,
        timeout_s=args.timeout_s,
        uci_options=[],
    )

    read = 0
    written = 0
    skipped_cp = 0
    tb_hits = 0
    tb_misses = 0
    sf_used = 0
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                read += 1
                fen = row["fen"]
                board = chess.Board(fen)
                piece_count = len(board.piece_map())

                score = None
                source = "stockfish-static"
                if piece_count <= args.tb_pieces:
                    try:
                        wdl = tablebase.probe_wdl(board)
                        score = wdl_to_cp(wdl)
                        source = "syzygy-wdl"
                        tb_hits += 1
                    except (chess.syzygy.MissingTableError, KeyError):
                        tb_misses += 1

                if score is None:
                    score = engine.eval(fen)
                    sf_used += 1

                if abs(score) > args.max_abs_cp:
                    skipped_cp += 1
                else:
                    row["enyo_score"] = row.get("score")
                    row["score"] = score
                    row["score_source"] = source
                    dst.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1

                if read % args.progress_every == 0:
                    elapsed = max(time.monotonic() - start, 0.001)
                    print(
                        f"read={read} selected={read - skipped_cp} written={written} "
                        f"rate={read / elapsed:.1f}/s "
                        f"tb_hits={tb_hits} tb_misses={tb_misses} sf_used={sf_used}",
                        flush=True,
                    )
    finally:
        engine.close()
        tablebase.close()

    elapsed = max(time.monotonic() - start, 0.001)
    print(
        f"read={read} selected={read - skipped_cp} written={written} "
        f"rate={read / elapsed:.1f}/s "
        f"tb_hits={tb_hits} tb_misses={tb_misses} sf_used={sf_used}",
        flush=True,
    )
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "written": written,
        "skipped_cp": skipped_cp,
        "tb_hits": tb_hits,
        "tb_misses": tb_misses,
        "sf_used": sf_used,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
