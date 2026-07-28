#!/usr/bin/env python3
"""Correct existing self-play labels with exact Syzygy WDL.

Shares its encoding with tools/bullet/bullet_tb_relabel.py, which documents why
Syzygy corrects the evaluation rather than overwriting it with a saturated
+/-2045 ternary signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.syzygy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bullet"))

from bullet_tb_relabel import wdl_to_cp, wdl_to_result  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tb-pieces", type=int, default=6)
    ap.add_argument("--tb-dir", action="append", default=[])
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()
    tb_dirs = args.tb_dir or [str(Path.home() / "assets/tablebases/6-wdl"), str(Path.home() / "assets/tablebases/3-4-5-wdl")]
    tablebase = chess.syzygy.open_tablebase(tb_dirs[0])
    for extra in tb_dirs[1:]:
        tablebase.add_directory(extra)
    read = written = tb_hits = tb_misses = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                read += 1
                board = chess.Board(row["fen"])
                if len(board.piece_map()) <= args.tb_pieces:
                    try:
                        wdl = tablebase.probe_wdl(board)
                    except (chess.syzygy.MissingTableError, KeyError):
                        tb_misses += 1
                    else:
                        row["score"] = wdl_to_cp(wdl, int(round(float(row["score"]))))
                        row["score_source"] = "syzygy-wdl"
                        # Drop the noisy game outcome so bullet_text falls
                        # through to this side-to-move relative ground truth.
                        row.pop("result", None)
                        row["wdl"] = wdl_to_result(wdl) / 2.0
                        tb_hits += 1
                dst.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
                if read % args.progress_every == 0:
                    print(f"read={read} written={written} tb_hits={tb_hits} tb_misses={tb_misses}", flush=True)
    finally:
        tablebase.close()
    print(json.dumps({"read": read, "written": written, "tb_hits": tb_hits, "tb_misses": tb_misses}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
