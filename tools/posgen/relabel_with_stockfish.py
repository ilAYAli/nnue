#!/usr/bin/env python3
"""Replace the score field in a pgn_to_jsonl.py rows JSONL with a fresh
Stockfish static eval, keeping every other field (fen, wdl, result, side,
ply, ...) unchanged.

Reuses the same persistent-engine UCI eval mechanism already validated by
tools/validate/structural_net_audit.py all session (Enyo's engine binary
loading Stockfish's own .nnue file via the standard UCI `eval` command,
which is a static evaluation, not a search) rather than spawning a real
Stockfish binary or a new evaluation mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validate"))

from structural_net_audit import Subject, UciEvalEngine  # noqa: E402


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
    ap.add_argument("--progress-every", type=int, default=500, help="Print a read=/selected=/written=/rate= progress line every N input rows")
    args = ap.parse_args()

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
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sf_score = engine.eval(row["fen"])
                read += 1
                if abs(sf_score) > args.max_abs_cp:
                    skipped_cp += 1
                else:
                    row["enyo_score"] = row.get("score")
                    row["score"] = sf_score
                    row["score_source"] = "stockfish-static"
                    dst.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1
                if read % args.progress_every == 0:
                    elapsed = max(time.monotonic() - start, 0.001)
                    print(f"read={read} selected={read - skipped_cp} written={written} rate={read / elapsed:.1f}/s", flush=True)
    finally:
        engine.close()

    elapsed = max(time.monotonic() - start, 0.001)
    print(f"read={read} selected={read - skipped_cp} written={written} rate={read / elapsed:.1f}/s", flush=True)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "written": written,
        "skipped_cp": skipped_cp,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
