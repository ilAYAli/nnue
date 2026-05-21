#!/usr/bin/env python3
"""Report cheap distribution/novelty stats for NNUE JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PIECE_VALUE = {
    "P": 100,
    "N": 320,
    "B": 330,
    "R": 500,
    "Q": 900,
    "p": 100,
    "n": 320,
    "b": 330,
    "r": 500,
    "q": 900,
}

SCORE_BUCKETS = (
    ("0-25", 0, 25),
    ("25-75", 25, 75),
    ("75-150", 75, 150),
    ("150-300", 150, 300),
    ("300-600", 300, 600),
    ("600-1200", 600, 1200),
    ("1200+", 1200, math.inf),
)


def score_bucket(score: float) -> str:
    sign = "pos" if score > 0 else "neg" if score < 0 else "zero"
    value = abs(score)
    for name, lo, hi in SCORE_BUCKETS:
        if value >= lo and value < hi:
            return f"{sign}:{name}" if sign != "zero" else name
    return f"{sign}:unknown"


def phase_bucket(board_part: str) -> str:
    material = sum(PIECE_VALUE.get(ch, 0) for ch in board_part)
    if material <= 2600:
        return "endgame"
    if material <= 5200:
        return "late"
    if material <= 7600:
        return "middlegame"
    return "opening"


def material_bucket(board_part: str) -> str:
    material = sum(PIECE_VALUE.get(ch, 0) for ch in board_part)
    return f"{(material // 1000) * 1000}-{(material // 1000 + 1) * 1000}"


def fen_key(row: dict[str, Any]) -> str | None:
    fen = row.get("fen")
    return fen if isinstance(fen, str) and fen else None


def fen_parts(fen: str) -> tuple[str, str, str, str, str, str] | None:
    parts = fen.split()
    if len(parts) < 6:
        return None
    return tuple(parts[:6])  # type: ignore[return-value]


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def top(counter: Counter[str], total: int, n: int = 8) -> str:
    if not counter:
        return "none"
    return ", ".join(
        f"{key}={value}({pct(value, total)})"
        for key, value in counter.most_common(n)
    )


def iter_rows(path: Path, limit: int | None = None):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit is not None and idx > limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--compare", action="append", default=[], type=Path,
                    help="Optional older JSONL pool for exact-FEN overlap.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit input rows for quick smoke reports.")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    total = 0
    fens: set[str] = set()
    dup_fen = 0
    side = Counter()
    phase = Counter()
    material = Counter()
    score = Counter()
    source = Counter()
    teacher_depth = Counter()
    fullmove = Counter()
    bad_fen = 0
    missing_score = 0

    for row in iter_rows(args.input, limit):
        total += 1
        fen = fen_key(row)
        if fen is None:
            bad_fen += 1
            continue
        if fen in fens:
            dup_fen += 1
        else:
            fens.add(fen)
        parts = fen_parts(fen)
        if parts is None:
            bad_fen += 1
            continue
        board, stm, _castling, _ep, _halfmove, fullmove_no = parts
        side[stm] += 1
        phase[phase_bucket(board)] += 1
        material[material_bucket(board)] += 1
        fullmove[fullmove_no] += 1
        if "score" in row:
            try:
                score[score_bucket(float(row["score"]))] += 1
            except (TypeError, ValueError):
                missing_score += 1
        else:
            missing_score += 1
        if "source" in row:
            source[str(row["source"])] += 1
        if "teacher_depth" in row:
            teacher_depth[str(row["teacher_depth"])] += 1

    print(f"input={args.input}")
    print(f"rows={total}")
    print(f"unique_fen={len(fens)}")
    print(f"duplicate_fen={dup_fen} ({pct(dup_fen, total)})")
    print(f"bad_fen={bad_fen}")
    print(f"missing_score={missing_score}")
    print(f"side={top(side, total)}")
    print(f"phase={top(phase, total)}")
    print(f"material={top(material, total)}")
    print(f"score_buckets={top(score, total, 20)}")
    print(f"teacher_depth={top(teacher_depth, total)}")
    print(f"sources={top(source, total, 5)}")
    print(f"fullmove_top={top(fullmove, total, 12)}")

    for compare in args.compare:
        seen = 0
        overlap = 0
        for row in iter_rows(compare):
            seen += 1
            fen = fen_key(row)
            if fen is not None and fen in fens:
                overlap += 1
        print(
            f"compare={compare} rows={seen} overlap={overlap} "
            f"({pct(overlap, max(1, len(fens)))})")


if __name__ == "__main__":
    main()
