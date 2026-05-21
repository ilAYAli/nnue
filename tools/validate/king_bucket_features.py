#!/usr/bin/env python3
"""Sanity checks for Enyo's king-bucket NNUE feature mapping."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


OLD_KING_BUCKETS: tuple[int, ...] = (
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
)

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "8/8/8/8/8/8/8/2K2k2 w - - 0 1",
    "8/8/8/8/8/2k5/8/4K3 b - - 0 1",
    "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
]


def enyo_sq(coord: str) -> int:
    file_idx = ord(coord[0]) - ord("a")
    rank_idx = int(coord[1]) - 1
    return rank_idx * 8 + (7 - file_idx)


def king_bucket(square: int, view: int) -> int:
    kingsq = nn2.to_berserk_sq(square)
    ok = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ kingsq
    return nn2.KING_BUCKETS[ok]


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: {actual!r} != {expected!r}")


def check_bucket_shape() -> None:
    assert_equal(nn2.N_KING_BUCKETS, 32, "N_KING_BUCKETS")
    assert_equal(len(nn2.KING_BUCKETS), 64, "KING_BUCKETS length")
    assert_equal(set(nn2.KING_BUCKETS), set(range(32)), "KING_BUCKETS values")
    assert_equal(
        len(nn2.LEGACY_BUCKET_FOR_BUCKET),
        nn2.N_KING_BUCKETS,
        "legacy map length")

    for rank in range(8):
        row = nn2.KING_BUCKETS[rank * 8:(rank + 1) * 8]
        if tuple(row) != tuple(reversed(row)):
            raise SystemExit(f"rank {rank}: king buckets are not mirrored")

    observed: dict[int, int] = {}
    for old, new in zip(OLD_KING_BUCKETS, nn2.KING_BUCKETS):
        if new in observed and observed[new] != old:
            raise SystemExit(
                f"new bucket {new}: maps to both legacy {observed[new]} and {old}")
        observed[new] = old
    expected = tuple(observed[i] for i in range(nn2.N_KING_BUCKETS))
    assert_equal(expected, nn2.LEGACY_BUCKET_FOR_BUCKET, "legacy bucket map")


def check_perspective_coverage() -> None:
    for view in (nn2.WHITE, nn2.BLACK):
        buckets = {king_bucket(square, view) for square in range(64)}
        assert_equal(buckets, set(range(nn2.N_KING_BUCKETS)),
                     f"view {view} bucket coverage")
        for square in range(64):
            assert_equal(
                king_bucket(square, view),
                king_bucket(square ^ 7, view),
                f"view {view} horizontal mirror square {square}")


def check_known_moves() -> None:
    assert_equal(king_bucket(enyo_sq("e1"), nn2.WHITE),
                 king_bucket(enyo_sq("d1"), nn2.WHITE),
                 "center mirror e1/d1")
    if king_bucket(enyo_sq("e1"), nn2.WHITE) == king_bucket(enyo_sq("c1"), nn2.WHITE):
        raise SystemExit("white queenside castle must cross a bucket boundary")
    if king_bucket(enyo_sq("e1"), nn2.WHITE) == king_bucket(enyo_sq("e2"), nn2.WHITE):
        raise SystemExit("white king e1/e2 must use different rank buckets")
    if king_bucket(enyo_sq("e8"), nn2.BLACK) == king_bucket(enyo_sq("e7"), nn2.BLACK):
        raise SystemExit("black king e8/e7 must use different rank buckets")


def python_features(fen: str) -> tuple[list[int], list[int]]:
    pieces, _stm = nn2.parse_fen(fen)
    pieces.sort(key=lambda item: item[2])
    white = nn2.features_from_pieces(pieces, nn2.WHITE)
    black = nn2.features_from_pieces(pieces, nn2.BLACK)
    for feature in white + black:
        if not 0 <= feature < nn2.N_FEATURES:
            raise SystemExit(f"{fen}: feature out of range: {feature}")
    return white, black


def engine_features(engine: Path, fen: str) -> tuple[list[int], list[int]]:
    proc = subprocess.run(
        [str(engine)],
        input=f"position fen {fen}\neval dump\nquit\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"{engine}: rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "white_features" in line:
            row = json.loads(line)
            return list(row["white_features"]), list(row["black_features"])
    raise SystemExit(f"{engine}: no eval dump JSON in stdout:\n{proc.stdout}")


def check_feature_parity(engine: Path | None) -> None:
    for fen in FENS:
        white, black = python_features(fen)
        if engine is None:
            continue
        cxx_white, cxx_black = engine_features(engine, fen)
        assert_equal(cxx_white, white, f"{fen}: white C++ feature parity")
        assert_equal(cxx_black, black, f"{fen}: black C++ feature parity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path,
                    help="Optional Enyo binary for C++ feature parity checks.")
    args = ap.parse_args()

    check_bucket_shape()
    check_perspective_coverage()
    check_known_moves()
    check_feature_parity(args.engine)
    print("king bucket feature checks ok")


if __name__ == "__main__":
    main()
