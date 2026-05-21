#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


CASES = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        64,
    ),
    (
        "kings-only",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        0,
    ),
    (
        "one-rook",
        "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
        5,
    ),
    (
        "two-queens",
        "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
        20,
    ),
]


def main() -> None:
    for name, fen, expected_phase in CASES:
        pieces, _stm = nn2.parse_fen(fen)
        scale = nn2.phase_scale_from_pieces(pieces)
        phase = round((scale - 1.0) * 128.0)
        if phase != expected_phase:
            raise SystemExit(
                f"{name}: phase {phase} != expected {expected_phase}")
        head = nn2.phase_head_from_scale(scale)
        expected_head = expected_phase / 128.0
        if abs(head - expected_head) > 1e-7:
            raise SystemExit(
                f"{name}: head {head:.8f} != expected {expected_head:.8f}")
        print(f"{name}: phase={phase} head={head:.8f} scale={scale:.8f}")

    print("material phase ok")


if __name__ == "__main__":
    main()
