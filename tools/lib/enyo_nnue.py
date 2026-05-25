"""Python reference helpers for Enyo's exported NNUE.

The constants and feature-index formula mirror Enyo's Network path in
src/nnue_model.hpp. Legacy 16-bucket files can be expanded to the active
32-bucket feature map for architecture experiments.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

N_KING_BUCKETS = 32
LEGACY_N_KING_BUCKETS = 16
N_PIECE_TYPES = 12
N_SQUARES = 64
N_FEATURES = N_KING_BUCKETS * N_PIECE_TYPES * N_SQUARES
LEGACY_N_FEATURES = LEGACY_N_KING_BUCKETS * N_PIECE_TYPES * N_SQUARES
N_HIDDEN = 1024
SUPPORTED_HIDDEN = (1024, 1280)
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_OUTPUT = 1
N_OUTPUT_BUCKETS = 8
N_THREAT_FEATURES = N_PIECE_TYPES * N_PIECE_TYPES * N_SQUARES

QUANT1_BITS = 5
EVAL_DIVISOR = 32.0

NETWORK_SIZE = (
    N_FEATURES * N_HIDDEN * np.dtype(np.int16).itemsize
    + N_HIDDEN * np.dtype(np.int16).itemsize
    + N_L1 * N_L2 * np.dtype(np.int8).itemsize
    + N_L2 * np.dtype(np.int32).itemsize
    + N_L2 * N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
    + N_OUTPUT * np.dtype(np.float32).itemsize
)

LEGACY_NETWORK_SIZE = (
    LEGACY_N_FEATURES * N_HIDDEN * np.dtype(np.int16).itemsize
    + N_HIDDEN * np.dtype(np.int16).itemsize
    + N_L1 * N_L2 * np.dtype(np.int8).itemsize
    + N_L2 * np.dtype(np.int32).itemsize
    + N_L2 * N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
    + N_OUTPUT * np.dtype(np.float32).itemsize
)

BUCKETED_HEAD_NETWORK_SIZE = (
    N_FEATURES * N_HIDDEN * np.dtype(np.int16).itemsize
    + N_HIDDEN * np.dtype(np.int16).itemsize
    + N_L1 * N_L2 * np.dtype(np.int8).itemsize
    + N_L2 * np.dtype(np.int32).itemsize
    + N_OUTPUT_BUCKETS * N_L2 * N_L3 * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * N_L3 * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * np.dtype(np.float32).itemsize
)

LEGACY_BUCKETED_HEAD_NETWORK_SIZE = (
    LEGACY_N_FEATURES * N_HIDDEN * np.dtype(np.int16).itemsize
    + N_HIDDEN * np.dtype(np.int16).itemsize
    + N_L1 * N_L2 * np.dtype(np.int8).itemsize
    + N_L2 * np.dtype(np.int32).itemsize
    + N_OUTPUT_BUCKETS * N_L2 * N_L3 * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * N_L3 * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
    + N_OUTPUT_BUCKETS * np.dtype(np.float32).itemsize
)

THREAT_WEIGHTS_SIZE = N_THREAT_FEATURES * N_HIDDEN * np.dtype(np.int8).itemsize


def network_size(
    feature_count: int = N_FEATURES,
    hidden: int = N_HIDDEN,
    *,
    bucketed_head: bool = False,
    threat: bool = False,
) -> int:
    l1_width = 2 * hidden
    size = (
        feature_count * hidden * np.dtype(np.int16).itemsize
        + hidden * np.dtype(np.int16).itemsize
        + l1_width * N_L2 * np.dtype(np.int8).itemsize
        + N_L2 * np.dtype(np.int32).itemsize
    )
    if bucketed_head:
        size += (
            N_OUTPUT_BUCKETS * N_L2 * N_L3 * np.dtype(np.float32).itemsize
            + N_OUTPUT_BUCKETS * N_L3 * np.dtype(np.float32).itemsize
            + N_OUTPUT_BUCKETS * N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
            + N_OUTPUT_BUCKETS * np.dtype(np.float32).itemsize
        )
    else:
        size += (
            N_L2 * N_L3 * np.dtype(np.float32).itemsize
            + N_L3 * np.dtype(np.float32).itemsize
            + N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
            + N_OUTPUT * np.dtype(np.float32).itemsize
        )
    if threat:
        size += N_THREAT_FEATURES * hidden * np.dtype(np.int8).itemsize
    return size


def _load_specs() -> dict[int, tuple[int, bool, bool, bool]]:
    specs: dict[int, tuple[int, bool, bool, bool]] = {}
    for hidden in SUPPORTED_HIDDEN:
        for feature_count, legacy_layout in (
            (LEGACY_N_FEATURES, True),
            (N_FEATURES, False),
        ):
            for bucketed_head in (False, True):
                for threat_layout in (False, True):
                    size = network_size(
                        feature_count,
                        hidden,
                        bucketed_head=bucketed_head,
                        threat=threat_layout,
                    )
                    specs[size] = (
                        hidden,
                        legacy_layout,
                        bucketed_head,
                        threat_layout,
                    )
    return specs

KING_BUCKETS: tuple[int, ...] = (
    31, 30, 29, 28, 28, 29, 30, 31,
    27, 26, 25, 24, 24, 25, 26, 27,
    23, 22, 21, 20, 20, 21, 22, 23,
    19, 18, 17, 16, 16, 17, 18, 19,
    15, 14, 13, 12, 12, 13, 14, 15,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
)

LEGACY_BUCKET_FOR_BUCKET: tuple[int, ...] = (
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11,  8,  9, 10, 11,
    12, 12, 13, 13, 12, 12, 13, 13,
    14, 14, 15, 15, 14, 14, 15, 15,
)

_FEN_PIECE = {
    "p": PAWN,
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
    "k": KING,
}


@dataclass
class Net:
    input_weights: np.ndarray   # (N_FEATURES, N_HIDDEN) int16
    input_biases: np.ndarray    # (N_HIDDEN,) int16
    l1_weights: np.ndarray      # (N_L2, N_L1) int8
    l1_biases: np.ndarray       # (N_L2,) int32
    l2_weights: np.ndarray      # (N_L3, N_L2) or (buckets, N_L3, N_L2)
    l2_biases: np.ndarray       # (N_L3,) or (buckets, N_L3)
    output_weights: np.ndarray  # (N_L3,) or (buckets, N_L3)
    output_bias: float | np.ndarray
    threat_weights: np.ndarray | None = None  # optional (N_THREAT_FEATURES, N_HIDDEN) int8


def to_berserk_sq(enyo_sq: int) -> int:
    return enyo_sq ^ 63


def feature_index(piece_type: int, piece_color: int, enyo_sq: int,
                  enyo_kingsq: int, view: int) -> int:
    if piece_type == 0:
        return 0

    piece = ((piece_type - 1) << 1) | piece_color
    sq = to_berserk_sq(enyo_sq)
    kingsq = to_berserk_sq(enyo_kingsq)

    op = 6 * ((piece ^ view) & 0x1) + (piece >> 1)
    ok = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ kingsq
    osq = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ sq

    return KING_BUCKETS[ok] * 12 * 64 + op * 64 + osq


def parse_fen(fen: str) -> tuple[list[tuple[int, int, int]], int]:
    parts = fen.split()
    board_part, stm_part = parts[0], parts[1]
    pieces: list[tuple[int, int, int]] = []
    rank = 7
    file_idx = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file_idx = 0
            continue
        if ch.isdigit():
            file_idx += int(ch)
            continue

        # Enyo square convention: h1=0, g1=1, ..., a8=63.
        sq = rank * 8 + (7 - file_idx)
        color = WHITE if ch.isupper() else BLACK
        pieces.append((_FEN_PIECE[ch.lower()], color, sq))
        file_idx += 1

    return pieces, (WHITE if stm_part == "w" else BLACK)


def features_from_pieces(pieces: Sequence[tuple[int, int, int]],
                         view: int) -> list[int]:
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == view)
    return [feature_index(pt, color, sq, king_sq, view)
            for pt, color, sq in pieces]


def phase_scale_from_pieces(pieces: Sequence[tuple[int, int, int]]) -> float:
    minors = sum(1 for pt, _color, _sq in pieces
                 if pt in (KNIGHT, BISHOP))
    rooks = sum(1 for pt, _color, _sq in pieces if pt == ROOK)
    queens = sum(1 for pt, _color, _sq in pieces if pt == QUEEN)
    phase = 3 * minors + 5 * rooks + 10 * queens
    return (128.0 + float(phase)) / 128.0


def _rank(sq: int) -> int:
    return sq // 8


def _file(sq: int) -> int:
    return 7 - (sq % 8)


def _sq(rank: int, file_idx: int) -> int:
    return rank * 8 + (7 - file_idx)


def _on_board(rank: int, file_idx: int) -> bool:
    return 0 <= rank < 8 and 0 <= file_idx < 8


def _step_attacks(sq: int, deltas: Sequence[tuple[int, int]]) -> int:
    r = _rank(sq)
    f = _file(sq)
    attacks = 0
    for dr, df in deltas:
        rr = r + dr
        ff = f + df
        if _on_board(rr, ff):
            attacks |= 1 << _sq(rr, ff)
    return attacks


def _slide_attacks(
    sq: int,
    occ: int,
    deltas: Sequence[tuple[int, int]],
) -> int:
    r = _rank(sq)
    f = _file(sq)
    attacks = 0
    for dr, df in deltas:
        rr = r + dr
        ff = f + df
        while _on_board(rr, ff):
            dst = _sq(rr, ff)
            attacks |= 1 << dst
            if occ & (1 << dst):
                break
            rr += dr
            ff += df
    return attacks


def attacks_from_piece(piece_type: int, color: int, sq: int, occ: int) -> int:
    if piece_type == PAWN:
        direction = 1 if color == WHITE else -1
        return _step_attacks(sq, ((direction, -1), (direction, 1)))
    if piece_type == KNIGHT:
        return _step_attacks(
            sq,
            ((2, 1), (2, -1), (1, 2), (1, -2),
             (-1, 2), (-1, -2), (-2, 1), (-2, -1)),
        )
    if piece_type == BISHOP:
        return _slide_attacks(sq, occ, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
    if piece_type == ROOK:
        return _slide_attacks(sq, occ, ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if piece_type == QUEEN:
        return attacks_from_piece(BISHOP, color, sq, occ) | attacks_from_piece(
            ROOK, color, sq, occ)
    if piece_type == KING:
        return _step_attacks(
            sq,
            ((1, 1), (1, 0), (1, -1), (0, 1),
             (0, -1), (-1, 1), (-1, 0), (-1, -1)),
        )
    return 0


def king_pressure_bucket_from_pieces(
    pieces: Sequence[tuple[int, int, int]],
    stm: int,
) -> int:
    """Mirror ENYO_ENABLE_KING_PRESSURE_BUCKETS MaterialBucket()."""
    them = BLACK if stm == WHITE else WHITE
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == stm)
    occ = 0
    for _pt, _color, sq in pieces:
        occ |= 1 << sq
    king_zone = attacks_from_piece(KING, stm, king_sq, occ) | (1 << king_sq)

    pressure = 0
    for pt, color, sq in pieces:
        if color != them or pt == KING:
            continue
        if attacks_from_piece(pt, color, sq, occ) & king_zone:
            if pt == PAWN:
                pressure += 1
            elif pt in (KNIGHT, BISHOP):
                pressure += 2
            elif pt == ROOK:
                pressure += 3
            else:
                pressure += 4
    return max(0, min(pressure, N_OUTPUT_BUCKETS - 1))


def check_state_bucket_from_pieces(
    pieces: Sequence[tuple[int, int, int]],
    stm: int,
) -> int:
    them = BLACK if stm == WHITE else WHITE
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == stm)
    occ = 0
    for _pt, _color, sq in pieces:
        occ |= 1 << sq
    king_bit = 1 << king_sq
    king_zone = attacks_from_piece(KING, stm, king_sq, occ) | king_bit

    checkers = 0
    slider_checkers = 0
    pressure = 0
    for pt, color, sq in pieces:
        if color != them or pt == KING:
            continue
        attacks = attacks_from_piece(pt, color, sq, occ)
        if attacks & king_bit:
            checkers += 1
            if pt in (BISHOP, ROOK, QUEEN):
                slider_checkers += 1
        if attacks & king_zone:
            pressure += 1

    if checkers >= 2:
        return min(3, N_OUTPUT_BUCKETS - 1)
    if checkers == 1:
        return min(2 if slider_checkers else 1, N_OUTPUT_BUCKETS - 1)
    if pressure <= 0:
        return 0
    return min(3 + pressure, N_OUTPUT_BUCKETS - 1)


def load_net(path: str | Path) -> Net:
    data = Path(path).read_bytes()
    specs = _load_specs()
    if len(data) not in specs:
        valid_sizes = sorted(specs)
        raise ValueError(
            f"{path}: size {len(data)} != expected "
            f"{', '.join(str(v) for v in valid_sizes)}")
    hidden, legacy_layout, bucketed_head, threat_layout = specs[len(data)]
    l1_width = 2 * hidden
    head_buckets = N_OUTPUT_BUCKETS if bucketed_head else 1

    off = 0

    def take(dtype, count: int):
        nonlocal off
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    if legacy_layout:
        legacy_iw = take(np.int16, LEGACY_N_FEATURES * hidden).reshape(
            LEGACY_N_FEATURES, hidden)
        iw = np.empty((N_FEATURES, hidden), dtype=np.int16)
        bucket_stride = N_PIECE_TYPES * N_SQUARES
        for bucket, legacy_bucket in enumerate(LEGACY_BUCKET_FOR_BUCKET):
            src = slice(legacy_bucket * bucket_stride,
                        (legacy_bucket + 1) * bucket_stride)
            dst = slice(bucket * bucket_stride,
                        (bucket + 1) * bucket_stride)
            iw[dst] = legacy_iw[src]
    else:
        iw = take(np.int16, N_FEATURES * hidden).reshape(N_FEATURES, hidden)
    ib = take(np.int16, hidden)
    l1w = take(np.int8, l1_width * N_L2).reshape(N_L2, l1_width)
    l1b = take(np.int32, N_L2)
    if bucketed_head:
        l2w = take(np.float32, head_buckets * N_L2 * N_L3).reshape(
            head_buckets, N_L3, N_L2)
        l2b = take(np.float32, head_buckets * N_L3).reshape(
            head_buckets, N_L3)
        ow = take(np.float32, head_buckets * N_L3).reshape(
            head_buckets, N_L3)
        ob = take(np.float32, head_buckets)
    else:
        l2w = take(np.float32, N_L2 * N_L3).reshape(N_L3, N_L2)
        l2b = take(np.float32, N_L3)
        ow = take(np.float32, N_L3)
        ob = struct.unpack_from("<f", data, off)[0]
        off += 4
    threat_weights = None
    if threat_layout:
        threat_weights = take(np.int8, N_THREAT_FEATURES * hidden).reshape(
            N_THREAT_FEATURES, hidden)
    assert off == len(data)
    return Net(iw, ib, l1w, l1b, l2w, l2b, ow,
               ob if bucketed_head else float(ob), threat_weights)


def write_net(net: Net, path: str | Path) -> None:
    out = Path(path)
    with out.open("wb") as f:
        f.write(np.asarray(net.input_weights, dtype=np.int16).tobytes(order="C"))
        f.write(np.asarray(net.input_biases, dtype=np.int16).tobytes(order="C"))
        f.write(np.asarray(net.l1_weights, dtype=np.int8).tobytes(order="C"))
        f.write(np.asarray(net.l1_biases, dtype=np.int32).tobytes(order="C"))
        f.write(np.asarray(net.l2_weights, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(net.l2_biases, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(net.output_weights, dtype=np.float32).tobytes(order="C"))
        output_bias = np.asarray(net.output_bias, dtype=np.float32)
        if output_bias.ndim == 0:
            f.write(struct.pack("<f", float(output_bias)))
        else:
            f.write(output_bias.tobytes(order="C"))
        if net.threat_weights is not None:
            f.write(np.asarray(net.threat_weights, dtype=np.int8).tobytes(order="C"))
    size = out.stat().st_size
    input_weights = np.asarray(net.input_weights)
    hidden = int(input_weights.shape[1])
    expected = network_size(
        int(input_weights.shape[0]),
        hidden,
        bucketed_head=np.asarray(net.l2_weights).ndim == 3,
        threat=net.threat_weights is not None,
    )
    if size != expected:
        raise RuntimeError(f"wrote {size} bytes, expected {expected}")
