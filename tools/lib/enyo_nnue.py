"""Python reference helpers for Enyo's Berserk-format NNUE.

The constants and feature-index formula mirror src/nnue.hpp.  The file
format is Berserk v13's .nn layout as loaded by NNUE::LoadNetwork().
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
from typing import Sequence

import numpy as np


WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

DEFAULT_N_KING_BUCKETS = 16
SUPPORTED_N_KING_BUCKETS = (1, 2, 4, 8, 10, 16, 32)
SUPPORTED_TRAINED_HIDDEN = (512, 768, 1024)
DEFAULT_N_FEATURE_CHANNELS = 12
HALFKA_V2_FEATURE_CHANNELS = 11
SUPPORTED_N_FEATURE_LAYOUTS = (
    (1, 12), (2, 12), (4, 12), (8, 12), (10, 12), (16, 12), (32, 12),
    (10, 11), (16, 11), (32, 11),
)
DEFAULT_N_OUTPUT_BUCKETS = 1
SUPPORTED_N_OUTPUT_BUCKETS = (1, 2, 4, 8)
DEFAULT_N_OUTPUT_HEAD_FEATURES = 0
N_HEAD_FEATURES = 2
SUPPORTED_N_OUTPUT_HEAD_FEATURES = (0, N_HEAD_FEATURES)
N_THREAT_FEATURES = 60_720
N_KING_BUCKETS = DEFAULT_N_KING_BUCKETS
N_PIECE_TYPES = 12
N_SQUARES = 64
N_HIDDEN = 1024
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_OUTPUT = 1
NETWORK_V2_HEADER_MAGIC = b"ENYONN2\0"
NETWORK_V3_HEADER_MAGIC = b"ENYONN3\0"
NETWORK_V4_HEADER_MAGIC = b"ENYONN4\0"
NETWORK_HEADER_MAGIC = NETWORK_V2_HEADER_MAGIC
NETWORK_FORMAT_VERSION = 2
NETWORK_FLAG_FULL_THREATS = 1
NETWORK_FLAG_FULL_HEADS = 2
NETWORK_FLAG_MIXED_ACTIVATION = 4
NETWORK_HEADER = struct.Struct("<8s14I")
NETWORK_HEADER_SIZE = NETWORK_HEADER.size
N_FEATURES = N_KING_BUCKETS * N_PIECE_TYPES * N_SQUARES
LEGACY_N_FEATURES = N_FEATURES

QUANT1_BITS = 5
EVAL_DIVISOR = 32.0

def feature_count(
    input_buckets: int = DEFAULT_N_KING_BUCKETS,
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
) -> int:
    if (input_buckets, feature_channels) not in SUPPORTED_N_FEATURE_LAYOUTS:
        raise ValueError(
            "unsupported feature layout "
            f"{input_buckets} input buckets / {feature_channels} channels")
    return input_buckets * feature_channels * N_SQUARES


def input_feature_count(
    input_buckets: int = DEFAULT_N_KING_BUCKETS,
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
    full_threats: bool = False,
) -> int:
    return feature_count(input_buckets, feature_channels) + (
        N_THREAT_FEATURES if full_threats else 0)


def network_size(
    input_buckets: int = DEFAULT_N_KING_BUCKETS,
    output_buckets: int = DEFAULT_N_OUTPUT_BUCKETS,
    output_head_features: int = DEFAULT_N_OUTPUT_HEAD_FEATURES,
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
    full_threats: bool = False,
    full_heads: bool = False,
    mixed_activation: bool = False,
) -> int:
    features = input_feature_count(input_buckets, feature_channels, full_threats)
    if output_buckets not in SUPPORTED_N_OUTPUT_BUCKETS:
        raise ValueError(f"unsupported output bucket count {output_buckets}")
    if output_head_features not in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
        raise ValueError(
            f"unsupported output head feature count {output_head_features}")
    output_width = N_L3 + output_head_features
    head_count = output_buckets if full_heads else 1
    return (
        features * N_HIDDEN * np.dtype(np.int16).itemsize
        + N_HIDDEN * np.dtype(np.int16).itemsize
        + head_count * N_L1 * N_L2 * np.dtype(np.int8).itemsize
        + head_count * N_L2 * np.dtype(np.int32).itemsize
        + head_count * N_L2 * N_L3 * np.dtype(np.float32).itemsize
        + head_count * N_L3 * np.dtype(np.float32).itemsize
        + ((N_L2 * N_L3 + N_L3) * np.dtype(np.float32).itemsize
           if mixed_activation else 0)
        + output_buckets * output_width * N_OUTPUT * np.dtype(np.float32).itemsize
        + output_buckets * N_OUTPUT * np.dtype(np.float32).itemsize
    )


NETWORK_SIZE = network_size()
LEGACY_NETWORK_SIZE = NETWORK_SIZE

KING_BUCKETS_16: tuple[int, ...] = (
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
)

KING_BUCKETS_32: tuple[int, ...] = (
    31, 30, 29, 28, 28, 29, 30, 31,
    27, 26, 25, 24, 24, 25, 26, 27,
    23, 22, 21, 20, 20, 21, 22, 23,
    19, 18, 17, 16, 16, 17, 18, 19,
    15, 14, 13, 12, 12, 13, 14, 15,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
)

KING_BUCKETS_10: tuple[int, ...] = (
     9,  9,  8,  8,  8,  8,  9,  9,
     9,  9,  8,  8,  8,  8,  9,  9,
     8,  8,  7,  7,  7,  7,  8,  8,
     8,  8,  7,  7,  7,  7,  8,  8,
     6,  6,  5,  5,  5,  5,  6,  6,
     6,  6,  5,  5,  5,  5,  6,  6,
     4,  3,  3,  2,  2,  3,  3,  4,
     1,  1,  0,  0,  0,  0,  1,  1,
)

KING_BUCKETS = KING_BUCKETS_16


def king_buckets(input_buckets: int = DEFAULT_N_KING_BUCKETS) -> tuple[int, ...]:
    if input_buckets in (1, 2, 4, 8, 16):
        return tuple(bucket * input_buckets // 16 for bucket in KING_BUCKETS_16)
    if input_buckets == 10:
        return KING_BUCKETS_10
    if input_buckets == 32:
        return KING_BUCKETS_32
    raise ValueError(f"unsupported input bucket count {input_buckets}")


def detect_network_layout(size: int) -> tuple[int, int, int, int]:
    for payload_size in (size, size - NETWORK_HEADER_SIZE):
        if payload_size <= 0:
            continue
        for input_buckets, feature_channels in SUPPORTED_N_FEATURE_LAYOUTS:
            for output_buckets in SUPPORTED_N_OUTPUT_BUCKETS:
                for output_head_features in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
                    if payload_size == network_size(
                        input_buckets, output_buckets, output_head_features,
                        feature_channels
                    ):
                        return (
                            input_buckets,
                            feature_channels,
                            output_buckets,
                            output_head_features)
    expected = ", ".join(
        f"{network_size(i, o, h, c)} "
        f"({i} input, {c} channels, {o} output, {h} head)"
        for i, c in SUPPORTED_N_FEATURE_LAYOUTS
        for o in SUPPORTED_N_OUTPUT_BUCKETS
        for h in SUPPORTED_N_OUTPUT_HEAD_FEATURES)
    raise ValueError(f"size {size} does not match supported net sizes: {expected}")


def detect_input_buckets(size: int) -> int:
    return detect_network_layout(size)[0]


def detect_feature_channels(size: int) -> int:
    return detect_network_layout(size)[1]


def detect_output_buckets(size: int) -> int:
    return detect_network_layout(size)[2]


def detect_output_head_features(size: int) -> int:
    return detect_network_layout(size)[3]


def detect_feature_layout_from_count(feature_count_value: int) -> tuple[int, int]:
    for input_buckets, feature_channels in SUPPORTED_N_FEATURE_LAYOUTS:
        if feature_count(input_buckets, feature_channels) == feature_count_value:
            return input_buckets, feature_channels
    expected = ", ".join(
        f"{feature_count(i, c)} ({i} input, {c} channels)"
        for i, c in SUPPORTED_N_FEATURE_LAYOUTS)
    raise ValueError(
        f"feature count {feature_count_value} does not match supported layouts: {expected}")

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
    l2_weights: np.ndarray      # (N_L3, N_L2) float32
    l2_biases: np.ndarray       # (N_L3,) float32
    output_weights: np.ndarray  # (output_buckets, N_L3 + output_head_features) float32
    output_biases: np.ndarray   # (output_buckets,) float32
    input_buckets: int = DEFAULT_N_KING_BUCKETS
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS
    output_buckets: int = DEFAULT_N_OUTPUT_BUCKETS
    output_head_features: int = DEFAULT_N_OUTPUT_HEAD_FEATURES
    trained_hidden: int = N_HIDDEN
    format_version: int = 1
    full_threats: bool = False
    full_heads: bool = False
    mixed_activation: bool = False
    l2_squared_weights: np.ndarray | None = None
    l2_squared_biases: np.ndarray | None = None

    @property
    def output_bias(self) -> float:
        return float(np.asarray(self.output_biases, dtype=np.float32).reshape(-1)[0])

    @property
    def output_width(self) -> int:
        return N_L3 + self.output_head_features


def to_berserk_sq(enyo_sq: int) -> int:
    return enyo_sq ^ 63


def feature_channel(
    piece_type: int,
    piece_color: int,
    view: int,
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
) -> int:
    piece = ((piece_type - 1) << 1) | piece_color
    if feature_channels == HALFKA_V2_FEATURE_CHANNELS:
        zero_based_type = piece_type - 1
        if zero_based_type == 5:
            return 10
        return 5 * ((piece ^ view) & 0x1) + zero_based_type
    if feature_channels == DEFAULT_N_FEATURE_CHANNELS:
        return 6 * ((piece ^ view) & 0x1) + (piece >> 1)
    raise ValueError(f"unsupported feature channel count {feature_channels}")


def feature_index(piece_type: int, piece_color: int, enyo_sq: int,
                  enyo_kingsq: int, view: int,
                  input_buckets: int = DEFAULT_N_KING_BUCKETS,
                  feature_channels: int = DEFAULT_N_FEATURE_CHANNELS) -> int:
    if piece_type == 0:
        return 0

    sq = to_berserk_sq(enyo_sq)
    kingsq = to_berserk_sq(enyo_kingsq)

    op = feature_channel(piece_type, piece_color, view, feature_channels)
    ok = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ kingsq
    osq = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ sq

    return king_buckets(input_buckets)[ok] * feature_channels * 64 + op * 64 + osq


_VALID_THREAT_TARGETS = (0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0)
_THREAT_TARGET_MAP = (
    (0, 1, -1, 2, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)
_THREAT_PIECES = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)
_THREAT_NONE = 0xff


def _on_board(file_idx: int, rank: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank < 8


def _piece_color(piece: int) -> int:
    return piece >> 3


def _piece_type(piece: int) -> int:
    return piece & 7


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _leaper_attacks(piece_type: int, square: int) -> int:
    knight = (
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    )
    king = (
        (1, 1), (1, 0), (1, -1), (0, -1),
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
    )
    attacks = 0
    file_idx = square % 8
    rank = square // 8
    for df, dr in (knight if piece_type == KNIGHT else king):
        to_file = file_idx + df
        to_rank = rank + dr
        if _on_board(to_file, to_rank):
            attacks |= 1 << (to_rank * 8 + to_file)
    return attacks


def _slider_attacks(piece_type: int, square: int, occupied: int) -> int:
    bishop = ((1, 1), (1, -1), (-1, -1), (-1, 1))
    rook = ((1, 0), (0, -1), (-1, 0), (0, 1))
    attacks = 0
    file_idx = square % 8
    rank = square // 8
    rays = ()
    if piece_type in (BISHOP, QUEEN):
        rays += bishop
    if piece_type in (ROOK, QUEEN):
        rays += rook
    for df, dr in rays:
        to_file = file_idx + df
        to_rank = rank + dr
        while _on_board(to_file, to_rank):
            target = to_rank * 8 + to_file
            bit = 1 << target
            attacks |= bit
            if occupied & bit:
                break
            to_file += df
            to_rank += dr
    return attacks


def _pawn_push_or_attacks(color: int, square: int) -> int:
    attacks = 0
    file_idx = square % 8
    rank = square // 8
    rank_delta = 1 if color == WHITE else -1
    for file_delta in (-1, 0, 1):
        to_file = file_idx + file_delta
        to_rank = rank + rank_delta
        if _on_board(to_file, to_rank):
            attacks |= 1 << (to_rank * 8 + to_file)
    return attacks


def _pseudo_threat_attacks(piece: int, square: int) -> int:
    pt = _piece_type(piece)
    if pt == PAWN:
        return _pawn_push_or_attacks(_piece_color(piece), square)
    if pt in (KNIGHT, KING):
        return _leaper_attacks(pt, square)
    return _slider_attacks(pt, square, 0)


@lru_cache(maxsize=1)
def _threat_tables():
    offsets = [[0 for _ in range(N_SQUARES)] for _ in range(16)]
    target_offsets = [
        [[N_THREAT_FEATURES for _ in range(2)] for _ in range(16)]
        for _ in range(16)
    ]
    attack_offsets = [
        [[0 for _ in range(N_SQUARES)] for _ in range(N_SQUARES)]
        for _ in range(16)
    ]
    helper: list[tuple[int, int]] = [(0, 0)] * 16
    global_offset = 0
    for piece in _THREAT_PIECES:
        piece_span = 0
        for square in range(N_SQUARES):
            offsets[piece][square] = piece_span
            if _piece_type(piece) != PAWN or 8 <= square < 56:
                piece_span += _popcount(_pseudo_threat_attacks(piece, square))
        helper[piece] = (piece_span, global_offset)
        global_offset += _VALID_THREAT_TARGETS[piece] * piece_span

        for from_sq in range(N_SQUARES):
            attacks = _pseudo_threat_attacks(piece, from_sq)
            for to_sq in range(N_SQUARES):
                below = 0 if to_sq == 0 else (1 << to_sq) - 1
                attack_offsets[piece][from_sq][to_sq] = _popcount(attacks & below)
    if global_offset != N_THREAT_FEATURES:
        raise RuntimeError("FullThreats table size mismatch")

    for attacker in _THREAT_PIECES:
        for attacked in _THREAT_PIECES:
            attacker_type = _piece_type(attacker)
            attacked_type = _piece_type(attacked)
            mapped = _THREAT_TARGET_MAP[attacker_type - 1][attacked_type - 1]
            excluded = mapped < 0
            enemy = (attacker ^ attacked) == 8
            same_type_excluded = (
                attacker_type == attacked_type and (enemy or attacker_type != PAWN)
            )
            if excluded:
                base = N_THREAT_FEATURES
            else:
                piece_span, global_base = helper[attacker]
                base = (
                    global_base
                    + (_piece_color(attacked) * (_VALID_THREAT_TARGETS[attacker] // 2)
                       + mapped) * piece_span
                )
            target_offsets[attacker][attacked][0] = base
            target_offsets[attacker][attacked][1] = (
                N_THREAT_FEATURES if excluded or same_type_excluded else base
            )
    return offsets, target_offsets, attack_offsets


def _threat_make_index(perspective: int, attacker: int, from_sq: int,
                       to_sq: int, attacked: int, king_square: int) -> int:
    orientation = (0 if king_square % 8 < 4 else 7) ^ (56 * perspective)
    oriented_from = from_sq ^ orientation
    oriented_to = to_sq ^ orientation
    color_swap = 8 * perspective
    oriented_attacker = attacker ^ color_swap
    oriented_attacked = attacked ^ color_swap
    offsets, target_offsets, attack_offsets = _threat_tables()
    target_offset = target_offsets[oriented_attacker][oriented_attacked][
        int(oriented_from < oriented_to)]
    if target_offset >= N_THREAT_FEATURES:
        return N_THREAT_FEATURES
    return (
        target_offset
        + offsets[oriented_attacker][oriented_from]
        + attack_offsets[oriented_attacker][oriented_from][oriented_to]
    )


def _trailing_square(bitboard: int) -> int:
    if bitboard == 0:
        raise ValueError("missing king for FullThreats features")
    return (bitboard & -bitboard).bit_length() - 1


def threat_features_from_pieces(
    pieces: Sequence[tuple[int, int, int]],
    view: int,
) -> list[int]:
    occupied = 0
    pt_bb = [[0 for _ in range(7)] for _ in range(2)]
    piece_at = [_THREAT_NONE for _ in range(N_SQUARES)]
    color_at = [WHITE for _ in range(N_SQUARES)]
    for pt, color, sq in pieces:
        bit = 1 << sq
        occupied |= bit
        pt_bb[color][pt] |= bit
        piece_at[sq] = pt
        color_at[sq] = color

    both = lambda pt: pt_bb[WHITE][pt] | pt_bb[BLACK][pt]
    pawn_targets = both(PAWN) | both(KNIGHT) | both(ROOK)
    minor_slider_targets = pawn_targets | both(BISHOP)
    queen_targets = minor_slider_targets | both(QUEEN)
    king_square = [
        _trailing_square(pt_bb[WHITE][KING]) ^ 7,
        _trailing_square(pt_bb[BLACK][KING]) ^ 7,
    ]
    active: list[int] = []

    def emit(attacker: int, from_sq: int, to_sq: int) -> None:
        if piece_at[to_sq] == _THREAT_NONE:
            return
        attacked = piece_at[to_sq] + (color_at[to_sq] << 3)
        index = _threat_make_index(
            view, attacker, from_sq ^ 7, to_sq ^ 7, attacked, king_square[view])
        if index < N_THREAT_FEATURES:
            active.append(index)

    for color in (WHITE, BLACK):
        rank_delta = 1 if color == WHITE else -1
        pawn = PAWN + (color << 3)
        pawns = pt_bb[color][PAWN]
        while pawns:
            from_sq = _trailing_square(pawns)
            pawns &= pawns - 1
            file_idx = from_sq % 8
            target_rank = from_sq // 8 + rank_delta
            for file_delta in (-1, 1):
                target_file = file_idx + file_delta
                if _on_board(target_file, target_rank):
                    to_sq = target_rank * 8 + target_file
                    if pawn_targets & (1 << to_sq):
                        emit(pawn, from_sq, to_sq)
            if _on_board(file_idx, target_rank):
                to_sq = target_rank * 8 + file_idx
                if piece_at[to_sq] == PAWN:
                    emit(pawn, from_sq, to_sq)

        for pt in (KNIGHT, BISHOP, ROOK, QUEEN):
            attacker = pt + (color << 3)
            targets = queen_targets if pt in (KNIGHT, QUEEN) else minor_slider_targets
            attackers = pt_bb[color][pt]
            while attackers:
                from_sq = _trailing_square(attackers)
                attackers &= attackers - 1
                if pt == KNIGHT:
                    attacks = _leaper_attacks(KNIGHT, from_sq)
                elif pt == BISHOP:
                    attacks = _slider_attacks(BISHOP, from_sq, occupied)
                elif pt == ROOK:
                    attacks = _slider_attacks(ROOK, from_sq, occupied)
                else:
                    attacks = (
                        _slider_attacks(BISHOP, from_sq, occupied)
                        | _slider_attacks(ROOK, from_sq, occupied)
                    )
                hits = attacks & targets
                while hits:
                    to_sq = _trailing_square(hits)
                    hits &= hits - 1
                    emit(attacker, from_sq, to_sq)

    active.sort()
    return active


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
                         view: int,
                         input_buckets: int = DEFAULT_N_KING_BUCKETS,
                         feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
                         full_threats: bool = False) -> list[int]:
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == view)
    features = [
        feature_index(pt, color, sq, king_sq, view, input_buckets, feature_channels)
        for pt, color, sq in pieces
    ]
    if full_threats:
        base = feature_count(input_buckets, feature_channels)
        features.extend(base + index for index in threat_features_from_pieces(pieces, view))
    return features


def phase_scale_from_pieces(pieces: Sequence[tuple[int, int, int]]) -> float:
    minors = sum(1 for pt, _color, _sq in pieces
                 if pt in (KNIGHT, BISHOP))
    rooks = sum(1 for pt, _color, _sq in pieces if pt == ROOK)
    queens = sum(1 for pt, _color, _sq in pieces if pt == QUEEN)
    phase = 3 * minors + 5 * rooks + 10 * queens
    return (128.0 + float(phase)) / 128.0


def output_bucket_for_piece_count(
    piece_count: int,
    output_buckets: int = DEFAULT_N_OUTPUT_BUCKETS,
) -> int:
    if output_buckets <= 1:
        return 0
    divisor = (32 + output_buckets - 1) // output_buckets
    return max(0, min(output_buckets - 1, (piece_count - 2) // divisor))


def output_bucket_from_pieces(
    pieces: Sequence[tuple[int, int, int]],
    output_buckets: int = DEFAULT_N_OUTPUT_BUCKETS,
) -> int:
    return output_bucket_for_piece_count(len(pieces), output_buckets)


def material_head_features_from_pieces(
    pieces: Sequence[tuple[int, int, int]],
) -> tuple[float, float]:
    return (
        phase_scale_from_pieces(pieces) - 1.0,
        (float(len(pieces)) - 16.0) / 16.0,
    )


def load_net(path: str | Path) -> Net:
    data = Path(path).read_bytes()
    trained_hidden = N_HIDDEN
    format_version = 1
    full_threats = False
    full_heads = False
    mixed_activation = False
    payload = data
    if data.startswith((NETWORK_V2_HEADER_MAGIC, NETWORK_V3_HEADER_MAGIC, NETWORK_V4_HEADER_MAGIC)):
        if len(data) < NETWORK_HEADER_SIZE:
            raise ValueError(f"{path}: truncated Enyo NNUE header")
        (
            magic,
            format_version,
            header_size,
            input_buckets,
            feature_channels,
            trained_hidden,
            runtime_hidden,
            l2_size,
            l3_size,
            output_buckets,
            output_head_features,
            flags,
            payload_size,
            reserved0,
            reserved1,
        ) = NETWORK_HEADER.unpack_from(data)
        if (magic, format_version) not in (
            (NETWORK_V2_HEADER_MAGIC, 2),
            (NETWORK_V3_HEADER_MAGIC, 3),
            (NETWORK_V4_HEADER_MAGIC, 4),
        ):
            raise ValueError(f"{path}: unsupported Enyo NNUE header")
        if header_size != NETWORK_HEADER_SIZE:
            raise ValueError(f"{path}: invalid header size {header_size}")
        if runtime_hidden != N_HIDDEN or l2_size != N_L2 or l3_size != N_L3:
            raise ValueError(f"{path}: unsupported runtime dimensions")
        if trained_hidden not in SUPPORTED_TRAINED_HIDDEN:
            raise ValueError(f"{path}: unsupported trained hidden width {trained_hidden}")
        if (input_buckets, feature_channels) not in SUPPORTED_N_FEATURE_LAYOUTS:
            raise ValueError(f"{path}: unsupported feature layout")
        if output_buckets not in SUPPORTED_N_OUTPUT_BUCKETS:
            raise ValueError(f"{path}: unsupported output bucket count")
        if output_head_features not in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
            raise ValueError(f"{path}: unsupported output head feature count")
        allowed_flags = NETWORK_FLAG_FULL_THREATS | (
            NETWORK_FLAG_FULL_HEADS if format_version == 3 else 0) | (
            NETWORK_FLAG_MIXED_ACTIVATION if format_version == 4 else 0)
        if (flags & ~allowed_flags) or reserved0 or reserved1:
            raise ValueError(f"{path}: unsupported header flags or reserved fields")
        full_threats = bool(flags & NETWORK_FLAG_FULL_THREATS)
        full_heads = bool(flags & NETWORK_FLAG_FULL_HEADS)
        mixed_activation = bool(flags & NETWORK_FLAG_MIXED_ACTIVATION)
        if full_heads != (format_version == 3):
            raise ValueError(f"{path}: v3 and full-head flag must be used together")
        if full_heads and (output_buckets <= 1 or full_threats):
            raise ValueError(f"{path}: unsupported full-head architecture")
        if mixed_activation != (format_version == 4):
            raise ValueError(f"{path}: v4 and mixed-activation flag must be used together")
        if mixed_activation and (full_heads or full_threats or output_buckets != 8):
            raise ValueError(f"{path}: unsupported mixed-activation architecture")
        payload = data[header_size:]
        expected_payload = network_size(
            input_buckets, output_buckets, output_head_features,
            feature_channels, full_threats, full_heads, mixed_activation)
        if payload_size != expected_payload or len(payload) != expected_payload:
            raise ValueError(
                f"{path}: payload size {len(payload)} does not match {expected_payload}")
    else:
        try:
            input_buckets, feature_channels, output_buckets, output_head_features = detect_network_layout(
                len(data))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    n_features = input_feature_count(input_buckets, feature_channels, full_threats)
    output_width = N_L3 + output_head_features
    head_count = output_buckets if full_heads else 1

    off = 0

    def take(dtype, count: int):
        nonlocal off
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    iw = take(np.int16, n_features * N_HIDDEN).reshape(n_features, N_HIDDEN)
    ib = take(np.int16, N_HIDDEN)
    l1w = take(np.int8, head_count * N_L1 * N_L2).reshape(
        head_count, N_L2, N_L1)
    l1b = take(np.int32, head_count * N_L2).reshape(head_count, N_L2)
    l2w = take(np.float32, head_count * N_L2 * N_L3).reshape(
        head_count, N_L3, N_L2)
    l2b = take(np.float32, head_count * N_L3).reshape(head_count, N_L3)
    l2sw = None
    l2sb = None
    if mixed_activation:
        l2sw = take(np.float32, N_L2 * N_L3).reshape(N_L3, N_L2)
        l2sb = take(np.float32, N_L3)
    if not full_heads:
        l1w = l1w[0]
        l1b = l1b[0]
        l2w = l2w[0]
        l2b = l2b[0]
    ow = take(np.float32, output_buckets * output_width).reshape(
        output_buckets, output_width)
    ob = take(np.float32, output_buckets)
    assert off == len(payload)
    return Net(
        input_weights=iw,
        input_biases=ib,
        l1_weights=l1w,
        l1_biases=l1b,
        l2_weights=l2w,
        l2_biases=l2b,
        output_weights=ow,
        output_biases=ob,
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        output_head_features=output_head_features,
        trained_hidden=trained_hidden,
        format_version=format_version,
        full_threats=full_threats,
        full_heads=full_heads,
        mixed_activation=mixed_activation,
        l2_squared_weights=l2sw,
        l2_squared_biases=l2sb)


def write_net(net: Net, path: str | Path) -> None:
    expected_features = input_feature_count(
        net.input_buckets, net.feature_channels, net.full_threats)
    if net.trained_hidden not in SUPPORTED_TRAINED_HIDDEN:
        raise ValueError(f"unsupported trained hidden width {net.trained_hidden}")
    input_weights = np.asarray(net.input_weights, dtype=np.int16)
    if input_weights.shape == (expected_features, net.trained_hidden):
        padded = np.zeros((expected_features, N_HIDDEN), dtype=np.int16)
        padded[:, :net.trained_hidden] = input_weights
        input_weights = padded
    if input_weights.shape != (expected_features, N_HIDDEN):
        raise ValueError(
            f"input_weights shape {input_weights.shape} does not match "
            f"{net.input_buckets} buckets / {net.feature_channels} channels")
    input_biases = np.asarray(net.input_biases, dtype=np.int16).reshape(-1)
    if input_biases.shape == (net.trained_hidden,):
        padded = np.zeros(N_HIDDEN, dtype=np.int16)
        padded[:net.trained_hidden] = input_biases
        input_biases = padded
    if input_biases.shape != (N_HIDDEN,):
        raise ValueError(f"input_biases shape {input_biases.shape} does not match runtime")
    head_count = net.output_buckets if net.full_heads else 1
    l1_weights = np.asarray(net.l1_weights, dtype=np.int8)
    if not net.full_heads and l1_weights.ndim == 2:
        l1_weights = l1_weights.reshape(1, *l1_weights.shape)
    if l1_weights.shape == (head_count, N_L2, 2 * net.trained_hidden):
        padded = np.zeros((head_count, N_L2, N_L1), dtype=np.int8)
        padded[:, :, :net.trained_hidden] = l1_weights[:, :, :net.trained_hidden]
        padded[:, :, N_HIDDEN:N_HIDDEN + net.trained_hidden] = \
            l1_weights[:, :, net.trained_hidden:]
        l1_weights = padded
    if l1_weights.shape != (head_count, N_L2, N_L1):
        raise ValueError(f"l1_weights shape {l1_weights.shape} does not match runtime")
    l1_biases = np.asarray(net.l1_biases, dtype=np.int32)
    if not net.full_heads and l1_biases.ndim == 1:
        l1_biases = l1_biases.reshape(1, N_L2)
    if l1_biases.shape != (head_count, N_L2):
        raise ValueError(f"l1_biases shape {l1_biases.shape} does not match runtime")
    l2_weights = np.asarray(net.l2_weights, dtype=np.float32)
    if not net.full_heads and l2_weights.ndim == 2:
        l2_weights = l2_weights.reshape(1, N_L3, N_L2)
    if l2_weights.shape != (head_count, N_L3, N_L2):
        raise ValueError(f"l2_weights shape {l2_weights.shape} does not match runtime")
    l2_biases = np.asarray(net.l2_biases, dtype=np.float32)
    if not net.full_heads and l2_biases.ndim == 1:
        l2_biases = l2_biases.reshape(1, N_L3)
    if l2_biases.shape != (head_count, N_L3):
        raise ValueError(f"l2_biases shape {l2_biases.shape} does not match runtime")
    squared_payload = b""
    if net.mixed_activation:
        l2_squared_weights = np.asarray(net.l2_squared_weights, dtype=np.float32)
        l2_squared_biases = np.asarray(net.l2_squared_biases, dtype=np.float32)
        if l2_squared_weights.shape != (N_L3, N_L2):
            raise ValueError("mixed activation requires l2_squared_weights[32,16]")
        if l2_squared_biases.shape != (N_L3,):
            raise ValueError("mixed activation requires l2_squared_biases[32]")
        squared_payload = (
            l2_squared_weights.tobytes(order="C")
            + l2_squared_biases.tobytes(order="C")
        )
    if net.output_buckets not in SUPPORTED_N_OUTPUT_BUCKETS:
        raise ValueError(f"unsupported output bucket count {net.output_buckets}")
    if net.output_head_features not in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
        raise ValueError(
            f"unsupported output head feature count {net.output_head_features}")
    if net.full_heads and (
            net.output_buckets <= 1 or net.full_threats
            or net.output_head_features != 0):
        raise ValueError("unsupported full-head architecture")
    output_width = net.output_width
    output_weights = np.asarray(net.output_weights, dtype=np.float32)
    if output_weights.shape == (output_width,):
        output_weights = output_weights.reshape(1, output_width)
    if output_weights.shape != (net.output_buckets, output_width):
        raise ValueError(
            f"output_weights shape {output_weights.shape} does not match "
            f"{net.output_buckets} output buckets and "
            f"{net.output_head_features} output head features")
    output_biases = np.asarray(net.output_biases, dtype=np.float32).reshape(-1)
    if output_biases.shape != (net.output_buckets,):
        raise ValueError(
            f"output_biases shape {output_biases.shape} does not match "
            f"{net.output_buckets} output buckets")
    payload = b"".join((
        input_weights.tobytes(order="C"),
        input_biases.tobytes(order="C"),
        l1_weights.tobytes(order="C"),
        l1_biases.tobytes(order="C"),
        l2_weights.tobytes(order="C"),
        l2_biases.tobytes(order="C"),
        squared_payload,
        output_weights.tobytes(order="C"),
        output_biases.tobytes(order="C"),
    ))
    if net.format_version == 1:
        if net.full_heads:
            raise ValueError("full-head networks require enyo-native-v3")
        if net.trained_hidden != N_HIDDEN:
            raise ValueError("non-1024 hidden widths require enyo-native-v2")
        data = payload
    elif net.format_version in (2, 3, 4):
        if (net.format_version == 3) != net.full_heads:
            raise ValueError("enyo-native-v3 and full_heads must be used together")
        if (net.format_version == 4) != net.mixed_activation:
            raise ValueError("enyo-native-v4 and mixed activation must be used together")
        magic = (
            NETWORK_V4_HEADER_MAGIC if net.mixed_activation
            else NETWORK_V3_HEADER_MAGIC if net.full_heads
            else NETWORK_V2_HEADER_MAGIC
        )
        header = NETWORK_HEADER.pack(
            magic,
            net.format_version,
            NETWORK_HEADER_SIZE,
            net.input_buckets,
            net.feature_channels,
            net.trained_hidden,
            N_HIDDEN,
            N_L2,
            N_L3,
            net.output_buckets,
            net.output_head_features,
            (NETWORK_FLAG_FULL_THREATS if net.full_threats else 0)
            | (NETWORK_FLAG_FULL_HEADS if net.full_heads else 0)
            | (NETWORK_FLAG_MIXED_ACTIVATION if net.mixed_activation else 0),
            len(payload),
            0,
            0,
        )
        data = header + payload
    else:
        raise ValueError(f"unsupported format version {net.format_version}")

    out = Path(path)
    out.write_bytes(data)
    size = out.stat().st_size
    expected = network_size(
        net.input_buckets, net.output_buckets, net.output_head_features,
        net.feature_channels, net.full_threats, net.full_heads, net.mixed_activation)
    if net.format_version in (2, 3, 4):
        expected += NETWORK_HEADER_SIZE
    if size != expected:
        raise RuntimeError(f"wrote {size} bytes, expected {expected}")
