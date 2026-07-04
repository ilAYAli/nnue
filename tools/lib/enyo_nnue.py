"""Python reference helpers for Enyo's Berserk-format NNUE.

The constants and feature-index formula mirror src/nnue.hpp.  The file
format is Berserk v13's .nn layout as loaded by NNUE::LoadNetwork().
"""
from __future__ import annotations

from dataclasses import dataclass
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
N_KING_BUCKETS = DEFAULT_N_KING_BUCKETS
N_PIECE_TYPES = 12
N_SQUARES = 64
N_HIDDEN = 1024
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_OUTPUT = 1
NETWORK_HEADER_MAGIC = b"ENYONN2\0"
NETWORK_FORMAT_VERSION = 2
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


def network_size(
    input_buckets: int = DEFAULT_N_KING_BUCKETS,
    output_buckets: int = DEFAULT_N_OUTPUT_BUCKETS,
    output_head_features: int = DEFAULT_N_OUTPUT_HEAD_FEATURES,
    feature_channels: int = DEFAULT_N_FEATURE_CHANNELS,
) -> int:
    features = feature_count(input_buckets, feature_channels)
    if output_buckets not in SUPPORTED_N_OUTPUT_BUCKETS:
        raise ValueError(f"unsupported output bucket count {output_buckets}")
    if output_head_features not in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
        raise ValueError(
            f"unsupported output head feature count {output_head_features}")
    output_width = N_L3 + output_head_features
    return (
        features * N_HIDDEN * np.dtype(np.int16).itemsize
        + N_HIDDEN * np.dtype(np.int16).itemsize
        + N_L1 * N_L2 * np.dtype(np.int8).itemsize
        + N_L2 * np.dtype(np.int32).itemsize
        + N_L2 * N_L3 * np.dtype(np.float32).itemsize
        + N_L3 * np.dtype(np.float32).itemsize
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
                         feature_channels: int = DEFAULT_N_FEATURE_CHANNELS) -> list[int]:
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == view)
    return [feature_index(pt, color, sq, king_sq, view, input_buckets, feature_channels)
            for pt, color, sq in pieces]


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
    payload = data
    if data.startswith(NETWORK_HEADER_MAGIC):
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
        if magic != NETWORK_HEADER_MAGIC or format_version != NETWORK_FORMAT_VERSION:
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
        if flags or reserved0 or reserved1:
            raise ValueError(f"{path}: unsupported header flags or reserved fields")
        payload = data[header_size:]
        expected_payload = network_size(
            input_buckets, output_buckets, output_head_features,
            feature_channels)
        if payload_size != expected_payload or len(payload) != expected_payload:
            raise ValueError(
                f"{path}: payload size {len(payload)} does not match {expected_payload}")
    else:
        try:
            input_buckets, feature_channels, output_buckets, output_head_features = detect_network_layout(
                len(data))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    n_features = feature_count(input_buckets, feature_channels)
    output_width = N_L3 + output_head_features

    off = 0

    def take(dtype, count: int):
        nonlocal off
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    iw = take(np.int16, n_features * N_HIDDEN).reshape(n_features, N_HIDDEN)
    ib = take(np.int16, N_HIDDEN)
    l1w = take(np.int8, N_L1 * N_L2).reshape(N_L2, N_L1)
    l1b = take(np.int32, N_L2)
    l2w = take(np.float32, N_L2 * N_L3).reshape(N_L3, N_L2)
    l2b = take(np.float32, N_L3)
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
        format_version=format_version)


def write_net(net: Net, path: str | Path) -> None:
    expected_features = feature_count(net.input_buckets, net.feature_channels)
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
    l1_weights = np.asarray(net.l1_weights, dtype=np.int8)
    if l1_weights.shape == (N_L2, 2 * net.trained_hidden):
        padded = np.zeros((N_L2, N_L1), dtype=np.int8)
        padded[:, :net.trained_hidden] = l1_weights[:, :net.trained_hidden]
        padded[:, N_HIDDEN:N_HIDDEN + net.trained_hidden] = l1_weights[:, net.trained_hidden:]
        l1_weights = padded
    if l1_weights.shape != (N_L2, N_L1):
        raise ValueError(f"l1_weights shape {l1_weights.shape} does not match runtime")
    if net.output_buckets not in SUPPORTED_N_OUTPUT_BUCKETS:
        raise ValueError(f"unsupported output bucket count {net.output_buckets}")
    if net.output_head_features not in SUPPORTED_N_OUTPUT_HEAD_FEATURES:
        raise ValueError(
            f"unsupported output head feature count {net.output_head_features}")
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
        np.asarray(net.l1_biases, dtype=np.int32).tobytes(order="C"),
        np.asarray(net.l2_weights, dtype=np.float32).tobytes(order="C"),
        np.asarray(net.l2_biases, dtype=np.float32).tobytes(order="C"),
        output_weights.tobytes(order="C"),
        output_biases.tobytes(order="C"),
    ))
    if net.format_version == 1:
        if net.trained_hidden != N_HIDDEN:
            raise ValueError("non-1024 hidden widths require enyo-native-v2")
        data = payload
    elif net.format_version == NETWORK_FORMAT_VERSION:
        header = NETWORK_HEADER.pack(
            NETWORK_HEADER_MAGIC,
            NETWORK_FORMAT_VERSION,
            NETWORK_HEADER_SIZE,
            net.input_buckets,
            net.feature_channels,
            net.trained_hidden,
            N_HIDDEN,
            N_L2,
            N_L3,
            net.output_buckets,
            net.output_head_features,
            0,
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
        net.feature_channels)
    if net.format_version == NETWORK_FORMAT_VERSION:
        expected += NETWORK_HEADER_SIZE
    if size != expected:
        raise RuntimeError(f"wrote {size} bytes, expected {expected}")
