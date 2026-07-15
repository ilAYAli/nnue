"""Shared trainer/runtime unit conversions for NNUE audits."""

from __future__ import annotations

import math


def phase_scale(minors: int, rooks: int, queens: int) -> float:
    return (128.0 + 3 * minors + 5 * rooks + 10 * queens) / 128.0


def normalize_training_score(score_cp: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("phase scale must be positive")
    return score_cp / scale


def runtime_score(normalized_cp: float, scale: float) -> float:
    return normalized_cp * scale


def wdl_target(score_cp: float, result: float, blend: float, eval_scale: float = 400.0) -> float:
    if not 0.0 <= result <= 1.0 or not 0.0 <= blend <= 1.0:
        raise ValueError("result and blend must be in [0, 1]")
    cp = 1.0 / (1.0 + math.exp(-score_cp / eval_scale))
    return blend * result + (1.0 - blend) * cp


def output_bucket(piece_count: int, buckets: int = 8) -> int:
    if buckets <= 1:
        return 0
    divisor = (32 + buckets - 1) // buckets
    return max(0, min(buckets - 1, (piece_count - 2) // divisor))


def quantize_eval(raw_cp: float, scale: float = 32.0) -> int:
    return int(round(raw_cp * scale))


def dequantize_eval(value: int, scale: float = 32.0) -> float:
    return value / scale
