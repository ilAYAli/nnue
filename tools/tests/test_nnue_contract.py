from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate.nnue_contract import (  # noqa: E402
    dequantize_eval,
    normalize_training_score,
    output_bucket,
    phase_scale,
    quantize_eval,
    runtime_score,
    wdl_target,
)


def test_phase_normalization_roundtrip() -> None:
    for pieces in ((0, 0, 1), (4, 2, 1), (0, 0, 0)):
        scale = phase_scale(*pieces)
        normalized = normalize_training_score(640.0, scale)
        assert math.isclose(runtime_score(normalized, scale), 640.0, abs_tol=1e-6)


def test_wdl_target_matches_bullet_formula() -> None:
    target = wdl_target(400.0, 1.0, 0.05)
    expected = 0.05 + 0.95 / (1.0 + math.exp(-1.0))
    assert math.isclose(target, expected, rel_tol=1e-7)


def test_output_bucket_matches_runtime_boundaries() -> None:
    assert [output_bucket(count) for count in (2, 6, 10, 14, 18, 22, 26, 30, 32)] == [
        0, 1, 2, 3, 4, 5, 6, 7, 7
    ]


def test_export_quantization_roundtrip_is_bounded() -> None:
    for value in (-2045.0, -123.25, 0.0, 123.25, 2045.0):
        assert abs(dequantize_eval(quantize_eval(value)) - value) <= 1.0 / 32.0
