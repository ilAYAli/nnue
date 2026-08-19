#!/usr/bin/env python3
"""Focused parity checks for NumPy Enyo forward inference."""
from __future__ import annotations

import numpy as np

from tools.lib import enyo_nnue as nn2
from tools.lib.nnue_forward import EnyoNNUE


def test_shared_screlu_branch_uses_its_only_head() -> None:
    input_bias = np.zeros(nn2.N_HIDDEN, dtype=np.float32)
    input_bias[0] = 1 << nn2.QUANT1_BITS
    l1_weight = np.zeros((nn2.N_L2, 2 * nn2.N_HIDDEN), dtype=np.float32)
    l1_weight[0, 0] = 1.0
    l2_squared_weight = np.zeros((1, nn2.N_L3, nn2.N_L2), dtype=np.float32)
    l2_squared_weight[0, 0, 0] = 1.0
    output_weight = np.zeros((8, nn2.N_L3), dtype=np.float32)
    output_weight[:, 0] = nn2.EVAL_DIVISOR
    model = EnyoNNUE(
        input_weights=np.zeros((1, nn2.N_HIDDEN), dtype=np.float32),
        input_bias=input_bias,
        l1_weight=l1_weight,
        l1_bias=np.zeros(nn2.N_L2, dtype=np.float32),
        l2_weight=np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32),
        l2_bias=np.zeros(nn2.N_L3, dtype=np.float32),
        output_weight=output_weight,
        output_bias=np.zeros(8, dtype=np.float32),
        input_buckets=16,
        feature_channels=12,
        output_buckets=8,
        output_head_features=0,
        trained_hidden=nn2.N_HIDDEN,
        format_version=8,
        full_threats=False,
        slider_xray_threats=False,
        full_heads=False,
        mixed_activation=True,
        l2_squared_weight=l2_squared_weight,
        l2_squared_bias=np.zeros((1, nn2.N_L3), dtype=np.float32),
    )

    actual = model.raw_forward(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_allclose(actual, np.asarray([1.0 / 127.0], dtype=np.float32))


if __name__ == "__main__":
    test_shared_screlu_branch_uses_its_only_head()
