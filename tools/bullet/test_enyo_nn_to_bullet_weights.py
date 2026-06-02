#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from enyo_nn_to_bullet_weights import (
    bullet_l3_weights,
    expand_output_head,
)
from lib.enyo_nnue import N_L3


class SingleHeadNet:
    output_weights = np.arange(N_L3, dtype=np.float32).reshape(1, N_L3)
    output_biases = np.asarray([7.0], dtype=np.float32)
    output_buckets = 1


def test_expanded_l3_weights_use_bullet_internal_orientation() -> None:
    output_weights, output_biases = expand_output_head(SingleHeadNet(), 4)
    bullet_weights = bullet_l3_weights(output_weights)

    assert output_weights.shape == (4, N_L3)
    assert bullet_weights.shape == (N_L3, 4)
    np.testing.assert_array_equal(output_biases, np.asarray([7.0] * 4))
    np.testing.assert_array_equal(bullet_weights[0], np.asarray([0.0] * 4))
    np.testing.assert_array_equal(bullet_weights[1], np.asarray([1.0] * 4))
    np.testing.assert_array_equal(
        bullet_weights[:, 0],
        np.arange(N_L3, dtype=np.float32),
    )
