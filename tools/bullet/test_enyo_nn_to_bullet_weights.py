#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from enyo_nn_to_bullet_weights import (
    bullet_l3_weights,
    expand_output_head,
    source_bucket_for_target,
    source_channel_for_target,
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


def test_32_bucket_init_uses_legacy_parent_buckets() -> None:
    assert source_bucket_for_target(11, 16, 32) == 11
    assert source_bucket_for_target(12, 16, 32) == 8
    assert source_bucket_for_target(31, 16, 32) == 15


def test_clean_bucket_expansion_repeats_parent_buckets() -> None:
    assert [source_bucket_for_target(bucket, 8, 16) for bucket in range(16)] == [
        0, 0, 1, 1, 2, 2, 3, 3,
        4, 4, 5, 5, 6, 6, 7, 7,
    ]


def test_unsupported_bucket_mapping_fails() -> None:
    try:
        source_bucket_for_target(0, 8, 12)
    except SystemExit as err:
        assert "cannot map input buckets 8 -> 12" in str(err)
    else:
        raise AssertionError("expected unsupported bucket mapping to fail")


def test_halfka_v2_init_merges_king_channels_by_square_legality() -> None:
    assert source_channel_for_target(0, 0, 31, 12, 11) == 0
    assert source_channel_for_target(4, 0, 31, 12, 11) == 4
    assert source_channel_for_target(5, 0, 31, 12, 11) == 6
    assert source_channel_for_target(9, 0, 31, 12, 11) == 10
    assert source_channel_for_target(10, 0, 31, 12, 11) == 5
    assert source_channel_for_target(10, 1, 31, 12, 11) == 11

def main() -> None:
    test_expanded_l3_weights_use_bullet_internal_orientation()
    test_32_bucket_init_uses_legacy_parent_buckets()
    test_clean_bucket_expansion_repeats_parent_buckets()
    test_unsupported_bucket_mapping_fails()
    test_halfka_v2_init_merges_king_channels_by_square_legality()


if __name__ == "__main__":
    main()
