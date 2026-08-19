#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from enyo_nn_to_bullet_weights import (
    add_full_threat_rows,
    bullet_initial_l3_weights,
    bullet_l3_weights,
    expand_dense_heads,
    expand_output_head,
    fresh_dense_tail,
    mixed_activation_weights,
    source_bucket_for_target,
    source_output_bucket_for_target,
    source_channel_for_target,
)
from lib.enyo_nnue import N_L1, N_L2, N_L3


class SingleHeadNet:
    output_weights = np.arange(N_L3, dtype=np.float32).reshape(1, N_L3)
    output_biases = np.asarray([7.0], dtype=np.float32)
    output_buckets = 1


class FourHeadNet:
    output_weights = np.arange(4 * N_L3, dtype=np.float32).reshape(4, N_L3)
    output_biases = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    output_buckets = 4


class SharedDenseNet:
    l1_weights = np.arange(N_L2 * N_L1, dtype=np.float32).reshape(N_L2, N_L1)
    l1_biases = np.arange(N_L2, dtype=np.float32)
    l2_weights = np.arange(N_L3 * N_L2, dtype=np.float32).reshape(N_L3, N_L2)
    l2_biases = np.arange(N_L3, dtype=np.float32)
    full_heads = False


class MixedActivationNet:
    l2_squared_weights = np.arange(
        N_L3 * N_L2, dtype=np.float32).reshape(N_L3, N_L2)
    l2_squared_biases = np.arange(N_L3, dtype=np.float32) + 0.5


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


def test_initial_l3_weights_use_bullet_target_scale() -> None:
    output_weights = np.asarray([[1.0, 2.0]], dtype=np.float32)
    actual = bullet_initial_l3_weights(
        output_weights,
        eval_scale=400.0,
        eval_divisor=32.0,
    )

    np.testing.assert_allclose(
        actual,
        np.asarray([[1.0 / 12800.0], [2.0 / 12800.0]], dtype=np.float32),
    )


def test_expanded_multi_bucket_head_repeats_parent_ranges() -> None:
    output_weights, output_biases = expand_output_head(FourHeadNet(), 8)

    assert [source_output_bucket_for_target(bucket, 4, 8) for bucket in range(8)] == [
        0, 0, 1, 1, 2, 2, 3, 3,
    ]
    assert output_weights.shape == (8, N_L3)
    np.testing.assert_array_equal(output_biases, np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]))
    np.testing.assert_array_equal(output_weights[0], FourHeadNet.output_weights[0])
    np.testing.assert_array_equal(output_weights[1], FourHeadNet.output_weights[0])
    np.testing.assert_array_equal(output_weights[6], FourHeadNet.output_weights[3])
    np.testing.assert_array_equal(output_weights[7], FourHeadNet.output_weights[3])


def test_full_dense_heads_repeat_shared_native_head() -> None:
    l1w, l1b, l2w, l2b = expand_dense_heads(SharedDenseNet(), 8, True)

    assert l1w.shape == (8, N_L2, N_L1)
    assert l1b.shape == (8, N_L2)
    assert l2w.shape == (8, N_L3, N_L2)
    assert l2b.shape == (8, N_L3)
    for head in range(8):
        np.testing.assert_array_equal(l1w[head], SharedDenseNet.l1_weights)
        np.testing.assert_array_equal(l1b[head], SharedDenseNet.l1_biases)
        np.testing.assert_array_equal(l2w[head], SharedDenseNet.l2_weights)
        np.testing.assert_array_equal(l2b[head], SharedDenseNet.l2_biases)


def test_mixed_activation_init_preserves_parent_branch() -> None:
    weights, biases = mixed_activation_weights(MixedActivationNet(), 1, False)
    np.testing.assert_array_equal(weights, MixedActivationNet.l2_squared_weights)
    np.testing.assert_array_equal(biases, MixedActivationNet.l2_squared_biases)
    np.testing.assert_array_equal(
        weights.ravel(order="F"),
        MixedActivationNet.l2_squared_weights.T.ravel(),
    )


def test_full_threat_init_is_deterministic_and_nonzero() -> None:
    parent = np.asarray([
        [-3.0, 2.0],
        [-1.0, 4.0],
        [1.0, 6.0],
        [3.0, 8.0],
    ], dtype=np.float32)
    first = add_full_threat_rows(parent, rows=8)
    second = add_full_threat_rows(parent, rows=8)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:len(parent)], parent)
    assert np.count_nonzero(first[len(parent):]) == 16
    np.testing.assert_allclose(
        first[len(parent):].std(axis=0),
        parent.std(axis=0) * 0.1,
        rtol=0.6,
    )


def test_fresh_dense_tail_is_deterministic_and_has_native_v2_shapes() -> None:
    first = fresh_dense_tail(
        hidden=1024, output_buckets=1, full_heads=False, l1_std=1.0, seed=17)
    same = fresh_dense_tail(
        hidden=1024, output_buckets=1, full_heads=False, l1_std=1.0, seed=17)
    other = fresh_dense_tail(
        hidden=1024, output_buckets=1, full_heads=False, l1_std=1.0, seed=18)

    l1w, l1b, l2w, l2b, l3w, l3b = first
    assert l1w.shape == (N_L2, N_L1)
    assert l1b.shape == (N_L2,)
    assert l2w.shape == (N_L3, N_L2)
    assert l2b.shape == (N_L3,)
    assert l3w.shape == (1, N_L3)
    assert l3b.shape == (1,)
    assert np.count_nonzero(l1w) == l1w.size
    assert np.count_nonzero(l2w) == l2w.size
    assert np.count_nonzero(l3w) == l3w.size
    assert not np.any(l1b) and not np.any(l2b) and not np.any(l3b)
    for actual, expected in zip(first, same):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(first[0], other[0])




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
    test_initial_l3_weights_use_bullet_target_scale()
    test_expanded_multi_bucket_head_repeats_parent_ranges()
    test_full_dense_heads_repeat_shared_native_head()
    test_mixed_activation_init_preserves_parent_branch()
    test_full_threat_init_is_deterministic_and_nonzero()
    test_fresh_dense_tail_is_deterministic_and_has_native_v2_shapes()
    test_32_bucket_init_uses_legacy_parent_buckets()
    test_clean_bucket_expansion_repeats_parent_buckets()
    test_unsupported_bucket_mapping_fails()
    test_halfka_v2_init_merges_king_channels_by_square_legality()


if __name__ == "__main__":
    main()
