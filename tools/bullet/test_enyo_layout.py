#!/usr/bin/env python3
"""Layout checks for Bullet Enyo checkpoint export helpers."""
from __future__ import annotations

from bullet import (
    ENYO_DEFAULT_INPUT_BUCKETS,
    ENYO_FEATURE_STRIDE,
    ENYO_HIDDEN,
    ENYO_NETWORK_HEADER,
    ENYO_NETWORK_HEADER_MAGIC,
    ENYO_NETWORK_SIZE,
    enyo_v2_container,
    enyo_network_size,
    enyo_parent_bucket,
    expand_enyo_input_buckets,
    pad_enyo_hidden,
)


def test_default_size() -> None:
    assert enyo_network_size() == ENYO_NETWORK_SIZE


def test_variable_hidden_size() -> None:
    assert enyo_network_size(hidden=1280) > enyo_network_size(hidden=1024)
    assert enyo_network_size(8, hidden=1280) < enyo_network_size(32, hidden=1280)
    assert enyo_network_size(
        32, feature_channels=11) < enyo_network_size(32, feature_channels=12)


def test_expand_variable_hidden() -> None:
    hidden = 2
    l2 = 1
    l3 = 2
    input_buckets = 1
    feature_bytes = hidden * 2
    size = enyo_network_size(input_buckets, hidden=hidden, l2=l2, l3=l3)
    raw = bytearray(size)
    for source_feature in range(input_buckets * ENYO_FEATURE_STRIDE):
        start = source_feature * feature_bytes
        raw[start:start + feature_bytes] = bytes([source_feature % 251]) * feature_bytes

    expanded = expand_enyo_input_buckets(
        bytes(raw),
        input_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )

    target_l0_bytes = ENYO_DEFAULT_INPUT_BUCKETS * ENYO_FEATURE_STRIDE * feature_bytes
    assert len(expanded) == enyo_network_size(
        ENYO_DEFAULT_INPUT_BUCKETS,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    for target_bucket in range(ENYO_DEFAULT_INPUT_BUCKETS):
        source_bucket = enyo_parent_bucket(target_bucket, input_buckets)
        for offset in (0, 17, ENYO_FEATURE_STRIDE - 1):
            target_feature = target_bucket * ENYO_FEATURE_STRIDE + offset
            source_feature = source_bucket * ENYO_FEATURE_STRIDE + offset
            start = target_feature * feature_bytes
            expected = bytes([source_feature % 251]) * feature_bytes
            assert expanded[start:start + feature_bytes] == expected
    assert expanded[target_l0_bytes:] == raw[input_buckets * ENYO_FEATURE_STRIDE * feature_bytes:]


def test_expand_preserves_output_bucket_tail() -> None:
    hidden = 2
    l2 = 1
    l3 = 2
    input_buckets = 8
    runtime_buckets = 16
    output_buckets = 4
    feature_bytes = hidden * 2
    size = enyo_network_size(
        input_buckets,
        output_buckets=output_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    raw = bytearray(size)
    source_l0_bytes = input_buckets * ENYO_FEATURE_STRIDE * feature_bytes
    raw[source_l0_bytes:] = bytes(range(1, 1 + len(raw[source_l0_bytes:])))

    expanded = expand_enyo_input_buckets(
        bytes(raw),
        input_buckets,
        runtime_input_buckets=runtime_buckets,
        output_buckets=output_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )

    target_l0_bytes = runtime_buckets * ENYO_FEATURE_STRIDE * feature_bytes
    assert len(expanded) == enyo_network_size(
        runtime_buckets,
        output_buckets=output_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    assert expanded[target_l0_bytes:] == raw[source_l0_bytes:]


def test_export_exact_runtime_bucket_count() -> None:
    hidden = 2
    l2 = 1
    l3 = 2
    input_buckets = 16
    size = enyo_network_size(input_buckets, hidden=hidden, l2=l2, l3=l3)
    raw = bytes([7]) * size

    exported = expand_enyo_input_buckets(
        raw,
        input_buckets,
        runtime_input_buckets=16,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )

    assert exported == raw


def test_expand_to_16_runtime_buckets() -> None:
    hidden = 2
    l2 = 1
    l3 = 2
    input_buckets = 8
    runtime_buckets = 16
    feature_bytes = hidden * 2
    size = enyo_network_size(input_buckets, hidden=hidden, l2=l2, l3=l3)
    raw = bytearray(size)
    for source_feature in range(input_buckets * ENYO_FEATURE_STRIDE):
        start = source_feature * feature_bytes
        raw[start:start + feature_bytes] = bytes([source_feature % 251]) * feature_bytes

    expanded = expand_enyo_input_buckets(
        bytes(raw),
        input_buckets,
        runtime_input_buckets=runtime_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )

    target_l0_bytes = runtime_buckets * ENYO_FEATURE_STRIDE * feature_bytes
    assert len(expanded) == enyo_network_size(
        runtime_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    for target_bucket in range(runtime_buckets):
        source_bucket = enyo_parent_bucket(
            target_bucket,
            input_buckets,
            runtime_input_buckets=runtime_buckets,
        )
        for offset in (0, 17, ENYO_FEATURE_STRIDE - 1):
            target_feature = target_bucket * ENYO_FEATURE_STRIDE + offset
            source_feature = source_bucket * ENYO_FEATURE_STRIDE + offset
            start = target_feature * feature_bytes
            expected = bytes([source_feature % 251]) * feature_bytes
            assert expanded[start:start + feature_bytes] == expected
    assert expanded[target_l0_bytes:] == raw[input_buckets * ENYO_FEATURE_STRIDE * feature_bytes:]


def test_expand_to_32_runtime_buckets_uses_legacy_mapping() -> None:
    hidden = 2
    l2 = 1
    l3 = 2
    input_buckets = 16
    runtime_buckets = 32
    feature_bytes = hidden * 2
    size = enyo_network_size(input_buckets, hidden=hidden, l2=l2, l3=l3)
    raw = bytearray(size)
    for source_feature in range(input_buckets * ENYO_FEATURE_STRIDE):
        start = source_feature * feature_bytes
        raw[start:start + feature_bytes] = bytes([source_feature % 251]) * feature_bytes

    expanded = expand_enyo_input_buckets(
        bytes(raw),
        input_buckets,
        runtime_input_buckets=runtime_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )

    target_l0_bytes = runtime_buckets * ENYO_FEATURE_STRIDE * feature_bytes
    assert len(expanded) == enyo_network_size(
        runtime_buckets,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    for target_bucket in range(runtime_buckets):
        source_bucket = enyo_parent_bucket(
            target_bucket,
            input_buckets,
            runtime_input_buckets=runtime_buckets,
        )
        for offset in (0, 17, ENYO_FEATURE_STRIDE - 1):
            target_feature = target_bucket * ENYO_FEATURE_STRIDE + offset
            source_feature = source_bucket * ENYO_FEATURE_STRIDE + offset
            start = target_feature * feature_bytes
            expected = bytes([source_feature % 251]) * feature_bytes
            assert expanded[start:start + feature_bytes] == expected
    assert expanded[target_l0_bytes:] == raw[input_buckets * ENYO_FEATURE_STRIDE * feature_bytes:]


def test_pad_hidden_preserves_two_perspective_l1_halves() -> None:
    input_buckets = 1
    feature_channels = 1
    hidden = 2
    l2 = 1
    l3 = 2
    size = enyo_network_size(
        input_buckets,
        feature_channels=feature_channels,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    raw = bytearray(size)
    source_l0w = input_buckets * feature_channels * 64 * hidden * 2
    source_l0b = hidden * 2
    source_l1 = source_l0w + source_l0b
    raw[source_l1:source_l1 + 2 * hidden] = bytes((1, 2, 3, 4))

    padded = pad_enyo_hidden(
        bytes(raw),
        input_buckets,
        feature_channels=feature_channels,
        hidden=hidden,
        l2=l2,
        l3=l3,
    )
    target_l0w = input_buckets * feature_channels * 64 * ENYO_HIDDEN * 2
    target_l0b = ENYO_HIDDEN * 2
    target_l1 = target_l0w + target_l0b
    assert padded[target_l1:target_l1 + hidden] == bytes((1, 2))
    assert padded[
        target_l1 + ENYO_HIDDEN:target_l1 + ENYO_HIDDEN + hidden
    ] == bytes((3, 4))


def test_v2_header_records_trained_layout() -> None:
    payload = bytes(enyo_network_size(10, feature_channels=11, output_buckets=8))
    container = enyo_v2_container(
        payload,
        input_buckets=10,
        feature_channels=11,
        hidden=768,
        output_buckets=8,
    )
    fields = ENYO_NETWORK_HEADER.unpack_from(container)
    assert fields[0] == ENYO_NETWORK_HEADER_MAGIC
    assert fields[3:11] == (10, 11, 768, ENYO_HIDDEN, 16, 32, 8, 0)
    assert fields[12] == len(payload)


def main() -> None:
    test_default_size()
    test_variable_hidden_size()
    test_expand_variable_hidden()
    test_expand_preserves_output_bucket_tail()
    test_export_exact_runtime_bucket_count()
    test_expand_to_16_runtime_buckets()
    test_expand_to_32_runtime_buckets_uses_legacy_mapping()
    test_pad_hidden_preserves_two_perspective_l1_halves()
    test_v2_header_records_trained_layout()


if __name__ == "__main__":
    main()
