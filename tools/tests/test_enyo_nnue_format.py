#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.lib import enyo_nnue as nn2


def zero_net(*, input_buckets: int = 16,
             feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
             output_buckets: int = 1,
             output_head_features: int = 0,
             trained_hidden: int = nn2.N_HIDDEN,
             format_version: int = 1,
             full_threats: bool = False,
             slider_xray_threats: bool = False,
             full_heads: bool = False) -> nn2.Net:
    features = nn2.input_feature_count(
        input_buckets, feature_channels,
        full_threats or slider_xray_threats)
    head_count = output_buckets if full_heads else 1
    l1_shape = (head_count, nn2.N_L2, 2 * trained_hidden)
    l2_shape = (head_count, nn2.N_L3, nn2.N_L2)
    if not full_heads:
        l1_shape = l1_shape[1:]
        l2_shape = l2_shape[1:]
    return nn2.Net(
        input_weights=np.zeros((features, trained_hidden), dtype=np.int16),
        input_biases=np.zeros((trained_hidden,), dtype=np.int16),
        l1_weights=np.zeros(l1_shape, dtype=np.int8),
        l1_biases=np.zeros(
            (head_count, nn2.N_L2) if full_heads else (nn2.N_L2,),
            dtype=np.int32),
        l2_weights=np.zeros(l2_shape, dtype=np.float32),
        l2_biases=np.zeros(
            (head_count, nn2.N_L3) if full_heads else (nn2.N_L3,),
            dtype=np.float32),
        output_weights=np.zeros(
            (output_buckets, nn2.N_L3 + output_head_features),
            dtype=np.float32),
        output_biases=np.arange(output_buckets, dtype=np.float32),
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        output_head_features=output_head_features,
        trained_hidden=trained_hidden,
        format_version=format_version,
        full_threats=full_threats,
        slider_xray_threats=slider_xray_threats,
        full_heads=full_heads,
    )


class EnyoNNUEFormatTests(unittest.TestCase):
    def test_slider_xray_v2_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slider-xray.nn"
            nn2.write_net(zero_net(
                output_buckets=8,
                format_version=2,
                slider_xray_threats=True,
            ), path)
            loaded = nn2.load_net(path)
            self.assertFalse(loaded.full_threats)
            self.assertTrue(loaded.slider_xray_threats)
            self.assertEqual(
                loaded.input_weights.shape,
                (nn2.input_feature_count(16, 12, True), nn2.N_HIDDEN),
            )

    def test_full_head_v3_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-heads.nn"
            net = zero_net(
                output_buckets=8,
                format_version=3,
                full_heads=True,
            )
            net.l1_biases[:, 0] = np.arange(8, dtype=np.int32)
            net.l2_biases[:, 0] = np.arange(8, dtype=np.float32) + 0.5
            nn2.write_net(net, path)

            self.assertEqual(
                path.stat().st_size,
                nn2.NETWORK_HEADER_SIZE
                + nn2.network_size(16, 8, full_heads=True))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.format_version, 3)
            self.assertTrue(loaded.full_heads)
            self.assertEqual(loaded.l1_weights.shape, (8, nn2.N_L2, nn2.N_L1))
            self.assertEqual(loaded.l2_weights.shape, (8, nn2.N_L3, nn2.N_L2))
            np.testing.assert_array_equal(
                loaded.l1_biases[:, 0], np.arange(8, dtype=np.int32))
            np.testing.assert_array_equal(
                loaded.l2_biases[:, 0], np.arange(8, dtype=np.float32) + 0.5)

    def test_full_head_mixed_skip_v11_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-head-mixed-skip.nn"
            net = zero_net(
                input_buckets=32,
                output_buckets=8,
                format_version=11,
                full_heads=True,
            )
            net.mixed_activation = True
            net.l2_output_skip = True
            net.l2_squared_weights = np.zeros(
                (8, nn2.N_L3, nn2.N_L2), dtype=np.float32)
            net.l2_squared_biases = np.zeros((8, nn2.N_L3), dtype=np.float32)
            net.l2_squared_biases[:, 0] = np.arange(8, dtype=np.float32)
            net.l2_output_skip_weights = np.zeros(
                (8, nn2.N_L2), dtype=np.float32)
            nn2.write_net(net, path)

            self.assertEqual(
                path.stat().st_size,
                nn2.NETWORK_HEADER_SIZE
                + nn2.network_size(32, 8, full_heads=True,
                                   mixed_activation=True, l2_output_skip=True))
            loaded = nn2.load_net(path)
            self.assertEqual(loaded.format_version, 11)
            self.assertTrue(loaded.full_heads)
            self.assertTrue(loaded.mixed_activation)
            self.assertTrue(loaded.l2_output_skip)
            self.assertEqual(loaded.l2_squared_weights.shape, (8, nn2.N_L3, nn2.N_L2))
            np.testing.assert_array_equal(
                loaded.l2_squared_biases[:, 0], np.arange(8, dtype=np.float32))

    def test_single_head_roundtrip_keeps_legacy_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "single.nn"
            nn2.write_net(zero_net(), path)

            self.assertEqual(path.stat().st_size, nn2.network_size(16, 1))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.input_buckets, 16)
            self.assertEqual(loaded.feature_channels, 12)
            self.assertEqual(loaded.output_buckets, 1)
            self.assertEqual(loaded.output_weights.shape, (1, nn2.N_L3))
            self.assertEqual(loaded.output_bias, 0.0)

    def test_output_bucket_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bucketed.nn"
            nn2.write_net(zero_net(output_buckets=4), path)

            self.assertEqual(path.stat().st_size, nn2.network_size(16, 4))
            self.assertEqual(
                nn2.detect_network_layout(path.stat().st_size),
                (16, 12, 4, 0))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.output_buckets, 4)
            self.assertEqual(loaded.output_weights.shape, (4, nn2.N_L3))
            np.testing.assert_array_equal(
                loaded.output_biases,
                np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            )

    def test_material_phase_head_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "material-head.nn"
            nn2.write_net(
                zero_net(output_buckets=4,
                         output_head_features=nn2.N_HEAD_FEATURES),
                path)

            self.assertEqual(
                path.stat().st_size,
                nn2.network_size(16, 4, nn2.N_HEAD_FEATURES))
            self.assertEqual(
                nn2.detect_network_layout(path.stat().st_size),
                (16, 12, 4, nn2.N_HEAD_FEATURES))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.output_head_features, nn2.N_HEAD_FEATURES)
            self.assertEqual(
                loaded.output_weights.shape,
                (4, nn2.N_L3 + nn2.N_HEAD_FEATURES))

    def test_halfka_v2_channel_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "halfka-v2.nn"
            nn2.write_net(
                zero_net(input_buckets=32, feature_channels=11),
                path)

            self.assertEqual(path.stat().st_size, nn2.network_size(32, 1, 0, 11))
            self.assertEqual(
                nn2.detect_network_layout(path.stat().st_size),
                (32, 11, 1, 0))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.input_buckets, 32)
            self.assertEqual(loaded.feature_channels, 11)
            self.assertEqual(
                loaded.input_weights.shape,
                (nn2.feature_count(32, 11), nn2.N_HIDDEN))

    def test_material_count_bucket_matches_bullet_formula(self) -> None:
        self.assertEqual(nn2.output_bucket_for_piece_count(32, 4), 3)
        self.assertEqual(nn2.output_bucket_for_piece_count(24, 4), 2)
        self.assertEqual(nn2.output_bucket_for_piece_count(16, 4), 1)
        self.assertEqual(nn2.output_bucket_for_piece_count(8, 4), 0)
        self.assertEqual(nn2.output_bucket_for_piece_count(32, 1), 0)

    def test_versioned_narrow_hidden_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "narrow.nn"
            nn2.write_net(
                zero_net(
                    input_buckets=10,
                    feature_channels=11,
                    output_buckets=8,
                    trained_hidden=512,
                    format_version=nn2.NETWORK_FORMAT_VERSION,
                ),
                path,
            )

            self.assertEqual(
                path.stat().st_size,
                nn2.NETWORK_HEADER_SIZE + nn2.network_size(10, 8, 0, 11),
            )
            loaded = nn2.load_net(path)
            self.assertEqual(loaded.input_buckets, 10)
            self.assertEqual(loaded.feature_channels, 11)
            self.assertEqual(loaded.output_buckets, 8)
            self.assertEqual(loaded.trained_hidden, 512)
            self.assertEqual(loaded.format_version, nn2.NETWORK_FORMAT_VERSION)
            self.assertEqual(
                loaded.input_weights.shape,
                (nn2.feature_count(10, 11), nn2.N_HIDDEN),
            )

    def test_versioned_full_threats_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-threats.nn"
            nn2.write_net(
                zero_net(
                    input_buckets=16,
                    output_buckets=8,
                    format_version=nn2.NETWORK_FORMAT_VERSION,
                    full_threats=True,
                ),
                path,
            )

            self.assertEqual(
                path.stat().st_size,
                nn2.NETWORK_HEADER_SIZE
                + nn2.network_size(16, 8, 0, 12, True),
            )
            loaded = nn2.load_net(path)
            self.assertTrue(loaded.full_threats)
            self.assertEqual(loaded.output_buckets, 8)
            self.assertEqual(
                loaded.input_weights.shape,
                (nn2.input_feature_count(16, 12, True), nn2.N_HIDDEN),
            )

    def test_l2_output_skip_v8_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l2-output-skip.nn"
            net = zero_net(output_buckets=8, format_version=8)
            net.mixed_activation = True
            net.l2_squared_weights = np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32)
            net.l2_squared_biases = np.zeros(nn2.N_L3, dtype=np.float32)
            net.l2_output_skip = True
            net.l2_output_skip_weights = np.arange(
                8 * nn2.N_L2, dtype=np.float32).reshape(8, nn2.N_L2)
            nn2.write_net(net, path)

            self.assertEqual(
                path.stat().st_size,
                nn2.NETWORK_HEADER_SIZE
                + nn2.network_size(16, 8, mixed_activation=True,
                                   l2_output_skip=True))
            loaded = nn2.load_net(path)
            self.assertEqual(loaded.format_version, 8)
            self.assertTrue(loaded.mixed_activation)
            self.assertTrue(loaded.l2_output_skip)
            np.testing.assert_array_equal(
                loaded.l2_output_skip_weights, net.l2_output_skip_weights)


    def test_full_threats_l2_output_skip_v9_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-threats-l2-output-skip.nn"
            net = zero_net(
                input_buckets=32,
                output_buckets=8,
                format_version=9,
                full_threats=True,
            )
            net.mixed_activation = True
            net.l2_squared_weights = np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32)
            net.l2_squared_biases = np.zeros(nn2.N_L3, dtype=np.float32)
            net.l2_output_skip = True
            net.l2_output_skip_weights = np.arange(
                8 * nn2.N_L2, dtype=np.float32).reshape(8, nn2.N_L2)
            nn2.write_net(net, path)

            loaded = nn2.load_net(path)
            self.assertEqual(loaded.format_version, 9)
            self.assertTrue(loaded.full_threats)
            self.assertTrue(loaded.mixed_activation)
            self.assertTrue(loaded.l2_output_skip)
            np.testing.assert_array_equal(
                loaded.l2_output_skip_weights, net.l2_output_skip_weights)


if __name__ == "__main__":
    unittest.main()
