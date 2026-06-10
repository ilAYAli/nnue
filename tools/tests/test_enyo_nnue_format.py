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
             threat_inputs: int = 0) -> nn2.Net:
    features = nn2.feature_count(input_buckets, feature_channels)
    threat_weights = threat_biases = threat_output_weights = threat_output_bias = None
    if threat_inputs:
        threat_weights = np.zeros(
            (nn2.N_THREAT_FEATURES, nn2.N_THREAT_HIDDEN), dtype=np.int16)
        threat_biases = np.arange(nn2.N_THREAT_HIDDEN, dtype=np.int16)
        threat_output_weights = np.linspace(
            -0.25, 0.25, 2 * nn2.N_THREAT_HIDDEN, dtype=np.float32)
        threat_output_bias = np.float32(1.25)
    return nn2.Net(
        input_weights=np.zeros((features, nn2.N_HIDDEN), dtype=np.int16),
        input_biases=np.zeros((nn2.N_HIDDEN,), dtype=np.int16),
        l1_weights=np.zeros((nn2.N_L2, nn2.N_L1), dtype=np.int8),
        l1_biases=np.zeros((nn2.N_L2,), dtype=np.int32),
        l2_weights=np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32),
        l2_biases=np.zeros((nn2.N_L3,), dtype=np.float32),
        output_weights=np.zeros(
            (output_buckets, nn2.N_L3 + output_head_features),
            dtype=np.float32),
        output_biases=np.arange(output_buckets, dtype=np.float32),
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        output_head_features=output_head_features,
        threat_inputs=threat_inputs,
        threat_weights=threat_weights,
        threat_biases=threat_biases,
        threat_output_weights=threat_output_weights,
        threat_output_bias=threat_output_bias,
    )


class EnyoNNUEFormatTests(unittest.TestCase):
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
                (16, 12, 4, 0, 0))
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
                (16, 12, 4, nn2.N_HEAD_FEATURES, 0))
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
                (32, 11, 1, 0, 0))
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.input_buckets, 32)
            self.assertEqual(loaded.feature_channels, 11)
            self.assertEqual(
                loaded.input_weights.shape,
                (nn2.feature_count(32, 11), nn2.N_HIDDEN))

    def test_threat_input_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threat.nn"
            nn2.write_net(
                zero_net(output_buckets=4,
                         output_head_features=nn2.N_HEAD_FEATURES,
                         threat_inputs=1),
                path)

            self.assertEqual(
                path.stat().st_size,
                nn2.network_size(16, 4, nn2.N_HEAD_FEATURES, 12, 1))
            self.assertEqual(
                nn2.detect_network_layout(path.stat().st_size),
                (16, 12, 4, nn2.N_HEAD_FEATURES, 1))
            self.assertEqual(nn2.detect_threat_inputs(path.stat().st_size), 1)
            loaded = nn2.load_net(path)

            self.assertEqual(loaded.threat_inputs, 1)
            self.assertIsNotNone(loaded.threat_weights)
            self.assertIsNotNone(loaded.threat_biases)
            self.assertIsNotNone(loaded.threat_output_weights)
            self.assertEqual(loaded.threat_weights.shape,
                             (nn2.N_THREAT_FEATURES, nn2.N_THREAT_HIDDEN))
            np.testing.assert_array_equal(
                loaded.threat_biases,
                np.arange(nn2.N_THREAT_HIDDEN, dtype=np.int16))
            np.testing.assert_allclose(float(loaded.threat_output_bias), 1.25)

    def test_material_count_bucket_matches_bullet_formula(self) -> None:
        self.assertEqual(nn2.output_bucket_for_piece_count(32, 4), 3)
        self.assertEqual(nn2.output_bucket_for_piece_count(24, 4), 2)
        self.assertEqual(nn2.output_bucket_for_piece_count(16, 4), 1)
        self.assertEqual(nn2.output_bucket_for_piece_count(8, 4), 0)
        self.assertEqual(nn2.output_bucket_for_piece_count(32, 1), 0)

    def test_supported_layout_sizes_do_not_collide(self) -> None:
        seen: dict[int, tuple[int, int, int, int, int]] = {}
        for input_buckets, feature_channels in nn2.SUPPORTED_N_FEATURE_LAYOUTS:
            for output_buckets in nn2.SUPPORTED_N_OUTPUT_BUCKETS:
                for output_head_features in nn2.SUPPORTED_N_OUTPUT_HEAD_FEATURES:
                    for threat_inputs in nn2.SUPPORTED_N_THREAT_INPUTS:
                        layout = (
                            input_buckets,
                            feature_channels,
                            output_buckets,
                            output_head_features,
                            threat_inputs,
                        )
                        size = nn2.network_size(
                            input_buckets,
                            output_buckets,
                            output_head_features,
                            feature_channels,
                            threat_inputs,
                        )
                        self.assertNotIn(size, seen)
                        seen[size] = layout
                        self.assertEqual(nn2.detect_network_layout(size), layout)


if __name__ == "__main__":
    unittest.main()
