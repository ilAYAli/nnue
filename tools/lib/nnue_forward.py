"""Numpy inference model matching Enyo NNUE's Berserk-format architecture.

A dependency-free (no torch) reimplementation of the forward pass in
tools/lib/nnue_model.py, for gate/diagnostic scripts that only ever need
zero-gradient inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import enyo_nnue as nn2


class EnyoNNUE:
    def __init__(self, *, input_weights: np.ndarray, input_bias: np.ndarray,
                 l1_weight: np.ndarray, l1_bias: np.ndarray,
                 l2_weight: np.ndarray, l2_bias: np.ndarray,
                 output_weight: np.ndarray, output_bias: np.ndarray,
                 input_buckets: int, feature_channels: int,
                 output_buckets: int, output_head_features: int,
                 trained_hidden: int, format_version: int,
                 full_threats: bool, slider_xray_threats: bool,
                 full_heads: bool, mixed_activation: bool = False,
                 l2_squared_weight: np.ndarray | None = None,
                 l2_squared_bias: np.ndarray | None = None,
                 l2_output_skip_weight: np.ndarray | None = None):
        self.input_weights = input_weights
        self.input_bias = input_bias
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.input_buckets = input_buckets
        self.feature_channels = feature_channels
        self.output_buckets = output_buckets
        self.output_head_features = output_head_features
        self.trained_hidden = trained_hidden
        self.format_version = format_version
        self.full_threats = full_threats
        self.slider_xray_threats = slider_xray_threats
        self.full_heads = full_heads
        self.mixed_activation = mixed_activation
        self.l2_squared_weight = l2_squared_weight
        self.l2_squared_bias = l2_squared_bias
        self.l2_output_skip_weight = l2_output_skip_weight
        if mixed_activation:
            if full_heads:
                raise ValueError("mixed activation does not support full heads")
            if l2_squared_weight is None or l2_squared_bias is None:
                raise ValueError("mixed activation requires the squared branch")
        if l2_output_skip_weight is not None and l2_output_skip_weight.shape != (
                output_buckets, nn2.N_L2):
            raise ValueError("L2-output skip requires [output_buckets,16] weights")

    def accumulator(self, feats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        rows = self.input_weights[feats]
        summed = np.add.reduceat(rows, offsets, axis=0)
        return summed + self.input_bias

    def output_bucket_from_offsets(
        self, feats: np.ndarray, offsets: np.ndarray,
    ) -> np.ndarray:
        if self.output_buckets <= 1:
            return np.zeros_like(offsets)
        counts = np.empty_like(offsets)
        if len(offsets) > 1:
            counts[:-1] = offsets[1:] - offsets[:-1]
        counts[-1] = len(feats) - offsets[-1]
        divisor = (32 + self.output_buckets - 1) // self.output_buckets
        return np.clip((counts - 2) // divisor, 0, self.output_buckets - 1)

    def output_bucket_from_piece_count(self, piece_count: np.ndarray) -> np.ndarray:
        if self.output_buckets <= 1:
            return np.zeros_like(piece_count)
        divisor = (32 + self.output_buckets - 1) // self.output_buckets
        return np.clip((piece_count - 2) // divisor, 0, self.output_buckets - 1)

    @staticmethod
    def material_head_features(
        phase_scale: np.ndarray, piece_count: np.ndarray,
    ) -> np.ndarray:
        return np.stack((
            phase_scale - 1.0,
            (piece_count.astype(np.float32) - 16.0) / 16.0,
        ), axis=-1)

    def piece_counts_from_offsets(
        self, feats: np.ndarray, offsets: np.ndarray,
    ) -> np.ndarray:
        counts = np.empty_like(offsets)
        if len(offsets) > 1:
            counts[:-1] = offsets[1:] - offsets[:-1]
        counts[-1] = len(feats) - offsets[-1]
        return counts

    @staticmethod
    def _quantized_input_relu(acc: np.ndarray) -> np.ndarray:
        x = np.clip(acc, 0.0, float(127 << nn2.QUANT1_BITS))
        scaled = x / float(1 << nn2.QUANT1_BITS)
        # The torch original adds a straight-through-estimator term for
        # gradients only ((floored - scaled).detach()); at inference the
        # forward value is exactly the floor.
        return np.floor(scaled)

    def raw_forward(self, w_feats: np.ndarray, b_feats: np.ndarray,
                     w_offsets: np.ndarray, b_offsets: np.ndarray,
                     stm: np.ndarray,
                     output_bucket: np.ndarray | None = None,
                     head_features: np.ndarray | None = None) -> np.ndarray:
        w_acc = self.accumulator(w_feats, w_offsets)
        b_acc = self.accumulator(b_feats, b_offsets)

        stm_f = stm.astype(np.float32)[:, None]
        us = (1.0 - stm_f) * w_acc + stm_f * b_acc
        them = stm_f * w_acc + (1.0 - stm_f) * b_acc
        acc = np.concatenate([us, them], axis=-1)

        x0 = self._quantized_input_relu(acc)
        if output_bucket is None:
            output_bucket = self.output_bucket_from_offsets(w_feats, w_offsets)
        output_bucket = output_bucket.astype(np.int64)
        if self.full_heads:
            l1_weight = self.l1_weight[output_bucket]
            l1_bias = self.l1_bias[output_bucket]
            x1 = np.maximum(
                np.einsum("bij,bj->bi", l1_weight, x0) + l1_bias, 0.0)
            l2_weight = self.l2_weight[output_bucket]
            l2_bias = self.l2_bias[output_bucket]
            x2 = np.maximum(
                np.einsum("bij,bj->bi", l2_weight, x1) + l2_bias, 0.0)
        else:
            x1_pre = x0 @ self.l1_weight.T + self.l1_bias
            x1 = np.maximum(x1_pre, 0.0)
            x2_pre = x1 @ self.l2_weight.T + self.l2_bias
            if self.mixed_activation:
                squared = np.clip(x1_pre, 0.0, 127.0) ** 2 / 127.0
                x2_pre += (
                    squared @ self.l2_squared_weight.T
                    + self.l2_squared_bias
                )
            x2 = np.maximum(x2_pre, 0.0)
        if self.output_head_features:
            if head_features is None:
                raise ValueError("head_features is required for output-head nets")
            x2 = np.concatenate([x2, head_features], axis=-1)
        raw = x2 @ self.output_weight.T + self.output_bias
        if self.l2_output_skip_weight is not None:
            skip_input = np.clip(x1, 0.0, 127.0) / 127.0
            raw += skip_input @ self.l2_output_skip_weight.T
        raw /= nn2.EVAL_DIVISOR
        if self.output_buckets == 1:
            return raw[:, 0]
        return np.take_along_axis(raw, output_bucket[:, None], axis=1)[:, 0]

    def forward(self, w_feats: np.ndarray, b_feats: np.ndarray,
                w_offsets: np.ndarray, b_offsets: np.ndarray,
                stm: np.ndarray, phase_scale: np.ndarray,
                output_bucket: np.ndarray | None = None,
                piece_count: np.ndarray | None = None) -> np.ndarray:
        head_features = None
        if output_bucket is None and piece_count is not None:
            output_bucket = self.output_bucket_from_piece_count(piece_count)
        if self.output_head_features:
            if piece_count is None:
                piece_count = self.piece_counts_from_offsets(w_feats, w_offsets)
            head_features = self.material_head_features(phase_scale, piece_count)
        raw = self.raw_forward(
            w_feats, b_feats, w_offsets, b_offsets, stm, output_bucket,
            head_features)
        return np.clip(raw * phase_scale, -2045.0, 2045.0)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)


def load_model_from_nn(
    path: str | Path,
    output_head_features: int | None = None,
) -> EnyoNNUE:
    net = nn2.load_net(path)
    if output_head_features is None:
        output_head_features = net.output_head_features
    output_width = nn2.N_L3 + output_head_features
    if net.output_weights.shape[1] > output_width:
        raise ValueError(
            f"{path}: cannot shrink output width "
            f"{net.output_weights.shape[1]} to {output_width}")
    output_weights = np.zeros(
        (net.output_buckets, output_width), dtype=np.float32)
    output_weights[:, :net.output_weights.shape[1]] = net.output_weights
    return EnyoNNUE(
        input_weights=net.input_weights.astype(np.float32),
        input_bias=net.input_biases.astype(np.float32),
        l1_weight=net.l1_weights.astype(np.float32),
        l1_bias=net.l1_biases.astype(np.float32),
        l2_weight=net.l2_weights.astype(np.float32),
        l2_bias=net.l2_biases.astype(np.float32),
        output_weight=output_weights,
        output_bias=net.output_biases.astype(np.float32),
        input_buckets=net.input_buckets,
        feature_channels=net.feature_channels,
        output_buckets=net.output_buckets,
        output_head_features=output_head_features,
        trained_hidden=net.trained_hidden,
        format_version=net.format_version,
        full_threats=net.full_threats,
        slider_xray_threats=net.slider_xray_threats,
        full_heads=net.full_heads,
        mixed_activation=net.mixed_activation,
        l2_squared_weight=(
            None if net.l2_squared_weights is None
            else net.l2_squared_weights.astype(np.float32)
        ),
        l2_squared_bias=(
            None if net.l2_squared_biases is None
            else net.l2_squared_biases.astype(np.float32)
        ),
        l2_output_skip_weight=(
            None if net.l2_output_skip_weights is None
            else net.l2_output_skip_weights.astype(np.float32)
        ),
    )


def _clip_round(arr: np.ndarray, lo: int, hi: int, name: str) -> np.ndarray:
    rounded = np.rint(arr)
    clipped = np.clip(rounded, lo, hi)
    n_clip = int(np.sum(rounded != clipped))
    if n_clip:
        print(f"warning: {name}: clipped {n_clip} values")
    return clipped


def export_model(model: EnyoNNUE, path: str | Path) -> None:
    iw = _clip_round(
        model.input_weights, np.iinfo(np.int16).min, np.iinfo(np.int16).max,
        "input_weights").astype(np.int16)
    ib = _clip_round(
        model.input_bias, np.iinfo(np.int16).min, np.iinfo(np.int16).max,
        "input_biases").astype(np.int16)
    l1w = _clip_round(
        model.l1_weight, np.iinfo(np.int8).min, np.iinfo(np.int8).max,
        "l1_weights").astype(np.int8)
    l1b = _clip_round(
        model.l1_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        "l1_biases").astype(np.int32)
    net = nn2.Net(
        input_weights=iw,
        input_biases=ib,
        l1_weights=l1w,
        l1_biases=l1b,
        l2_weights=model.l2_weight.astype(np.float32),
        l2_biases=model.l2_bias.astype(np.float32),
        output_weights=model.output_weight.astype(np.float32),
        output_biases=model.output_bias.astype(np.float32),
        input_buckets=model.input_buckets,
        feature_channels=model.feature_channels,
        output_buckets=model.output_buckets,
        output_head_features=model.output_head_features,
        trained_hidden=model.trained_hidden,
        format_version=model.format_version,
        full_threats=model.full_threats,
        slider_xray_threats=model.slider_xray_threats,
        full_heads=model.full_heads,
        mixed_activation=model.mixed_activation,
        l2_squared_weights=model.l2_squared_weight,
        l2_squared_biases=model.l2_squared_bias,
        l2_output_skip= model.l2_output_skip_weight is not None,
        l2_output_skip_weights=model.l2_output_skip_weight,
    )
    nn2.write_net(net, path)
