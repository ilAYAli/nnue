"""PyTorch model matching Enyo NNUE's Berserk-format architecture."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn_pt

from . import enyo_nnue as nn2


class EnyoNNUE(nn_pt.Module):
    def __init__(self, init: str = "kaiming",
                 input_buckets: int = nn2.DEFAULT_N_KING_BUCKETS,
                 feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
                 output_buckets: int = nn2.DEFAULT_N_OUTPUT_BUCKETS,
                 output_head_features: int = nn2.DEFAULT_N_OUTPUT_HEAD_FEATURES,
                 trained_hidden: int = nn2.N_HIDDEN,
                 format_version: int = 1,
                 full_threats: bool = False,
                 slider_xray_threats: bool = False,
                 full_heads: bool = False):
        super().__init__()
        self.input_buckets = input_buckets
        self.feature_channels = feature_channels
        self.output_buckets = output_buckets
        self.output_head_features = output_head_features
        self.trained_hidden = trained_hidden
        self.format_version = format_version
        self.full_threats = full_threats
        self.slider_xray_threats = slider_xray_threats
        self.full_heads = full_heads
        self.embed = nn_pt.EmbeddingBag(
            nn2.input_feature_count(
                input_buckets, feature_channels,
                full_threats or slider_xray_threats),
            nn2.N_HIDDEN,
            mode="sum")
        self.input_bias = nn_pt.Parameter(torch.zeros(nn2.N_HIDDEN))
        head_count = output_buckets if full_heads else 1
        self.l1_weight = nn_pt.Parameter(torch.zeros(
            (head_count, nn2.N_L2, nn2.N_L1)
            if full_heads else (nn2.N_L2, nn2.N_L1)))
        self.l1_bias = nn_pt.Parameter(torch.zeros(
            (head_count, nn2.N_L2) if full_heads else (nn2.N_L2,)))
        if full_heads:
            self.l2 = None
            self.l2_weight = nn_pt.Parameter(torch.zeros(
                head_count, nn2.N_L3, nn2.N_L2))
            self.l2_bias = nn_pt.Parameter(torch.zeros(head_count, nn2.N_L3))
        else:
            self.l2 = nn_pt.Linear(nn2.N_L2, nn2.N_L3)
            self.register_parameter("l2_weight", None)
            self.register_parameter("l2_bias", None)
        self.output = nn_pt.Linear(
            nn2.N_L3 + output_head_features, output_buckets)

        if init == "kaiming":
            nn_pt.init.normal_(self.embed.weight, std=math.sqrt(2.0 / 32.0))
            nn_pt.init.normal_(self.l1_weight, std=math.sqrt(2.0 / nn2.N_L1))
            if self.full_heads:
                nn_pt.init.normal_(self.l2_weight, std=math.sqrt(2.0 / nn2.N_L2))
            else:
                nn_pt.init.normal_(self.l2.weight, std=math.sqrt(2.0 / nn2.N_L2))
            nn_pt.init.normal_(
                self.output.weight,
                std=math.sqrt(2.0 / (nn2.N_L3 + output_head_features)))
        elif init == "berserk-ish":
            nn_pt.init.normal_(self.embed.weight, std=64.0)
            nn_pt.init.normal_(self.l1_weight, std=4.0)
            if self.full_heads:
                nn_pt.init.normal_(self.l2_weight, std=0.02)
            else:
                nn_pt.init.normal_(self.l2.weight, std=0.02)
            nn_pt.init.normal_(self.output.weight, std=0.02)
        else:
            raise ValueError(f"unknown init '{init}'")

    def accumulator(self, feats: torch.Tensor, offsets: torch.Tensor
                    ) -> torch.Tensor:
        return self.embed(feats, offsets) + self.input_bias

    def output_bucket_from_offsets(
        self,
        feats: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self.output_buckets <= 1:
            return torch.zeros_like(offsets)
        counts = torch.empty_like(offsets)
        if len(offsets) > 1:
            counts[:-1] = offsets[1:] - offsets[:-1]
        counts[-1] = feats.numel() - offsets[-1]
        divisor = (32 + self.output_buckets - 1) // self.output_buckets
        return torch.clamp(
            torch.div(counts - 2, divisor, rounding_mode="floor"),
            min=0,
            max=self.output_buckets - 1,
        )

    def output_bucket_from_piece_count(self, piece_count: torch.Tensor) -> torch.Tensor:
        if self.output_buckets <= 1:
            return torch.zeros_like(piece_count)
        divisor = (32 + self.output_buckets - 1) // self.output_buckets
        return torch.clamp(
            torch.div(piece_count - 2, divisor, rounding_mode="floor"),
            min=0,
            max=self.output_buckets - 1,
        )

    @staticmethod
    def material_head_features(
        phase_scale: torch.Tensor,
        piece_count: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack((
            phase_scale - 1.0,
            (piece_count.float() - 16.0) / 16.0,
        ), dim=-1)

    def piece_counts_from_offsets(
        self,
        feats: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        counts = torch.empty_like(offsets)
        if len(offsets) > 1:
            counts[:-1] = offsets[1:] - offsets[:-1]
        counts[-1] = feats.numel() - offsets[-1]
        return counts

    @staticmethod
    def _quantized_input_relu(acc: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(acc, min=0.0, max=float(127 << nn2.QUANT1_BITS))
        scaled = x / float(1 << nn2.QUANT1_BITS)
        floored = torch.floor(scaled)
        return scaled + (floored - scaled).detach()

    def raw_forward(self, w_feats: torch.Tensor, b_feats: torch.Tensor,
                    w_offsets: torch.Tensor, b_offsets: torch.Tensor,
                    stm: torch.Tensor,
                    output_bucket: torch.Tensor | None = None,
                    head_features: torch.Tensor | None = None) -> torch.Tensor:
        w_acc = self.accumulator(w_feats, w_offsets)
        b_acc = self.accumulator(b_feats, b_offsets)

        stm_f = stm.unsqueeze(-1).float()
        us = (1.0 - stm_f) * w_acc + stm_f * b_acc
        them = stm_f * w_acc + (1.0 - stm_f) * b_acc
        acc = torch.cat([us, them], dim=-1)

        x0 = self._quantized_input_relu(acc)
        if output_bucket is None:
            output_bucket = self.output_bucket_from_offsets(w_feats, w_offsets)
        output_bucket = output_bucket.long()
        if self.full_heads:
            l1_weight = self.l1_weight[output_bucket]
            l1_bias = self.l1_bias[output_bucket]
            x1 = torch.relu(
                torch.bmm(l1_weight, x0.unsqueeze(-1)).squeeze(-1) + l1_bias)
            l2_weight = self.l2_weight[output_bucket]
            l2_bias = self.l2_bias[output_bucket]
            x2 = torch.relu(
                torch.bmm(l2_weight, x1.unsqueeze(-1)).squeeze(-1) + l2_bias)
        else:
            x1 = torch.relu(x0 @ self.l1_weight.t() + self.l1_bias)
            x2 = torch.relu(self.l2(x1))
        if self.output_head_features:
            if head_features is None:
                raise ValueError("head_features is required for output-head nets")
            x2 = torch.cat([x2, head_features], dim=-1)
        raw = self.output(x2) / nn2.EVAL_DIVISOR
        if self.output_buckets == 1:
            return raw.squeeze(-1)
        return raw.gather(1, output_bucket.view(-1, 1)).squeeze(-1)

    def forward(self, w_feats: torch.Tensor, b_feats: torch.Tensor,
                w_offsets: torch.Tensor, b_offsets: torch.Tensor,
                stm: torch.Tensor, phase_scale: torch.Tensor,
                output_bucket: torch.Tensor | None = None,
                piece_count: torch.Tensor | None = None,
                ) -> torch.Tensor:
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
        return torch.clamp(raw * phase_scale, min=-2045.0, max=2045.0)


def load_model_from_nn(
    path: str | Path,
    device: str = "cpu",
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
    model = EnyoNNUE(
        init="kaiming",
        input_buckets=net.input_buckets,
        feature_channels=net.feature_channels,
        output_buckets=net.output_buckets,
        output_head_features=output_head_features,
        trained_hidden=net.trained_hidden,
        format_version=net.format_version,
        full_threats=net.full_threats,
        slider_xray_threats=net.slider_xray_threats,
        full_heads=net.full_heads)
    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(net.input_weights.astype(np.float32)))
        model.input_bias.copy_(torch.from_numpy(net.input_biases.astype(np.float32)))
        model.l1_weight.copy_(torch.from_numpy(net.l1_weights.astype(np.float32)))
        model.l1_bias.copy_(torch.from_numpy(net.l1_biases.astype(np.float32)))
        if net.full_heads:
            model.l2_weight.copy_(torch.from_numpy(net.l2_weights.astype(np.float32)))
            model.l2_bias.copy_(torch.from_numpy(net.l2_biases.astype(np.float32)))
        else:
            model.l2.weight.copy_(torch.from_numpy(net.l2_weights.astype(np.float32)))
            model.l2.bias.copy_(torch.from_numpy(net.l2_biases.astype(np.float32)))
        output_weights = np.zeros(
            (net.output_buckets, output_width), dtype=np.float32)
        output_weights[:, :net.output_weights.shape[1]] = net.output_weights
        model.output.weight.copy_(torch.from_numpy(output_weights))
        model.output.bias.copy_(torch.from_numpy(net.output_biases.astype(np.float32)))
    return model.to(device)


def _clip_round(arr: np.ndarray, lo: int, hi: int, name: str) -> np.ndarray:
    rounded = np.rint(arr)
    clipped = np.clip(rounded, lo, hi)
    n_clip = int(np.sum(rounded != clipped))
    if n_clip:
        print(f"warning: {name}: clipped {n_clip} values")
    return clipped


def _tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def export_model(model: EnyoNNUE, path: str | Path) -> None:
    iw = _clip_round(
        _tensor_numpy(model.embed.weight),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
        "input_weights").astype(np.int16)
    ib = _clip_round(
        _tensor_numpy(model.input_bias),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
        "input_biases").astype(np.int16)
    l1w = _clip_round(
        _tensor_numpy(model.l1_weight),
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max,
        "l1_weights").astype(np.int8)
    l1b = _clip_round(
        _tensor_numpy(model.l1_bias),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        "l1_biases").astype(np.int32)
    if model.full_heads:
        l2_weights = _tensor_numpy(model.l2_weight).astype(np.float32)
        l2_biases = _tensor_numpy(model.l2_bias).astype(np.float32)
    else:
        l2_weights = _tensor_numpy(model.l2.weight).astype(np.float32)
        l2_biases = _tensor_numpy(model.l2.bias).astype(np.float32)
    net = nn2.Net(
        input_weights=iw,
        input_biases=ib,
        l1_weights=l1w,
        l1_biases=l1b,
        l2_weights=l2_weights,
        l2_biases=l2_biases,
        output_weights=_tensor_numpy(model.output.weight).astype(np.float32),
        output_biases=_tensor_numpy(model.output.bias).astype(np.float32),
        input_buckets=model.input_buckets,
        feature_channels=model.feature_channels,
        output_buckets=model.output_buckets,
        output_head_features=model.output_head_features,
        trained_hidden=model.trained_hidden,
        format_version=model.format_version,
        full_threats=model.full_threats,
        slider_xray_threats=model.slider_xray_threats,
        full_heads=model.full_heads,
    )
    nn2.write_net(net, path)
