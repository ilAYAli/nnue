#!/usr/bin/env python3
"""Create a full-head warm-start net from an existing shared-scope Enyo net.

Replicates the shared L1/L2 (post-feature-transformer) weights into
`output_buckets` independent copies, one per output bucket, as an identical
warm start - the feature transformer and final output layer (already
per-bucket) are left unchanged. Training then differentiates the per-bucket
L1/L2 copies from this shared starting point.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


def make_full_head(source: Path, output: Path) -> None:
    net = nn2.load_net(source)
    if net.full_heads:
        raise ValueError(f"{source} is already a full-head net")

    l1_weights = np.asarray(net.l1_weights, dtype=np.int8)
    l1_biases = np.asarray(net.l1_biases, dtype=np.int32)
    l2_weights = np.asarray(net.l2_weights, dtype=np.float32)
    l2_biases = np.asarray(net.l2_biases, dtype=np.float32)

    heads = net.output_buckets
    l1_weights_full = np.tile(l1_weights[None, :, :], (heads, 1, 1))
    l1_biases_full = np.tile(l1_biases[None, :], (heads, 1))
    l2_weights_full = np.tile(l2_weights[None, :, :], (heads, 1, 1))
    l2_biases_full = np.tile(l2_biases[None, :], (heads, 1))

    warm = nn2.Net(
        input_weights=np.asarray(net.input_weights, dtype=np.int16).copy(),
        input_biases=np.asarray(net.input_biases, dtype=np.int16).copy(),
        l1_weights=l1_weights_full,
        l1_biases=l1_biases_full,
        l2_weights=l2_weights_full,
        l2_biases=l2_biases_full,
        output_weights=np.asarray(net.output_weights, dtype=np.float32).copy(),
        output_biases=np.asarray(net.output_biases, dtype=np.float32).copy(),
        input_buckets=net.input_buckets,
        feature_channels=net.feature_channels,
        output_buckets=net.output_buckets,
        output_head_features=net.output_head_features,
        trained_hidden=net.trained_hidden,
        format_version=3,
        full_threats=net.full_threats,
        slider_xray_threats=net.slider_xray_threats,
        full_heads=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nn2.write_net(warm, output)
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    make_full_head(args.input.expanduser(), args.output.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
