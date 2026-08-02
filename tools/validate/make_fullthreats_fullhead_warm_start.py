#!/usr/bin/env python3
"""Create a combined FullThreats + full-head warm-start net from an existing net.

Applies both transformations in one step:
  - zero-pads new FullThreats input rows (feature transformer gains
    N_THREAT_FEATURES new rows, initialized to zero - matches
    make_full_threats_warm_start.py)
  - replicates the shared L1/L2 (post-FT) weights into `output_buckets`
    independent copies, one per output bucket, as an identical warm start
    (matches make_fullhead_warm_start.py)

The final output layer (already per-bucket regardless of full_heads) is
left unchanged in both respects.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


def expand_output_head(net: nn2.Net, output_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(net.output_weights, dtype=np.float32)
    biases = np.asarray(net.output_biases, dtype=np.float32).reshape(-1)
    if net.output_buckets == output_buckets:
        return weights.copy(), biases.copy()
    if net.output_buckets != 1 or output_buckets < 1:
        raise ValueError(
            f"cannot map output buckets {net.output_buckets} -> {output_buckets}")
    return (
        np.repeat(weights, output_buckets, axis=0),
        np.repeat(biases, output_buckets),
    )


def make_combined(source: Path, output: Path, output_buckets: int | None) -> None:
    net = nn2.load_net(source)
    if net.full_threats:
        raise ValueError(f"{source} is already a FullThreats net")
    if net.full_heads:
        raise ValueError(f"{source} is already a full-head net")

    buckets = output_buckets or net.output_buckets
    output_weights, output_biases = expand_output_head(net, buckets)

    threat_rows = np.zeros((nn2.N_THREAT_FEATURES, nn2.N_HIDDEN), dtype=np.int16)
    input_weights = np.concatenate(
        (np.asarray(net.input_weights, dtype=np.int16), threat_rows), axis=0,
    )

    l1_weights = np.asarray(net.l1_weights, dtype=np.int8)
    l1_biases = np.asarray(net.l1_biases, dtype=np.int32)
    l2_weights = np.asarray(net.l2_weights, dtype=np.float32)
    l2_biases = np.asarray(net.l2_biases, dtype=np.float32)

    l1_weights_full = np.tile(l1_weights[None, :, :], (buckets, 1, 1))
    l1_biases_full = np.tile(l1_biases[None, :], (buckets, 1))
    l2_weights_full = np.tile(l2_weights[None, :, :], (buckets, 1, 1))
    l2_biases_full = np.tile(l2_biases[None, :], (buckets, 1))

    warm = nn2.Net(
        input_weights=input_weights,
        input_biases=np.asarray(net.input_biases, dtype=np.int16).copy(),
        l1_weights=l1_weights_full,
        l1_biases=l1_biases_full,
        l2_weights=l2_weights_full,
        l2_biases=l2_biases_full,
        output_weights=output_weights,
        output_biases=output_biases,
        input_buckets=net.input_buckets,
        feature_channels=net.feature_channels,
        output_buckets=buckets,
        output_head_features=net.output_head_features,
        trained_hidden=net.trained_hidden,
        format_version=6,
        full_threats=True,
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
    parser.add_argument("--output-buckets", type=int, choices=nn2.SUPPORTED_N_OUTPUT_BUCKETS)
    args = parser.parse_args()
    make_combined(args.input.expanduser(), args.output.expanduser(), args.output_buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
