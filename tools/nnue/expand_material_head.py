#!/usr/bin/env python3
"""Expand a single-head Enyo .nn into an 8-bucket material-head .nn.

The copied-head output is an exact functional no-op when loaded by an engine
that supports material output buckets. This is a parity/bootstrap tool; train
the buckets only after copied-head parity is proven.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    net = nn2.load_net(args.input.expanduser())
    if net.l2_weights.ndim != 2:
        raise SystemExit(f"{args.input}: already has a bucketed head")

    bucketed = nn2.Net(
        input_weights=net.input_weights,
        input_biases=net.input_biases,
        l1_weights=net.l1_weights,
        l1_biases=net.l1_biases,
        l2_weights=np.repeat(
            net.l2_weights[np.newaxis, :, :],
            nn2.N_OUTPUT_BUCKETS,
            axis=0,
        ),
        l2_biases=np.repeat(
            net.l2_biases[np.newaxis, :],
            nn2.N_OUTPUT_BUCKETS,
            axis=0,
        ),
        output_weights=np.repeat(
            net.output_weights[np.newaxis, :],
            nn2.N_OUTPUT_BUCKETS,
            axis=0,
        ),
        output_bias=np.repeat(
            np.asarray([net.output_bias], dtype=np.float32),
            nn2.N_OUTPUT_BUCKETS,
            axis=0,
        ),
    )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    nn2.write_net(bucketed, args.output.expanduser())
    print(args.output.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
