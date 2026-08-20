#!/usr/bin/env python3
"""Reject FullThreats checkpoints whose threat rows collapsed under quantization.

The prior FullThreats attempt (enyo-fullhead-threats-v1-rc1, rejected at
Elo -119.2) exported a net with only 16 nonzero values among 61,243,392
threat weights: the warm-started rows never grew past the int16
quantization floor during training, so the SPRT result reflected a net
with the threat feature effectively absent, not a real test of the
hypothesis. This checks that failure mode directly instead of trusting
Elo alone to catch it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import N_THREAT_FEATURES, Net, feature_count, load_net


def threat_rows(net: Net) -> np.ndarray:
    if not net.full_threats:
        raise SystemExit("net was not exported with full_threats")
    base = feature_count(net.input_buckets, net.feature_channels)
    return net.input_weights[base:base + N_THREAT_FEATURES]


def report(label: str, rows: np.ndarray) -> float:
    nonzero = int(np.count_nonzero(rows))
    total = rows.size
    fraction = nonzero / total
    print(f"{label}_nonzero={nonzero}")
    print(f"{label}_total={total}")
    print(f"{label}_nonzero_fraction={fraction:.8f}")
    print(f"{label}_mean_abs={float(np.abs(rows.astype(np.int32)).mean()):.4f}")
    return fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", required=True, type=Path,
                         help="exported .nn checkpoint after training")
    parser.add_argument("--initial", type=Path,
                         help="exported .nn checkpoint immediately after the "
                              "initialize_from conversion, before training "
                              "(reported only, not required for pass/fail)")
    parser.add_argument("--min-nonzero-fraction", type=float, default=0.01,
                         help="minimum fraction of threat weights that must be "
                              "nonzero after quantization (default 1%%; the "
                              "prior collapse measured ~0.000026%%)")
    args = parser.parse_args()

    trained = load_net(args.trained)
    trained_fraction = report("trained", threat_rows(trained))

    if args.initial is not None:
        initial = load_net(args.initial)
        report("initial", threat_rows(initial))

    if trained_fraction < args.min_nonzero_fraction:
        raise SystemExit(
            "trained threat weights collapsed under quantization: "
            f"{trained_fraction:.6%} nonzero, expected at least "
            f"{args.min_nonzero_fraction:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
