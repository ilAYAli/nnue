#!/usr/bin/env python3
"""Scale the tensor delta between two Enyo .nn files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lib import enyo_nnue as nnue


def scaled_int(base: np.ndarray, cand: np.ndarray, scale: float, dtype) -> np.ndarray:
    values = np.rint(base.astype(np.float64) + scale * (
        cand.astype(np.float64) - base.astype(np.float64)))
    info = np.iinfo(dtype)
    return np.clip(values, info.min, info.max).astype(dtype)


def scaled_float(base: np.ndarray, cand: np.ndarray, scale: float) -> np.ndarray:
    return (base.astype(np.float32) + np.float32(scale) * (
        cand.astype(np.float32) - base.astype(np.float32))).astype(np.float32)


def require_shape(label: str, base, cand) -> None:
    if np.asarray(base).shape != np.asarray(cand).shape:
        raise SystemExit(
            f"{label}: shape mismatch {np.asarray(base).shape} != "
            f"{np.asarray(cand).shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base/reference .nn")
    parser.add_argument("--candidate", required=True, help="Candidate .nn")
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = nnue.load_net(args.base)
    cand = nnue.load_net(args.candidate)

    for label in (
        "input_weights", "input_biases", "l1_weights", "l1_biases",
        "l2_weights", "l2_biases", "output_weights", "output_bias",
    ):
        require_shape(label, getattr(base, label), getattr(cand, label))

    if (base.threat_weights is None) != (cand.threat_weights is None):
        raise SystemExit("threat layout mismatch")
    if base.threat_weights is not None:
        require_shape("threat_weights", base.threat_weights, cand.threat_weights)

    out = nnue.Net(
        input_weights=scaled_int(base.input_weights, cand.input_weights,
                                 args.scale, np.int16),
        input_biases=scaled_int(base.input_biases, cand.input_biases,
                                args.scale, np.int16),
        l1_weights=scaled_int(base.l1_weights, cand.l1_weights,
                              args.scale, np.int8),
        l1_biases=scaled_int(base.l1_biases, cand.l1_biases,
                             args.scale, np.int32),
        l2_weights=scaled_float(base.l2_weights, cand.l2_weights, args.scale),
        l2_biases=scaled_float(base.l2_biases, cand.l2_biases, args.scale),
        output_weights=scaled_float(
            base.output_weights, cand.output_weights, args.scale),
        output_bias=scaled_float(
            np.asarray(base.output_bias), np.asarray(cand.output_bias), args.scale),
        threat_weights=(
            scaled_int(base.threat_weights, cand.threat_weights, args.scale, np.int8)
            if base.threat_weights is not None else None
        ),
    )
    nnue.write_net(out, args.output)
    print(f"wrote {args.output}")
    print(f"scale={args.scale:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
