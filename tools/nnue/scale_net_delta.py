#!/usr/bin/env python3
"""Scale the exported tensor delta between two Enyo .nn files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lib import enyo_nnue as nnue


def scaled_array(base: np.ndarray, candidate: np.ndarray, scale: float) -> np.ndarray:
    if base.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {base.shape} != {candidate.shape}")
    out = base.astype(np.float64) + scale * (
        candidate.astype(np.float64) - base.astype(np.float64)
    )
    if np.issubdtype(base.dtype, np.integer):
        info = np.iinfo(base.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(base.dtype)
    else:
        out = out.astype(base.dtype)
    return out


def scaled_optional(
    base: np.ndarray | None,
    candidate: np.ndarray | None,
    scale: float,
) -> np.ndarray | None:
    if base is None and candidate is None:
        return None
    if base is None or candidate is None:
        raise ValueError("optional tensor exists in only one input net")
    return scaled_array(base, candidate, scale)


def changed_count(candidate: np.ndarray | None, base: np.ndarray | None) -> tuple[int, int]:
    if candidate is None and base is None:
        return 0, 0
    if candidate is None or base is None:
        raise ValueError("optional tensor exists in only one compared net")
    delta = np.asarray(candidate) - np.asarray(base)
    return int(np.count_nonzero(delta)), int(delta.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=float)
    args = parser.parse_args()

    base = nnue.load_net(args.base.expanduser())
    candidate = nnue.load_net(args.candidate.expanduser())

    out = nnue.Net(
        input_weights=scaled_array(base.input_weights, candidate.input_weights, args.scale),
        input_biases=scaled_array(base.input_biases, candidate.input_biases, args.scale),
        l1_weights=scaled_array(base.l1_weights, candidate.l1_weights, args.scale),
        l1_biases=scaled_array(base.l1_biases, candidate.l1_biases, args.scale),
        l2_weights=scaled_array(base.l2_weights, candidate.l2_weights, args.scale),
        l2_biases=scaled_array(base.l2_biases, candidate.l2_biases, args.scale),
        output_weights=scaled_array(
            np.asarray(base.output_weights),
            np.asarray(candidate.output_weights),
            args.scale,
        ),
        output_bias=scaled_array(
            np.asarray(base.output_bias),
            np.asarray(candidate.output_bias),
            args.scale,
        ),
        threat_weights=scaled_optional(
            base.threat_weights,
            candidate.threat_weights,
            args.scale,
        ),
    )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    nnue.write_net(out, args.output.expanduser())
    print(f"wrote {args.output.expanduser()}")
    print(f"scale={args.scale:g}")
    for name in (
        "input_weights", "input_biases", "l1_weights", "l1_biases",
        "l2_weights", "l2_biases", "output_weights",
    ):
        changed, total = changed_count(getattr(out, name), getattr(base, name))
        print(f"{name}: changed={changed}/{total}")
    changed, total = changed_count(np.asarray(out.output_bias), np.asarray(base.output_bias))
    print(f"output_bias: changed={changed}/{total}")
    changed, total = changed_count(out.threat_weights, base.threat_weights)
    print(f"threat_weights: changed={changed}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
