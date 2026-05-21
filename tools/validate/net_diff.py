#!/usr/bin/env python3
"""Compare exported Enyo NNUE tensors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import load_net


def count_delta(name: str, candidate: np.ndarray, reference: np.ndarray) -> tuple[str, int, int, float]:
    delta = np.asarray(candidate) - np.asarray(reference)
    changed = int(np.count_nonzero(delta))
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return name, changed, int(delta.size), max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--fail-if-identical",
        action="store_true",
        help="Exit 1 when all exported tensors are identical.",
    )
    args = parser.parse_args()

    candidate = load_net(args.candidate)
    reference = load_net(args.reference)

    rows = [
        count_delta("input_weights", candidate.input_weights, reference.input_weights),
        count_delta("input_biases", candidate.input_biases, reference.input_biases),
        count_delta("l1_weights", candidate.l1_weights, reference.l1_weights),
        count_delta("l1_biases", candidate.l1_biases, reference.l1_biases),
        count_delta("l2_weights", candidate.l2_weights, reference.l2_weights),
        count_delta("l2_biases", candidate.l2_biases, reference.l2_biases),
        count_delta("output_weights", candidate.output_weights, reference.output_weights),
    ]
    output_bias_delta = float(candidate.output_bias - reference.output_bias)

    total_changed = sum(changed for _, changed, _, _ in rows)
    total_values = sum(total for _, _, total, _ in rows) + 1
    if output_bias_delta != 0.0:
        total_changed += 1

    for name, changed, total, max_abs in rows:
        print(f"{name:14s} changed={changed}/{total} max_abs={max_abs:g}")
    print(f"{'output_bias':14s} changed={1 if output_bias_delta != 0.0 else 0}/1 max_abs={abs(output_bias_delta):g}")
    print(f"{'total':14s} changed={total_changed}/{total_values}")

    if args.fail_if_identical and total_changed == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
