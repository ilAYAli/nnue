#!/usr/bin/env python3
"""Compare exported Enyo NNUE tensors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import load_net


def count_delta(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float = 0.0,
) -> tuple[str, int, int, float]:
    delta = np.asarray(candidate) - np.asarray(reference)
    changed = int(np.count_nonzero(np.abs(delta) > atol))
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return name, changed, int(delta.size), max_abs


def optional_delta(name: str, candidate: np.ndarray | None,
                   reference: np.ndarray | None) -> tuple[str, int, int, float]:
    if candidate is None and reference is None:
        return name, 0, 0, 0.0
    if candidate is None:
        candidate = np.zeros_like(reference)
    if reference is None:
        reference = np.zeros_like(candidate)
    return count_delta(name, candidate, reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--fail-if-identical",
        action="store_true",
        help="Exit 1 when all exported tensors are identical.",
    )
    parser.add_argument(
        "--fail-if-different",
        action="store_true",
        help="Exit 1 when any exported tensor differs beyond tolerance.",
    )
    parser.add_argument(
        "--float-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for float tensors.",
    )
    args = parser.parse_args()

    candidate = load_net(args.candidate)
    reference = load_net(args.reference)

    rows = [
        count_delta("input_weights", candidate.input_weights, reference.input_weights),
        count_delta("input_biases", candidate.input_biases, reference.input_biases),
        count_delta("l1_weights", candidate.l1_weights, reference.l1_weights),
        count_delta("l1_biases", candidate.l1_biases, reference.l1_biases),
        count_delta("l2_weights", candidate.l2_weights, reference.l2_weights, atol=args.float_atol),
        count_delta("l2_biases", candidate.l2_biases, reference.l2_biases, atol=args.float_atol),
        count_delta("output_weights", candidate.output_weights, reference.output_weights, atol=args.float_atol),
    ]
    output_bias_row = count_delta(
        "output_bias",
        np.asarray(candidate.output_bias),
        np.asarray(reference.output_bias),
        atol=args.float_atol,
    )
    threat_row = optional_delta(
        "threat_weights",
        candidate.threat_weights,
        reference.threat_weights,
    )

    total_changed = (
        sum(changed for _, changed, _, _ in rows)
        + output_bias_row[1]
        + threat_row[1]
    )
    total_values = (
        sum(total for _, _, total, _ in rows)
        + output_bias_row[2]
        + threat_row[2]
    )

    for name, changed, total, max_abs in rows:
        print(f"{name:14s} changed={changed}/{total} max_abs={max_abs:g}")
    name, changed, total, max_abs = output_bias_row
    print(f"{name:14s} changed={changed}/{total} max_abs={max_abs:g}")
    name, changed, total, max_abs = threat_row
    print(f"{name:14s} changed={changed}/{total} max_abs={max_abs:g}")
    print(f"{'total':14s} changed={total_changed}/{total_values}")

    if args.fail_if_identical and total_changed == 0:
        return 1
    if args.fail_if_different and total_changed != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
