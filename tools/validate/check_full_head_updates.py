#!/usr/bin/env python3
"""Reject full-head checkpoints whose material-specific dense paths did not update."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_weights(path: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        while name := handle.readline():
            count_raw = handle.read(8)
            if len(count_raw) != 8:
                raise ValueError(f"{path}: truncated tensor count for {name!r}")
            count = struct.unpack("<Q", count_raw)[0]
            raw = handle.read(count * 4)
            if len(raw) != count * 4:
                raise ValueError(f"{path}: truncated tensor {name!r}")
            tensors[name.rstrip(b"\n").decode("ascii")] = np.frombuffer(
                raw, dtype=np.float32).copy()
    return tensors


def require_shape(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{name}: got {values.size} values, expected {int(np.prod(shape))}")
    return values.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output-buckets", required=True, type=int)
    parser.add_argument("--expect-frozen-input", action="store_true")
    args = parser.parse_args()

    initial = read_weights(args.initial)
    trained = read_weights(args.trained)
    required = {"l0w", "l0b", "l1w", "l2w", "l3w"}
    missing = sorted(required - initial.keys() | required - trained.keys())
    if missing:
        raise SystemExit(f"missing tensors: {', '.join(missing)}")

    buckets = args.output_buckets
    layers = {
        "l1w": (2048, buckets, 16),
        "l2w": (16, buckets, 32),
        "l3w": (32, buckets),
    }
    for name, shape in layers.items():
        before = require_shape(initial[name], shape, f"initial {name}")
        after = require_shape(trained[name], shape, f"trained {name}")
        for bucket in range(buckets):
            axis = 1
            changed = np.count_nonzero(
                np.take(before, bucket, axis=axis) != np.take(after, bucket, axis=axis))
            print(f"{name}_bucket_{bucket}_changed={changed}")
            if changed == 0:
                raise SystemExit(f"{name} material bucket {bucket} did not update")

    if args.expect_frozen_input:
        for name in ("l0w", "l0b"):
            changed = np.count_nonzero(initial[name] != trained[name])
            print(f"{name}_changed={changed}")
            if changed != 0:
                raise SystemExit(f"{name} changed in dense-head mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
