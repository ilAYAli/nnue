#!/usr/bin/env python3
"""Convert an exported Enyo .nn into Bullet trainer weights.bin format."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import load_net


def write_tensor(handle, name: str, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).ravel(order="C")
    handle.write(name.encode("ascii") + b"\n")
    handle.write(struct.pack("<Q", int(flat.size)))
    handle.write(flat.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-scale", type=float, default=400.0)
    parser.add_argument("--eval-divisor", type=float, default=32.0)
    parser.add_argument("--l1-export-scale", type=float, default=1.0)
    args = parser.parse_args()

    net = load_net(args.input.expanduser())
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)

    output_scale = args.eval_scale * args.eval_divisor
    with args.output.expanduser().open("wb") as handle:
        write_tensor(handle, "l0w", net.input_weights.astype(np.float32))
        write_tensor(handle, "l0b", net.input_biases.astype(np.float32))
        write_tensor(
            handle,
            "l1w",
            np.asarray(net.l1_weights, dtype=np.float32).ravel(order="F"),
        )
        write_tensor(handle, "l1b", net.l1_biases.astype(np.float32) / args.l1_export_scale)
        write_tensor(
            handle,
            "l2w",
            np.asarray(net.l2_weights, dtype=np.float32).ravel(order="F"),
        )
        write_tensor(handle, "l2b", net.l2_biases.astype(np.float32))
        write_tensor(
            handle,
            "l3w",
            np.asarray(net.output_weights, dtype=np.float32).reshape(1, -1) / output_scale,
        )
        write_tensor(handle, "l3b", np.asarray([net.output_bias / output_scale], dtype=np.float32))

    print(f"wrote {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
