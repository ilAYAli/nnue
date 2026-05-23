#!/usr/bin/env python3
"""Convert an exported Enyo .nn into Bullet trainer weights.bin format."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.enyo_nnue import (
    LEGACY_N_FEATURES,
    LEGACY_NETWORK_SIZE,
    N_HIDDEN,
    N_L1,
    N_L2,
    N_L3,
    load_net,
)


def load_legacy_net_raw(path: Path):
    data = path.read_bytes()
    if len(data) != LEGACY_NETWORK_SIZE:
        raise SystemExit(
            f"{path}: --legacy-inputs requires {LEGACY_NETWORK_SIZE} byte legacy net, got {len(data)}")
    off = 0

    def take(dtype, count: int):
        nonlocal off
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    input_weights = take(np.int16, LEGACY_N_FEATURES * N_HIDDEN).reshape(
        LEGACY_N_FEATURES, N_HIDDEN)
    input_biases = take(np.int16, N_HIDDEN)
    l1_weights = take(np.int8, N_L1 * N_L2).reshape(N_L2, N_L1)
    l1_biases = take(np.int32, N_L2)
    l2_weights = take(np.float32, N_L2 * N_L3).reshape(N_L3, N_L2)
    l2_biases = take(np.float32, N_L3)
    output_weights = take(np.float32, N_L3)
    output_bias = struct.unpack_from("<f", data, off)[0]
    off += 4
    assert off == len(data)

    class RawNet:
        pass

    net = RawNet()
    net.input_weights = input_weights
    net.input_biases = input_biases
    net.l1_weights = l1_weights
    net.l1_biases = l1_biases
    net.l2_weights = l2_weights
    net.l2_biases = l2_biases
    net.output_weights = output_weights
    net.output_bias = output_bias
    return net


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
    parser.add_argument(
        "--legacy-inputs",
        action="store_true",
        help="Keep the 16-king-bucket legacy input tensor instead of expanding to 32 buckets.",
    )
    args = parser.parse_args()

    net = (
        load_legacy_net_raw(args.input.expanduser())
        if args.legacy_inputs
        else load_net(args.input.expanduser())
    )
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
