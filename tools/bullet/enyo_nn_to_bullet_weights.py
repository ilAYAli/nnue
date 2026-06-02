#!/usr/bin/env python3
"""Convert an exported Enyo .nn into Bullet trainer weights.bin format."""
from __future__ import annotations

import argparse
import json
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
    net.output_weights = output_weights.reshape(1, N_L3)
    net.output_biases = np.asarray([output_bias], dtype=np.float32)
    net.output_buckets = 1
    return net


def expand_output_head(net, output_buckets: int):
    weights = np.asarray(net.output_weights, dtype=np.float32)
    if weights.shape == (N_L3,):
        weights = weights.reshape(1, N_L3)
    biases = np.asarray(net.output_biases, dtype=np.float32).reshape(-1)
    current = int(getattr(net, "output_buckets", weights.shape[0]))
    if current == output_buckets:
        return weights, biases
    if current == 1:
        return np.repeat(weights, output_buckets, axis=0), np.repeat(biases, output_buckets)
    raise SystemExit(
        f"cannot expand {current} output buckets to {output_buckets}; "
        "only single-head init can be repeated")


def bullet_l3_weights(output_weights: np.ndarray) -> np.ndarray:
    # Bullet optimiser-state affine weights use input-major orientation. The
    # saved Enyo .nn format is bucket-major, and the trainer save_format
    # transposes l3w back to that layout during export.
    return np.asarray(output_weights, dtype=np.float32).T


def write_tensor(handle, name: str, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).ravel(order="C")
    handle.write(name.encode("ascii") + b"\n")
    handle.write(struct.pack("<Q", int(flat.size)))
    handle.write(flat.tobytes(order="C"))


def write_metadata(path: Path, args: argparse.Namespace) -> None:
    meta = {
        "kind": "enyo_bullet_weights",
        "source_net": str(args.input.expanduser().resolve()),
        "eval_scale": args.eval_scale,
        "eval_divisor": args.eval_divisor,
        "l1_export_scale": args.l1_export_scale,
        "output_buckets": args.output_buckets,
        "legacy_inputs": bool(args.legacy_inputs),
    }
    (path.parent / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-scale", type=float, default=400.0)
    parser.add_argument("--eval-divisor", type=float, default=32.0)
    parser.add_argument("--l1-export-scale", type=float, default=1.0)
    parser.add_argument("--output-buckets", type=int, default=1, choices=[1, 2, 4, 8])
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
    output_weights, output_biases = expand_output_head(net, args.output_buckets)
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
            bullet_l3_weights(output_weights) / output_scale,
        )
        write_tensor(handle, "l3b", output_biases / output_scale)

    write_metadata(args.output.expanduser(), args)
    print(f"wrote {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
