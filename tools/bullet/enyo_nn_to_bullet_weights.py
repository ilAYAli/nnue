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
    DEFAULT_N_FEATURE_CHANNELS,
    HALFKA_V2_FEATURE_CHANNELS,
    KING_BUCKETS_32,
    LEGACY_N_FEATURES,
    LEGACY_NETWORK_SIZE,
    N_THREAT_FEATURES,
    N_HIDDEN,
    N_L1,
    N_L2,
    N_L3,
    load_net,
)

N_PAWN_PAIR_FEATURES = 4_560

LEGACY_BUCKET_FOR_32 = (
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 8, 9, 10, 11,
    12, 12, 13, 13, 12, 12, 13, 13,
    14, 14, 15, 15, 14, 14, 15, 15,
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
    net.input_buckets = 16
    net.feature_channels = 12
    net.output_buckets = 1
    return net


def source_bucket_for_target(target_bucket: int, source_buckets: int, target_buckets: int) -> int:
    if source_buckets == target_buckets:
        return target_bucket
    if source_buckets == 1:
        return 0
    if source_buckets == 16 and target_buckets == 32:
        return LEGACY_BUCKET_FOR_32[target_bucket]
    if target_buckets % source_buckets == 0:
        return target_bucket * source_buckets // target_buckets
    raise SystemExit(
        f"cannot map input buckets {source_buckets} -> {target_buckets}")


def source_channel_for_target(
    target_channel: int,
    target_square: int,
    target_bucket: int,
    source_channels: int,
    target_channels: int,
) -> int:
    if source_channels == target_channels:
        return target_channel
    if source_channels == DEFAULT_N_FEATURE_CHANNELS and target_channels == HALFKA_V2_FEATURE_CHANNELS:
        if target_channel < 5:
            return target_channel
        if target_channel < 10:
            return target_channel + 1
        king_squares = {
            square for square, bucket in enumerate(KING_BUCKETS_32)
            if bucket == target_bucket
        }
        return 5 if target_square in king_squares else 11
    raise SystemExit(
        f"cannot map feature channels {source_channels} -> {target_channels}")


def convert_input_weights(
    input_weights: np.ndarray,
    *,
    source_buckets: int,
    source_channels: int,
    target_buckets: int,
    target_channels: int,
) -> np.ndarray:
    if source_buckets == target_buckets and source_channels == target_channels:
        return np.asarray(input_weights, dtype=np.float32)

    target = np.empty(
        (target_buckets * target_channels * 64, N_HIDDEN),
        dtype=np.float32)
    for target_bucket in range(target_buckets):
        source_bucket = source_bucket_for_target(
            target_bucket, source_buckets, target_buckets)
        for target_channel in range(target_channels):
            for target_square in range(64):
                source_channel = source_channel_for_target(
                    target_channel,
                    target_square,
                    target_bucket,
                    source_channels,
                    target_channels)
                source_feature = (
                    source_bucket * source_channels * 64
                    + source_channel * 64
                    + target_square)
                target_feature = (
                    target_bucket * target_channels * 64
                    + target_channel * 64
                    + target_square)
                target[target_feature] = input_weights[source_feature]
    return target


def add_full_threat_rows(
        input_weights: np.ndarray,
        *,
        rows: int = N_THREAT_FEATURES,
        seed: int = 0,
        scale: float = 0.1) -> np.ndarray:
    """Append deterministic, export-visible FullThreat rows.

    A zero warm start leaves the new rows far below input quantization after a
    continuation run.  Match each hidden column's parent scale at a conservative
    fraction instead, so threats begin measurable without disturbing the parent.
    """
    parent = np.asarray(input_weights, dtype=np.float32)
    column_std = parent.std(axis=0, dtype=np.float64).astype(np.float32)
    rng = np.random.default_rng(seed)
    threats = (
        rng.standard_normal((rows, parent.shape[1])).astype(np.float32)
        * column_std[np.newaxis, :]
        * scale
    )
    return np.concatenate((parent, threats), axis=0)


def source_output_bucket_for_target(target_bucket: int, source_buckets: int, target_buckets: int) -> int:
    if source_buckets == target_buckets:
        return target_bucket
    if source_buckets == 1:
        return 0
    if target_buckets > source_buckets and target_buckets % source_buckets == 0:
        return target_bucket * source_buckets // target_buckets
    raise SystemExit(
        f"cannot map output buckets {source_buckets} -> {target_buckets}")


def expand_output_head(net, output_buckets: int):
    weights = np.asarray(net.output_weights, dtype=np.float32)
    if weights.shape == (N_L3,):
        weights = weights.reshape(1, N_L3)
    biases = np.asarray(net.output_biases, dtype=np.float32).reshape(-1)
    current = int(getattr(net, "output_buckets", weights.shape[0]))
    if current == output_buckets:
        return weights, biases

    target_weights = np.empty((output_buckets, N_L3), dtype=np.float32)
    target_biases = np.empty(output_buckets, dtype=np.float32)
    for target_bucket in range(output_buckets):
        source_bucket = source_output_bucket_for_target(target_bucket, current, output_buckets)
        target_weights[target_bucket] = weights[source_bucket]
        target_biases[target_bucket] = biases[source_bucket]
    return target_weights, target_biases


def bullet_l3_weights(output_weights: np.ndarray) -> np.ndarray:
    # Bullet optimiser-state affine weights use input-major orientation. The
    # saved Enyo .nn format is bucket-major, and the trainer save_format
    # transposes l3w back to that layout during export.
    return np.asarray(output_weights, dtype=np.float32).T


def mixed_activation_weights(
    net,
    output_buckets: int,
    full_heads: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve a SCReLU parent branch, or add a zero branch to a ReLU parent."""
    source_weights = getattr(net, "l2_squared_weights", None)
    source_biases = getattr(net, "l2_squared_biases", None)
    target_heads = output_buckets if full_heads else 1
    if source_weights is None and source_biases is None:
        weights = np.zeros((N_L3, N_L2), dtype=np.float32)
        biases = np.zeros(N_L3, dtype=np.float32)
    else:
        if source_weights is None or source_biases is None:
            raise SystemExit("source SCReLU branch is incomplete")
        weights = np.asarray(source_weights, dtype=np.float32)
        biases = np.asarray(source_biases, dtype=np.float32)
        if weights.shape == (N_L3, N_L2) and biases.shape == (N_L3,):
            pass
        elif weights.ndim == 3 and weights.shape[1:] == (N_L3, N_L2) \
                and biases.shape == (weights.shape[0], N_L3):
            if not full_heads:
                raise SystemExit("cannot collapse full SCReLU heads into a shared branch")
            indices = [
                source_output_bucket_for_target(bucket, weights.shape[0], output_buckets)
                for bucket in range(output_buckets)
            ]
            return weights[indices], biases[indices]
        else:
            raise SystemExit("source SCReLU branch has incompatible shape")
    if full_heads:
        return (
            np.repeat(weights[np.newaxis, ...], target_heads, axis=0),
            np.repeat(biases[np.newaxis, ...], target_heads, axis=0),
        )
    return weights, biases


def expand_dense_heads(net, output_buckets: int, full_heads: bool):
    l1_weights = np.asarray(net.l1_weights, dtype=np.float32)
    l1_biases = np.asarray(net.l1_biases, dtype=np.float32)
    l2_weights = np.asarray(net.l2_weights, dtype=np.float32)
    l2_biases = np.asarray(net.l2_biases, dtype=np.float32)
    source_full_heads = bool(getattr(net, "full_heads", False))

    if source_full_heads:
        if not full_heads:
            raise SystemExit("cannot collapse full material heads into a shared dense head")
        source_heads = int(getattr(net, "output_buckets", l1_weights.shape[0]))
        indices = [
            source_output_bucket_for_target(bucket, source_heads, output_buckets)
            for bucket in range(output_buckets)
        ]
        return (
            l1_weights[indices],
            l1_biases[indices],
            l2_weights[indices],
            l2_biases[indices],
        )

    if l1_weights.shape != (N_L2, N_L1):
        raise SystemExit(f"unexpected shared L1 weight shape: {l1_weights.shape}")
    if l1_biases.shape != (N_L2,):
        raise SystemExit(f"unexpected shared L1 bias shape: {l1_biases.shape}")
    if l2_weights.shape != (N_L3, N_L2):
        raise SystemExit(f"unexpected shared L2 weight shape: {l2_weights.shape}")
    if l2_biases.shape != (N_L3,):
        raise SystemExit(f"unexpected shared L2 bias shape: {l2_biases.shape}")
    if not full_heads:
        return l1_weights, l1_biases, l2_weights, l2_biases
    return (
        np.repeat(l1_weights[np.newaxis, ...], output_buckets, axis=0),
        np.repeat(l1_biases[np.newaxis, ...], output_buckets, axis=0),
        np.repeat(l2_weights[np.newaxis, ...], output_buckets, axis=0),
        np.repeat(l2_biases[np.newaxis, ...], output_buckets, axis=0),
    )


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
        "target_input_buckets": args.input_buckets,
        "target_feature_channels": args.feature_channels,
        "target_hidden": args.hidden,
        "full_threats": bool(args.full_threats),
        "slider_xray_threats": bool(args.slider_xray_threats),
        "pawn_pairs": bool(args.pawn_pairs),
        "full_heads": bool(args.full_heads),
        "mixed_activation": bool(args.mixed_activation),
        "l2_output_skip": bool(args.l2_output_skip),
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
    parser.add_argument("--input-buckets", type=int, default=0,
                        help="Target Enyo input buckets; 0 keeps source layout.")
    parser.add_argument("--feature-channels", type=int, default=0, choices=[0, 11, 12],
                        help="Target feature channels; 0 keeps source layout.")
    parser.add_argument("--hidden", type=int, default=0, choices=[0, 512, 768, 1024],
                        help="Target hidden width; 0 keeps source trained width.")
    parser.add_argument(
        "--full-threats",
        action="store_true",
        help="Append zero-initialized FullThreats rows for warm-starting that architecture.",
    )
    parser.add_argument(
        "--slider-xray-threats",
        action="store_true",
        help="Append zero-initialized slider x-ray rows for warm-starting that architecture.",
    )
    parser.add_argument(
        "--pawn-pairs",
        action="store_true",
        help="Append deterministic pawn-pair rows for a warm-started architecture.",
    )
    parser.add_argument(
        "--full-heads",
        action="store_true",
        help="Duplicate a shared native dense head into every output bucket.",
    )
    parser.add_argument(
        "--mixed-activation",
        action="store_true",
        help="Append a zero-initialized squared-activation residual affine.",
    )
    parser.add_argument(
        "--psqt-residual",
        action="store_true",
        help="Append a zero-initialized material-bucketed PSQT residual table.",
    )
    parser.add_argument(
        "--l2-output-skip",
        action="store_true",
        help="Append zero-initialized activated L2-to-output skip weights.",
    )
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
    source_buckets = int(getattr(net, "input_buckets", 16))
    source_channels = int(getattr(net, "feature_channels", 12))
    args.input_buckets = args.input_buckets or source_buckets
    args.feature_channels = args.feature_channels or source_channels
    args.hidden = args.hidden or int(getattr(net, "trained_hidden", N_HIDDEN))
    if args.feature_channels == 11 and args.input_buckets not in (10, 16, 32):
        raise SystemExit("--feature-channels 11 requires --input-buckets 10, 16, or 32")
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)

    output_scale = args.eval_scale * args.eval_divisor
    output_weights, output_biases = expand_output_head(net, args.output_buckets)
    l1_weights, l1_biases, l2_weights, l2_biases = expand_dense_heads(
        net, args.output_buckets, args.full_heads)
    input_weights = convert_input_weights(
        net.input_weights,
        source_buckets=source_buckets,
        source_channels=source_channels,
        target_buckets=args.input_buckets,
        target_channels=args.feature_channels,
    )[:, :args.hidden]
    if sum((args.full_threats, args.slider_xray_threats, args.pawn_pairs)) > 1:
        raise SystemExit("extended input modes are mutually exclusive")
    source_threat_features = (
        bool(getattr(net, "full_threats", False))
        or bool(getattr(net, "slider_xray_threats", False))
    )
    if (args.full_threats or args.slider_xray_threats) and not source_threat_features:
        input_weights = add_full_threat_rows(input_weights)
    if args.pawn_pairs:
        if source_threat_features:
            raise SystemExit("pawn-pair warm start requires a base-input parent")
        input_weights = add_full_threat_rows(
            input_weights, rows=N_PAWN_PAIR_FEATURES, seed=1)
    input_biases = np.asarray(net.input_biases, dtype=np.float32)[:args.hidden]
    l1_weights = np.concatenate((
        l1_weights[..., :args.hidden],
        l1_weights[..., N_HIDDEN:N_HIDDEN + args.hidden],
    ), axis=-1)
    l1_output_rows = args.output_buckets * N_L2 if args.full_heads else N_L2
    l2_output_rows = args.output_buckets * N_L3 if args.full_heads else N_L3
    l1_weights = l1_weights.reshape(l1_output_rows, 2 * args.hidden)
    l1_biases = l1_biases.reshape(l1_output_rows)
    l2_weights = l2_weights.reshape(l2_output_rows, N_L2)
    l2_biases = l2_biases.reshape(l2_output_rows)
    with args.output.expanduser().open("wb") as handle:
        write_tensor(handle, "l0w", input_weights)
        write_tensor(handle, "l0b", input_biases)
        write_tensor(
            handle,
            "l1w",
            l1_weights.ravel(order="F"),
        )
        write_tensor(handle, "l1b", l1_biases / args.l1_export_scale)
        write_tensor(
            handle,
            "l2w",
            l2_weights.ravel(order="F"),
        )
        write_tensor(handle, "l2b", l2_biases)
        if args.mixed_activation:
            l2_squared_weights, l2_squared_biases = mixed_activation_weights(
                net, args.output_buckets, args.full_heads)
            l2_squared_weights = l2_squared_weights.reshape(
                l2_output_rows, N_L2)
            l2_squared_biases = l2_squared_biases.reshape(l2_output_rows)
            write_tensor(handle, "l2sw", l2_squared_weights.ravel(order="F"))
            write_tensor(handle, "l2sb", l2_squared_biases)
        write_tensor(
            handle,
            "l3w",
            bullet_l3_weights(output_weights) / output_scale,
        )
        write_tensor(handle, "l3b", output_biases / output_scale)
        if args.l2_output_skip:
            source_skip = getattr(net, "l2_output_skip_weights", None)
            l2_output_skip = (
                np.asarray(source_skip, dtype=np.float32)
                if source_skip is not None
                else np.zeros((args.output_buckets, N_L2), dtype=np.float32)
            )
            if l2_output_skip.shape != (args.output_buckets, N_L2):
                raise SystemExit("source L2-output skip has incompatible shape")
            write_tensor(
                handle,
                "l2skipw",
                bullet_l3_weights(l2_output_skip) / output_scale,
            )
        if args.psqt_residual:
            write_tensor(
                handle,
                "psqtw",
                np.zeros((args.input_buckets * args.feature_channels * 64,
                          args.output_buckets), dtype=np.float32),
            )
            write_tensor(handle, "psqtb", np.zeros(args.output_buckets, dtype=np.float32))

    write_metadata(args.output.expanduser(), args)
    print(f"wrote {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
