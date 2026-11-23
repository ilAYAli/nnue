#!/usr/bin/env python3
"""Widen an architecture-26 net's accumulator (e.g. 1024 -> 1280) as an
initialisation for training.

Each perspective's accumulator is two halves whose lanes are multiplied
pairwise (lane j of the first half with lane j of the second) to give fc0's
inputs. The new lanes at the end of each half are copies of that half's first
lanes plus +-1 noise, so they carry realistic activations. Their fc0 columns
are zero, so the widened net evaluates exactly as its source until training
moves those columns.

usage: widen_net.py SOURCE TARGET --hidden 1280 [--name NAME] [--seed 1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enyo_net_v1 as v1  # noqa: E402


def lane_map(narrow: int, wide: int) -> np.ndarray:
    """Source lane of each wide lane: the original lane, or the lane a copy
    duplicates."""
    half_n, half_w = narrow // 2, wide // 2
    extra = half_w - half_n
    if not 0 < extra <= half_n:
        raise ValueError(f"cannot widen {narrow} to {wide}")
    src = np.empty(wide, dtype=np.int64)
    for half in range(2):
        base_w, base_n = half * half_w, half * half_n
        src[base_w:base_w + half_n] = base_n + np.arange(half_n)
        src[base_w + half_n:base_w + half_w] = base_n + np.arange(extra)
    return src


def is_copy(narrow: int, wide: int) -> np.ndarray:
    half_n, half_w = narrow // 2, wide // 2
    lane = np.arange(wide) % half_w
    return lane >= half_n


def widen(net: v1.NetV1, wide: int, seed: int) -> v1.NetV1:
    a = net.arrays
    narrow = net.layout["hidden"]
    src = lane_map(narrow, wide)
    copy = is_copy(narrow, wide)
    rng = np.random.default_rng(seed)

    def lanes(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
        out = values[..., src].copy()
        noise = rng.integers(-1, 2, size=out[..., copy].shape)
        out[..., copy] = np.clip(out[..., copy].astype(np.int64) + noise, lo, hi)
        return out

    out = dict(a)
    out["ft_biases"] = lanes(a["ft_biases"], -32768, 32767).astype(np.int16)
    out["halfka_weights"] = lanes(a["halfka_weights"], -32768, 32767).astype(np.int16)
    out["threat_weights"] = lanes(a["threat_weights"], -128, 127).astype(np.int8)

    # fc0 reads H columns: per perspective the H/2 products of its lane pairs.
    stacks, f0 = net.layout["stacks"], net.layout["fc0_out"]
    old = a["fc0_weights"].reshape(stacks, f0, narrow)
    new = np.zeros((stacks, f0, wide), dtype=np.int8)
    half_n, half_w = narrow // 2, wide // 2
    for p in range(2):
        new[:, :, p * half_w:p * half_w + half_n] = old[:, :, p * half_n:(p + 1) * half_n]
    out["fc0_weights"] = new

    layout = dict(net.layout, hidden=wide)
    return v1.NetV1(net.architecture, net.int_l1_shift, net.int_l2_shift, dict(net.metadata), out, layout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    data = Path(args.source).read_bytes()
    net = v1.parse(data, args.source) if not v1.is_nnue(data) else v1.from_nnue(data)[0]
    wide = widen(net, args.hidden, args.seed)
    wide.metadata["name"] = args.name or Path(args.target).stem
    v1.write(args.target, wide)
    print(f"wrote {args.target}: hidden {net.layout['hidden']} -> {args.hidden}, {wide.metadata}")


if __name__ == "__main__":
    main()
