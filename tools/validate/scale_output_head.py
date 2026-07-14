#!/usr/bin/env python3
"""Scale an Enyo NNUE final score head without changing feature/dense layers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib import enyo_nnue as nn2


def parse_bucket_scales(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("--bucket-scales must be positive floats")
    return scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scale", type=float)
    ap.add_argument(
        "--bucket-scales",
        type=parse_bucket_scales,
        help="comma-separated per-output-bucket scales",
    )
    ap.add_argument("--scale-psqt", action="store_true")
    args = ap.parse_args()

    if args.scale is None and args.bucket_scales is None:
        raise SystemExit("provide --scale or --bucket-scales")
    if args.scale is not None and args.scale <= 0.0:
        raise SystemExit("--scale must be positive")
    net = nn2.load_net(args.input)
    if args.bucket_scales is not None:
        if len(args.bucket_scales) != net.output_buckets:
            raise SystemExit(
                f"--bucket-scales has {len(args.bucket_scales)} entries, "
                f"but net has {net.output_buckets} output buckets"
            )
        scales = nn2.np.asarray(args.bucket_scales, dtype=nn2.np.float32)
    else:
        scales = nn2.np.full(net.output_buckets, args.scale, dtype=nn2.np.float32)

    net.output_weights *= scales[:, None]
    net.output_biases *= scales
    if args.scale_psqt and net.psqt_residual:
        if net.psqt_weights is not None:
            net.psqt_weights *= scales[None, :]
        if net.psqt_biases is not None:
            net.psqt_biases *= scales
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nn2.write_net(net, args.output)
    print(
        f"wrote {args.output} scales={','.join(f'{scale:g}' for scale in scales)}"
        f" input_buckets={net.input_buckets} feature_channels={net.feature_channels}"
        f" output_buckets={net.output_buckets} psqt_residual={net.psqt_residual}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
