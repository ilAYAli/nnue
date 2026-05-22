#!/usr/bin/env python3
"""Mix selected material-head buckets from one bucketed Enyo net into another."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lib import enyo_nnue as nnue


def parse_buckets(text: str) -> list[int]:
    buckets: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            buckets.update(range(lo, hi + 1))
        else:
            buckets.add(int(part))
    result = sorted(buckets)
    bad = [b for b in result if b < 0 or b >= nnue.N_OUTPUT_BUCKETS]
    if bad:
        raise argparse.ArgumentTypeError(f"bad material buckets: {bad}")
    return result


def require_bucketed(net: nnue.Net, label: str) -> None:
    if np.asarray(net.output_weights).ndim != 2:
        raise SystemExit(f"{label} must be a bucketed-head net")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True,
                        help="Bucketed base net used for all unselected buckets")
    parser.add_argument("--candidate", required=True,
                        help="Bucketed candidate net providing selected buckets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-buckets", required=True,
                        type=parse_buckets,
                        help="Buckets to copy from candidate, e.g. 0-4 or 0,1,2")
    args = parser.parse_args()

    base = nnue.load_net(args.base)
    cand = nnue.load_net(args.candidate)
    require_bucketed(base, "base")
    require_bucketed(cand, "candidate")

    out = nnue.Net(
        input_weights=np.array(base.input_weights, copy=True),
        input_biases=np.array(base.input_biases, copy=True),
        l1_weights=np.array(base.l1_weights, copy=True),
        l1_biases=np.array(base.l1_biases, copy=True),
        l2_weights=np.array(base.l2_weights, copy=True),
        l2_biases=np.array(base.l2_biases, copy=True),
        output_weights=np.array(base.output_weights, copy=True),
        output_bias=np.array(base.output_bias, copy=True),
    )
    for bucket in args.candidate_buckets:
        out.l2_weights[bucket] = cand.l2_weights[bucket]
        out.l2_biases[bucket] = cand.l2_biases[bucket]
        out.output_weights[bucket] = cand.output_weights[bucket]
        out.output_bias[bucket] = cand.output_bias[bucket]

    nnue.write_net(out, args.output)
    print(f"wrote {args.output}")
    print("candidate_buckets=" + ",".join(map(str, args.candidate_buckets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
