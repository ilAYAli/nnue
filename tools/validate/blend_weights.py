"""Blend a trained net's dense-head/output weights back toward a reference
net's weights, keeping the input embedding at full trained strength.

Usage: blend_weights.py --ref REF.nn --trained TRAINED.nn --alpha A --out OUT.nn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.nnue_forward import load_model_from_nn, export_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                     help="weight on the trained net for l1/l2/output; "
                          "0=pure reference, 1=pure trained")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = load_model_from_nn(args.ref)
    trained = load_model_from_nn(args.trained)
    a = args.alpha

    trained.l1_weight = a * trained.l1_weight + (1 - a) * ref.l1_weight
    trained.l1_bias = a * trained.l1_bias + (1 - a) * ref.l1_bias
    trained.l2_weight = a * trained.l2_weight + (1 - a) * ref.l2_weight
    trained.l2_bias = a * trained.l2_bias + (1 - a) * ref.l2_bias
    trained.output_weight = a * trained.output_weight + (1 - a) * ref.output_weight
    trained.output_bias = a * trained.output_bias + (1 - a) * ref.output_bias
    # input_weights / input_bias stay fully at the trained values (untouched).

    export_model(trained, args.out)
    print(f"wrote {args.out} (alpha={a})")


if __name__ == "__main__":
    main()
