"""Blend a trained net's dense-head/output weights back toward a reference
net's weights, keeping the input embedding at full trained strength.

Usage: blend_weights.py --ref REF.nn --trained TRAINED.nn --alpha A --out OUT.nn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lib.nnue_model import load_model_from_nn, export_model


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

    with torch.no_grad():
        trained.l1_weight.copy_(a * trained.l1_weight + (1 - a) * ref.l1_weight)
        trained.l1_bias.copy_(a * trained.l1_bias + (1 - a) * ref.l1_bias)
        trained.l2.weight.copy_(a * trained.l2.weight + (1 - a) * ref.l2.weight)
        trained.l2.bias.copy_(a * trained.l2.bias + (1 - a) * ref.l2.bias)
        trained.output.weight.copy_(a * trained.output.weight + (1 - a) * ref.output.weight)
        trained.output.bias.copy_(a * trained.output.bias + (1 - a) * ref.output.bias)
        # embed.weight / input_bias stay fully at the trained values (untouched).

    export_model(trained, args.out)
    print(f"wrote {args.out} (alpha={a})")


if __name__ == "__main__":
    main()
