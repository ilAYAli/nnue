from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.nnue_model import EnyoNNUE, export_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="PyTorch state_dict .pt")
    ap.add_argument("--out", required=True, help="Output Berserk-format .nn")
    args = ap.parse_args()

    model = EnyoNNUE()
    model.load_state_dict(torch.load(args.state, map_location="cpu"))
    export_model(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
