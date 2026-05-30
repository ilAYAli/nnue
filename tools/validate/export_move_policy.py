#!/usr/bin/env python3
"""Export a trained sidecar move-policy checkpoint to JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.move_policy import FEATURE_VERSION  # noqa: E402


SCHEMA = "enyo.move_policy.v1"


def tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=18.0)
    args = ap.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    state = checkpoint["model"]
    payload = {
        "schema": SCHEMA,
        "feature_version": checkpoint.get("feature_version", FEATURE_VERSION),
        "feature_set": checkpoint.get("feature_set", "compact"),
        "input_dim": int(checkpoint["input_dim"]),
        "hidden": int(checkpoint["hidden"]),
        "threshold": float(args.threshold),
        "mean": tensor_list(checkpoint["mean"]),
        "std": tensor_list(checkpoint["std"]),
        "layers": [
            {
                "activation": "relu",
                "weight": tensor_list(state["net.0.weight"]),
                "bias": tensor_list(state["net.0.bias"]),
            },
            {
                "activation": "relu",
                "weight": tensor_list(state["net.2.weight"]),
                "bias": tensor_list(state["net.2.bias"]),
            },
            {
                "activation": "linear",
                "weight": tensor_list(state["net.4.weight"]),
                "bias": tensor_list(state["net.4.bias"]),
            },
        ],
        "config": checkpoint.get("config", {}),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print(
        f"exported {out} input_dim={payload['input_dim']} "
        f"hidden={payload['hidden']} feature_set={payload['feature_set']} "
        f"threshold={payload['threshold']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
