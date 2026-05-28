#!/usr/bin/env python3
"""Export a trained sidecar policy ranker to an engine-loadable artifact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.policy_ranker import export_policy_ranker_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    export_policy_ranker_checkpoint(args.model, args.out, args.threshold)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
