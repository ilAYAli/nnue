#!/usr/bin/env python3
"""Append zero-initialized threat-feature weights to an Enyo .nn file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nnue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    net = nnue.load_net(Path(args.input).expanduser())
    if net.threat_weights is None:
        net.threat_weights = np.zeros(
            (nnue.N_THREAT_FEATURES, nnue.N_HIDDEN), dtype=np.int8)
    nnue.write_net(net, Path(args.output).expanduser())
    print(f"wrote {Path(args.output).expanduser()}")
    print(f"threat_features={nnue.N_THREAT_FEATURES}")
    print(f"threat_values={nnue.N_THREAT_FEATURES * nnue.N_HIDDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
