#!/usr/bin/env python3
"""Count BulletFormat ChessBoard records."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import bullet_format  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()
    records = bullet_format.record_count(args.data)
    print(f"{records} {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
