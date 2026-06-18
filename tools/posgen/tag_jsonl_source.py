#!/usr/bin/env python3
"""Copy JSONL rows while assigning a source_type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-type", required=True)
    ap.add_argument("--progress", type=int, default=100000)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open(encoding="utf-8", errors="replace") as src, \
            args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_type"] = args.source_type
            dst.write(json.dumps(row, separators=(",", ":")))
            dst.write("\n")
            written += 1
            if args.progress > 0 and written % args.progress == 0:
                print(f"tagged {written}", flush=True)
    print(f"wrote {written} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
