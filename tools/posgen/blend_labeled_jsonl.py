#!/usr/bin/env python3
"""Blend a base labeled JSONL with repeated extra labeled rows."""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_lines(path: Path, *, limit: int = 0) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(line)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--extra-repeat", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    base_rows = read_lines(args.base.expanduser(), limit=args.base_limit)
    extra_rows = read_lines(args.extra.expanduser())
    repeat = max(0, args.extra_repeat)
    rows = base_rows + extra_rows * repeat
    if args.shuffle_seed:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)
            handle.write("\n")

    print(f"base_rows={len(base_rows)}")
    print(f"extra_rows={len(extra_rows)}")
    print(f"extra_repeat={repeat}")
    print(f"total_rows={len(rows)}")
    print(f"wrote {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
