#!/usr/bin/env python3
"""Set policy base_move in child-ranking targets from a move origin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def move_origins(move: dict[str, Any]) -> set[str]:
    origins = set(as_list(move.get("origins")))
    origins.update(as_list(move.get("role")))
    return origins


def find_base_move(row: dict[str, Any], origin: str) -> str:
    for move in row.get("moves", []):
        if origin in move_origins(move):
            return str(move.get("move", ""))
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--base-origin", required=True)
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--keep-missing", action=argparse.BooleanOptionalAction,
                    default=False)
    args = ap.parse_args()

    rows = 0
    written = 0
    stamped = 0
    skipped_missing = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.out.open(
            "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            base_move = find_base_move(row, args.base_origin)
            if not base_move:
                skipped_missing += 1
                if not args.keep_missing:
                    continue
            else:
                row["base_move"] = base_move
                stamped += 1
            dst.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1

    if written < args.min_groups:
        raise SystemExit(
            f"written_groups={written} below min_groups={args.min_groups}")

    summary = (
        f"rows={rows}\n"
        f"written_groups={written}\n"
        f"base_origin={args.base_origin}\n"
        f"base_moves_stamped={stamped}\n"
        f"skipped_missing={skipped_missing}\n"
        f"output={args.out}\n"
    )
    print(summary, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
