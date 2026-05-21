#!/usr/bin/env python3
"""Convert Enyo scored JSONL rows to Bullet's text import format.

Bullet text rows are:

    <FEN> | <white-relative cp score> | <white-relative result>

Enyo rows store `score` and `wdl` side-to-move relative, so black-to-move rows
must be flipped before Bullet conversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def white_result_from_row(row: dict) -> float:
    result = row.get("result")
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5

    wdl = float(row.get("wdl", 0.5))
    stm = row["fen"].split()[1]
    return wdl if stm == "w" else 1.0 - wdl


def convert(input_path: Path, output_path: Path, *, limit: int,
            max_abs_cp: int) -> dict[str, int | str]:
    stats: dict[str, int | str] = {
        "input": str(input_path),
        "output": str(output_path),
        "read": 0,
        "written": 0,
        "skipped_cp": 0,
        "skipped_bad": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stats["read"] = int(stats["read"]) + 1
            try:
                row = json.loads(line)
                fen = str(row["fen"])
                stm = fen.split()[1]
                score = int(round(float(row["score"])))
                if stm == "b":
                    score = -score
                if max_abs_cp > 0 and abs(score) > max_abs_cp:
                    stats["skipped_cp"] = int(stats["skipped_cp"]) + 1
                    continue
                result = white_result_from_row(row)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                stats["skipped_bad"] = int(stats["skipped_bad"]) + 1
                continue

            dst.write(f"{fen} | {score} | {result:.1f}\n")
            stats["written"] = int(stats["written"]) + 1
            if limit > 0 and int(stats["written"]) >= limit:
                break

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Enyo scored JSONL to Bullet text format."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-abs-cp", type=int, default=1600)
    args = parser.parse_args()

    stats = convert(
        args.input.expanduser(),
        args.output.expanduser(),
        limit=args.limit,
        max_abs_cp=args.max_abs_cp,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
