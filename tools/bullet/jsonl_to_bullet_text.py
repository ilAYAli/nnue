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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import bullet_text


def convert(input_path: Path, output_path: Path, *, limit: int,
            max_abs_cp: int, enyo_runtime_target: bool,
            zero_score: bool = False) -> dict[str, int | str]:
    stats: dict[str, int | str] = {
        "input": str(input_path),
        "output": str(output_path),
        "read": 0,
        "written": 0,
        "skipped_cp": 0,
        "skipped_bad": 0,
        "target": "zero-score" if zero_score else (
            "enyo-runtime" if enyo_runtime_target else "raw-cp"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stats["read"] = int(stats["read"]) + 1
            try:
                row = json.loads(line)
                score = bullet_text.white_score_from_row(
                    row,
                    enyo_runtime_target=enyo_runtime_target,
                    zero_score=zero_score,
                )
                if not zero_score and max_abs_cp > 0 and abs(score) > max_abs_cp:
                    stats["skipped_cp"] = int(stats["skipped_cp"]) + 1
                    continue
                text = bullet_text.row_to_text(
                    row,
                    enyo_runtime_target=enyo_runtime_target,
                    zero_score=zero_score,
                )
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                stats["skipped_bad"] = int(stats["skipped_bad"]) + 1
                continue

            dst.write(text + "\n")
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
    parser.add_argument(
        "--enyo-runtime-target",
        action="store_true",
        help=(
            "Pre-divide labels by Enyo's runtime phase scale. Enyo applies "
            "that scale after Propagate(), so Bullet should train the raw "
            "network output against the inverse-scaled target when exporting "
            "to Enyo .nn format."
        ),
    )
    parser.add_argument(
        "--zero-score",
        action="store_true",
        help=(
            "Discard the per-move search score and emit score=0 for every "
            "row, keeping only the game result. For outcome-only lineages "
            "that must not learn from any engine eval as a label."
        ),
    )
    args = parser.parse_args()

    if args.zero_score and args.enyo_runtime_target:
        parser.error("--zero-score and --enyo-runtime-target are mutually exclusive")

    stats = convert(
        args.input.expanduser(),
        args.output.expanduser(),
        limit=args.limit,
        max_abs_cp=args.max_abs_cp,
        enyo_runtime_target=args.enyo_runtime_target,
        zero_score=args.zero_score,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
