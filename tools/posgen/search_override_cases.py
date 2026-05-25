#!/usr/bin/env python3
"""Build hard root-move cases from joined model/search diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def int_value(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def include_row(row: dict[str, str], subset: str) -> bool:
    model_top1 = truthy(row["model_top1"])
    engine_equals_model = truthy(row["engine_equals_model"])
    reference_better = int_value(row["diff_cp"]) < 0
    if subset == "model-top1-engine-not-model":
        return model_top1 and not engine_equals_model
    if subset == "model-top1-engine-not-model-reference-better":
        return model_top1 and not engine_equals_model and reference_better
    if subset == "engine-not-model-reference-better":
        return not engine_equals_model and reference_better
    raise ValueError(f"unknown subset {subset}")


def loss_cp(row: dict[str, str]) -> int:
    diff = int_value(row["capped_diff_cp"], int_value(row["diff_cp"]))
    return max(0, -diff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--subset",
        default="model-top1-engine-not-model-reference-better",
        choices=[
            "model-top1-engine-not-model",
            "model-top1-engine-not-model-reference-better",
            "engine-not-model-reference-better",
        ],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-loss-cp", type=int, default=1)
    args = parser.parse_args()

    with args.joined_csv.expanduser().open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if include_row(row, args.subset) and loss_cp(row) >= args.min_loss_cp
        ]

    rows.sort(key=lambda row: (loss_cp(row), int_value(row["model_gap_cp"])), reverse=True)
    if args.limit > 0:
        rows = rows[:args.limit]

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps({
                "id": row["id"],
                "fen": row["fen"],
                "line": index,
                "ply": row["ply"],
                "source": row["source"],
                "severity": args.subset,
                "played": row["engine_move"],
                "best": row["model_move"],
                "loss_cp": loss_cp(row),
                "diff_cp": int_value(row["diff_cp"]),
                "capped_diff_cp": int_value(row["capped_diff_cp"]),
                "engine_gap_cp": int_value(row["gap_cp"]),
                "model_gap_cp": int_value(row["model_gap_cp"]),
                "reference_move": row["reference_move"],
            }, sort_keys=True) + "\n")

    print(f"wrote {len(rows)} cases to {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
