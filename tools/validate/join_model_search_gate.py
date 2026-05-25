#!/usr/bin/env python3
"""Join exported-model target choices with engine search-gate choices."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def truthy(row: dict[str, object], key: str) -> bool:
    return str(row[key]).lower() in {"1", "true", "yes"}


def median_nonzero(values: list[int]) -> float:
    nonzero = [value for value in values if value != 0]
    return float(median(nonzero)) if nonzero else 0.0


def summarize(name: str, rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return [f"[{name}]", "n=0"]
    diffs = [int(row["diff_cp"]) for row in rows]
    capped = [int(row["capped_diff_cp"]) for row in rows]
    gaps = [int(float(row["gap_cp"])) for row in rows]
    return [
        f"[{name}]",
        f"n={len(rows)}",
        f"engine_top1={sum(truthy(row, 'engine_top1') for row in rows)}",
        f"model_top1={sum(truthy(row, 'model_top1') for row in rows)}",
        f"engine_equals_model="
        f"{sum(truthy(row, 'engine_equals_model') for row in rows)}",
        f"candidate_better={sum(int(row['diff_cp']) > 0 for row in rows)}",
        f"reference_better={sum(int(row['diff_cp']) < 0 for row in rows)}",
        f"capped_sum_diff_cp={sum(capped)}",
        f"median_nonzero_diff_cp={median_nonzero(diffs):.1f}",
        f"median_engine_gap_cp={median(gaps)}",
        f"worst_regression_cp={min(diffs)}",
        f"best_gain_cp={max(diffs)}",
    ]


def load_targets(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            targets[str(row["id"])] = line
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--compare-csv", required=True, type=Path)
    parser.add_argument("--targets-jsonl", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--write-model-top1-overrides",
        action="store_true",
        help="Write JSONL targets where model top1 is not selected by search.",
    )
    args = parser.parse_args()

    model = {row["id"]: row for row in read_csv(args.model_csv)}
    candidate = {row["id"]: row for row in read_csv(args.candidate_csv)}
    compare = {row["id"]: row for row in read_csv(args.compare_csv)}

    joined: list[dict[str, object]] = []
    for target_id, cand in candidate.items():
        mod = model[target_id]
        comp = compare[target_id]
        row: dict[str, object] = dict(cand)
        row.update({
            "model_move": mod["model_move"],
            "model_top1": mod["model_top1"],
            "model_top3": mod["model_top3"],
            "model_gap_cp": mod["model_gap_cp"],
            "engine_equals_model": str(cand["engine_move"] == mod["model_move"]),
            "engine_top1": str(int(cand["rank"]) == 1),
            "engine_top3": str(int(cand["rank"]) <= 3),
            "reference_move": comp["reference_move"],
            "reference_rank": comp["reference_rank"],
            "reference_gap_cp": comp["reference_gap_cp"],
            "diff_cp": comp["diff_cp"],
            "capped_diff_cp": comp["capped_diff_cp"],
        })
        joined.append(row)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "joined.csv", joined)

    subsets = {
        "all": joined,
        "engine_equals_model": [
            row for row in joined if truthy(row, "engine_equals_model")
        ],
        "engine_not_model": [
            row for row in joined if not truthy(row, "engine_equals_model")
        ],
        "model_top1": [row for row in joined if truthy(row, "model_top1")],
        "model_top1_engine_not_model": [
            row for row in joined
            if truthy(row, "model_top1") and not truthy(row, "engine_equals_model")
        ],
        "model_not_top1_engine_top1": [
            row for row in joined
            if not truthy(row, "model_top1") and truthy(row, "engine_top1")
        ],
        "mate_like": [
            row for row in joined if "mate_like" in str(row["tags"]).split(";")
        ],
        "non_mate": [
            row for row in joined if "non_mate" in str(row["tags"]).split(";")
        ],
    }
    lines: list[str] = []
    for name, rows in subsets.items():
        lines.extend(summarize(name, rows))
        lines.append("")

    losses = [row for row in joined if int(row["diff_cp"]) < 0]
    by_source: dict[tuple[str, str], list[int]] = {}
    for row in losses:
        key = (str(row.get("source", "")), str(row.get("log", "")))
        stats = by_source.setdefault(key, [0, 0])
        stats[0] += 1
        stats[1] += int(row["capped_diff_cp"])
    lines.append("[reference_better_by_source_log_top10]")
    for (source, log), (count, capsum) in sorted(
        by_source.items(), key=lambda item: item[1][1])[:10]:
        lines.append(f"{count}\t{capsum}\t{source}\t{log}")
    lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")

    if args.write_model_top1_overrides:
        if args.targets_jsonl is None:
            raise SystemExit("--targets-jsonl is required for override target output")
        targets = load_targets(args.targets_jsonl)
        output = out_dir / "model_top1_search_override_targets.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in subsets["model_top1_engine_not_model"]:
                target_id = str(row["id"])
                handle.write(targets[target_id])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
