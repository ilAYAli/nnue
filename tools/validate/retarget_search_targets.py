#!/usr/bin/env python3
"""Rewrite search-aware targets to use engine-selected moves as targets."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_reference_moves(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["id"]: row["engine_move"].lower()
            for row in csv.DictReader(handle)
            if row.get("id") and row.get("engine_move")
        }


def soft_policy(gaps: list[float], temperature_cp: float) -> list[float]:
    if not gaps:
        return []
    if temperature_cp <= 0:
        return [1.0] + [0.0] * (len(gaps) - 1)
    values = [-max(0.0, gap) / temperature_cp for gap in gaps]
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def retarget_row(row: dict[str, object], target_move: str,
                 *, gap_floor: int, temperature_cp: float) -> dict[str, object] | None:
    moves = list(row.get("moves", []))
    by_uci = {str(move.get("uci", "")).lower(): move for move in moves}
    target = by_uci.get(target_move)
    if target is None:
        return None

    target_score = int(target.get("score_cp", 0))
    rewritten = []
    for move in moves:
        uci = str(move.get("uci", "")).lower()
        score = int(move.get("score_cp", 0))
        out = dict(move)
        flags = set(str(flag) for flag in out.get("flags", []))
        if uci == target_move:
            out["rank"] = 1
            out["gap_cp"] = 0
            flags.add("reference")
            flags.add("target")
        else:
            out["rank"] = max(2, int(out.get("rank", 999)))
            out["gap_cp"] = max(gap_floor, abs(target_score - score))
        out["flags"] = sorted(flags)
        rewritten.append(out)

    rewritten.sort(key=lambda item: (int(item.get("rank", 999)), int(item.get("gap_cp", 0))))
    policies = soft_policy(
        [float(move.get("gap_cp", 0.0)) for move in rewritten],
        temperature_cp,
    )
    for move, policy in zip(rewritten, policies):
        move["policy"] = round(policy, 8)

    out_row = dict(row)
    tags = set(str(tag) for tag in out_row.get("tags", []))
    tags.add("reference_distill")
    out_row["tags"] = sorted(tags)
    out_row["best_move"] = target_move
    out_row["reference_moves"] = [target_move]
    out_row["moves"] = rewritten
    return out_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--gap-floor-cp", type=int, default=25)
    parser.add_argument("--policy-temperature-cp", type=float, default=200.0)
    args = parser.parse_args()

    reference_moves = read_reference_moves(args.reference_csv)
    total = 0
    written = 0
    missing_reference = 0
    missing_move = 0

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.targets.expanduser().open(encoding="utf-8") as src, \
            args.output.expanduser().open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            target_id = str(row.get("id", ""))
            target_move = reference_moves.get(target_id)
            if not target_move:
                missing_reference += 1
                continue
            rewritten = retarget_row(
                row,
                target_move,
                gap_floor=args.gap_floor_cp,
                temperature_cp=args.policy_temperature_cp,
            )
            if rewritten is None:
                missing_move += 1
                continue
            dst.write(json.dumps(rewritten, sort_keys=True) + "\n")
            written += 1

    lines = [
        f"targets={total}",
        f"written={written}",
        f"missing_reference={missing_reference}",
        f"missing_move={missing_move}",
        f"gap_floor_cp={args.gap_floor_cp}",
        f"policy_temperature_cp={args.policy_temperature_cp:g}",
        f"output={args.output.expanduser()}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.summary:
        args.summary.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.summary.expanduser().write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
