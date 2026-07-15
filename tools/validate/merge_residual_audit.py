#!/usr/bin/env python3
"""Merge subject residuals with Enyo raw/runtime activation audit rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    var = mean([(x - mx) ** 2 for x in xs])
    return mean([(x - mx) * (y - my) for x, y in zip(xs, ys)]) / max(var, 1e-12)


def group(rows: list[dict], key: str) -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    for name, items in sorted(buckets.items()):
        target = [float(item["stockfish"]) for item in items]
        candidate = [float(item["candidate"]) for item in items]
        errors = [c - t for c, t in zip(candidate, target)]
        print(
            f"{key}={name:12} rows={len(items):6d}"
            f" mae={mean([abs(e) for e in errors]):8.2f}"
            f" slope={slope(target, candidate):8.4f}"
            f" raw_mean={mean([float(item['raw']) for item in items]):8.2f}"
            f" runtime_mean={mean([float(item['scaled']) for item in items]):8.2f}"
            f" clamp={sum(bool(item['clamped_flag']) for item in items):5d}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structural", required=True, type=Path)
    ap.add_argument("--activation", required=True, type=Path)
    ap.add_argument("--candidate", default="enyo")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    structural = json.loads(args.structural.read_text())
    activation = json.loads(args.activation.read_text())
    scores = structural["scores"]
    subjects = {item["name"] for item in structural["subjects"]}
    if args.candidate not in subjects or args.stockfish not in subjects:
        raise SystemExit("structural audit lacks candidate or Stockfish subject")

    by_fen = {row["fen"]: row for row in activation["rows"]}
    rows = []
    for index, fen in enumerate(structural["fens"]):
        act = by_fen.get(fen)
        if act is None:
            continue
        feature = structural["features"][index]
        rows.append({
            "fen": fen,
            "candidate": scores[args.candidate][index],
            "stockfish": scores[args.stockfish][index],
            "material_bucket": feature["material_bucket"],
            "output_bucket": feature["output_bucket"],
            "eval_bucket": (
                "000-049" if abs(scores[args.stockfish][index]) < 50 else
                "050-099" if abs(scores[args.stockfish][index]) < 100 else
                "100-299" if abs(scores[args.stockfish][index]) < 300 else
                "300-799" if abs(scores[args.stockfish][index]) < 800 else
                "800+"
            ),
            **{key: act[key] for key in ("raw", "scaled", "clamped", "clamped_flag",
                                          "phase_scale", "output_bucket")},
        })

    if len(rows) < len(structural["fens"]):
        print(f"activation_rows_missing={len(structural['fens']) - len(rows)}")
    print(f"rows={len(rows)}")
    for key in ("material_bucket", "eval_bucket", "output_bucket"):
        group(rows, key)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
