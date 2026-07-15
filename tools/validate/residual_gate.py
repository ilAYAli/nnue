#!/usr/bin/env python3
"""Gate candidate promotion on stratified residual improvement."""

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


def eval_bucket(value: float) -> str:
    value = abs(value)
    if value < 50:
        return "000-049"
    if value < 100:
        return "050-099"
    if value < 300:
        return "100-299"
    if value < 800:
        return "300-799"
    return "800+"


def audit_rows(payload: dict) -> list[dict]:
    if "rows" in payload:
        return payload["rows"]
    scores = payload["scores"]
    features = payload["features"]
    required = ("reference", "candidate", "stockfish")
    missing = [name for name in required if name not in scores]
    if missing:
        raise ValueError(f"audit is missing score subjects: {missing}")
    return [
        {
            "reference": scores["reference"][index],
            "candidate": scores["candidate"][index],
            "stockfish": scores["stockfish"][index],
            "material_bucket": feature["material_bucket"],
            "output_bucket": feature["output_bucket"],
        }
        for index, feature in enumerate(features)
    ]


def metrics(rows: list[dict], key: str) -> dict[str, float]:
    target = [float(row["stockfish"]) for row in rows]
    values = [float(row[key]) for row in rows]
    errors = [value - expected for value, expected in zip(values, target)]
    return {
        "rows": len(rows),
        "mae": mean([abs(error) for error in errors]),
        "slope": slope(target, values),
        "slope_distance": abs(slope(target, values) - 1.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--min-improvement", type=float, default=0.0)
    args = ap.parse_args()
    rows = audit_rows(json.loads(args.audit.read_text()))

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"phase:{row['material_bucket']}"].append(row)
        groups[f"eval:{eval_bucket(row['stockfish'])}"].append(row)
        groups[f"output:{row['output_bucket']}"].append(row)

    required = ("phase:endgame", "eval:800+", "eval:300-799")
    failed = False
    for name in required:
        group = groups.get(name, [])
        if not group:
            print(f"residual_gate {name}: missing rows", flush=True)
            failed = True
            continue
        baseline = metrics(group, "reference")
        candidate = metrics(group, "candidate")
        mae_gain = baseline["mae"] - candidate["mae"]
        slope_gain = baseline["slope_distance"] - candidate["slope_distance"]
        print(
            f"residual_gate {name} rows={len(group)} "
            f"mae={baseline['mae']:.3f}->{candidate['mae']:.3f} "
            f"slope={baseline['slope']:.6f}->{candidate['slope']:.6f} "
            f"mae_gain={mae_gain:.3f} slope_gain={slope_gain:.6f}",
            flush=True,
        )
        if mae_gain <= args.min_improvement or slope_gain <= args.min_improvement:
            failed = True

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
