#!/usr/bin/env python3
"""Fit simple static eval transforms between subjects from structural_net_audit JSON."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fit:
    scale: float
    bias: float


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def rmse(values: list[float]) -> float:
    return math.sqrt(mean([v * v for v in values])) if values else 0.0


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = mean([(x - mx) ** 2 for x in xs])
    vy = mean([(y - my) ** 2 for y in ys])
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)])
    return cov / math.sqrt(vx * vy)


def fit_linear(xs: list[float], ys: list[float], *, bias: bool) -> Fit:
    if not xs:
        return Fit(1.0, 0.0)
    if not bias:
        denom = sum(x * x for x in xs)
        return Fit(sum(x * y for x, y in zip(xs, ys, strict=True)) / denom if denom else 1.0, 0.0)
    mx = mean(xs)
    my = mean(ys)
    var = mean([(x - mx) ** 2 for x in xs])
    if var <= 0.0:
        return Fit(1.0, my - mx)
    cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)])
    scale = cov / var
    return Fit(scale, my - scale * mx)


def material_bucket(feature: dict[str, object]) -> str:
    return str(feature.get("material_bucket", "?"))


def eval_bucket(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 50:
        return "000-049"
    if abs_value < 100:
        return "050-099"
    if abs_value < 300:
        return "100-299"
    if abs_value < 800:
        return "300-799"
    return "800+"


def summarize(label: str, xs: list[float], ys: list[float], fit: Fit) -> dict[str, float | int | str]:
    pred = [fit.scale * x + fit.bias for x in xs]
    before = [x - y for x, y in zip(xs, ys, strict=True)]
    after = [p - y for p, y in zip(pred, ys, strict=True)]
    return {
        "label": label,
        "rows": len(xs),
        "scale": fit.scale,
        "bias": fit.bias,
        "mae_before": mean([abs(v) for v in before]),
        "mae_after": mean([abs(v) for v in after]),
        "rmse_before": rmse(before),
        "rmse_after": rmse(after),
        "corr": corr(xs, ys),
        "source_abs_ge_2000": sum(abs(x) >= 2000 for x in xs),
        "target_abs_ge_2000": sum(abs(y) >= 2000 for y in ys),
    }


def print_summary(row: dict[str, float | int | str]) -> None:
    print(
        f"{row['label']:18} rows={row['rows']:6d}"
        f" scale={row['scale']:8.4f} bias={row['bias']:8.2f}"
        f" mae {row['mae_before']:8.2f}->{row['mae_after']:8.2f}"
        f" rmse {row['rmse_before']:8.2f}->{row['rmse_after']:8.2f}"
        f" corr={row['corr']:8.4f}"
        f" clip {row['source_abs_ge_2000']}/{row['target_abs_ge_2000']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--source", default=None, help="source subject; defaults to first subject")
    ap.add_argument("--target", action="append", default=[], help="target subject; defaults to all non-source subjects")
    ap.add_argument("--no-bias", action="store_true", help="fit scale only through origin")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    scores: dict[str, list[float]] = {k: [float(v) for v in vals] for k, vals in payload["scores"].items()}
    subjects = list(scores)
    source = args.source or subjects[0]
    targets = args.target or [name for name in subjects if name != source]
    features = payload.get("features", [{} for _ in scores[source]])

    results = []
    for target in targets:
        xs = scores[source]
        ys = scores[target]
        global_fit = fit_linear(xs, ys, bias=not args.no_bias)
        row = summarize(f"{target} global", xs, ys, global_fit)
        results.append(row)
        print_summary(row)

        print(f"{target} by material")
        for bucket in sorted({material_bucket(f) for f in features}):
            idx = [i for i, f in enumerate(features) if material_bucket(f) == bucket]
            bx = [xs[i] for i in idx]
            by = [ys[i] for i in idx]
            fit = fit_linear(bx, by, bias=not args.no_bias)
            row = summarize(bucket, bx, by, fit)
            results.append({"target": target, "group": "material", **row})
            print_summary(row)

        print(f"{target} by source eval")
        for bucket in sorted({eval_bucket(x) for x in xs}):
            idx = [i for i, x in enumerate(xs) if eval_bucket(x) == bucket]
            bx = [xs[i] for i in idx]
            by = [ys[i] for i in idx]
            fit = fit_linear(bx, by, bias=not args.no_bias)
            row = summarize(bucket, bx, by, fit)
            results.append({"target": target, "group": "source_eval", **row})
            print_summary(row)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"source": source, "results": results}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
