#!/usr/bin/env python3
"""Evaluate exported sidecar move-policy JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.move_policy_features import move_features  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def dot(row: list[float], values: list[float]) -> float:
    return sum(a * b for a, b in zip(row, values))


def layer_forward(x: list[float], layer: dict[str, Any]) -> list[float]:
    out = [
        dot(weights, x) + float(bias)
        for weights, bias in zip(layer["weight"], layer["bias"])
    ]
    if layer.get("activation") == "relu":
        out = [max(0.0, item) for item in out]
    return out


def load_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema") != "enyo.move_policy.v1":
        raise ValueError(f"{path}: unsupported schema {model.get('schema')}")
    return model


def score(model: dict[str, Any], fen: str, move: str) -> float:
    feature_set = str(model.get("feature_set", "compact"))
    x = move_features(fen, move, feature_set)
    mean = model["mean"]
    std = model["std"]
    if len(x) != len(mean) or len(x) != len(std):
        raise ValueError(
            f"feature dimension {len(x)} does not match model {len(mean)}")
    y = [(value - m) / s for value, m, s in zip(x, mean, std)]
    for layer in model["layers"]:
        y = layer_forward(y, layer)
    if len(y) != 1:
        raise ValueError(f"expected scalar output, got {len(y)}")
    return float(y[0])


def score_many(model: dict[str, Any], fen: str, moves: list[str]) -> list[float]:
    try:
        import numpy as np
    except ImportError:
        return [score(model, fen, move) for move in moves]

    feature_set = str(model.get("feature_set", "compact"))
    rows = [move_features(fen, move, feature_set) for move in moves]
    if not rows:
        return []
    x = np.asarray(rows, dtype=np.float32)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    if x.shape[1] != mean.shape[0] or x.shape[1] != std.shape[0]:
        raise ValueError(
            f"feature dimension {x.shape[1]} does not match model {mean.shape[0]}")
    y = (x - mean) / std
    for layer in model["layers"]:
        weight = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        y = y @ weight.T + bias
        if layer.get("activation") == "relu":
            y = np.maximum(y, 0.0)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"expected scalar output, got shape {y.shape}")
    return [float(item) for item in y[:, 0]]


def eval_gate(model: dict[str, Any], rows: list[dict[str, Any]],
              thresholds: list[float]) -> dict[str, Any]:
    margins = [
        score(model, str(row["fen"]), str(row["best"]))
        - score(model, str(row["fen"]), str(row["played"]))
        for row in rows
    ]
    out = []
    for threshold in thresholds:
        selected = sum(1 for margin in margins if margin >= threshold)
        correct = sum(1 for margin in margins if margin > 0)
        out.append({
            "threshold": threshold,
            "selected": selected,
            "correct": correct,
            "cases": len(rows),
        })
        print(
            f"gate threshold={threshold:g} selected={selected}/{len(rows)} "
            f"correct={correct}/{len(rows)}",
            flush=True,
        )
    return {"cases": len(rows), "thresholds": out}


def eval_guard(model: dict[str, Any], rows: list[dict[str, Any]],
               thresholds: list[float], bad_tolerance_cp: float,
               good_tolerance_cp: float) -> dict[str, Any]:
    per_row: list[tuple[float, float]] = []
    for row in rows:
        current_score = score(model, str(row["fen"]), str(row["current"]))
        best_margin = None
        best_delta = 0.0
        for alt in row.get("alternatives", []):
            margin = score(model, str(row["fen"]), str(alt["move"])) - current_score
            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_delta = float(alt.get("delta_cp", 0.0))
        if best_margin is not None:
            per_row.append((best_margin, best_delta))

    out = []
    for threshold in thresholds:
        overrides = harmful = neutral = helpful = 0
        for margin, delta in per_row:
            if margin < threshold:
                continue
            overrides += 1
            if delta < -bad_tolerance_cp:
                harmful += 1
            elif delta > good_tolerance_cp:
                helpful += 1
            else:
                neutral += 1
        out.append({
            "threshold": threshold,
            "overrides": overrides,
            "harmful": harmful,
            "neutral": neutral,
            "helpful": helpful,
            "cases": len(rows),
        })
        print(
            f"guard threshold={threshold:g} overrides={overrides}/{len(rows)} "
            f"helpful={helpful} neutral={neutral} harmful={harmful}",
            flush=True,
        )
    return {"cases": len(rows), "thresholds": out}


def parse_thresholds(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate")
    ap.add_argument("--guard")
    ap.add_argument("--thresholds", default="18")
    ap.add_argument("--bad-tolerance-cp", type=float, default=10.0)
    ap.add_argument("--good-tolerance-cp", type=float, default=10.0)
    ap.add_argument("--summary-json")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    thresholds = parse_thresholds(args.thresholds)
    summary: dict[str, Any] = {"model": args.model}
    if args.gate:
        summary["gate"] = eval_gate(model, read_jsonl(Path(args.gate)), thresholds)
    if args.guard:
        summary["guard"] = eval_guard(
            model,
            read_jsonl(Path(args.guard)),
            thresholds,
            args.bad_tolerance_cp,
            args.good_tolerance_cp,
        )
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
